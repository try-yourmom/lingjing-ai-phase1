from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torchvision.models import ResNet18_Weights, resnet18 as torchvision_resnet18

from src.dataset import DEFAULT_MANIFEST_PATH, build_dataloaders
from src.train import (
    evaluate,
    save_confusion_matrix,
    save_training_curves,
    set_seed,
    train_one_epoch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["TORCH_HOME"] = str(PROJECT_ROOT / ".cache" / "torch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer learning ResNet18")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--run-name", type=str, default="transfer_resnet18")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def build_pretrained_model(num_classes: int) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = torchvision_resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model_dir = PROJECT_ROOT / "models"
    report_dir = PROJECT_ROOT / "reports" / "training" / args.run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        manifest_path=args.manifest_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    class_names = train_loader.dataset.classes
    model = build_pretrained_model(len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01,
    )

    config = vars(args).copy()
    config["manifest_path"] = str(args.manifest_path)
    config["device"] = str(device)
    config["class_names"] = class_names
    config["pretrained_weights"] = "IMAGENET1K_V1"
    (report_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    history: list[dict[str, float]] = []
    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    best_checkpoint_path = model_dir / f"{args.run_name}_best.pt"

    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
    print("pretrained weights: IMAGENET1K_V1")
    print("train size:", len(train_loader.dataset))
    print("val size:", len(val_loader.dataset))
    print("test size:", len(test_loader.dataset))
    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=0,
        )
        val_metrics, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=0,
        )
        scheduler.step()

        elapsed = time.perf_counter() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "elapsed_seconds": elapsed,
        }
        history.append(row)

        print(
            f"epoch {epoch:02d}/{args.epochs:02d} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"train_f1={row['train_macro_f1']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} "
            f"val_f1={row['val_macro_f1']:.4f} "
            f"lr={current_lr:.6f} "
            f"time={elapsed:.1f}s"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "config": config,
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        history_df = pd.DataFrame(history)
        history_df.to_csv(
            report_dir / "history.csv",
            index=False,
            encoding="utf-8-sig",
        )
        save_training_curves(history_df, report_dir / "training_curves.png")

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break
    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, test_targets, test_predictions = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        max_batches=0,
    )

    test_report = classification_report(
        test_targets,
        test_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(test_report).transpose().to_csv(
        report_dir / "classification_report.csv",
        encoding="utf-8-sig",
    )

    save_confusion_matrix(
        targets=test_targets,
        predictions=test_predictions,
        class_names=class_names,
        output_path=report_dir / "confusion_matrix.png",
    )

    final_summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "checkpoint": str(best_checkpoint_path),
        "pretrained_weights": "IMAGENET1K_V1",
    }
    (report_dir / "final_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("transfer learning complete")
    print("best epoch:", best_epoch)
    print("best val macro f1:", round(best_macro_f1, 6))
    print("test accuracy:", round(test_metrics["accuracy"], 6))
    print("test macro f1:", round(test_metrics["macro_f1"], 6))
    print("checkpoint:", best_checkpoint_path)
    print("report dir:", report_dir)


if __name__ == "__main__":
    main()