from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from src.dataset import DEFAULT_MANIFEST_PATH, build_dataloaders
from src.model import resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train handwritten ResNet18")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--run-name", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def limit_reached(batch_index: int, limit: int) -> bool:
    return limit > 0 and batch_index >= limit


def compute_classification_metrics(
    targets: list[int],
    predictions: list[int],
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for batch_index, (images, targets) in enumerate(loader):
        if limit_reached(batch_index, max_batches):
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        predictions = outputs.argmax(dim=1)
        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    metrics = compute_classification_metrics(all_targets, all_predictions)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for batch_index, (images, targets) in enumerate(loader):
        if limit_reached(batch_index, max_batches):
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)
        predictions = outputs.argmax(dim=1)
        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    metrics = compute_classification_metrics(all_targets, all_predictions)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics, all_targets, all_predictions


def save_training_curves(history: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(history["epoch"], history["train_macro_f1"], label="train")
    axes[2].plot(history["epoch"], history["val_macro_f1"], label="val")
    axes[2].set_title("Macro F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix(
    targets: list[int],
    predictions: list[int],
    class_names: list[str],
    output_path: Path,
) -> None:
    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(len(class_names))),
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title("Test Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

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
    model = resnet18(num_classes=len(class_names)).to(device)

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
    print("classes:", class_names)
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
            max_batches=args.max_train_batches,
        )
        val_metrics, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches,
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
    }
    (report_dir / "final_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("training complete")
    print("best epoch:", best_epoch)
    print("best val macro f1:", round(best_macro_f1, 6))
    print("test accuracy:", round(test_metrics["accuracy"], 6))
    print("test macro f1:", round(test_metrics["macro_f1"], 6))
    print("checkpoint:", best_checkpoint_path)
    print("report dir:", report_dir)


if __name__ == "__main__":
    main()