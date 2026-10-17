from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import DEFAULT_MANIFEST_PATH, build_dataloaders
from src.model import resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "resnet18_best.pt"
REPORT_ROOT = PROJECT_ROOT / "reports" / "error_analysis" / "resnet18"


def load_model(device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    class_names = checkpoint["class_names"]
    model = resnet18(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    sample_offset = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        probabilities = F.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)

        batch_size = targets.size(0)
        for batch_index in range(batch_size):
            sample_index = sample_offset + batch_index
            image_path, _, label_name = loader.dataset.samples[sample_index]

            records.append(
                {
                    "path": str(image_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "true_index": int(targets[batch_index].item()),
                    "true_label": label_name,
                    "pred_index": int(predictions[batch_index].item()),
                    "pred_label": loader.dataset.classes[int(predictions[batch_index].item())],
                    "confidence": float(confidences[batch_index].item()),
                    "correct": bool(predictions[batch_index].item() == targets[batch_index].item()),
                }
            )

        sample_offset += batch_size

    return records


def save_error_grid(errors: pd.DataFrame, output_path: Path) -> None:
    selected = errors.head(16)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()

    for axis, (_, row) in zip(axes, selected.iterrows()):
        image_path = PROJECT_ROOT / str(row["path"])
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(
            f"true: {row['true_label']}\npred: {row['pred_label']}\n"
            f"conf: {row['confidence']:.3f}",
            fontsize=8,
        )
        axis.axis("off")

    for axis in axes[len(selected):]:
        axis.axis("off")

    fig.suptitle("Top Misclassified Test Images", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(device)

    _, _, test_loader = build_dataloaders(
        manifest_path=DEFAULT_MANIFEST_PATH,
        batch_size=64,
        num_workers=0,
        image_size=224,
    )

    records = collect_predictions(model, test_loader, device)
    dataframe = pd.DataFrame(records)

    errors = dataframe[~dataframe["correct"]].copy()
    errors = errors.sort_values("confidence", ascending=False)

    dataframe.to_csv(
        REPORT_ROOT / "all_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    errors.to_csv(
        REPORT_ROOT / "misclassified.csv",
        index=False,
        encoding="utf-8-sig",
    )

    confusion_pairs = (
        errors.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    confusion_pairs.to_csv(
        REPORT_ROOT / "confusion_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    class_accuracy = (
        dataframe.groupby("true_label")["correct"]
        .agg(["sum", "count"])
        .reset_index()
    )
    class_accuracy["accuracy"] = class_accuracy["sum"] / class_accuracy["count"]
    class_accuracy.to_csv(
        REPORT_ROOT / "class_accuracy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_error_grid(errors, REPORT_ROOT / "top_errors.png")

    summary = {
        "test_samples": int(len(dataframe)),
        "correct": int(dataframe["correct"].sum()),
        "errors": int(len(errors)),
        "accuracy": float(dataframe["correct"].mean()),
        "mean_error_confidence": float(errors["confidence"].mean()),
        "top_confusion_pairs": confusion_pairs.head(5).to_dict(orient="records"),
        "worst_classes": class_accuracy.sort_values("accuracy").head(3).to_dict(orient="records"),
    }
    (REPORT_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# ResNet18 错误分析",
        "",
        "## 总体情况",
        "",
        f"- 测试图片数：{summary['test_samples']}",
        f"- 正确数量：{summary['correct']}",
        f"- 错误数量：{summary['errors']}",
        f"- 测试准确率：{summary['accuracy']:.4f}",
        f"- 错误样本平均置信度：{summary['mean_error_confidence']:.4f}",
        "",
        "## 主要混淆组合",
        "",
        "| 真实类别 | 预测类别 | 数量 |",
        "| --- | --- | ---: |",
    ]

    for row in confusion_pairs.head(10).itertuples(index=False):
        lines.append(f"| {row.true_label} | {row.pred_label} | {row.count} |")

    lines.extend(
        [
            "",
            "## 表现最差的类别",
            "",
            "| 类别 | 正确数 | 总数 | 准确率 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    for row in class_accuracy.sort_values("accuracy").head(5).itertuples(index=False):
        lines.append(
            f"| {row.true_label} | {row.sum} | {row.count} | {row.accuracy:.4f} |"
        )

    (REPORT_ROOT / "error_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("report dir:", REPORT_ROOT)


if __name__ == "__main__":
    main()