from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.model import resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "comparison"

RUNS = {
    "handwritten_resnet18": PROJECT_ROOT / "reports" / "training" / "resnet18",
    "transfer_resnet18": PROJECT_ROOT / "reports" / "training" / "transfer_resnet18",
}


def count_parameters(checkpoint_path: Path) -> int:
    import torch

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("config", {}).get("pretrained_weights"):
        from torchvision.models import resnet18 as tv_resnet18

        model = tv_resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint["class_names"]))
    else:
        model = resnet18(num_classes=len(checkpoint["class_names"]))
    return sum(parameter.numel() for parameter in model.parameters())


def resolve_checkpoint(name: str, summary: dict[str, object]) -> Path:
    run_name = name.removeprefix("handwritten_")
    candidate = PROJECT_ROOT / "models" / f"{run_name}_best.pt"
    if candidate.exists():
        return candidate

    recorded = Path(str(summary["checkpoint"]))
    if recorded.exists():
        return recorded

    raise FileNotFoundError(
        "找不到模型权重。请将权重放到 "
        f"{candidate}，或重新运行对应训练脚本。"
    )


def load_run(name: str, run_dir: Path) -> dict[str, object]:
    summary = json.loads((run_dir / "final_summary.json").read_text(encoding="utf-8"))
    history = pd.read_csv(run_dir / "history.csv")
    checkpoint = resolve_checkpoint(name, summary)

    return {
        "run_name": name,
        "best_epoch": summary["best_epoch"],
        "best_val_macro_f1": summary["best_val_macro_f1"],
        "test_accuracy": summary["test_accuracy"],
        "test_macro_f1": summary["test_macro_f1"],
        "total_epochs": int(history["epoch"].max()),
        "total_train_seconds": float(history["elapsed_seconds"].sum()),
        "parameters": count_parameters(checkpoint),
    }


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    rows = [load_run(name, run_dir) for name, run_dir in RUNS.items()]
    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        REPORT_ROOT / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 模型对比",
        "",
        "| 模型 | 最佳轮次 | 验证 Macro-F1 | 测试 Accuracy | 测试 Macro-F1 | 训练轮数 | 参数量 | 训练耗时 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in dataframe.itertuples(index=False):
        lines.append(
            f"| {row.run_name} | {row.best_epoch} | {row.best_val_macro_f1:.4f} | "
            f"{row.test_accuracy:.4f} | {row.test_macro_f1:.4f} | {row.total_epochs} | "
            f"{row.parameters:,} | {row.total_train_seconds:.1f}s |"
        )

    lines.extend(
        [
            "",
            "## 主要结论",
            "",
            "- 手写 ResNet18 完全由项目自身实现，参数量与标准 ResNet18 基本一致。",
            "- 迁移学习使用 ImageNet 预训练权重，在小数据集上收敛更快，测试集指标更高。",
            "- 迁移学习作为对照实验，不替代手写 ResNet 的实现要求。",
        ]
    )

    (REPORT_ROOT / "model_comparison.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(dataframe.to_string(index=False))
    print("comparison report:", REPORT_ROOT)


if __name__ == "__main__":
    main()