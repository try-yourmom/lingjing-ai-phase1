from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = PROJECT_ROOT / "reports" / "data_audit"
ANALYSIS_ROOT = PROJECT_ROOT / "reports" / "data_analysis"

RAW_COUNTS_PATH = AUDIT_ROOT / "class_counts_raw.csv"
CLEAN_COUNTS_PATH = AUDIT_ROOT / "class_counts_clean.csv"
IMAGE_INFO_PATH = AUDIT_ROOT / "image_info.csv"
DUPLICATES_PATH = AUDIT_ROOT / "duplicates.csv"
CORRUPT_PATH = AUDIT_ROOT / "corrupt_images.csv"
NEAR_DUPLICATES_PATH = AUDIT_ROOT / "near_duplicates.csv"
MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "plantvillage8_manifest.csv"
)

SPLIT_ORDER = ["train", "val", "test"]

MANIFEST_COLUMNS = ["path", "label", "split", "sha256", "width", "height"]
IMAGE_INFO_COLUMNS = [
    "path", "label", "sha256", "width", "height", "mode", "format",
]
NEAR_DUPLICATES_COLUMNS = [
    "kept_path",
    "kept_label",
    "duplicate_path",
    "duplicate_label",
    "hamming_distance",
    "kept_dhash",
    "duplicate_dhash",
    "kept_split",
    "duplicate_split",
]

def read_csv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def percentage(value: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(value / total * 100, 2)


def array_stats(values: np.ndarray) -> dict[str, int | float]:
    """返回数组的 min/max/mean/median；空数组返回 0，避免 .min() 崩溃。"""
    if values.size == 0:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": round(float(values.mean()), 2),
        "median": round(float(np.median(values)), 2),
    }


def main() -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)

    raw_counts = pd.read_csv(RAW_COUNTS_PATH)
    clean_counts = pd.read_csv(CLEAN_COUNTS_PATH)
    image_info = read_csv_or_empty(IMAGE_INFO_PATH, IMAGE_INFO_COLUMNS)

    duplicates = read_csv_or_empty(
        DUPLICATES_PATH,
        [
            "sha256",
            "kept_path",
            "kept_label",
            "duplicate_path",
            "duplicate_label",
        ],
    )

    corrupt = read_csv_or_empty(
        CORRUPT_PATH,
        ["path", "label", "error"],
    )

    near_duplicates = read_csv_or_empty(
        NEAR_DUPLICATES_PATH, NEAR_DUPLICATES_COLUMNS
    )

    manifest = read_csv_or_empty(MANIFEST_PATH, MANIFEST_COLUMNS)

    class_order = clean_counts["label"].tolist()
    raw_counts = raw_counts.set_index("label").reindex(class_order)
    clean_counts = clean_counts.set_index("label").reindex(class_order)

    split_table = (
        manifest.groupby(["label", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=class_order, columns=SPLIT_ORDER, fill_value=0)
    )

    duplicate_by_label = (
        duplicates.groupby("duplicate_label")
        .size()
        .reindex(class_order, fill_value=0)
        .astype(int)
    )

    width_values = manifest["width"].to_numpy()
    height_values = manifest["height"].to_numpy()
    aspect_ratios = width_values / np.maximum(height_values, 1)

    width_stats = array_stats(width_values)
    height_stats = array_stats(height_values)
    aspect_stats = (
        {"min": 0.0, "max": 0.0, "mean": 0.0}
        if aspect_ratios.size == 0
        else {
            "min": round(float(aspect_ratios.min()), 4),
            "max": round(float(aspect_ratios.max()), 4),
            "mean": round(float(aspect_ratios.mean()), 4),
        }
    )

    raw_total = int(raw_counts["count"].sum())
    clean_total = int(clean_counts["count"].sum())
    duplicate_total = int(len(duplicates))
    corrupt_total = int(len(corrupt))
    cross_label_duplicates = 0

    if not duplicates.empty:
        cross_label_duplicates = int(
            (duplicates["kept_label"] != duplicates["duplicate_label"]).sum()
        )

    near_duplicate_total = int(len(near_duplicates))
    cross_label_near_duplicates = 0
    cross_split_near_duplicates = 0
    if not near_duplicates.empty:
        cross_label_near_duplicates = int(
            (near_duplicates["kept_label"] != near_duplicates["duplicate_label"]).sum()
        )
        cross_split_near_duplicates = int(
            (near_duplicates["kept_split"] != near_duplicates["duplicate_split"]).sum()
        )

    min_class_count = int(clean_counts["count"].min())
    max_class_count = int(clean_counts["count"].max())
    imbalance_ratio = round(max_class_count / max(min_class_count, 1), 3)

    split_counts = {
        split: int((manifest["split"] == split).sum())
        for split in SPLIT_ORDER
    }

    summary = {
        "raw_total": raw_total,
        "clean_total": clean_total,
        "removed_duplicates": duplicate_total,
        "corrupt_images": corrupt_total,
        "cross_label_duplicates": cross_label_duplicates,
        "removed_near_duplicates": near_duplicate_total,
        "cross_label_near_duplicates": cross_label_near_duplicates,
        "cross_split_near_duplicates": cross_split_near_duplicates,
        "class_counts_raw": {
            label: int(raw_counts.loc[label, "count"]) for label in class_order
        },
        "class_counts_clean": {
            label: int(clean_counts.loc[label, "count"]) for label in class_order
        },
        "duplicate_by_label": {
            label: int(duplicate_by_label.loc[label]) for label in class_order
        },
        "split_counts": split_counts,
        "split_percentages": {
            split: percentage(count, clean_total)
            for split, count in split_counts.items()
        },
        "imbalance_ratio": imbalance_ratio,
        "image_width": width_stats,
        "image_height": height_stats,
        "aspect_ratio": aspect_stats,
        "color_modes": image_info["mode"].value_counts().to_dict(),
        "image_formats": image_info["format"].value_counts().to_dict(),
    }

    summary_path = ANALYSIS_ROOT / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    split_table.to_csv(
        ANALYSIS_ROOT / "split_by_class.csv",
        encoding="utf-8-sig",
    )
    duplicate_by_label.rename("duplicate_count").to_csv(
        ANALYSIS_ROOT / "duplicates_by_class.csv",
        encoding="utf-8-sig",
    )

    x = np.arange(len(class_order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(x - width / 2, raw_counts["count"], width, label="Raw")
    ax.bar(x + width / 2, clean_counts["count"], width, label="Clean")
    ax.set_title("Class Distribution Before and After Deduplication")
    ax.set_ylabel("Image Count")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label.replace("___", "\n") for label in class_order],
        rotation=20,
        ha="right",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_ROOT / "class_distribution.png", dpi=200)
    plt.close(fig)

    bottom = np.zeros(len(class_order))
    colors = {
        "train": "#3B82F6",
        "val": "#F59E0B",
        "test": "#EF4444",
    }

    fig, ax = plt.subplots(figsize=(15, 7))
    for split in SPLIT_ORDER:
        values = split_table[split].to_numpy()
        ax.bar(
            np.arange(len(class_order)),
            values,
            bottom=bottom,
            label=split,
            color=colors[split],
        )
        bottom += values

    ax.set_title("Train / Validation / Test Split by Class")
    ax.set_ylabel("Image Count")
    ax.set_xticks(np.arange(len(class_order)))
    ax.set_xticklabels(
        [label.replace("___", "\n") for label in class_order],
        rotation=20,
        ha="right",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_ROOT / "split_by_class.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].scatter(
        width_values,
        height_values,
        alpha=0.28,
        s=10,
        color="#2563EB",
    )
    axes[0].set_title("Image Width vs Height")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Height")
    axes[0].grid(alpha=0.2)

    axes[1].hist(
        aspect_ratios,
        bins=30,
        color="#10B981",
        alpha=0.85,
    )
    axes[1].set_title("Aspect Ratio Distribution")
    axes[1].set_xlabel("Width / Height")
    axes[1].set_ylabel("Image Count")
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(ANALYSIS_ROOT / "image_geometry.png", dpi=200)
    plt.close(fig)

    recommendations: list[str] = []

    if imbalance_ratio >= 1.5:
        recommendations.append(
            f"类别数量不平衡，最多与最少类别之比为 {imbalance_ratio:.3f}。"
            "训练时建议记录 Macro-F1，并考虑 class weight 或 WeightedRandomSampler。"
        )
    else:
        recommendations.append(
            f"类别数量相对均衡，最多与最少类别之比为 {imbalance_ratio:.3f}。"
        )

    if cross_label_duplicates > 0:
        recommendations.append(
            f"发现 {cross_label_duplicates} 张重复图片具有不同标签，"
            "需要人工抽查，避免标签冲突和训练集泄漏。"
        )
    else:
        recommendations.append(
            "未发现跨标签重复图片，当前去重策略没有明显标签冲突。"
        )

    if near_duplicate_total > 0:
        recommendations.append(
            f"检测并移除 {near_duplicate_total} 张近重复图片，其中 "
            f"{cross_split_near_duplicates} 对原本会落在不同划分，"
            "已避免训练/验证/测试之间的数据泄漏。"
        )
        if cross_label_near_duplicates > 0:
            recommendations.append(
                f"其中 {cross_label_near_duplicates} 对近重复图片标签不一致，"
                "建议人工抽查是否存在标注错误。"
            )
    else:
        recommendations.append("未发现近重复图片。")

    recommendations.append(
        "训练图像建议统一缩放到 160x160 或 224x224，"
        "训练集使用随机裁剪和翻转，验证集与测试集只做确定性缩放。"
    )

    report_lines = [
        "# PlantVillage 8 类数据报告",
        "",
        "## 数据规模",
        "",
        f"- 原始图片：{raw_total}",
        f"- 清洗后图片：{clean_total}",
        f"- 删除重复图片：{duplicate_total}",
        f"- 删除近重复图片：{near_duplicate_total}",
        f"- 损坏图片：{corrupt_total}",
        f"- 跨标签重复图片：{cross_label_duplicates}",
        f"- 跨标签近重复图片：{cross_label_near_duplicates}",
        "",
        "## 数据划分",
        "",
        f"- 训练集：{split_counts['train']}，"
        f"占 {percentage(split_counts['train'], clean_total)}%",
        f"- 验证集：{split_counts['val']}，"
        f"占 {percentage(split_counts['val'], clean_total)}%",
        f"- 测试集：{split_counts['test']}，"
        f"占 {percentage(split_counts['test'], clean_total)}%",
        "",
        "## 图片属性",
        "",
        f"- 宽度：{summary['image_width']}",
        f"- 高度：{summary['image_height']}",
        f"- 宽高比：{summary['aspect_ratio']}",
        f"- 颜色模式：{summary['color_modes']}",
        f"- 图片格式：{summary['image_formats']}",
        "",
        "## 分析与建议",
        "",
    ]

    report_lines.extend(f"- {item}" for item in recommendations)

    report_path = ANALYSIS_ROOT / "analysis_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("数据分析完成")
    print(f"原始图片：{raw_total}")
    print(f"清洗后图片：{clean_total}")
    print(f"删除重复图片：{duplicate_total}")
    print(f"删除近重复图片：{near_duplicate_total}")
    print(f"损坏图片：{corrupt_total}")
    print(f"跨标签重复：{cross_label_duplicates}")
    print(f"跨标签近重复：{cross_label_near_duplicates}")
    print(f"类别不平衡比例：{imbalance_ratio}")
    print(f"训练集：{split_counts['train']}")
    print(f"验证集：{split_counts['val']}")
    print(f"测试集：{split_counts['test']}")
    print(f"分析报告：{report_path}")


if __name__ == "__main__":
    main()