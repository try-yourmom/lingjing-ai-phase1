from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "plantvillage"
    / "plantvillage dataset"
    / "color"
)
REPORT_ROOT = PROJECT_ROOT / "reports" / "data_audit"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_ROOT / "plantvillage8_manifest.csv"

TARGET_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_SEED = 42
# 感知哈希近重复判定阈值：64 位 dHash 的汉明距离 <= 该值即视为近重复。
NEAR_DUP_HAMMING_THRESHOLD = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """返回图片的 64 位差异哈希（dHash），用于近重复检测。

    将灰度图缩放到 (hash_size + 1) x hash_size，逐行比较相邻像素，
    得到 hash_size * hash_size 个比特。不同压缩/缩放的同一张图距离通常很小。
    """
    grayscale = image.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(grayscale, dtype=np.uint8)
    diff = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    """两个等长整数之间置位不同的比特数。"""
    return (left ^ right).bit_count()


def inspect_image(path: Path) -> dict[str, str | int | None]:
    with Image.open(path) as image:
        image.verify()

    with Image.open(path) as image:
        width, height = image.size
        return {
            "width": width,
            "height": height,
            "mode": image.mode,
            "format": image.format,
            "dhash": dhash(image),
        }


def relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def main() -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"找不到原始数据目录：{RAW_ROOT}")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    raw_counts: dict[str, int] = {}
    valid_records: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    corrupt_rows: list[dict[str, object]] = []

    print(f"原始数据目录：{RAW_ROOT}")

    for label in TARGET_CLASSES:
        class_dir = RAW_ROOT / label
        if not class_dir.exists():
            raise FileNotFoundError(f"缺少类别目录：{class_dir}")

        image_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        raw_counts[label] = len(image_paths)
        print(f"检查 {label}: {len(image_paths)} 张图片")

        for image_path in image_paths:
            try:
                image_info = inspect_image(image_path)
            except Exception as error:
                corrupt_rows.append(
                    {
                        "path": relative_path(image_path),
                        "label": label,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue

            digest = sha256_file(image_path)
            record = {
                "path": relative_path(image_path),
                "label": label,
                "sha256": digest,
                **image_info,
            }
            valid_records.append(record)
            image_rows.append(record)

    unique_records: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []
    hash_to_record: dict[str, dict[str, object]] = {}

    for record in valid_records:
        digest = str(record["sha256"])
        if digest in hash_to_record:
            kept = hash_to_record[digest]
            duplicate_rows.append(
                {
                    "sha256": digest,
                    "kept_path": kept["path"],
                    "kept_label": kept["label"],
                    "duplicate_path": record["path"],
                    "duplicate_label": record["label"],
                }
            )
            continue

        hash_to_record[digest] = record
        unique_records.append(record)

    # 在精确去重的基础上，再做一次感知哈希近重复检测。
    # 贪心聚类：每条记录与已有簇的代表比对，距离超过阈值则另起新簇。
    clusters: list[list[dict[str, object]]] = []
    for record in unique_records:
        record_hash = int(record["dhash"])
        for cluster in clusters:
            if (
                hamming_distance(record_hash, int(cluster[0]["dhash"]))
                <= NEAR_DUP_HAMMING_THRESHOLD
            ):
                cluster.append(record)
                break
        else:
            clusters.append([record])

    near_duplicate_pairs: list[
        tuple[dict[str, object], dict[str, object], int]
    ] = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        kept = cluster[0]
        for duplicate in cluster[1:]:
            distance = hamming_distance(
                int(kept["dhash"]), int(duplicate["dhash"])
            )
            near_duplicate_pairs.append((kept, duplicate, distance))

    # 每个近重复簇只保留一个代表，其余作为近重复图片从干净集中移除。
    final_records = [cluster[0] for cluster in clusters]

    for label in TARGET_CLASSES:
        class_records = [
            record for record in unique_records if record["label"] == label
        ]
        random.Random(RANDOM_SEED + len(label)).shuffle(class_records)

        total = len(class_records)
        train_end = int(total * 0.70)
        val_end = train_end + int(total * 0.15)

        for index, record in enumerate(class_records):
            if index < train_end:
                record["split"] = "train"
            elif index < val_end:
                record["split"] = "val"
            else:
                record["split"] = "test"

    unique_records.sort(key=lambda record: (str(record["label"]), str(record["path"])))

    # 统计落在不同划分里的近重复对，用于发现训练/验证/测试之间的数据泄漏。
    cross_split_near_duplicates = sum(
        1
        for kept, duplicate, _ in near_duplicate_pairs
        if kept["split"] != duplicate["split"]
    )

    raw_count_series = pd.Series(raw_counts, dtype="int64").reindex(
        TARGET_CLASSES, fill_value=0
    )
    clean_count_series = pd.Series(
        Counter(str(record["label"]) for record in final_records),
        dtype="int64",
    ).reindex(TARGET_CLASSES, fill_value=0)

    raw_count_df = raw_count_series.rename("count").reset_index()
    raw_count_df.columns = ["label", "count"]
    raw_count_df.to_csv(
        REPORT_ROOT / "class_counts_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    clean_count_df = clean_count_series.rename("count").reset_index()
    clean_count_df.columns = ["label", "count"]
    clean_count_df.to_csv(
        REPORT_ROOT / "class_counts_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    image_info_df = pd.DataFrame(image_rows)
    image_info_df.to_csv(
        REPORT_ROOT / "image_info.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(duplicate_rows).to_csv(
        REPORT_ROOT / "duplicates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(corrupt_rows).to_csv(
        REPORT_ROOT / "corrupt_images.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "kept_path": kept["path"],
                "kept_label": kept["label"],
                "duplicate_path": duplicate["path"],
                "duplicate_label": duplicate["label"],
                "hamming_distance": distance,
                "kept_dhash": kept["dhash"],
                "duplicate_dhash": duplicate["dhash"],
                "kept_split": kept["split"],
                "duplicate_split": duplicate["split"],
            }
            for kept, duplicate, distance in near_duplicate_pairs
        ]
    ).to_csv(
        REPORT_ROOT / "near_duplicates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest_rows = [
        {
            "path": record["path"],
            "label": record["label"],
            "split": record["split"],
            "sha256": record["sha256"],
            "width": record["width"],
            "height": record["height"],
        }
        for record in final_records
    ]
    pd.DataFrame(manifest_rows).to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    x = np.arange(len(TARGET_CLASSES))
    width = 0.38

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(x - width / 2, raw_count_series.values, width, label="Raw")
    ax.bar(x + width / 2, clean_count_series.values, width, label="Clean")
    ax.set_title("PlantVillage 8-Class Distribution")
    ax.set_ylabel("Image Count")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label.replace("___", "\n") for label in TARGET_CLASSES],
        rotation=20,
        ha="right",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_ROOT / "class_distribution.png", dpi=200)
    plt.close(fig)

    sample_records: list[dict[str, object]] = []
    sampler = random.Random(RANDOM_SEED)
    for label in TARGET_CLASSES:
        candidates = [
            record for record in final_records if record["label"] == label
        ]
        sampler.shuffle(candidates)
        sample_records.extend(candidates[:2])

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.ravel()

    for axis, record in zip(axes, sample_records):
        with Image.open(PROJECT_ROOT / str(record["path"])) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(
            str(record["label"]).replace("___", "\n"),
            fontsize=9,
        )
        axis.axis("off")

    for axis in axes[len(sample_records):]:
        axis.axis("off")

    fig.suptitle("PlantVillage Sample Images", fontsize=16)
    fig.tight_layout()
    fig.savefig(REPORT_ROOT / "sample_grid.png", dpi=180)
    plt.close(fig)

    if not image_info_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            image_info_df["width"],
            image_info_df["height"],
            alpha=0.25,
            s=8,
        )
        ax.set_title("Image Size Distribution")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        fig.tight_layout()
        fig.savefig(REPORT_ROOT / "image_sizes.png", dpi=200)
        plt.close(fig)

    split_counts = Counter(
        str(record["split"]) for record in final_records
    )

    summary = {
        "source": str(RAW_ROOT),
        "target_classes": TARGET_CLASSES,
        "raw_total": int(raw_count_series.sum()),
        "valid_total": len(valid_records),
        "corrupt_count": len(corrupt_rows),
        "duplicate_count": len(duplicate_rows),
        "near_duplicate_count": len(near_duplicate_pairs),
        "cross_split_near_duplicates": cross_split_near_duplicates,
        "clean_total": len(final_records),
        "raw_class_counts": {
            label: int(raw_count_series[label]) for label in TARGET_CLASSES
        },
        "clean_class_counts": {
            label: int(clean_count_series[label]) for label in TARGET_CLASSES
        },
        "split_counts": {
            "train": int(split_counts["train"]),
            "val": int(split_counts["val"]),
            "test": int(split_counts["test"]),
        },
    }

    summary_path = REPORT_ROOT / "audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("数据审计完成")
    print(f"原始图片：{summary['raw_total']}")
    print(f"有效图片：{summary['valid_total']}")
    print(f"损坏图片：{summary['corrupt_count']}")
    print(f"重复图片：{summary['duplicate_count']}")
    print(f"近重复图片：{summary['near_duplicate_count']}")
    print(f"跨划分近重复对：{summary['cross_split_near_duplicates']}")
    print(f"去重后图片：{summary['clean_total']}")
    print(f"划分统计：{summary['split_counts']}")
    print(f"清单文件：{MANIFEST_PATH}")
    print(f"审计报告：{REPORT_ROOT}")


if __name__ == "__main__":
    main()