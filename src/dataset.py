from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "plantvillage8_manifest.csv"
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PlantVillageDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        transform: Callable | None = None,
    ) -> None:
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"找不到数据清单：{manifest_path}")

        dataframe = pd.read_csv(manifest_path)
        required_columns = {"path", "label", "split"}
        missing_columns = required_columns - set(dataframe.columns)
        if missing_columns:
            raise ValueError(f"数据清单缺少字段：{sorted(missing_columns)}")

        valid_splits = {"train", "val", "test"}
        if split not in valid_splits:
            raise ValueError(f"split 必须是 {sorted(valid_splits)} 之一")

        self.classes = sorted(dataframe["label"].unique().tolist())
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }

        split_dataframe = dataframe[dataframe["split"] == split].copy()
        if split_dataframe.empty:
            raise ValueError(f"{split} 数据集为空")

        self.samples: list[tuple[Path, int, str]] = []

        for row in split_dataframe.itertuples(index=False):
            image_path = PROJECT_ROOT / str(row.path)
            label_index = self.class_to_idx[str(row.label)]
            self.samples.append((image_path, label_index, str(row.label)))

        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label_index, _ = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = (
                self.transform(image)
                if self.transform is not None
                else transforms.ToTensor()(image)
            )

        return image_tensor, label_index


def build_transform(split: str, image_size: int = 224) -> transforms.Compose:
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def create_weighted_sampler(
    dataset: PlantVillageDataset,
) -> WeightedRandomSampler:
    labels = torch.tensor(
        [label for _, label, _ in dataset.samples],
        dtype=torch.long,
    )
    class_counts = torch.bincount(
        labels,
        minlength=len(dataset.classes),
    ).float()

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )


def build_dataloaders(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = PlantVillageDataset(
        manifest_path=manifest_path,
        split="train",
        transform=build_transform("train", image_size),
    )
    val_dataset = PlantVillageDataset(
        manifest_path=manifest_path,
        split="val",
        transform=build_transform("val", image_size),
    )
    test_dataset = PlantVillageDataset(
        manifest_path=manifest_path,
        split="test",
        transform=build_transform("test", image_size),
    )

    train_sampler = create_weighted_sampler(train_dataset)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader