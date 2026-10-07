from __future__ import annotations

import torch

from src.dataset import build_dataloaders


def main() -> None:
    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=32,
        num_workers=0,
        image_size=224,
    )

    print("CUDA 可用：", torch.cuda.is_available())
    print("训练集大小：", len(train_loader.dataset))
    print("验证集大小：", len(val_loader.dataset))
    print("测试集大小：", len(test_loader.dataset))
    print("类别数量：", len(train_loader.dataset.classes))

    for index, class_name in enumerate(train_loader.dataset.classes):
        print(f"{index}: {class_name}")

    images, labels = next(iter(train_loader))

    print()
    print("一个批次的图片形状：", images.shape)
    print("图片张量类型：", images.dtype)
    print("标签形状：", labels.shape)
    print("标签：", labels.tolist())
    print("标签范围：", int(labels.min()), int(labels.max()))

    class_counts = torch.bincount(
        labels,
        minlength=len(train_loader.dataset.classes),
    )
    print("当前批次各类数量：", class_counts.tolist())

    val_images, val_labels = next(iter(val_loader))
    test_images, test_labels = next(iter(test_loader))

    print("验证批次：", val_images.shape, val_labels.shape)
    print("测试批次：", test_images.shape, test_labels.shape)


if __name__ == "__main__":
    main()