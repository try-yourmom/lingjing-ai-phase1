from __future__ import annotations

import torch

from src.model import BasicBlock, resnet18


def test_basic_block_downsample_shape() -> None:
    downsample = torch.nn.Sequential(
        torch.nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
        torch.nn.BatchNorm2d(128),
    )
    block = BasicBlock(
        in_channels=64,
        out_channels=128,
        stride=2,
        downsample=downsample,
    )

    inputs = torch.randn(2, 64, 32, 32)
    outputs = block(inputs)

    assert outputs.shape == (2, 128, 16, 16)


def test_resnet18_output_shape() -> None:
    model = resnet18(num_classes=8)
    model.eval()

    inputs = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (2, 8)