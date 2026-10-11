from __future__ import annotations

import torch
import torch.nn as nn

from src.model import resnet18


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")
    torch.manual_seed(42)

    model = resnet18(num_classes=8).to(device)
    model.train()

    inputs = torch.randn(4, 3, 224, 224, device=device)
    targets = torch.tensor([0, 3, 4, 7], device=device)

    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    loss.backward()

    total_parameters, trainable_parameters = count_parameters(model)
    gradient_norm = sum(
        parameter.grad.detach().norm().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )

    print("device:", torch.cuda.get_device_name(0))
    print("input shape:", tuple(inputs.shape))
    print("output shape:", tuple(outputs.shape))
    print("loss:", round(loss.item(), 6))
    print("total parameters:", f"{total_parameters:,}")
    print("trainable parameters:", f"{trainable_parameters:,}")
    print("gradient norm sum:", round(gradient_norm, 6))
    print("last block bn2 weight mean:", round(model.layer4[1].bn2.weight.mean().item(), 6))

    assert outputs.shape == (4, 8)
    assert torch.isfinite(loss)
    assert gradient_norm > 0

    print("model forward/backward check passed")


if __name__ == "__main__":
    main()