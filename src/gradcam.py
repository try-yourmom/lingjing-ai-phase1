from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import DEFAULT_MANIFEST_PATH
from src.predict import DEFAULT_CHECKPOINT, DISPLAY_NAMES, ImageClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "gradcam" / "transfer_resnet18"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-class", type=int, default=1)
    return parser.parse_args()


def normalize_cam(cam: torch.Tensor) -> np.ndarray:
    cam = cam.detach().cpu().numpy()
    minimum = float(cam.min())
    maximum = float(cam.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(cam)
    return (cam - minimum) / (maximum - minimum)

def generate_cam(
    classifier: ImageClassifier,
    image: Image.Image,
    target_index: int | None = None,
) -> tuple[np.ndarray, int, float]:
    model = classifier.model
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    target_layer = model.layer4[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        image = image.convert("RGB")
        tensor = classifier.transform(image).unsqueeze(0).to(classifier.device)
        logits = model(tensor)

        if target_index is None:
            target_index = int(logits.argmax(dim=1).item())

        model.zero_grad(set_to_none=True)
        logits[0, target_index].backward()

        activation = activations[0]
        gradient = gradients[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        probability = float(
            torch.softmax(logits, dim=1)[0, target_index].detach().cpu().item()
        )
        return normalize_cam(cam), target_index, probability
    finally:
        forward_handle.remove()
        backward_handle.remove()


def save_overlay(
    image: Image.Image,
    cam: np.ndarray,
    output_path: Path,
) -> None:
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    overlay = np.clip(0.55 * base + 0.45 * heatmap, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(base)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    classifier = ImageClassifier(checkpoint_path=args.checkpoint)
    manifest = pd.read_csv(args.manifest_path)
    test_manifest = manifest[manifest["split"] == "test"].copy()

    sampled_frames = []
    for label in classifier.class_names:
        class_samples = test_manifest[test_manifest["label"] == label].head(
            args.samples_per_class
        )
        sampled_frames.append(class_samples)

    samples = pd.concat(sampled_frames, ignore_index=True)
    rows = []

    for index, row in samples.iterrows():
        image_path = PROJECT_ROOT / str(row["path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            cam, target_index, probability = generate_cam(classifier, image)

        true_label = str(row["label"])
        predicted_label = classifier.class_names[target_index]
        output_path = args.output_dir / f"{index:04d}_{predicted_label}.png"
        save_overlay(image, cam, output_path)

        rows.append(
            {
                "path": str(row["path"]),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "predicted_display_name": DISPLAY_NAMES.get(
                    predicted_label, predicted_label
                ),
                "probability": probability,
                "correct": true_label == predicted_label,
                "overlay_path": str(output_path.relative_to(PROJECT_ROOT)),
            }
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(
        args.output_dir / "gradcam_index.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"generated {len(result_df)} Grad-CAM images")
    print(f"output dir: {args.output_dir}")
    print(result_df[["true_label", "predicted_label", "probability", "correct"]].to_string(index=False))


if __name__ == "__main__":
    main()