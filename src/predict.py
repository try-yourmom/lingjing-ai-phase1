from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18 as torchvision_resnet18

from src.model import resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "transfer_resnet18_best.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DISPLAY_NAMES = {
    "Apple___Apple_scab": "苹果黑星病",
    "Apple___Black_rot": "苹果黑腐病",
    "Apple___Cedar_apple_rust": "苹果锈病",
    "Apple___healthy": "苹果健康",
    "Tomato___Early_blight": "番茄早疫病",
    "Tomato___Late_blight": "番茄晚疫病",
    "Tomato___Leaf_Mold": "番茄叶霉病",
    "Tomato___healthy": "番茄健康",
}


class ImageClassifier:
    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.class_names: list[str] = checkpoint["class_names"]
        self.image_size = int(checkpoint.get("image_size", 224))
        config = checkpoint.get("config", {})
        self.uses_pretrained = bool(config.get("pretrained_weights"))

        if self.uses_pretrained:
            model = torchvision_resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(self.class_names))
        else:
            model = resnet18(num_classes=len(self.class_names))

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def model_info(self) -> dict[str, object]:
        return {
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
            "classes": self.class_names,
            "image_size": self.image_size,
            "uses_pretrained": self.uses_pretrained,
        }
    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict[str, object]]:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probabilities = F.softmax(logits, dim=1)[0]
        k = min(max(top_k, 1), len(self.class_names))
        values, indices = probabilities.topk(k)

        results: list[dict[str, object]] = []
        for probability, index in zip(
            values.detach().cpu().tolist(),
            indices.detach().cpu().tolist(),
        ):
            class_name = self.class_names[index]
            results.append(
                {
                    "class_name": class_name,
                    "display_name": DISPLAY_NAMES.get(class_name, class_name),
                    "probability": float(probability),
                }
            )
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a plant leaf disease image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = ImageClassifier(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    with Image.open(args.image) as image:
        results = classifier.predict(image, top_k=args.top_k)

    output = {
        "image": str(args.image),
        "model": classifier.model_info(),
        "predictions": results,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()