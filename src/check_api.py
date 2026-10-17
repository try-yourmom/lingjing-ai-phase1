from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.api import app
from src.dataset import DEFAULT_MANIFEST_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = pd.read_csv(DEFAULT_MANIFEST_PATH)
    sample_relative_path = str(manifest.iloc[0]["path"])
    sample_path = PROJECT_ROOT / sample_relative_path

    with TestClient(app) as client:
        health = client.get("/health")
        health.raise_for_status()
        print("health:", json.dumps(health.json(), ensure_ascii=False))

        info = client.get("/model-info")
        info.raise_for_status()
        print("model-info:", json.dumps(info.json(), ensure_ascii=False))

        with sample_path.open("rb") as image_file:
            response = client.post(
                "/predict",
                files={"file": (sample_path.name, image_file, "image/jpeg")},
                data={"top_k": "3"},
            )

        response.raise_for_status()
        result = response.json()
        print("predict:", json.dumps(result, ensure_ascii=False, indent=2))

        invalid_response = client.post(
            "/predict",
            files={"file": ("bad.txt", b"not an image", "text/plain")},
        )
        print("invalid status:", invalid_response.status_code)

        assert health.status_code == 200
        assert info.status_code == 200
        assert len(result["predictions"]) == 3
        assert invalid_response.status_code == 400

    print("API check passed")


if __name__ == "__main__":
    main()