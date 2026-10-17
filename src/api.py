from __future__ import annotations

import time
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.predict import DEFAULT_CHECKPOINT, ImageClassifier


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = ImageClassifier(checkpoint_path=DEFAULT_CHECKPOINT)
    yield
    app.state.classifier = None


app = FastAPI(
    title="Plant Leaf Disease Diagnosis API",
    description="Handwritten ResNet and transfer learning based inference service.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_classifier() -> ImageClassifier:
    classifier = getattr(app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return classifier


@app.get("/health")
def health() -> dict[str, object]:
    classifier = get_classifier()
    return {
        "status": "ok",
        "device": str(classifier.device),
        "checkpoint": str(classifier.checkpoint_path),
    }


@app.get("/model-info")
def model_info() -> dict[str, object]:
    return get_classifier().model_info()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = Form(3),
) -> dict[str, object]:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 10 MB")

    try:
        with Image.open(BytesIO(payload)) as image:
            image = image.convert("RGB")
            start_time = time.perf_counter()
            predictions = get_classifier().predict(image, top_k=top_k)
            latency_ms = (time.perf_counter() - start_time) * 1000
    except UnidentifiedImageError as error:
        raise HTTPException(status_code=400, detail="Invalid image file") from error

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "latency_ms": round(latency_ms, 3),
        "predictions": predictions,
    }