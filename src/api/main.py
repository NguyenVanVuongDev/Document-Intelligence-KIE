import asyncio
import base64
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from src.kie.extractor import extract_spatial_kie
from src.ocr.engine import OCREngine
from src.preprocessing.pipeline import preprocess_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.getenv("CONFIG_PATH", PROJECT_ROOT / "configs" / "config.yaml"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
MAX_IMAGE_PIXELS = 16_000_000
ALLOWED_MIME_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG"}


def _read_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def _validate_and_decode(payload: bytes, content_type: str) -> np.ndarray:
    expected_format = ALLOWED_MIME_TYPES.get(content_type.split(";")[0].strip().lower())
    if expected_format is None:
        raise HTTPException(status_code=415, detail="Only JPEG and PNG images are supported.")
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format != expected_format:
                raise HTTPException(status_code=415, detail="Image content does not match its MIME type.")
            width, height = image.size
            if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions exceed the allowed limit.")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="OpenCV could not decode the uploaded image.")
    return image


async def _read_limited_upload(upload: UploadFile) -> bytes:
    chunks = []
    total_bytes = 0
    while chunk := await upload.read(1024 * 1024):
        total_bytes += len(chunk)
        if total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Maximum upload size is 10 MB.")
        chunks.append(chunk)
    return b"".join(chunks)


def _jpeg_base64(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("Could not encode an output image.")
    return base64.b64encode(encoded).decode("ascii")


def _is_key_value(text: str, fields: dict[str, str]) -> bool:
    normalized_text = re.sub(r"\s+", " ", text.casefold()).strip()
    normalized_fields = [value.casefold() for value in fields.values() if value]
    if any(value in normalized_text or normalized_text in value for value in normalized_fields):
        return True
    if re.search(r"\b(?:tổng|tong|total|thanh toán|thanh toan|date|ngày|ngay)\b", text, re.IGNORECASE):
        return True
    text_digits = re.sub(r"\D", "", text)
    total_digits = re.sub(r"\D", "", fields.get("total", ""))
    return bool(text_digits and total_digits and text_digits == total_digits)


def _annotate(image: np.ndarray, boxes: list[Any], fields: dict[str, str]) -> np.ndarray:
    output = image.copy()
    for box in boxes:
        is_key = _is_key_value(box.text, fields)
        polygon = np.asarray(box.polygon, dtype=np.int32).reshape(-1, 1, 2)
        color = (0, 0, 255) if is_key else (255, 100, 0)
        cv2.polylines(output, [polygon], isClosed=True, color=color, thickness=2)
    return output


def _analyze(image: np.ndarray, config: dict[str, Any], engine: OCREngine) -> dict[str, Any]:
    timings: dict[str, float] = {}

    started = time.perf_counter()
    preprocessing = preprocess_image(image, config)
    preprocessed = preprocessing.processed
    timings["preprocessing_ms"] = round((time.perf_counter() - started) * 1000, 2)

    started = time.perf_counter()
    boxes = engine.run(preprocessed, max_side=int(config.get("ocr", {}).get("max_side_for_ocr", 960)))
    timings["ocr_ms"] = round((time.perf_counter() - started) * 1000, 2)

    started = time.perf_counter()
    extraction = extract_spatial_kie(boxes, config, img_shape=preprocessed.shape)
    timings["kie_ms"] = round((time.perf_counter() - started) * 1000, 2)

    annotated = _annotate(preprocessed, boxes, extraction["fields"])
    return {
        "fields": extraction["fields"],
        "full_text": extraction["full_text"],
        "lines": extraction["lines"],
        "boxes": [
            {
                "bbox": box.polygon,
                "text": box.text,
                "confidence": box.confidence,
                "is_key_value": _is_key_value(box.text, extraction["fields"]),
            }
            for box in boxes
        ],
        "preprocessing": {"steps": preprocessing.steps},
        "timings_ms": timings,
        "images": {
            "preprocessed_jpeg_base64": _jpeg_base64(preprocessed),
            "annotated_jpeg_base64": _jpeg_base64(annotated),
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = _read_config()
    engine = OCREngine(**config.get("ocr", {}))
    engine.pipeline.predict(np.full((64, 64, 3), 255, dtype=np.uint8))
    app.state.config = config
    app.state.ocr_engine = engine
    app.state.inference_lock = asyncio.Lock()
    logger.info("OCR engine initialized and warmed up")
    yield


app = FastAPI(title="Document Intelligence KIE", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> dict[str, Any]:
    payload = await _read_limited_upload(image)
    decoded = _validate_and_decode(payload, image.content_type or "")
    async with app.state.inference_lock:
        return await run_in_threadpool(_analyze, decoded, app.state.config, app.state.ocr_engine)


app.mount("/", StaticFiles(directory=PROJECT_ROOT / "receipt-web", html=True), name="web")