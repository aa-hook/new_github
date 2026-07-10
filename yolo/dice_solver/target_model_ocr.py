# -*- coding: utf-8 -*-
"""ONNX target-number classifier for 2400x400 dice FunCaptcha strips.

The model predicts the bottom-left target sum directly as one of 26 classes:
5, 6, ..., 30.  It is optional: callers can fall back to template OCR when the
ONNX model is absent or low-confidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageOps

from .layout import DiceLayout

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_MODEL = PACKAGE_ROOT / "models" / "target_classifier.onnx"
TARGET_VALUES = list(range(5, 31))
INPUT_WIDTH = 96
INPUT_HEIGHT = 128
_SESSION_CACHE = {}


def _get_session(model_path: Path):
    key = str(model_path.resolve())
    sess = _SESSION_CACHE.get(key)
    if sess is not None:
        return sess
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    _SESSION_CACHE[key] = sess
    return sess


def crop_target_from_image(image: Image.Image) -> Image.Image:
    """Accept a full 2400x400 strip or an already-cropped target panel."""
    img = image.convert("RGB")
    layout = DiceLayout()
    if img.width >= layout.candidate_width * layout.candidate_count and img.height >= layout.candidate_height + layout.target_height:
        return img.crop(layout.target_box())
    return img


def preprocess_target_image(image: Image.Image) -> np.ndarray:
    """Return NCHW float32 tensor shaped (1, 1, 128, 96)."""
    target = crop_target_from_image(image)
    gray = ImageOps.grayscale(target)
    gray = ImageOps.autocontrast(gray)
    gray = gray.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    # Center around 0 for the small CNN.
    arr = (arr - 0.5) / 0.5
    return arr[None, None, :, :].astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def predict_target_number(
    image: Image.Image | str | Path,
    *,
    model_path: str | Path = DEFAULT_TARGET_MODEL,
    min_conf: float = 0.45,
) -> Dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        return {"target": None, "status": "missing_model", "model": str(model_path)}

    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
        image_name = str(Path(image).resolve())
    else:
        img = image.convert("RGB")
        image_name = None

    tensor = preprocess_target_image(img)
    try:
        sess = _get_session(model_path)
        input_name = sess.get_inputs()[0].name
        output = sess.run(None, {input_name: tensor})[0]
    except ImportError as e:
        return {"target": None, "status": "missing_onnxruntime", "error": f"{type(e).__name__}: {e}"}
    except Exception as e:
        return {"target": None, "status": "inference_error", "error": f"{type(e).__name__}: {e}"}

    logits = np.asarray(output)[0]
    probs = softmax(logits)
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    target = TARGET_VALUES[idx]
    top3_idx = list(np.argsort(-probs)[:3])
    top3 = [{"target": TARGET_VALUES[i], "confidence": float(probs[i])} for i in top3_idx]
    return {
        "target": int(target) if conf >= min_conf else None,
        "raw_target": int(target),
        "confidence": conf,
        "min_conf": min_conf,
        "top3": top3,
        "status": "ok" if conf >= min_conf else "low_confidence",
        "model": str(model_path.resolve()),
        "image": image_name,
    }
