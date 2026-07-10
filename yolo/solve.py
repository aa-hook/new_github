# -*- coding: utf-8 -*-
"""Standalone FunCaptcha dice solver.

Usage:
    python solve.py path/to/captcha.jpg

Default stdout is ONLY the answer index 0-11, suitable for subprocess parsing.
Use --json for full debug output.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from PIL import Image

from dice_solver.batch_eval_yolo import recognize_target_number
from dice_solver.target_model_ocr import DEFAULT_TARGET_MODEL, predict_target_number
from dice_solver.yolo_topface_infer import predict_strip_with_model

DEFAULT_WEIGHTS = SCRIPT_DIR / "models" / "best.onnx"

_YOLO_MODEL_CACHE: Dict[str, Any] = {}


def _get_yolo_model(weights: Path, *, verbose: bool = False):
    """Load YOLO once per process; register.py may call solve_image for every wave."""
    key = str(weights.resolve())
    model = _YOLO_MODEL_CACHE.get(key)
    if model is not None:
        return model
    with quiet(not verbose):
        from ultralytics import YOLO
        try:
            model = YOLO(str(weights), task="detect")
        except TypeError:
            model = YOLO(str(weights))
    _YOLO_MODEL_CACHE[key] = model
    return model


def _maybe_rescue_target_from_classifier_top3(result: Dict[str, Any], ocr: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recover Linux/template OCR mistakes when classifier top-3 matches one candidate sum.

    This is intentionally conservative: it only runs after the solver produced no
    exact match, and only for template fallback cases that still carry the target
    classifier's top-3 candidates. It prevents cases like target panel "20"
    being read as "28" by the template fallback on CI/Linux.
    """
    if result.get("status") == "unique_match" or result.get("matches"):
        return result
    if not isinstance(ocr, dict) or ocr.get("method") != "template_fallback":
        return result
    previous = ocr.get("previous") or {}
    if previous.get("method") != "target_classifier_onnx":
        return result

    sums_raw = result.get("candidate_sums") or {}
    sums: Dict[int, int] = {}
    for idx, value in sums_raw.items():
        try:
            sums[int(idx)] = int(value)
        except Exception:
            pass
    if not sums:
        return result

    candidates = []
    for item in previous.get("top3") or []:
        try:
            target_value = int(item.get("target"))
            confidence = float(item.get("confidence", 0.0))
        except Exception:
            continue
        matches = [idx for idx, total in sums.items() if total == target_value]
        if len(matches) == 1:
            candidates.append((confidence, target_value, matches[0]))

    if not candidates:
        return result
    candidates.sort(reverse=True)
    confidence, recovered_target, answer_index = candidates[0]
    if confidence < 0.15:
        return result
    if len(candidates) > 1 and (confidence - candidates[1][0]) < 0.05:
        return result

    old_target = result.get("target_number")
    rescued = dict(result)
    rescued.update({
        "target_number": recovered_target,
        "matches": [answer_index],
        "answer_index": answer_index,
        "status": "unique_match",
        "target_recovery": {
            "method": "classifier_top3_candidate_sum_rescue",
            "old_target": old_target,
            "recovered_target": recovered_target,
            "classifier_confidence": confidence,
            "reason": "template target had no exact candidate sum; classifier top3 had exactly one unique candidate-sum match",
        },
    })
    return rescued


@contextlib.contextmanager
def quiet(enabled: bool = True):
    if not enabled:
        yield
        return
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield


def solve_image(
    image_path: str | Path,
    *,
    weights: str | Path = DEFAULT_WEIGHTS,
    target: Optional[int] = None,
    conf: float = 0.25,
    imgsz: int = 224,
    iou: float = 0.50,
    device: str = "cpu",
    ocr_min_score: float = 0.42,
    target_model: str | Path | None = DEFAULT_TARGET_MODEL,
    target_model_min_conf: float = 0.45,
    verbose: bool = False,
) -> Dict[str, Any]:
    image_path = Path(image_path)
    weights = Path(weights)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not weights.exists():
        raise FileNotFoundError(f"model not found: {weights}")

    img = Image.open(image_path).convert("RGB")
    if target is None:
        ocr = None
        if target_model and Path(target_model).exists():
            model_ocr = predict_target_number(img, model_path=target_model, min_conf=target_model_min_conf)
            if model_ocr.get("target") is not None:
                ocr = {"method": "target_classifier_onnx", **model_ocr}
                target = int(model_ocr["target"])
            else:
                ocr = {"method": "target_classifier_onnx", **model_ocr}
        if target is None:
            template_ocr = recognize_target_number(img, min_score=ocr_min_score, image_stem=image_path.stem)
            if ocr:
                template_ocr = {"method": "template_fallback", "previous": ocr, **template_ocr}
            else:
                template_ocr = {"method": "template", **template_ocr}
            ocr = template_ocr
            target = ocr.get("target")
    else:
        ocr = {"target": int(target), "manual": True}

    if target is None:
        return {
            "image": str(image_path.resolve()),
            "target_number": None,
            "target_ocr": ocr,
            "answer_index": None,
            "status": "no_target",
            "error": "target OCR failed",
        }

    model = _get_yolo_model(weights, verbose=verbose)
    with quiet(not verbose):
        result = predict_strip_with_model(
            model,
            image_path,
            target_number=int(target),
            conf=conf,
            imgsz=imgsz,
            device=device,
            iou_threshold=iou,
        )

    result = _maybe_rescue_target_from_classifier_top3(result, ocr)
    result["target_ocr"] = ocr
    result["weights"] = str(weights.resolve())
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Output FunCaptcha dice answer index 0-11")
    parser.add_argument("image", help="Full 2400x400 dice captcha image")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="ONNX/PT weights path")
    parser.add_argument("--target", type=int, default=None, help="Manual target number; normally auto-OCR")
    parser.add_argument("--device", default="cpu", help="Inference device, default cpu")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=224, help="YOLO image size")
    parser.add_argument("--iou", type=float, default=0.50, help="Duplicate face suppression IoU")
    parser.add_argument("--ocr-min-score", type=float, default=0.42, help="Target OCR min score")
    parser.add_argument("--target-model", default=str(DEFAULT_TARGET_MODEL), help="Optional target-number ONNX classifier")
    parser.add_argument("--target-model-min-conf", type=float, default=0.45, help="Min confidence for target classifier")
    parser.add_argument("--no-target-model", action="store_true", help="Disable target classifier and use template OCR only")
    parser.add_argument("--json", action="store_true", help="Print full JSON instead of only answer index")
    parser.add_argument("--allow-fallback", action="store_true", help="Allow nearest_fallback answer when no exact unique match")
    parser.add_argument("--verbose", action="store_true", help="Show backend logs")
    args = parser.parse_args(argv)

    try:
        result = solve_image(
            args.image,
            weights=args.weights,
            target=args.target,
            conf=args.conf,
            imgsz=args.imgsz,
            iou=args.iou,
            device=args.device,
            ocr_min_score=args.ocr_min_score,
            target_model=None if args.no_target_model else args.target_model,
            target_model_min_conf=args.target_model_min_conf,
            verbose=args.verbose,
        )
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False), file=sys.stderr)
        return 1

    status = result.get("status")
    answer = result.get("answer_index")
    ok = status == "unique_match" or (args.allow_fallback and status == "nearest_fallback")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if ok and answer is not None else 2

    if ok and answer is not None and 0 <= int(answer) <= 11:
        print(int(answer))
        return 0

    # Keep stdout clean for callers; details go to stderr.
    print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
