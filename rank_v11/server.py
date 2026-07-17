from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image

from route_ranker.route_v11_predict import RouteV11Predictor


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "models" / "model_manifest.json"
SUPPORTED_IMAGE_SIZE = (2000, 400)


def resolve_device(value: str) -> str:
    selected = str(value).strip().lower()
    if selected == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if selected not in {"cpu", "cuda"}:
        raise ValueError(f"device must be auto, cpu, or cuda; got {value}")
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return selected


class ModelService:
    def __init__(
        self,
        *,
        manifest: Path,
        device: str,
        cache_dir: Path,
        default_mode: str,
        warmup: bool,
    ) -> None:
        self.started_at = time.time()
        self.device = device
        self.default_mode = default_mode
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        load_started = time.perf_counter()
        self.predictor = RouteV11Predictor(
            manifest,
            device=device,
            cache_dir=self.cache_dir,
            verify_hashes=True,
        )
        self.model_load_seconds = time.perf_counter() - load_started
        self.warmup_seconds = 0.0
        if warmup:
            warmup_image = self.cache_dir / "_rank_v11_warmup_2000x400.jpg"
            if not warmup_image.is_file():
                Image.new("RGB", (2000, 400), (245, 245, 245)).save(
                    warmup_image, format="JPEG", quality=90
                )
            warmup_started = time.perf_counter()
            self.predictor.predict(warmup_image, mode=default_mode)
            self.warmup_seconds = time.perf_counter() - warmup_started

    def health(self) -> dict:
        return {
            "ok": True,
            "status": "ready",
            "pid": os.getpid(),
            "device": self.device,
            "default_mode": self.default_mode,
            "model_load_seconds": self.model_load_seconds,
            "warmup_seconds": self.warmup_seconds,
            "uptime_seconds": max(0.0, time.time() - self.started_at),
        }

    def solve(self, image_value: str, mode: str | None = None) -> dict:
        image = Path(str(image_value)).expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"image not found: {image}")
        with Image.open(image) as opened:
            if opened.size != SUPPORTED_IMAGE_SIZE:
                raise ValueError(
                    f"expected challenge size {SUPPORTED_IMAGE_SIZE}, got {opened.size}"
                )
        selected_mode = str(mode or self.default_mode).lower()
        if selected_mode not in {"fast", "accurate"}:
            raise ValueError(f"mode must be fast or accurate; got {selected_mode}")
        started = time.perf_counter()
        with self.lock:
            result = self.predictor.predict(image, mode=selected_mode)
        return {
            "ok": True,
            "request_id": uuid.uuid4().hex,
            "image": str(image),
            "mode": selected_mode,
            "answer_index": int(result["answer_index"]),
            "confidence": float(result["confidence"]),
            "margin": float(result["margin"]),
            "fast_index": int(result["fast_index"]),
            "legacy_v9_index": (
                int(result["legacy_v9_index"])
                if result["legacy_v9_index"] is not None
                else None
            ),
            "expert_index": (
                int(result["expert_index"])
                if result["expert_index"] is not None
                else None
            ),
            "switched_to_v10": bool(result["switched_to_v10"]),
            "model_seconds": float(result["total_seconds"]),
            "service_seconds": float(time.perf_counter() - started),
        }


def build_handler(service: ModelService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format_string: str, *args) -> None:
            print(
                f"[rank_v11] {self.address_string()} {format_string % args}",
                file=sys.stderr,
                flush=True,
            )

        def send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.send_json(200, service.health())
                return
            self.send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            if self.path != "/solve":
                self.send_json(404, {"ok": False, "error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1024 * 1024:
                    raise ValueError("request body must be between 1 byte and 1 MiB")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict) or not payload.get("image"):
                    raise ValueError("JSON field 'image' is required")
                result = service.solve(str(payload["image"]), payload.get("mode"))
                self.send_json(200, result)
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
                self.send_json(
                    400,
                    {"ok": False, "error": type(exc).__name__, "message": str(exc)},
                )
            except Exception as exc:
                self.send_json(
                    500,
                    {"ok": False, "error": type(exc).__name__, "message": str(exc)},
                )

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Persistent Route V11 inference service.")
    parser.add_argument("--host", default=os.environ.get("RANK_V11_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("RANK_V11_PORT", "8765"))
    )
    parser.add_argument("--device", default=os.environ.get("RANK_V11_DEVICE", "auto"))
    parser.add_argument("--mode", choices=["fast", "accurate"], default="accurate")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("RANK_V11_CACHE_DIR", str(ROOT / ".cache"))),
    )
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if args.cpu_threads > 0:
        torch.set_num_threads(int(args.cpu_threads))
    service = ModelService(
        manifest=args.manifest.resolve(),
        device=device,
        cache_dir=args.cache_dir,
        default_mode=args.mode,
        warmup=not args.no_warmup,
    )
    server = ThreadingHTTPServer((args.host, int(args.port)), build_handler(service))
    server.daemon_threads = True
    print(
        json.dumps(
            {
                "event": "ready",
                "url": f"http://{args.host}:{args.port}",
                **service.health(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
