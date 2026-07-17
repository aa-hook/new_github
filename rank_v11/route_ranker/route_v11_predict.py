from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from .route_v11_gate import select_v11_predictions
from .route_v7_direct import RouteV7Dataset, load_route_v7_checkpoint


MODEL_MANIFEST_SCHEMA = "route_v11_model_manifest_v1"


def _member_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class RouteV11Predictor:
    def __init__(
        self,
        manifest: str | Path,
        *,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        verify_hashes: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest).resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8-sig"))
        if payload.get("schema_version") != MODEL_MANIFEST_SCHEMA:
            raise ValueError(f"unexpected V11 manifest schema: {payload.get('schema_version')}")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        self.fast_checkpoint = _member_path(
            self.manifest_path, str(payload["fast_checkpoint"])
        ).resolve()
        self.legacy_checkpoint = _member_path(
            self.manifest_path, str(payload["legacy_checkpoint"])
        ).resolve()
        self.expert_checkpoint = _member_path(
            self.manifest_path, str(payload["expert_checkpoint"])
        ).resolve()
        if verify_hashes:
            expected = dict(payload.get("sha256") or {})
            for key, path in (
                ("fast", self.fast_checkpoint),
                ("legacy", self.legacy_checkpoint),
                ("expert", self.expert_checkpoint),
            ):
                if expected.get(key) and _sha256(path) != str(expected[key]).lower():
                    raise ValueError(f"V11 checkpoint SHA-256 mismatch: {key}={path}")
        self.fast = load_route_v7_checkpoint(self.fast_checkpoint, device=self.device)
        self.legacy = load_route_v7_checkpoint(self.legacy_checkpoint, device=self.device)
        self.expert = load_route_v7_checkpoint(self.expert_checkpoint, device=self.device)
        self.fast.model.eval()
        self.legacy.model.eval()
        self.expert.model.eval()
        self.margin_advantage = float(payload["margin_advantage"])
        self.require_legacy_consensus = bool(payload.get("require_legacy_consensus", True))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.manifest = payload

    def _cache_for(self, name: str, loaded) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{name}_{loaded.image_size}_{loaded.foot_crop_size}"

    def _ranker_logits(self, name: str, loaded, image: str | Path) -> tuple[torch.Tensor, float]:
        started = time.perf_counter()
        dataset = RouteV7Dataset(
            [{"image": str(image), "answer_index": 0}],
            image_size=loaded.image_size,
            foot_crop_size=loaded.foot_crop_size,
            training=False,
            permute_candidates=False,
            cache_dir=self._cache_for(name, loaded),
        )
        item = dataset[0]
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            logits = loaded.model(
                item["prompt"][None].to(self.device, non_blocking=True),
                item["full_candidates"][None].to(self.device, non_blocking=True),
                item["foot_candidates"][None].to(self.device, non_blocking=True),
                item["geometry"][None].to(self.device, non_blocking=True),
            )
        return logits.float().cpu(), float(time.perf_counter() - started)

    @staticmethod
    def _ranker_result(logits: torch.Tensor) -> dict:
        probabilities = logits.softmax(dim=1)
        top2 = probabilities.topk(2, dim=1)
        return {
            "prediction": int(top2.indices[0, 0].item()),
            "top_indices": [int(value) for value in top2.indices[0].tolist()],
            "scores": [float(value) for value in probabilities[0].tolist()],
            "confidence": float(top2.values[0, 0].item()),
            "margin": float(top2.values[0, 0].item() - top2.values[0, 1].item()),
        }

    def predict(self, image: str | Path, *, mode: str = "accurate") -> dict:
        selected_mode = str(mode).lower()
        if selected_mode not in {"fast", "accurate"}:
            raise ValueError(f"mode must be fast or accurate, got {mode}")
        started = time.perf_counter()
        fast_logits, fast_seconds = self._ranker_logits("fast", self.fast, image)
        fast = self._ranker_result(fast_logits)
        answer = int(fast["prediction"])
        chosen = fast
        legacy = expert = None
        legacy_seconds = expert_seconds = 0.0
        switched = False
        legacy_consensus = False

        if selected_mode == "accurate":
            legacy_logits, legacy_seconds = self._ranker_logits(
                "legacy", self.legacy, image
            )
            expert_logits, expert_seconds = self._ranker_logits(
                "expert", self.expert, image
            )
            legacy = self._ranker_result(legacy_logits)
            expert = self._ranker_result(expert_logits)
            selected = select_v11_predictions(
                fast_logits,
                legacy_logits,
                expert_logits,
                margin_advantage=self.margin_advantage,
                require_legacy_consensus=self.require_legacy_consensus,
            )
            answer = int(selected["predictions"][0].item())
            switched = bool(selected["switched"][0].item())
            legacy_consensus = bool(selected["legacy_consensus"][0].item())
            chosen = fast if switched else expert

        return {
            "schema_version": "route_v11_prediction_v1",
            "image": str(Path(image)),
            "mode": selected_mode,
            "answer_index": answer,
            "top_indices": list(chosen["top_indices"]),
            "scores": list(chosen["scores"]),
            "confidence": float(chosen["confidence"]),
            "margin": float(chosen["margin"]),
            "fast_index": int(fast["prediction"]),
            "legacy_v9_index": int(legacy["prediction"]) if legacy else None,
            "expert_index": int(expert["prediction"]) if expert else None,
            "fast_margin": float(fast["margin"]),
            "legacy_v9_margin": float(legacy["margin"]) if legacy else None,
            "expert_margin": float(expert["margin"]) if expert else None,
            "legacy_consensus": legacy_consensus,
            "switched_to_v10": switched,
            "margin_advantage": self.margin_advantage,
            "require_legacy_consensus": self.require_legacy_consensus,
            "fast_seconds": fast_seconds,
            "legacy_v9_seconds": legacy_seconds,
            "expert_seconds": expert_seconds,
            "total_seconds": float(time.perf_counter() - started),
        }
