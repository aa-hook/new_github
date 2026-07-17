from __future__ import annotations

import os
import statistics
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F


def center_crop_image(image: Image.Image, center: tuple[float, float], *, size: int = 64) -> Image.Image:
    side = max(16, int(size))
    half = side / 2.0
    left = int(round(float(center[0]) - half))
    top = int(round(float(center[1]) - half))
    right, bottom = left + side, top + side
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(image.width, right), min(image.height, bottom)
    if sx1 > sx0 and sy1 > sy0:
        source = image.crop((sx0, sy0, sx1, sy1)).convert("RGB")
        canvas.paste(source, (sx0 - left, sy0 - top))
    return canvas


def patch_similarity_scores(
    prompt_features: torch.Tensor,
    icon_features: torch.Tensor,
    *,
    top_k: int = 75,
    icon_batch_size: int = 12,
) -> torch.Tensor:
    """Asymmetric patch matching: every prompt patch finds its best icon patch."""
    if prompt_features.ndim != 3 or icon_features.ndim != 3:
        raise ValueError("expected prompt AxPxD and icons NxCxD")
    prompt = F.normalize(prompt_features.float(), dim=-1)
    icons = F.normalize(icon_features.float(), dim=-1)
    outputs: list[torch.Tensor] = []
    for start in range(0, icons.shape[0], max(1, int(icon_batch_size))):
        chunk = icons[start : start + max(1, int(icon_batch_size))]
        similarities = torch.einsum("apd,ncd->napc", prompt, chunk)
        prompt_best = similarities.max(dim=-1).values
        count = max(1, min(int(top_k), prompt_best.shape[-1]))
        per_rotation = prompt_best.topk(count, dim=-1).values.mean(dim=-1)
        outputs.append(per_rotation.max(dim=-1).values)
    if not outputs:
        return torch.empty((0,), dtype=torch.float32, device=prompt_features.device)
    return torch.cat(outputs, dim=0)


def orientation_invariant_arc_residual(foot_arc: float, target_arc: float) -> float:
    residual = min(abs(float(foot_arc) - float(target_arc)), abs((1.0 - float(foot_arc)) - float(target_arc)))
    return round(float(residual), 12)


def rank_arc_hypotheses(
    candidate_foot_arcs: Sequence[float | None],
    target_arcs: Sequence[float | None],
) -> tuple[int, list[float]]:
    clean_targets = [float(value) for value in target_arcs if value is not None and np.isfinite(float(value))]
    scores: list[float] = []
    for foot_arc in candidate_foot_arcs:
        if foot_arc is None or not clean_targets:
            scores.append(-1_000_000.0)
            continue
        residuals = [orientation_invariant_arc_residual(float(foot_arc), target) for target in clean_targets]
        scores.append(-float(statistics.median(residuals)))
    return (int(np.argmax(np.asarray(scores, dtype=np.float64))) if scores else -1), scores


def _default_dinov2_repo() -> Path:
    return Path(os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main"))


class DinoV2PatchMatcher:
    def __init__(
        self,
        *,
        device: str = "cuda",
        repo_path: str | Path | None = None,
        image_size: int = 224,
        rotations: Sequence[int] = tuple(range(0, 360, 30)),
        central_low: int = 3,
        central_high: int = 13,
        top_k: int = 75,
    ) -> None:
        self.device = str(device)
        self.image_size = int(image_size)
        self.rotations = tuple(int(angle) for angle in rotations)
        self.central_low = int(central_low)
        self.central_high = int(central_high)
        self.top_k = int(top_k)
        repo = Path(repo_path) if repo_path is not None else _default_dinov2_repo()
        if not repo.exists():
            raise FileNotFoundError(f"DINOv2 torch-hub source directory is missing: {repo}")
        self.model = torch.hub.load(str(repo), "dinov2_vits14", source="local")
        self.model.eval().to(self.device)
        self._indices = torch.tensor(
            [row * 16 + col for row in range(self.central_low, self.central_high) for col in range(self.central_low, self.central_high)],
            dtype=torch.long,
        )

    def _to_tensor(self, image: Image.Image, *, angle: int = 0) -> torch.Tensor:
        image = ImageOps.autocontrast(image.convert("L")).convert("RGB")
        if angle:
            image = image.rotate(
                int(angle),
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=(128, 128, 128),
            )
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _encode(self, tensors: Sequence[torch.Tensor], *, batch_size: int = 32) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(tensors), max(1, int(batch_size))):
                batch = torch.stack(tensors[start : start + max(1, int(batch_size))]).to(self.device)
                patches = self.model.forward_features(batch)["x_norm_patchtokens"]
                patches = F.normalize(patches[:, self._indices.to(patches.device)], dim=-1)
                outputs.append(patches)
        return torch.cat(outputs, dim=0) if outputs else torch.empty((0, len(self._indices), 384), device=self.device)

    def score(self, prompt: Image.Image, icons: Sequence[Image.Image], *, batch_size: int = 32) -> list[float]:
        if not icons:
            return []
        prompt_features = self._encode([self._to_tensor(prompt, angle=angle) for angle in self.rotations], batch_size=batch_size)
        icon_features = self._encode([self._to_tensor(icon) for icon in icons], batch_size=batch_size)
        scores = patch_similarity_scores(prompt_features, icon_features, top_k=self.top_k)
        return [float(value) for value in scores.detach().cpu().tolist()]

