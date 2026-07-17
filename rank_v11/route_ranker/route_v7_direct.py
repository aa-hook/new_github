from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import functional as TF

from .geometry import detect_person
from .layout import DEFAULT_LAYOUT
from .route_v3_pair_dataset import crop_prompt_icon
from .route_v4_dino_matcher import center_crop_image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GEOMETRY_DIM = 6
CHECKPOINT_SCHEMA = "route_v7_dualview_set_ranker_v1"


def _square(image: Image.Image, *, fill: tuple[int, int, int] = (245, 245, 245)) -> Image.Image:
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image.convert("RGB"), ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


@dataclass(frozen=True)
class RouteV7Tiles:
    prompt: Image.Image
    full_candidates: list[Image.Image]
    foot_candidates: list[Image.Image]
    geometry: np.ndarray


def extract_route_v7_tiles(
    image: Image.Image,
    *,
    image_size: int = 128,
    foot_crop_size: int = 112,
) -> RouteV7Tiles:
    size = int(image_size)
    crop_size = int(foot_crop_size)
    if size <= 0:
        raise ValueError("image_size must be positive")
    if crop_size < 32:
        raise ValueError("foot_crop_size must be at least 32")
    source = image.convert("RGB")
    expected = (DEFAULT_LAYOUT.full_width, DEFAULT_LAYOUT.full_height)
    if source.size != expected:
        raise ValueError(f"expected challenge size {expected}, got {source.size}")

    prompt_raw = crop_prompt_icon(source.crop(DEFAULT_LAYOUT.prompt_box))
    prompt = _square(prompt_raw).resize((size, size), Image.Resampling.BILINEAR)
    full_candidates: list[Image.Image] = []
    foot_candidates: list[Image.Image] = []
    geometry_rows: list[list[float]] = []
    tile_size = float(DEFAULT_LAYOUT.candidate_size)
    for index in range(DEFAULT_LAYOUT.candidate_count):
        tile = source.crop(DEFAULT_LAYOUT.candidate_box(index))
        person = detect_person(tile)
        center = person.foot if person.found else (tile_size / 2.0, tile_size / 2.0)
        foot_crop = center_crop_image(tile, center, size=crop_size)
        full_candidates.append(tile.resize((size, size), Image.Resampling.BILINEAR))
        foot_candidates.append(foot_crop.resize((size, size), Image.Resampling.BILINEAR))
        if person.found:
            x0, y0, x1, y1 = person.bbox
            geometry_rows.append(
                [
                    1.0,
                    float(person.foot[0]) / tile_size,
                    float(person.foot[1]) / tile_size,
                    float(x1 - x0) / tile_size,
                    float(y1 - y0) / tile_size,
                    float(person.area) / (tile_size * tile_size),
                ]
            )
        else:
            geometry_rows.append([0.0, 0.5, 0.5, 0.0, 0.0, 0.0])
    geometry = np.asarray(geometry_rows, dtype=np.float32)
    return RouteV7Tiles(prompt, full_candidates, foot_candidates, geometry)


@dataclass(frozen=True)
class PermutedCandidateData:
    full_candidates: list
    foot_candidates: list
    geometry: np.ndarray
    answer_index: int
    order: list[int]


def permute_candidate_data(
    full_candidates: Sequence,
    foot_candidates: Sequence,
    geometry: np.ndarray,
    *,
    answer_index: int,
    order: Sequence[int],
) -> PermutedCandidateData:
    clean_order = [int(index) for index in order]
    if sorted(clean_order) != list(range(DEFAULT_LAYOUT.candidate_count)):
        raise ValueError("order must be a permutation of candidate indices 0-9")
    answer = int(answer_index)
    if not 0 <= answer < DEFAULT_LAYOUT.candidate_count:
        raise ValueError(f"answer_index out of range: {answer}")
    matrix = np.asarray(geometry, dtype=np.float32)
    if matrix.shape != (DEFAULT_LAYOUT.candidate_count, GEOMETRY_DIM):
        raise ValueError(f"expected geometry [10,{GEOMETRY_DIM}], got {matrix.shape}")
    return PermutedCandidateData(
        full_candidates=[full_candidates[index] for index in clean_order],
        foot_candidates=[foot_candidates[index] for index in clean_order],
        geometry=matrix[clean_order].copy(),
        answer_index=clean_order.index(answer),
        order=clean_order,
    )


def _cache_key(image_path: str | Path, *, image_size: int, foot_crop_size: int, content_hash: str = "") -> str:
    path = Path(image_path)
    if content_hash:
        source = str(content_hash)
    else:
        stat = path.stat()
        source = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(f"route-v7|{source}|{image_size}|{foot_crop_size}".encode()).hexdigest()


def _cache_paths(
    cache_dir: str | Path,
    image_path: str | Path,
    *,
    image_size: int,
    foot_crop_size: int,
    content_hash: str = "",
) -> tuple[Path, Path]:
    key = _cache_key(
        image_path,
        image_size=image_size,
        foot_crop_size=foot_crop_size,
        content_hash=content_hash,
    )
    root = Path(cache_dir) / key[:2]
    return root / f"{key}.jpg", root / f"{key}.json"


def _write_tile_cache(panel_path: Path, metadata_path: Path, tiles: RouteV7Tiles, *, image_size: int) -> None:
    size = int(image_size)
    panel = Image.new("RGB", (size * 11, size * 2), (245, 245, 245))
    panel.paste(tiles.prompt, (0, 0))
    for index, tile in enumerate(tiles.full_candidates):
        panel.paste(tile, ((index + 1) * size, 0))
    for index, tile in enumerate(tiles.foot_candidates):
        panel.paste(tile, ((index + 1) * size, size))
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel_tmp = panel_path.with_suffix(".tmp.jpg")
    metadata_tmp = metadata_path.with_suffix(".tmp.json")
    panel.save(panel_tmp, format="JPEG", quality=95, subsampling=0, optimize=False)
    metadata_tmp.write_text(
        json.dumps({"geometry": tiles.geometry.tolist(), "image_size": size}),
        encoding="utf-8",
    )
    os.replace(panel_tmp, panel_path)
    os.replace(metadata_tmp, metadata_path)


def _read_tile_cache(panel_path: Path, metadata_path: Path, *, image_size: int) -> RouteV7Tiles:
    size = int(image_size)
    with Image.open(panel_path) as opened:
        panel = opened.convert("RGB")
    if panel.size != (size * 11, size * 2):
        raise ValueError(f"invalid Route V7 cache panel size: {panel.size}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    geometry = np.asarray(metadata["geometry"], dtype=np.float32)
    if geometry.shape != (10, GEOMETRY_DIM):
        raise ValueError(f"invalid Route V7 cache geometry: {geometry.shape}")
    prompt = panel.crop((0, 0, size, size))
    full = [panel.crop(((index + 1) * size, 0, (index + 2) * size, size)) for index in range(10)]
    foot = [
        panel.crop(((index + 1) * size, size, (index + 2) * size, size * 2))
        for index in range(10)
    ]
    return RouteV7Tiles(prompt, full, foot, geometry)


def load_or_build_route_v7_tiles(
    image_path: str | Path,
    *,
    image_size: int,
    foot_crop_size: int,
    cache_dir: str | Path | None = None,
    content_hash: str = "",
) -> RouteV7Tiles:
    if cache_dir:
        panel_path, metadata_path = _cache_paths(
            cache_dir,
            image_path,
            image_size=image_size,
            foot_crop_size=foot_crop_size,
            content_hash=content_hash,
        )
        if panel_path.is_file() and metadata_path.is_file():
            try:
                return _read_tile_cache(panel_path, metadata_path, image_size=image_size)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
    with Image.open(image_path) as opened:
        tiles = extract_route_v7_tiles(opened, image_size=image_size, foot_crop_size=foot_crop_size)
    if cache_dir:
        _write_tile_cache(panel_path, metadata_path, tiles, image_size=image_size)
    return tiles


def _color_augment(image: Image.Image) -> Image.Image:
    result = image.convert("RGB")
    if random.random() < 0.75:
        result = ImageEnhance.Brightness(result).enhance(random.uniform(0.82, 1.18))
    if random.random() < 0.75:
        result = ImageEnhance.Contrast(result).enhance(random.uniform(0.78, 1.25))
    if random.random() < 0.45:
        result = ImageEnhance.Color(result).enhance(random.uniform(0.65, 1.30))
    if random.random() < 0.10:
        result = ImageOps.grayscale(result).convert("RGB")
    if random.random() < 0.08:
        result = ImageOps.invert(result)
    return result


def _augment_tiles(tiles: RouteV7Tiles) -> RouteV7Tiles:
    prompt = _color_augment(tiles.prompt)
    if random.random() < 0.65:
        angle = random.uniform(-180.0, 180.0)
        prompt = prompt.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(245, 245, 245))
    full: list[Image.Image] = []
    foot: list[Image.Image] = []
    geometry = tiles.geometry.copy()
    for index, (full_tile, foot_tile) in enumerate(zip(tiles.full_candidates, tiles.foot_candidates)):
        full_aug = _color_augment(full_tile)
        foot_aug = _color_augment(foot_tile)
        if random.random() < 0.5:
            full_aug = ImageOps.mirror(full_aug)
            foot_aug = ImageOps.mirror(foot_aug)
            geometry[index, 1] = 1.0 - geometry[index, 1]
        full.append(full_aug)
        foot.append(foot_aug)
    return RouteV7Tiles(prompt, full, foot, geometry)


def normalize_route_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
    values = tensor.float().div(255.0) if tensor.dtype == torch.uint8 else tensor.float()
    shape = [1] * values.ndim
    shape[-3] = 3
    mean = values.new_tensor(IMAGENET_MEAN).view(shape)
    std = values.new_tensor(IMAGENET_STD).view(shape)
    return values.sub(mean).div(std)


def _image_tensor(image: Image.Image, *, normalize: bool = True) -> torch.Tensor:
    tensor = TF.pil_to_tensor(image)
    return normalize_route_image_tensor(tensor) if normalize else tensor


class RouteV7Dataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        image_size: int = 128,
        foot_crop_size: int = 112,
        training: bool = False,
        permute_candidates: bool = True,
        cache_dir: str | Path | None = None,
        defer_normalization: bool = False,
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.image_size = int(image_size)
        self.foot_crop_size = int(foot_crop_size)
        self.training = bool(training)
        self.permute_candidates = bool(permute_candidates)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.defer_normalization = bool(defer_normalization)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = str(row["image"])
        answer = int(row.get("answer_index", 0))
        if not 0 <= answer < 10:
            raise ValueError(f"answer_index out of range: {answer}")
        tiles = load_or_build_route_v7_tiles(
            image_path,
            image_size=self.image_size,
            foot_crop_size=self.foot_crop_size,
            cache_dir=self.cache_dir,
            content_hash=str(row.get("content_sha256") or ""),
        )
        if self.training:
            tiles = _augment_tiles(tiles)
        order = list(range(10))
        if self.training and self.permute_candidates:
            random.shuffle(order)
        permuted = permute_candidate_data(
            tiles.full_candidates,
            tiles.foot_candidates,
            tiles.geometry,
            answer_index=answer,
            order=order,
        )
        return {
            "prompt": _image_tensor(tiles.prompt, normalize=not self.defer_normalization),
            "full_candidates": torch.stack(
                [
                    _image_tensor(tile, normalize=not self.defer_normalization)
                    for tile in permuted.full_candidates
                ]
            ),
            "foot_candidates": torch.stack(
                [
                    _image_tensor(tile, normalize=not self.defer_normalization)
                    for tile in permuted.foot_candidates
                ]
            ),
            "geometry": torch.from_numpy(permuted.geometry.copy()),
            "answer": torch.tensor(permuted.answer_index, dtype=torch.long),
            "candidate_order": torch.tensor(permuted.order, dtype=torch.long),
            "image": image_path,
        }


class RouteV7SetRanker(nn.Module):
    """Permutation-equivariant prompt/full/foot candidate-set ranker."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        embedding_dim: int = 128,
        hidden: int = 192,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        if int(hidden) % int(attention_heads):
            raise ValueError("hidden must be divisible by attention_heads")
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        network = mobilenet_v3_small(weights=weights)
        feature_dim = int(network.classifier[0].in_features)
        self.backbone = network.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        def projection() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(feature_dim, int(embedding_dim)),
                nn.LayerNorm(int(embedding_dim)),
                nn.GELU(),
            )

        self.prompt_projection = projection()
        self.full_projection = projection()
        self.foot_projection = projection()
        relation_dim = int(embedding_dim) * 8 + GEOMETRY_DIM
        self.fusion = nn.Sequential(
            nn.Linear(relation_dim, int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden),
            nhead=int(attention_heads),
            dim_feedforward=int(hidden) * 3,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(transformer_layers),
            enable_nested_tensor=False,
        )
        self.scorer = nn.Sequential(
            nn.Linear(int(hidden) * 4, int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 1),
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = bool(trainable)

    def _encode_raw(self, images: torch.Tensor) -> torch.Tensor:
        return self.pool(self.backbone(images)).flatten(1)

    def forward(
        self,
        prompt: torch.Tensor,
        full_candidates: torch.Tensor,
        foot_candidates: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        if prompt.ndim != 4:
            raise ValueError(f"expected prompt [B,3,H,W], got {tuple(prompt.shape)}")
        if full_candidates.ndim != 5 or full_candidates.shape[1] != 10:
            raise ValueError(f"expected full_candidates [B,10,3,H,W], got {tuple(full_candidates.shape)}")
        if foot_candidates.shape != full_candidates.shape:
            raise ValueError("foot_candidates must have the same shape as full_candidates")
        if geometry.shape != (prompt.shape[0], 10, GEOMETRY_DIM):
            raise ValueError(f"expected geometry [B,10,{GEOMETRY_DIM}], got {tuple(geometry.shape)}")
        batch = int(prompt.shape[0])
        images = torch.cat((prompt[:, None], full_candidates, foot_candidates), dim=1).reshape(
            batch * 21, *prompt.shape[1:]
        )
        raw = self._encode_raw(images).reshape(batch, 21, -1)
        prompt_embedding = self.prompt_projection(raw[:, 0])[:, None].expand(-1, 10, -1)
        full_embedding = self.full_projection(raw[:, 1:11])
        foot_embedding = self.foot_projection(raw[:, 11:21])
        relation = torch.cat(
            (
                full_embedding,
                foot_embedding,
                prompt_embedding,
                torch.abs(foot_embedding - prompt_embedding),
                foot_embedding * prompt_embedding,
                torch.abs(full_embedding - prompt_embedding),
                full_embedding * prompt_embedding,
                torch.abs(full_embedding - foot_embedding),
                geometry.to(dtype=full_embedding.dtype),
            ),
            dim=-1,
        )
        token = self.fusion(relation)
        contextual = self.set_encoder(token)
        mean = contextual.mean(dim=1, keepdim=True).expand_as(contextual)
        maximum = contextual.max(dim=1, keepdim=True).values.expand_as(contextual)
        return self.scorer(torch.cat((contextual, token, mean, maximum), dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class LoadedRouteV7:
    model: RouteV7SetRanker
    image_size: int
    foot_crop_size: int
    embedding_dim: int
    hidden: int
    transformer_layers: int
    attention_heads: int
    dropout: float
    epoch: int
    checkpoint_path: Path
    metadata: dict


def build_route_v7_checkpoint_payload(
    model: RouteV7SetRanker,
    *,
    image_size: int,
    foot_crop_size: int,
    embedding_dim: int,
    hidden: int,
    transformer_layers: int,
    attention_heads: int,
    dropout: float,
    epoch: int,
    config: Mapping[str, object],
    validation_metrics: Mapping[str, float] | None = None,
    training_metrics: Mapping[str, float] | None = None,
    train_images: Sequence[str] | None = None,
    val_images: Sequence[str] | None = None,
    locked_images: Sequence[str] | None = None,
) -> dict:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "model_state": model.state_dict(),
        "image_size": int(image_size),
        "foot_crop_size": int(foot_crop_size),
        "embedding_dim": int(embedding_dim),
        "hidden": int(hidden),
        "transformer_layers": int(transformer_layers),
        "attention_heads": int(attention_heads),
        "dropout": float(dropout),
        "epoch": int(epoch),
        "config": dict(config),
        "validation_metrics": dict(validation_metrics or {}),
        "training_metrics": dict(training_metrics or {}),
        "train_images": list(train_images or []),
        "val_images": list(val_images or []),
        "locked_images": list(locked_images or []),
    }


def load_route_v7_checkpoint(checkpoint: str | Path, *, device: str | torch.device) -> LoadedRouteV7:
    path = Path(checkpoint)
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unexpected Route V7 checkpoint schema: {payload.get('schema_version')}")
    model = RouteV7SetRanker(
        pretrained=False,
        embedding_dim=int(payload["embedding_dim"]),
        hidden=int(payload["hidden"]),
        transformer_layers=int(payload["transformer_layers"]),
        attention_heads=int(payload["attention_heads"]),
        dropout=float(payload["dropout"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return LoadedRouteV7(
        model=model,
        image_size=int(payload["image_size"]),
        foot_crop_size=int(payload["foot_crop_size"]),
        embedding_dim=int(payload["embedding_dim"]),
        hidden=int(payload["hidden"]),
        transformer_layers=int(payload["transformer_layers"]),
        attention_heads=int(payload["attention_heads"]),
        dropout=float(payload["dropout"]),
        epoch=int(payload.get("epoch", 0)),
        checkpoint_path=path,
        metadata=dict(payload),
    )


def predict_route_v7_tensors(
    loaded: LoadedRouteV7,
    prompt: torch.Tensor,
    full_candidates: torch.Tensor,
    foot_candidates: torch.Tensor,
    geometry: torch.Tensor,
) -> dict:
    device = next(loaded.model.parameters()).device
    loaded.model.eval()
    started = time.perf_counter()
    # Keep the exported single-image path in float32. It is deterministic across
    # batch sizes and is the precision used by the packaged locked-test report.
    with torch.inference_mode():
        logits = loaded.model(
            prompt[None].to(device, non_blocking=True),
            full_candidates[None].to(device, non_blocking=True),
            foot_candidates[None].to(device, non_blocking=True),
            geometry[None].to(device, non_blocking=True),
        )[0]
        scores = torch.softmax(logits.float(), dim=0)
    values = [float(value) for value in scores.detach().cpu().tolist()]
    raw_logits = [float(value) for value in logits.detach().cpu().tolist()]
    order = sorted(range(10), key=lambda index: values[index], reverse=True)
    return {
        "answer_index": int(order[0]),
        "top_indices": [int(order[0]), int(order[1])],
        "scores": values,
        "logits": raw_logits,
        "confidence": float(values[order[0]]),
        "margin": float(values[order[0]] - values[order[1]]),
        "inference_seconds": float(time.perf_counter() - started),
    }


def make_route_v7_tta_variants(
    prompt: torch.Tensor,
    full_candidates: torch.Tensor,
    foot_candidates: torch.Tensor,
    geometry: torch.Tensor,
    *,
    prompt_quarter_turns: Sequence[int] = (0, 1, 2, 3),
    include_mirror: bool = True,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    turns = [int(value) % 4 for value in prompt_quarter_turns]
    if not turns:
        raise ValueError("prompt_quarter_turns is empty")
    variants: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for mirrored in ([False, True] if include_mirror else [False]):
        if mirrored:
            full = torch.flip(full_candidates, dims=(-1,))
            foot = torch.flip(foot_candidates, dims=(-1,))
            geo = geometry.clone()
            geo[:, 1] = 1.0 - geo[:, 1]
            base_prompt = torch.flip(prompt, dims=(-1,))
        else:
            full = full_candidates
            foot = foot_candidates
            geo = geometry
            base_prompt = prompt
        for turn in turns:
            variants.append(
                (
                    torch.rot90(base_prompt, turn, dims=(-2, -1)),
                    full,
                    foot,
                    geo,
                )
            )
    return variants


def predict_route_v7_tta_tensors(
    loaded: LoadedRouteV7,
    prompt: torch.Tensor,
    full_candidates: torch.Tensor,
    foot_candidates: torch.Tensor,
    geometry: torch.Tensor,
    *,
    prompt_quarter_turns: Sequence[int] = (0, 1, 2, 3),
    include_mirror: bool = True,
) -> dict:
    variants = make_route_v7_tta_variants(
        prompt,
        full_candidates,
        foot_candidates,
        geometry,
        prompt_quarter_turns=prompt_quarter_turns,
        include_mirror=include_mirror,
    )
    device = next(loaded.model.parameters()).device
    prompt_batch = torch.stack([variant[0] for variant in variants]).to(device, non_blocking=True)
    full_batch = torch.stack([variant[1] for variant in variants]).to(device, non_blocking=True)
    foot_batch = torch.stack([variant[2] for variant in variants]).to(device, non_blocking=True)
    geometry_batch = torch.stack([variant[3] for variant in variants]).to(device, non_blocking=True)
    started = time.perf_counter()
    loaded.model.eval()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        logits_batch = loaded.model(prompt_batch, full_batch, foot_batch, geometry_batch)
        probability_batch = torch.softmax(logits_batch.float(), dim=1)
        scores = probability_batch.mean(dim=0)
        logits = torch.log(scores.clamp_min(1e-12))
    values = [float(value) for value in scores.detach().cpu().tolist()]
    raw_logits = [float(value) for value in logits.detach().cpu().tolist()]
    order = sorted(range(10), key=lambda index: values[index], reverse=True)
    return {
        "answer_index": int(order[0]),
        "top_indices": [int(order[0]), int(order[1])],
        "scores": values,
        "logits": raw_logits,
        "confidence": float(values[order[0]]),
        "margin": float(values[order[0]] - values[order[1]]),
        "inference_seconds": float(time.perf_counter() - started),
        "tta_variants": len(variants),
    }


def predict_route_v7_image(
    loaded: LoadedRouteV7,
    image_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> dict:
    started = time.perf_counter()
    dataset = RouteV7Dataset(
        [{"image": str(image_path), "answer_index": 0}],
        image_size=loaded.image_size,
        foot_crop_size=loaded.foot_crop_size,
        training=False,
        permute_candidates=False,
        cache_dir=cache_dir,
    )
    item = dataset[0]
    preprocess_seconds = time.perf_counter() - started
    result = predict_route_v7_tensors(
        loaded,
        item["prompt"],
        item["full_candidates"],
        item["foot_candidates"],
        item["geometry"],
    )
    result["image"] = str(Path(image_path))
    result["preprocess_seconds"] = float(preprocess_seconds)
    result["total_seconds"] = float(time.perf_counter() - started)
    return result


def predict_route_v7_image_tta(
    loaded: LoadedRouteV7,
    image_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    prompt_quarter_turns: Sequence[int] = (0, 1, 2, 3),
    include_mirror: bool = True,
) -> dict:
    started = time.perf_counter()
    dataset = RouteV7Dataset(
        [{"image": str(image_path), "answer_index": 0}],
        image_size=loaded.image_size,
        foot_crop_size=loaded.foot_crop_size,
        training=False,
        permute_candidates=False,
        cache_dir=cache_dir,
    )
    item = dataset[0]
    preprocess_seconds = time.perf_counter() - started
    result = predict_route_v7_tta_tensors(
        loaded,
        item["prompt"],
        item["full_candidates"],
        item["foot_candidates"],
        item["geometry"],
        prompt_quarter_turns=prompt_quarter_turns,
        include_mirror=include_mirror,
    )
    result["image"] = str(Path(image_path))
    result["preprocess_seconds"] = float(preprocess_seconds)
    result["total_seconds"] = float(time.perf_counter() - started)
    return result
