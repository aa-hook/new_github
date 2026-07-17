from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, List, Sequence

import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .layout import DEFAULT_LAYOUT, RouteLayout, load_rgb, split_challenge


class RouteRankerDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict],
        image_size: int = 160,
        prompt_size: int = 128,
        augment: bool = False,
        layout: RouteLayout = DEFAULT_LAYOUT,
        prompt_rotate_max: float = 180.0,
        candidate_rotate_max: float = 25.0,
        candidate_full_rotate_prob: float = 0.15,
    ):
        self.rows = list(rows)
        self.image_size = int(image_size)
        self.prompt_size = int(prompt_size)
        self.augment = augment
        self.layout = layout
        self.prompt_rotate_max = float(prompt_rotate_max)
        self.candidate_rotate_max = float(candidate_rotate_max)
        self.candidate_full_rotate_prob = float(candidate_full_rotate_prob)

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _pad_to_square(im: Image.Image, fill: tuple[int, int, int] = (92, 117, 146)) -> Image.Image:
        """Pad a non-square prompt crop before rotation so the icon is not cropped.

        Candidate tiles are already square. Prompt crops are about 135x200, so a
        60/90/180-degree rotation would otherwise cut off the icon.
        """
        if im.width == im.height:
            return im
        side = max(im.width, im.height)
        bg = Image.new("RGB", (side, side), fill)
        bg.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        return bg

    @staticmethod
    def _rotate_safe(im: Image.Image, angle: float, fill: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
        if abs(angle) < 1e-3:
            return im
        return im.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=fill)

    def _augment_common(self, im: Image.Image) -> Image.Image:
        if not self.augment:
            return im
        if random.random() < 0.35:
            im = ImageOps.autocontrast(im)
        if random.random() < 0.30:
            im = ImageOps.invert(im)
        if random.random() < 0.45:
            im = ImageEnhance.Brightness(im).enhance(random.uniform(0.75, 1.25))
        if random.random() < 0.45:
            im = ImageEnhance.Contrast(im).enhance(random.uniform(0.75, 1.35))
        return im

    def _augment_prompt(self, im: Image.Image) -> Image.Image:
        # The prompt and candidate icon can differ by large rotations or color
        # inversion. Padding + random large rotation teaches the prompt encoder
        # to represent icon identity rather than canvas orientation.
        im = self._pad_to_square(im)
        if self.augment and self.prompt_rotate_max > 0:
            im = self._rotate_safe(im, random.uniform(-self.prompt_rotate_max, self.prompt_rotate_max), fill=(92, 117, 146))
        return self._augment_common(im)

    def _augment_candidate(self, im: Image.Image) -> Image.Image:
        # Candidate tiles preserve the person/icon overlap under global rotation,
        # so rotating the whole tile is label-preserving.  Most samples only need
        # moderate viewpoint jitter; occasionally use a full 0..360 rotation to
        # prevent overfitting to canvas orientation.
        if self.augment:
            if random.random() < self.candidate_full_rotate_prob:
                angle = random.uniform(-180, 180)
            else:
                angle = random.uniform(-self.candidate_rotate_max, self.candidate_rotate_max)
            im = self._rotate_safe(im, angle, fill=(128, 128, 128))
        return self._augment_common(im)

    @staticmethod
    def _to_tensor(im: Image.Image, size: int) -> torch.Tensor:
        # Preserve aspect ratio for prompt crops; candidates are already square.
        if im.width != im.height:
            side = max(im.width, im.height)
            bg = Image.new("RGB", (side, side), (128, 128, 128))
            bg.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
            im = bg
        im = im.resize((size, size), Image.Resampling.BILINEAR)
        x = TF.to_tensor(im)
        # Images are largely grayscale. Imagenet normalization is not required;
        # keep values in a stable small range.
        return (x - 0.5) / 0.5

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image_path = row["image"]
        answer = int(row["answer_index"])
        img = load_rgb(image_path)
        prompt, candidates = split_challenge(img, self.layout)
        prompt = self._augment_prompt(prompt)
        cand_tensors = []
        for c in candidates:
            cand_tensors.append(self._to_tensor(self._augment_candidate(c), self.image_size))
        prompt_tensor = self._to_tensor(prompt, self.prompt_size)
        return {
            "prompt": prompt_tensor,
            "candidates": torch.stack(cand_tensors, dim=0),  # 10,C,H,W
            "answer": torch.tensor(answer, dtype=torch.long),
            "image": image_path,
        }


def read_labels(path: str | Path) -> List[dict]:
    rows = []
    p = Path(path)
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "image" not in rec or "answer_index" not in rec:
            continue
        if not Path(rec["image"]).exists():
            continue
        ans = int(rec["answer_index"])
        if 0 <= ans <= 9:
            rec["answer_index"] = ans
            rows.append(rec)
    return rows


def split_rows(rows: Sequence[dict], val_ratio: float = 0.2, seed: int = 1337) -> tuple[list[dict], list[dict]]:
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if len(rows) < 4:
        return rows, []
    n_val = max(1, int(round(len(rows) * val_ratio)))
    return rows[n_val:], rows[:n_val]
