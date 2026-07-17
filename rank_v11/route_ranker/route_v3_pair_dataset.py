from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from .dataset import read_labels
from .layout import DEFAULT_LAYOUT, RouteLayout, load_rgb
from .route_v3_nodes import _crop_with_margin, _safe_stem, analyze_challenge


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    rows: list[dict] = []
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def crop_prompt_icon(prompt: Image.Image, *, margin: int = 8) -> Image.Image:
    """Tightly crop the prompt icon from the blue prompt panel when possible."""
    arr = np.asarray(prompt.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 45, 130)
    # Ignore crop borders; those are panel edges, not the icon.
    edges[:3, :] = 0
    edges[-3:, :] = 0
    edges[:, :3] = 0
    edges[:, -3:] = 0
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    ys, xs = np.nonzero(edges)
    if xs.size < 20:
        return prompt
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    # Avoid returning the entire panel if background texture creates too many edges.
    if (x1 - x0) > prompt.width * 0.88 or (y1 - y0) > prompt.height * 0.88:
        return prompt
    return _crop_with_margin(prompt, (x0, y0, x1, y1), margin=margin)


def build_foot_pair_rows(
    label_rows: Sequence[dict],
    *,
    out_dir: str | Path,
    layout: RouteLayout = DEFAULT_LAYOUT,
    max_nodes: int = 12,
) -> list[dict]:
    """Export one prompt-vs-foot-node pair per candidate.

    Label is 1 only for the answer candidate's foot node, 0 for every other
    candidate's foot node. This directly trains the final candidate scorer and
    does not require every cyan node proposal to be correct.
    """
    out = Path(out_dir)
    crops = out / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for row_i, row in enumerate(label_rows):
        image = str(row["image"])
        answer = int(row["answer_index"])
        if not Path(image).exists():
            print(f"skip missing image: {image}")
            continue
        img = load_rgb(image)
        prompt = crop_prompt_icon(img.crop(layout.prompt_box))
        stem = f"{row_i:04d}_{_safe_stem(image)}"
        prompt_path = crops / f"{stem}_prompt.jpg"
        prompt.save(prompt_path, quality=95)

        trace = analyze_challenge(image, answer_index=answer, layout=layout, max_nodes=max_nodes)
        for cand in trace.candidates:
            if cand.foot_node is None:
                continue
            tile = img.crop(layout.candidate_box(cand.index))
            node_path = crops / f"{stem}_c{cand.index}_footnode_y{1 if cand.index == answer else 0}.jpg"
            _crop_with_margin(tile, cand.foot_node.bbox, margin=10).save(node_path, quality=95)
            rows.append(
                {
                    "image": image,
                    "answer_index": answer,
                    "candidate_index": int(cand.index),
                    "label": 1 if int(cand.index) == answer else 0,
                    "prompt_crop": str(prompt_path),
                    "node_crop": str(node_path),
                    "node_bbox": list(cand.foot_node.bbox),
                    "node_center": [float(cand.foot_node.center[0]), float(cand.foot_node.center[1])],
                    "node_score": float(cand.foot_node.score),
                    "person_bbox": list(cand.person_bbox),
                    "person_foot": list(cand.person_foot) if cand.person_foot is not None else None,
                    "foot_node_distance": cand.foot_node_distance,
                    "source": "route_v3_foot_pair",
                }
            )

    write_jsonl(out / "foot_pairs.jsonl", rows)
    return rows


def _augment_image(im: Image.Image) -> Image.Image:
    if random.random() < 0.75:
        im = im.rotate(random.uniform(-180, 180), resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(128, 128, 128))
    if random.random() < 0.35:
        im = ImageOps.invert(im.convert("RGB"))
    if random.random() < 0.45:
        im = ImageEnhance.Contrast(im).enhance(random.uniform(0.70, 1.45))
    if random.random() < 0.45:
        im = ImageEnhance.Brightness(im).enhance(random.uniform(0.78, 1.25))
    if random.random() < 0.25:
        im = ImageOps.grayscale(im).convert("RGB")
    return im


def image_to_tensor(im: Image.Image, *, image_size: int = 96, augment: bool = False) -> torch.Tensor:
    im = im.convert("RGB")
    if augment:
        im = _augment_image(im)
    im = ImageOps.contain(im, (image_size, image_size), method=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (image_size, image_size), (128, 128, 128))
    canvas.paste(im, ((image_size - im.width) // 2, (image_size - im.height) // 2))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class RouteV3PairDataset(Dataset):
    def __init__(self, rows: Sequence[dict], *, image_size: int = 96, augment: bool = False) -> None:
        self.rows = list(rows)
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[int(idx)]
        prompt = Image.open(row["prompt_crop"]).convert("RGB")
        node = Image.open(row["node_crop"]).convert("RGB")
        return {
            "prompt": image_to_tensor(prompt, image_size=self.image_size, augment=self.augment),
            "node": image_to_tensor(node, image_size=self.image_size, augment=self.augment),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "image": str(row.get("image", "")),
            "candidate_index": int(row.get("candidate_index", -1)),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build V3 prompt-vs-foot-node pair dataset.")
    ap.add_argument("--labels", default="route_ranker/data/labels.jsonl")
    ap.add_argument("--out-dir", default="route_ranker/datasets/route_v3_foot_pairs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-nodes", type=int, default=12)
    args = ap.parse_args()

    rows = read_labels(args.labels)
    if args.limit and args.limit > 0:
        rows = rows[: int(args.limit)]
    pair_rows = build_foot_pair_rows(rows, out_dir=args.out_dir, max_nodes=args.max_nodes)
    positives = sum(int(r["label"]) for r in pair_rows)
    negatives = len(pair_rows) - positives
    print(f"labels      : {len(rows)}")
    print(f"pairs       : {len(pair_rows)}")
    print(f"positives   : {positives}")
    print(f"negatives   : {negatives}")
    print(f"out         : {Path(args.out_dir) / 'foot_pairs.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
