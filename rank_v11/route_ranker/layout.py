from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class RouteLayout:
    full_width: int = 2000
    full_height: int = 400
    candidate_count: int = 10
    candidate_size: int = 200
    prompt_box: Tuple[int, int, int, int] = (0, 200, 135, 400)

    def candidate_box(self, index: int) -> Tuple[int, int, int, int]:
        if not 0 <= index < self.candidate_count:
            raise ValueError(f"candidate index out of range: {index}")
        x0 = index * self.candidate_size
        return (x0, 0, x0 + self.candidate_size, self.candidate_size)


DEFAULT_LAYOUT = RouteLayout()


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def split_challenge(image: Image.Image, layout: RouteLayout = DEFAULT_LAYOUT) -> tuple[Image.Image, List[Image.Image]]:
    prompt = image.crop(layout.prompt_box)
    candidates = [image.crop(layout.candidate_box(i)) for i in range(layout.candidate_count)]
    return prompt, candidates


def make_review_sheet(
    image_path: str | Path,
    out_path: str | Path,
    answer_index: int | None = None,
    layout: RouteLayout = DEFAULT_LAYOUT,
    scale: int = 2,
) -> Path:
    img = load_rgb(image_path)
    prompt, candidates = split_challenge(img, layout)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tile = layout.candidate_size * scale
    cols = 5
    rows = 2
    gap_x = 20
    label_h = 26
    prompt_w = 135 * scale
    prompt_h = 200 * scale
    sheet_w = cols * tile + (cols - 1) * gap_x
    sheet_h = rows * (tile + label_h) + prompt_h + 70
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for i, cand in enumerate(candidates):
        x = (i % cols) * (tile + gap_x)
        y = (i // cols) * (tile + label_h) + label_h
        cand_big = cand.resize((tile, tile), Image.Resampling.LANCZOS)
        sheet.paste(cand_big, (x, y))
        color = (0, 180, 0) if answer_index == i else (220, 0, 0)
        draw.text((x + 4, y - label_h + 4), f"index {i}", fill=color)
        if answer_index == i:
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(0, 220, 0), width=5)

    py = rows * (tile + label_h) + 35
    prompt_big = prompt.resize((prompt_w, prompt_h), Image.Resampling.LANCZOS)
    sheet.paste(prompt_big, (0, py))
    draw.text((4, py - 24), "prompt", fill=(220, 0, 0))
    draw.text((prompt_w + 24, py + 10), Path(image_path).name, fill=(0, 0, 0))
    if answer_index is not None:
        draw.text((prompt_w + 24, py + 36), f"answer_index={answer_index}", fill=(0, 160, 0))

    sheet.save(out, quality=95)
    return out


def iter_images(image_dir: str | Path) -> Iterable[Path]:
    root = Path(image_dir)
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in suffixes:
            yield p
