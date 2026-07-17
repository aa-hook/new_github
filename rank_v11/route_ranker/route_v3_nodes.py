from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .dataset import read_labels
from .geometry import PersonDetection, detect_person
from .layout import DEFAULT_LAYOUT, RouteLayout, load_rgb, split_challenge


@dataclass(frozen=True)
class IconNode:
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    score: float
    area: float
    kind: str = "node"


@dataclass(frozen=True)
class CandidateTrace:
    index: int
    person_found: bool
    person_bbox: tuple[int, int, int, int]
    person_foot: tuple[float, float] | None
    nodes: list[IconNode]
    foot_node_index: int | None
    foot_node_distance: float | None
    positive: bool = False

    @property
    def foot_node(self) -> IconNode | None:
        if self.foot_node_index is None:
            return None
        if not 0 <= self.foot_node_index < len(self.nodes):
            return None
        return self.nodes[self.foot_node_index]


@dataclass(frozen=True)
class RouteV3Trace:
    image: str
    answer_index: int | None
    prompt_box: tuple[int, int, int, int]
    candidates: list[CandidateTrace]

    def to_dict(self) -> dict:
        return asdict(self)


def _pil_rgb_array(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"))


def _clip_box(box: Sequence[int | float], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    return max(0, min(x0, x1)), max(0, min(y0, y1)), min(w, max(x0, x1)), min(h, max(y0, y1))


def _iou(a: Sequence[int | float], b: Sequence[int | float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
    return float(inter / max(1.0, area_a + area_b - inter))


def _box_distance_to_box(a: Sequence[int | float], b: Sequence[int | float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return math.hypot(dx, dy)


def _contrast_score(gray: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    return float(crop.std() / 64.0)


def _edge_density(edges: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box
    crop = edges[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    return float((crop > 0).mean())


def _contour_nodes_from_mask(mask: np.ndarray, gray: np.ndarray, edges: np.ndarray) -> list[IconNode]:
    contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nodes: list[IconNode] = []
    h, w = gray.shape[:2]
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 45.0:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 9 or bh < 9 or bw > 70 or bh > 70:
            continue
        aspect = bw / max(1.0, float(bh))
        if aspect < 0.35 or aspect > 2.80:
            continue
        box = _clip_box((x - 2, y - 2, x + bw + 2, y + bh + 2), w, h)
        bx0, by0, bx1, by1 = box
        box_area = max(1, (bx1 - bx0) * (by1 - by0))
        extent = area / float(box_area)
        if extent < 0.10:
            continue
        score = 0.45 * min(1.5, extent * 2.0) + 0.35 * _edge_density(edges, box) + 0.20 * _contrast_score(gray, box)
        nodes.append(
            IconNode(
                bbox=box,
                center=((bx0 + bx1) / 2.0, (by0 + by1) / 2.0),
                score=float(score),
                area=float(area),
            )
        )
    return nodes


def _hough_circle_nodes(gray: np.ndarray, edges: np.ndarray) -> list[IconNode]:
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=18,
        param1=70,
        param2=14,
        minRadius=7,
        maxRadius=30,
    )
    if circles is None:
        return []
    h, w = gray.shape[:2]
    nodes: list[IconNode] = []
    for cx, cy, radius in np.round(circles[0, :]).astype(int):
        box = _clip_box((cx - radius - 3, cy - radius - 3, cx + radius + 3, cy + radius + 3), w, h)
        score = 0.55 + 0.25 * _edge_density(edges, box) + 0.20 * _contrast_score(gray, box)
        nodes.append(IconNode(bbox=box, center=(float(cx), float(cy)), score=float(score), area=float(math.pi * radius * radius), kind="circle"))
    return nodes


def non_max_nodes(nodes: Sequence[IconNode], *, iou_threshold: float = 0.38, center_threshold: float = 8.0) -> list[IconNode]:
    selected: list[IconNode] = []
    for node in sorted(nodes, key=lambda n: n.score, reverse=True):
        keep = True
        for prev in selected:
            if _iou(node.bbox, prev.bbox) > iou_threshold:
                keep = False
                break
            if math.hypot(node.center[0] - prev.center[0], node.center[1] - prev.center[1]) < center_threshold:
                keep = False
                break
        if keep:
            selected.append(node)
    return selected


def detect_icon_nodes(
    tile: Image.Image,
    *,
    max_nodes: int = 8,
    person_bbox: tuple[int, int, int, int] | None = None,
) -> list[IconNode]:
    """Detect visible circular/oval route icon nodes in one 200x200 candidate tile.

    This is intentionally a high-recall classical detector.  The later matcher
    decides which node is semantically the prompt icon; this function only
    proposes likely node locations.
    """
    arr = _pil_rgb_array(tile)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray_blur, 42, 118)

    # Route nodes are usually very dark/very bright bubbles relative to the
    # gray terrain background.  Edges alone catches outlines; threshold catches
    # filled black/white icons.
    _, high = cv2.threshold(gray_blur, 182, 255, cv2.THRESH_BINARY)
    _, low = cv2.threshold(gray_blur, 62, 255, cv2.THRESH_BINARY_INV)
    edge_mask = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    mask = cv2.bitwise_or(edge_mask, cv2.bitwise_or(high, low))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    nodes = _contour_nodes_from_mask(mask, gray, edges) + _hough_circle_nodes(gray, edges)
    if person_bbox is not None:
        # Do not discard occluded nodes under the person; only remove boxes that
        # are huge person-color artifacts far from node-like contrast.
        nodes = [
            n
            for n in nodes
            if _box_distance_to_box(n.bbox, person_bbox) < 95.0 or _contrast_score(gray, n.bbox) > 0.12
        ]
    selected = non_max_nodes(nodes)
    return selected[: max(0, int(max_nodes))]


def nearest_node_to_point(nodes: Sequence[IconNode], point: tuple[float, float]) -> tuple[int | None, IconNode | None, float | None]:
    if not nodes:
        return None, None, None
    px, py = point
    distances = [math.hypot(px - n.center[0], py - n.center[1]) for n in nodes]
    idx = int(np.argmin(distances))
    return idx, nodes[idx], float(distances[idx])


def _person_to_trace_fields(person: PersonDetection) -> tuple[bool, tuple[int, int, int, int], tuple[float, float] | None]:
    if not person.found:
        return False, (0, 0, 0, 0), None
    return True, tuple(int(v) for v in person.bbox), (float(person.foot[0]), float(person.foot[1]))


def analyze_challenge(
    image_path: str | Path,
    *,
    answer_index: int | None = None,
    layout: RouteLayout = DEFAULT_LAYOUT,
    max_nodes: int = 12,
) -> RouteV3Trace:
    img = load_rgb(image_path)
    _prompt, candidates = split_challenge(img, layout)
    traces: list[CandidateTrace] = []
    for idx, tile in enumerate(candidates):
        person = detect_person(tile)
        found, person_bbox, foot = _person_to_trace_fields(person)
        nodes = detect_icon_nodes(tile, max_nodes=max_nodes, person_bbox=person_bbox if found else None)
        if foot is not None:
            node_idx, _node, dist = nearest_node_to_point(nodes, foot)
        else:
            node_idx, dist = None, None
        traces.append(
            CandidateTrace(
                index=idx,
                person_found=found,
                person_bbox=person_bbox,
                person_foot=foot,
                nodes=nodes,
                foot_node_index=node_idx,
                foot_node_distance=dist,
                positive=(answer_index is not None and int(answer_index) == idx),
            )
        )
    return RouteV3Trace(
        image=str(image_path),
        answer_index=int(answer_index) if answer_index is not None else None,
        prompt_box=tuple(int(v) for v in layout.prompt_box),
        candidates=traces,
    )


def _safe_stem(path: str | Path) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in Path(path).stem)[:96]


def _crop_with_margin(im: Image.Image, box: Sequence[int | float], margin: int = 8) -> Image.Image:
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    w, h = im.size
    clipped = _clip_box((x0 - margin, y0 - margin, x1 + margin, y1 + margin), w, h)
    return im.crop(clipped)


def export_weak_pairs(
    trace: RouteV3Trace,
    out_dir: str | Path,
    *,
    layout: RouteLayout = DEFAULT_LAYOUT,
    include_all_nodes: bool = True,
) -> list[dict]:
    if trace.answer_index is None:
        raise ValueError("answer_index is required to export weak pairs")

    out = Path(out_dir)
    crops = out / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    img = load_rgb(trace.image)
    prompt = img.crop(layout.prompt_box)
    stem = _safe_stem(trace.image)
    prompt_path = crops / f"{stem}_prompt.jpg"
    prompt.save(prompt_path, quality=95)

    rows: list[dict] = []
    for cand in trace.candidates:
        tile = img.crop(layout.candidate_box(cand.index))
        positive_node_index = cand.foot_node_index if cand.index == trace.answer_index else None
        for node_idx, node in enumerate(cand.nodes):
            label = 1 if positive_node_index is not None and int(node_idx) == int(positive_node_index) else 0
            if not include_all_nodes and label == 0 and cand.index != trace.answer_index:
                continue
            node_crop_path = crops / f"{stem}_c{cand.index}_n{node_idx}_y{label}.jpg"
            _crop_with_margin(tile, node.bbox).save(node_crop_path, quality=95)
            rows.append(
                {
                    "image": trace.image,
                    "answer_index": trace.answer_index,
                    "candidate_index": cand.index,
                    "node_index": node_idx,
                    "label": int(label),
                    "prompt_crop": str(prompt_path),
                    "node_crop": str(node_crop_path),
                    "node_bbox": list(node.bbox),
                    "node_center": [float(node.center[0]), float(node.center[1])],
                    "node_score": float(node.score),
                    "person_bbox": list(cand.person_bbox),
                    "person_foot": list(cand.person_foot) if cand.person_foot is not None else None,
                    "foot_node_index": cand.foot_node_index,
                    "foot_node_distance": cand.foot_node_distance,
                }
            )

    jsonl = out / "weak_pairs.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def annotate_trace(
    trace: RouteV3Trace,
    out_path: str | Path,
    *,
    layout: RouteLayout = DEFAULT_LAYOUT,
) -> Path:
    img = load_rgb(trace.image)
    draw = ImageDraw.Draw(img)
    for cand in trace.candidates:
        ox, oy, x1, y1 = layout.candidate_box(cand.index)
        border = (0, 180, 0) if cand.positive else (115, 115, 115)
        width = 5 if cand.positive else 1
        draw.rectangle((ox, oy, x1 - 1, y1 - 1), outline=border, width=width)
        draw.text((ox + 4, oy + 4), f"c{cand.index} nodes={len(cand.nodes)}", fill=border)
        if cand.person_found:
            px0, py0, px1, py1 = cand.person_bbox
            draw.rectangle((ox + px0, oy + py0, ox + px1, oy + py1), outline=(255, 210, 0), width=3)
        if cand.person_foot is not None:
            fx, fy = cand.person_foot
            draw.ellipse((ox + fx - 4, oy + fy - 4, ox + fx + 4, oy + fy + 4), fill=(255, 0, 255))
        for node_idx, node in enumerate(cand.nodes):
            nx0, ny0, nx1, ny1 = node.bbox
            is_foot = cand.foot_node_index is not None and int(cand.foot_node_index) == node_idx
            if cand.positive and is_foot:
                color, nwidth = (255, 0, 0), 5
            elif is_foot:
                color, nwidth = (255, 140, 0), 4
            else:
                color, nwidth = (0, 255, 255), 2
            draw.rectangle((ox + nx0, oy + ny0, ox + nx1, oy + ny1), outline=color, width=nwidth)
            draw.text((ox + nx0, oy + max(0, ny0 - 12)), str(node_idx), fill=color)
        if cand.person_foot is not None and cand.foot_node is not None:
            fx, fy = cand.person_foot
            cx, cy = cand.foot_node.center
            draw.line((ox + fx, oy + fy, ox + cx, oy + cy), fill=(255, 0, 255), width=2)

    px0, py0, px1, py1 = layout.prompt_box
    draw.rectangle((px0, py0, px1 - 1, py1 - 1), outline=(0, 120, 255), width=4)
    draw.text((px0 + 4, py0 + 4), "prompt", fill=(0, 120, 255))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=95)
    return out


def _label_rows(path: str | Path, limit: int = 0) -> list[dict]:
    rows = read_labels(path)
    if limit and limit > 0:
        return rows[: int(limit)]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="V3 probe: detect person foot + visible icon nodes and export weak matcher pairs.")
    ap.add_argument("--image", default="")
    ap.add_argument("--answer-index", type=int, default=None)
    ap.add_argument("--labels", default="")
    ap.add_argument("--out-dir", default="route_ranker/review/route_v3_probe")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-nodes", type=int, default=12)
    ap.add_argument("--export-pairs", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict]
    if args.image:
        rows = [{"image": args.image, "answer_index": args.answer_index}]
    elif args.labels:
        rows = _label_rows(args.labels, args.limit)
    else:
        raise SystemExit("pass --image or --labels")

    all_pair_rows: list[dict] = []
    trace_rows: list[dict] = []
    for i, row in enumerate(rows):
        image = str(row["image"])
        answer = row.get("answer_index")
        trace = analyze_challenge(image, answer_index=int(answer) if answer is not None else None, max_nodes=args.max_nodes)
        stem = _safe_stem(image)
        ann = annotate_trace(trace, out / "annotated" / f"{i:04d}_a{answer}_{stem}.jpg")
        trace_rows.append({**trace.to_dict(), "annotated": str(ann)})
        if args.export_pairs and trace.answer_index is not None:
            pair_rows = export_weak_pairs(trace, out / "weak_pairs" / f"{i:04d}_{stem}")
            all_pair_rows.extend(pair_rows)
        node_counts = [len(c.nodes) for c in trace.candidates]
        foot_dists = [c.foot_node_distance for c in trace.candidates if c.foot_node_distance is not None]
        avg_dist = sum(foot_dists) / len(foot_dists) if foot_dists else 0.0
        print(f"{i+1:04d}/{len(rows)} answer={answer} nodes={node_counts} avg_foot_dist={avg_dist:.1f} annotated={ann}")

    (out / "traces.json").write_text(json.dumps(trace_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if all_pair_rows:
        with (out / "weak_pairs_all.jsonl").open("w", encoding="utf-8") as f:
            for row in all_pair_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"traces: {out / 'traces.json'}")
    if all_pair_rows:
        print(f"weak pairs: {out / 'weak_pairs_all.jsonl'} count={len(all_pair_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
