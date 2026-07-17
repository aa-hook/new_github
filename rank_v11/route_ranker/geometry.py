from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .layout import DEFAULT_LAYOUT, RouteLayout, load_rgb, split_challenge


FEATURE_NAMES = [
    "person_found",
    "person_area_norm",
    "person_width_norm",
    "person_height_norm",
    "person_center_x",
    "person_center_y",
    "person_foot_x",
    "person_foot_y",
    "icon_match_score",
    "icon_center_x",
    "icon_center_y",
    "icon_width_norm",
    "icon_height_norm",
    "top2_icon_match_score",
    "top3_icon_match_score",
    "near_icon_match_score",
    "near_icon_center_x",
    "near_icon_center_y",
    "near_foot_to_icon_dist",
    "near_center_to_icon_dist",
    "near_person_icon_overlap",
    "near_icon_person_coverage",
    "foot_to_icon_dist",
    "center_to_icon_dist",
    "person_icon_overlap",
    "icon_person_coverage",
    "edge_density",
    "sat_density",
    "heuristic_score",
]


@dataclass(frozen=True)
class PersonDetection:
    found: bool
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    foot: tuple[float, float]
    area: int
    mask: np.ndarray


@dataclass(frozen=True)
class IconMatch:
    score: float
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    rotation: float
    target_size: int


def _pil_to_rgb_array(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"))


def _largest_component(mask: np.ndarray, min_area: int = 20) -> tuple[bool, np.ndarray, tuple[int, int, int, int], int]:
    mask_u8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return False, np.zeros_like(mask, dtype=bool), (0, 0, 0, 0), 0
    best_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[best_idx, cv2.CC_STAT_AREA])
    if area < min_area:
        return False, np.zeros_like(mask, dtype=bool), (0, 0, 0, 0), 0
    x = int(stats[best_idx, cv2.CC_STAT_LEFT])
    y = int(stats[best_idx, cv2.CC_STAT_TOP])
    w = int(stats[best_idx, cv2.CC_STAT_WIDTH])
    h = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
    comp = labels == best_idx
    return True, comp, (x, y, x + w, y + h), area


def detect_person(tile: Image.Image, *, min_area: int = 90) -> PersonDetection:
    """Detect the colored character in one 200x200 candidate tile.

    Arkose route tiles are mostly grayscale.  The character has yellow/green
    clothing and skin highlights, so saturation/colorfulness is a useful weak
    detector.  The mask is intentionally dilated to merge head/body/feet into a
    single geometry object for distance-to-icon scoring.
    """
    arr = _pil_to_rgb_array(tile)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    colorfulness = arr.max(axis=2).astype(np.int16) - arr.min(axis=2).astype(np.int16)
    mask = ((sat > 26) & (val > 60)) | ((colorfulness > 22) & (val > 75))

    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=2)
    found, comp, bbox, area = _largest_component(mask_u8 > 0, min_area=min_area)
    if not found:
        return PersonDetection(False, (0, 0, 0, 0), (0.0, 0.0), (0.0, 0.0), 0, comp)
    x0, y0, x1, y1 = bbox
    # The saturated mask reliably captures helmet/body highlights, but boots
    # are often gray/white and may be missed.  Expand the geometry bbox toward
    # the feet so the "standing on icon" distance uses a plausible foot point.
    pad_x = max(4, int(round((x1 - x0) * 0.18)))
    pad_top = max(2, int(round((y1 - y0) * 0.05)))
    pad_bottom = max(12, int(round((y1 - y0) * 0.35)))
    x0 = max(0, x0 - pad_x)
    x1 = min(arr.shape[1], x1 + pad_x)
    y0 = max(0, y0 - pad_top)
    y1 = min(arr.shape[0], y1 + pad_bottom)
    ys, xs = np.nonzero(comp)
    center = (float(xs.mean()) if xs.size else (x0 + x1) / 2.0, float(ys.mean()) if ys.size else (y0 + y1) / 2.0)
    foot = ((x0 + x1) / 2.0, float(y1))
    return PersonDetection(True, (x0, y0, x1, y1), center, foot, area, comp)


def _edge_map(im: Image.Image) -> np.ndarray:
    arr = _pil_to_rgb_array(im)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 120)
    return edges


def _bbox_from_mask(mask: np.ndarray, margin: int = 2) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return (0, 0, mask.shape[1], mask.shape[0])
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(mask.shape[1], int(xs.max()) + 1 + margin)
    y1 = min(mask.shape[0], int(ys.max()) + 1 + margin)
    return x0, y0, x1, y1


def _crop_nonzero(arr: np.ndarray, margin: int = 2) -> np.ndarray:
    x0, y0, x1, y1 = _bbox_from_mask(arr > 0, margin=margin)
    return arr[y0:y1, x0:x1]


def _rotate_binary_template(template: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-6:
        return template
    h, w = template.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    mat[0, 2] += nw / 2.0 - center[0]
    mat[1, 2] += nh / 2.0 - center[1]
    rot = cv2.warpAffine(template, mat, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    return _crop_nonzero(rot, margin=1)


def _resize_template_to_max_dim(template: np.ndarray, target_size: int) -> np.ndarray:
    h, w = template.shape[:2]
    max_dim = max(h, w)
    if max_dim <= 0:
        return template
    scale = float(target_size) / float(max_dim)
    nw = max(6, int(round(w * scale)))
    nh = max(6, int(round(h * scale)))
    return cv2.resize(template, (nw, nh), interpolation=cv2.INTER_AREA)


def _safe_match_template(image: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int]]:
    if template.shape[0] >= image.shape[0] or template.shape[1] >= image.shape[1]:
        return -1.0, (0, 0)
    img = image.astype(np.float32) / 255.0
    tmpl = template.astype(np.float32) / 255.0
    if float(tmpl.std()) < 1e-4:
        return -1.0, (0, 0)
    # Binary edge templates are sparse; normalized correlation rewards edge
    # overlap better than CCOEFF here and is less tempted by arbitrary route
    # line segments in the background.
    res = cv2.matchTemplate(img, tmpl, cv2.TM_CCORR_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    if not np.isfinite(max_val):
        return -1.0, (0, 0)
    return float(max_val), (int(max_loc[0]), int(max_loc[1]))


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return float(inter / max(1, area_a + area_b - inter))


def _candidate_from_template_response(
    res: np.ndarray,
    tmpl_shape: tuple[int, int],
    rotation: float,
    target_size: int,
    *,
    peaks_per_template: int,
) -> list[IconMatch]:
    work = res.copy()
    h, w = tmpl_shape[:2]
    out: list[IconMatch] = []
    for _ in range(peaks_per_template):
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(work)
        if not np.isfinite(max_val):
            break
        x0, y0 = int(max_loc[0]), int(max_loc[1])
        bbox = (x0, y0, x0 + w, y0 + h)
        out.append(IconMatch(float(max_val), bbox, (x0 + w / 2.0, y0 + h / 2.0), float(rotation), int(target_size)))
        # Suppress a neighborhood around the selected peak for this template.
        sx0 = max(0, x0 - max(4, w // 2))
        sy0 = max(0, y0 - max(4, h // 2))
        sx1 = min(work.shape[1], x0 + max(4, w // 2))
        sy1 = min(work.shape[0], y0 + max(4, h // 2))
        work[sy0:sy1, sx0:sx1] = -1.0
    return out


def find_prompt_icon_matches(
    prompt: Image.Image,
    tile: Image.Image,
    *,
    top_k: int = 6,
    rotations: Sequence[float] = tuple(range(0, 360, 30)),
    target_sizes: Sequence[int] = (18, 22, 26, 30, 34, 38, 44, 50),
) -> list[IconMatch]:
    """Find the prompt icon inside one candidate using edge template matching.

    This is deliberately rotation/color tolerant: matching is performed on Canny
    edges, and the prompt template is rotated and resized across a small search
    grid.  It is a weak detector, not a final solver.
    """
    prompt_edges = _crop_nonzero(_edge_map(prompt), margin=4)
    candidate_edges = _edge_map(tile)
    candidates: list[IconMatch] = []
    for rot in rotations:
        tmpl_rot = _rotate_binary_template(prompt_edges, float(rot))
        if tmpl_rot.size == 0 or tmpl_rot.shape[0] < 6 or tmpl_rot.shape[1] < 6:
            continue
        for size in target_sizes:
            tmpl = _resize_template_to_max_dim(tmpl_rot, int(size))
            if tmpl.shape[0] < 6 or tmpl.shape[1] < 6:
                continue
            if tmpl.shape[0] >= candidate_edges.shape[0] or tmpl.shape[1] >= candidate_edges.shape[1]:
                continue
            img = candidate_edges.astype(np.float32) / 255.0
            tmpl_f = tmpl.astype(np.float32) / 255.0
            if float(tmpl_f.std()) < 1e-4:
                continue
            res = cv2.matchTemplate(img, tmpl_f, cv2.TM_CCORR_NORMED)
            candidates.extend(
                _candidate_from_template_response(
                    res,
                    tmpl.shape,
                    float(rot),
                    int(size),
                    peaks_per_template=3,
                )
            )

    if not candidates:
        return [IconMatch(-1.0, (0, 0, 0, 0), (0.0, 0.0), 0.0, 0)]
    candidates.sort(key=lambda m: m.score, reverse=True)
    selected: list[IconMatch] = []
    for cand in candidates:
        if any(_bbox_iou(cand.bbox, prev.bbox) > 0.35 for prev in selected):
            continue
        selected.append(cand)
        if len(selected) >= top_k:
            break
    return selected


def match_prompt_icon(
    prompt: Image.Image,
    tile: Image.Image,
    *,
    rotations: Sequence[float] = tuple(range(0, 360, 30)),
    target_sizes: Sequence[int] = (18, 22, 26, 30, 34, 38, 44, 50),
) -> IconMatch:
    return find_prompt_icon_matches(prompt, tile, top_k=1, rotations=rotations, target_sizes=target_sizes)[0]


def _expanded_bbox(bbox: tuple[int, int, int, int], w: int, h: int, margin: int = 5) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return max(0, x0 - margin), max(0, y0 - margin), min(w, x1 + margin), min(h, y1 + margin)


def candidate_geometry_features(prompt: Image.Image, tile: Image.Image) -> tuple[np.ndarray, dict]:
    person = detect_person(tile)
    matches = find_prompt_icon_matches(prompt, tile, top_k=6)
    match = matches[0]
    w, h = tile.size
    diag = float((w * w + h * h) ** 0.5)

    def overlap_stats(icon_match: IconMatch) -> tuple[float, float, float, float]:
        icon_box = _expanded_bbox(icon_match.bbox, w, h, margin=6)
        ix0, iy0, ix1, iy1 = icon_box
        icon_area = max(1, (ix1 - ix0) * (iy1 - iy0))
        if person.found:
            person_slice = person.mask[iy0:iy1, ix0:ix1]
            overlap_px = int(person_slice.sum())
            p_overlap = overlap_px / max(1, person.area)
            i_coverage = overlap_px / icon_area
            f_dist = float(np.hypot(person.foot[0] - icon_match.center[0], person.foot[1] - icon_match.center[1]) / diag)
            c_dist = float(np.hypot(person.center[0] - icon_match.center[0], person.center[1] - icon_match.center[1]) / diag)
            return f_dist, c_dist, p_overlap, i_coverage
        return 1.0, 1.0, 0.0, 0.0

    def near_rank(m: IconMatch) -> float:
        f_dist, c_dist, p_overlap, i_coverage = overlap_stats(m)
        return float(m.score) - 1.20 * f_dist - 0.25 * c_dist + 0.75 * p_overlap + 0.35 * i_coverage

    near_match = max(matches, key=near_rank)
    foot_dist, center_dist, person_icon_overlap, icon_person_coverage = overlap_stats(match)
    near_foot_dist, near_center_dist, near_person_icon_overlap, near_icon_person_coverage = overlap_stats(near_match)
    top2 = float(matches[1].score) if len(matches) > 1 else 0.0
    top3 = float(matches[2].score) if len(matches) > 2 else 0.0

    arr = _pil_to_rgb_array(tile)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat_density = float((hsv[:, :, 1] > 26).mean())
    edge_density = float((_edge_map(tile) > 0).mean())

    x0, y0, x1, y1 = person.bbox
    mx0, my0, mx1, my1 = match.bbox
    heuristic = (
        float(near_match.score)
        - 1.35 * near_foot_dist
        - 0.35 * near_center_dist
        + 1.20 * near_person_icon_overlap
        + 0.65 * near_icon_person_coverage
        + (0.08 if person.found else -0.08)
    )

    feats = np.asarray(
        [
            1.0 if person.found else 0.0,
            person.area / float(w * h),
            (x1 - x0) / float(w),
            (y1 - y0) / float(h),
            person.center[0] / float(w),
            person.center[1] / float(h),
            person.foot[0] / float(w),
            person.foot[1] / float(h),
            float(match.score),
            match.center[0] / float(w),
            match.center[1] / float(h),
            (mx1 - mx0) / float(w),
            (my1 - my0) / float(h),
            top2,
            top3,
            float(near_match.score),
            near_match.center[0] / float(w),
            near_match.center[1] / float(h),
            near_foot_dist,
            near_center_dist,
            near_person_icon_overlap,
            near_icon_person_coverage,
            foot_dist,
            center_dist,
            person_icon_overlap,
            icon_person_coverage,
            edge_density,
            sat_density,
            heuristic,
        ],
        dtype=np.float32,
    )
    debug = {
        "person_found": person.found,
        "person_bbox": tuple(int(v) for v in person.bbox),
        "person_center": tuple(float(v) for v in person.center),
        "person_foot": tuple(float(v) for v in person.foot),
        "person_area": int(person.area),
        "icon_bbox": tuple(int(v) for v in match.bbox),
        "icon_center": tuple(float(v) for v in match.center),
        "icon_score": float(match.score),
        "icon_rotation": float(match.rotation),
        "icon_target_size": int(match.target_size),
        "near_icon_bbox": tuple(int(v) for v in near_match.bbox),
        "near_icon_center": tuple(float(v) for v in near_match.center),
        "near_icon_score": float(near_match.score),
        "near_icon_rotation": float(near_match.rotation),
        "matches": [
            {
                "score": float(m.score),
                "bbox": tuple(int(v) for v in m.bbox),
                "center": tuple(float(v) for v in m.center),
                "rotation": float(m.rotation),
                "target_size": int(m.target_size),
            }
            for m in matches
        ],
        "heuristic_score": float(heuristic),
    }
    return feats, debug


def extract_challenge_geometry_features(
    image_path: str | Path,
    *,
    layout: RouteLayout = DEFAULT_LAYOUT,
) -> tuple[np.ndarray, dict]:
    image_path = str(image_path)
    img = load_rgb(image_path)
    prompt, candidates = split_challenge(img, layout)
    rows: list[np.ndarray] = []
    cand_debug: list[dict] = []
    for index, candidate in enumerate(candidates):
        feats, dbg = candidate_geometry_features(prompt, candidate)
        dbg["index"] = index
        rows.append(feats)
        cand_debug.append(dbg)
    debug = {"image": image_path, "feature_names": FEATURE_NAMES, "candidates": cand_debug}
    return np.stack(rows, axis=0).astype(np.float32), debug


def annotate_geometry(
    image_path: str | Path,
    out_path: str | Path,
    *,
    answer_index: int | None = None,
    pred_index: int | None = None,
    scores: Sequence[float] | None = None,
    layout: RouteLayout = DEFAULT_LAYOUT,
) -> Path:
    img = load_rgb(image_path)
    _features, debug = extract_challenge_geometry_features(image_path, layout=layout)
    draw = ImageDraw.Draw(img)
    for cand in debug["candidates"]:
        i = int(cand["index"])
        ox, oy, _, _ = layout.candidate_box(i)
        pb = cand["person_bbox"]
        ib = cand["icon_bbox"]
        px0, py0, px1, py1 = [int(v) for v in pb]
        ix0, iy0, ix1, iy1 = [int(v) for v in ib]
        border = (80, 80, 80)
        width = 1
        if answer_index is not None and i == answer_index:
            border = (0, 140, 255)
            width = 5
        if pred_index is not None and i == pred_index:
            border = (0, 230, 0) if answer_index == pred_index else (255, 0, 0)
            width = 6
        x0, y0, x1, y1 = layout.candidate_box(i)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=border, width=width)
        draw.rectangle((ox + px0, oy + py0, ox + px1, oy + py1), outline=(255, 210, 0), width=3)
        draw.rectangle((ox + ix0, oy + iy0, ox + ix1, oy + iy1), outline=(0, 255, 255), width=3)
        fx, fy = cand["person_foot"]
        cx, cy = cand["icon_center"]
        draw.line((ox + fx, oy + fy, ox + cx, oy + cy), fill=(255, 0, 255), width=2)
        score_txt = f"h={cand['heuristic_score']:.2f} m={cand['icon_score']:.2f}"
        if scores is not None:
            score_txt = f"s={float(scores[i]):.2f} " + score_txt
        draw.text((ox + 4, oy + 4), f"{i} {score_txt}", fill=border)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=95)
    return out


def heuristic_scores_for_image(image_path: str | Path) -> tuple[np.ndarray, dict]:
    feats, debug = extract_challenge_geometry_features(image_path)
    h_idx = FEATURE_NAMES.index("heuristic_score")
    return feats[:, h_idx].copy(), debug
