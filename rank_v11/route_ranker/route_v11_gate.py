from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch


def _model_outputs(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    probabilities = logits.float().softmax(dim=1)
    top2 = probabilities.topk(2, dim=1)
    return {
        "probabilities": probabilities,
        "predictions": top2.indices[:, 0],
        "top2": top2.indices,
        "confidence": top2.values[:, 0],
        "margin": top2.values[:, 0] - top2.values[:, 1],
    }


def select_v11_predictions(
    v10_logits: torch.Tensor,
    v9_logits: torch.Tensor,
    expert_logits: torch.Tensor,
    *,
    margin_advantage: float,
    require_legacy_consensus: bool = True,
) -> dict[str, torch.Tensor]:
    shapes = {tuple(v10_logits.shape), tuple(v9_logits.shape), tuple(expert_logits.shape)}
    if len(shapes) != 1 or v10_logits.ndim != 2:
        raise ValueError(
            "V11 logits must have equal [N,C] shapes: "
            f"v10={tuple(v10_logits.shape)} v9={tuple(v9_logits.shape)} "
            f"expert={tuple(expert_logits.shape)}"
        )
    v10 = _model_outputs(v10_logits)
    v9 = _model_outputs(v9_logits)
    expert = _model_outputs(expert_logits)
    legacy_consensus = v10["predictions"] == v9["predictions"]
    disagreements = v10["predictions"] != expert["predictions"]
    stronger_v10 = v10["margin"] >= expert["margin"] + float(margin_advantage)
    switched = disagreements & stronger_v10
    if require_legacy_consensus:
        switched &= legacy_consensus
    predictions = torch.where(switched, v10["predictions"], expert["predictions"])
    selected_probabilities = torch.where(
        switched[:, None], v10["probabilities"], expert["probabilities"]
    )
    selected_top2 = selected_probabilities.topk(2, dim=1)
    return {
        "predictions": predictions,
        "probabilities": selected_probabilities,
        "top2": selected_top2.indices,
        "confidence": selected_top2.values[:, 0],
        "margin": selected_top2.values[:, 0] - selected_top2.values[:, 1],
        "v10_predictions": v10["predictions"],
        "v9_predictions": v9["predictions"],
        "expert_predictions": expert["predictions"],
        "v10_top2": v10["top2"],
        "v9_top2": v9["top2"],
        "expert_top2": expert["top2"],
        "v10_margins": v10["margin"],
        "v9_margins": v9["margin"],
        "expert_margins": expert["margin"],
        "legacy_consensus": legacy_consensus,
        "disagreements": disagreements,
        "switched": switched,
    }


def evaluate_v11_logits(
    images: Sequence[str],
    answers: torch.Tensor,
    v10_logits: torch.Tensor,
    v9_logits: torch.Tensor,
    expert_logits: torch.Tensor,
    *,
    margin_advantage: float,
    require_legacy_consensus: bool = True,
) -> tuple[dict, list[dict]]:
    if answers.shape != (len(images),):
        raise ValueError(
            f"answer count mismatch: images={len(images)} answers={tuple(answers.shape)}"
        )
    selected = select_v11_predictions(
        v10_logits,
        v9_logits,
        expert_logits,
        margin_advantage=margin_advantage,
        require_legacy_consensus=require_legacy_consensus,
    )
    model_predictions = {
        "fast": selected["v10_predictions"],
        "legacy_v9": selected["v9_predictions"],
        "expert": selected["expert_predictions"],
        "accurate": selected["predictions"],
    }
    oracle_mask = torch.stack(
        [prediction == answers for prediction in model_predictions.values()], dim=0
    ).any(dim=0)

    def metric(predictions: torch.Tensor, top2: torch.Tensor | None = None) -> dict:
        correct = int((predictions == answers).sum().item())
        payload = {"correct": correct, "rows": len(images), "top1": correct / max(1, len(images))}
        if top2 is not None:
            correct_top2 = int((top2 == answers[:, None]).any(dim=1).sum().item())
            payload.update(
                {"correct_top2": correct_top2, "top2": correct_top2 / max(1, len(images))}
            )
        return payload

    rows: list[dict] = []
    for index, image in enumerate(images):
        answer = int(answers[index].item())
        rows.append(
            {
                "image": str(image),
                "answer_index": answer,
                "fast_index": int(selected["v10_predictions"][index].item()),
                "legacy_v9_index": int(selected["v9_predictions"][index].item()),
                "expert_index": int(selected["expert_predictions"][index].item()),
                "accurate_index": int(selected["predictions"][index].item()),
                "fast_margin": float(selected["v10_margins"][index].item()),
                "legacy_v9_margin": float(selected["v9_margins"][index].item()),
                "expert_margin": float(selected["expert_margins"][index].item()),
                "legacy_consensus": bool(selected["legacy_consensus"][index].item()),
                "disagreement": bool(selected["disagreements"][index].item()),
                "switched_to_v10": bool(selected["switched"][index].item()),
                "fast_correct": int(selected["v10_predictions"][index].item()) == answer,
                "legacy_v9_correct": int(selected["v9_predictions"][index].item()) == answer,
                "expert_correct": int(selected["expert_predictions"][index].item()) == answer,
                "accurate_correct": int(selected["predictions"][index].item()) == answer,
                "oracle_correct": bool(oracle_mask[index].item()),
            }
        )

    oracle_correct = int(oracle_mask.sum().item())
    summary = {
        "schema_version": "route_v11_gate_evaluation_v1",
        "rows": len(images),
        "margin_advantage": float(margin_advantage),
        "require_legacy_consensus": bool(require_legacy_consensus),
        "selection_rule": (
            "default to V11 expert; switch to V10 only when V10 and V9 agree, "
            "V10 disagrees with V11, and V10 margin exceeds V11 margin by the threshold"
        ),
        "fast": metric(selected["v10_predictions"], selected["v10_top2"]),
        "legacy_v9": metric(selected["v9_predictions"], selected["v9_top2"]),
        "expert": metric(selected["expert_predictions"], selected["expert_top2"]),
        "accurate": metric(selected["predictions"], selected["top2"]),
        "oracle": {
            "correct": oracle_correct,
            "rows": len(images),
            "top1": oracle_correct / max(1, len(images)),
        },
        "disagreements": int(selected["disagreements"].sum().item()),
        "legacy_consensus_disagreements": int(
            (selected["disagreements"] & selected["legacy_consensus"]).sum().item()
        ),
        "switches": int(selected["switched"].sum().item()),
    }
    return summary, rows


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def evaluate_feature_files(
    *,
    v10_features: str | Path,
    v9_features: str | Path,
    expert_features: str | Path,
    output_dir: str | Path,
    margin_advantage: float,
    require_legacy_consensus: bool = True,
) -> dict:
    # Training-only dependency; runtime prediction only needs select_v11_predictions.
    from .route_v9_pair_train import load_feature_cache

    caches = [load_feature_cache(path) for path in (v10_features, v9_features, expert_features)]
    if any(list(caches[0]["images"]) != list(cache["images"]) for cache in caches[1:]):
        raise ValueError("V10, V9, and expert feature image order differs")
    if any(not torch.equal(caches[0]["answers"], cache["answers"]) for cache in caches[1:]):
        raise ValueError("V10, V9, and expert feature answers differ")
    summary, rows = evaluate_v11_logits(
        list(caches[0]["images"]),
        caches[0]["answers"],
        caches[0]["main_logits"],
        caches[1]["main_logits"],
        caches[2]["main_logits"],
        margin_advantage=margin_advantage,
        require_legacy_consensus=require_legacy_consensus,
    )
    summary.update(
        {
            "features": {
                "v10": str(Path(v10_features)),
                "v9": str(Path(v9_features)),
                "expert": str(Path(expert_features)),
            },
            "created_at": datetime.now().astimezone().isoformat(),
        }
    )
    output = Path(output_dir)
    _atomic_jsonl(output / "predictions.jsonl", rows)
    _atomic_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the frozen Route V11 three-model gate.")
    parser.add_argument("--v10-features", required=True)
    parser.add_argument("--v9-features", required=True)
    parser.add_argument("--expert-features", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--margin-advantage", type=float, default=0.01)
    parser.add_argument("--no-legacy-consensus", action="store_true")
    args = parser.parse_args()
    summary = evaluate_feature_files(
        v10_features=args.v10_features,
        v9_features=args.v9_features,
        expert_features=args.expert_features,
        output_dir=args.out_dir,
        margin_advantage=args.margin_advantage,
        require_legacy_consensus=not args.no_legacy_consensus,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
