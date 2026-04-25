"""
Build SFT training pairs from `data/structured/train.json` + teacher narratives.

Each input record contributes ONE SFT pair:
    prompt    = student-facing system + user prompt (scenario + peer cards,
                open-generation style — NO letters, NO rule menu)
    response  = CoT taxonomy preamble + qualitative body + final commitment

The CoT is rendered by training/cot_templates.build_full_cot using the
record's latent + gold_rule_id. We attach a lightweight lexical-audit pass
at save time and drop any pair whose CoT leaks MCQ vocab or forbidden
research terms (this should be zero after design, but the audit acts as a
belt-and-suspenders guard).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config.rule_label_templates import format_value  # noqa: E402
from data.config.vocabulary_pools import (  # noqa: E402
    find_forbidden_research_terms,
    find_mcq_rule_leaks_in_cot,
)
from training.cot_templates import (  # noqa: E402
    CoTContext,
    build_full_cot,
)


SFT_SCHEMA_VERSION = "sft_v1"


# ---------------------------------------------------------------------------
# Student-facing prompts (open-generation)
# ---------------------------------------------------------------------------

STUDENT_SYSTEM_PROMPT = (
    "You are a careful reasoner about everyday relational decisions. "
    "Read the scenario, think step by step, and then state the commitment you would pick. "
    "Be concrete about what the private floor is, how much weight (if any) the social circle "
    "carries, and why. Do not invent numbers that are not grounded in the scenario."
)


def _compose_student_user_prompt(record: Dict[str, Any]) -> str:
    scene = record["scene"]
    scenario_text = record.get("scenario_text") or ""
    action_label = scene.get("action_label", "the commitment level")
    action_unit = scene.get("action_unit", "")
    unit_suffix = f" (in {action_unit})" if action_unit else ""
    return (
        f"Scenario:\n{scenario_text}\n\n"
        f"Question: Based on this scene, what {action_label}{unit_suffix} should the protagonist "
        f"settle on, and why? Reason step by step first, then state the final commitment."
    )


# ---------------------------------------------------------------------------
# CoT generation
# ---------------------------------------------------------------------------

def _protagonist_for(record: Dict[str, Any]) -> str:
    # Keep it domain-generic; the teacher sometimes names the protagonist, but
    # the CoT stays abstract on purpose (the CoT must work even if the
    # teacher chose a different name than what CoT expects).
    return "the protagonist"


def _build_cot(record: Dict[str, Any], rng: random.Random) -> str:
    scene = record["scene"]
    gold = record["oracle"]["gold"]
    x_display = format_value(
        gold["x_value"],
        scene.get("value_format", "currency"),
        scene.get("action_unit", ""),
    )
    ctx = CoTContext(
        domain_key=scene["domain_key"],
        protagonist=_protagonist_for(record),
        peer_noun_singular=scene.get("peer_noun_singular") or "peer",
        peer_noun_plural=scene.get("peer_noun_plural") or "peers",
        action_label=scene.get("action_label") or "commitment",
        x_display=x_display,
    )
    return build_full_cot(ctx, rng=rng)


# ---------------------------------------------------------------------------
# Pair builder
# ---------------------------------------------------------------------------

@dataclass
class AuditStats:
    total_in: int = 0
    kept: int = 0
    dropped_no_scenario: int = 0
    dropped_cot_mcq_leak: int = 0
    dropped_cot_forbidden_term: int = 0
    dropped_scenario_leak: int = 0
    dropped_cot_too_short: int = 0


def _scenario_leaks(scenario_text: str) -> List[str]:
    """Check teacher narrative for forbidden research terms only.
    We do NOT enforce MCQ rule vocab in narratives (harmless if present)."""
    return find_forbidden_research_terms(scenario_text)


def build_pairs(
    records: Sequence[Dict[str, Any]],
    rng: random.Random,
    min_cot_chars: int = 80,
) -> Tuple[List[Dict[str, Any]], AuditStats]:
    stats = AuditStats(total_in=len(records))
    out: List[Dict[str, Any]] = []

    for rec in records:
        scenario_text = rec.get("scenario_text")
        if not scenario_text:
            stats.dropped_no_scenario += 1
            continue
        # Scenario audit: forbidden research terms only.
        scenario_forbidden = _scenario_leaks(scenario_text)
        if scenario_forbidden:
            stats.dropped_scenario_leak += 1
            continue

        cot = _build_cot(rec, rng)
        if len(cot) < min_cot_chars:
            stats.dropped_cot_too_short += 1
            continue
        mcq_leaks = find_mcq_rule_leaks_in_cot(cot)
        if mcq_leaks:
            stats.dropped_cot_mcq_leak += 1
            continue
        forbid = find_forbidden_research_terms(cot)
        if forbid:
            stats.dropped_cot_forbidden_term += 1
            continue

        user_prompt = _compose_student_user_prompt(rec)
        gold = rec["oracle"]["gold"]
        latent = rec["oracle"]["latent"]
        final_commitment = (
            f"\n\nFinal commitment: {gold['x_display']} for the "
            f"{rec['scene'].get('action_label') or 'commitment'}."
        )
        response = cot + final_commitment

        out.append({
            "sft_id": f"sft__{rec['record_id']}",
            "sft_schema_version": SFT_SCHEMA_VERSION,
            "source_record_id": rec["record_id"],
            "dataset_split": rec["dataset_split"],
            "scene_id": rec["scene"]["scene_id"],
            "domain_key": rec["scene"]["domain_key"],
            "is_placebo": rec["scene"]["domain_key"] == "domain2_non_positional_investment",
            "system_prompt": STUDENT_SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "response": response,
            "gold": gold,
            "cell_id": latent.get("cell_id"),
            "alpha_bucket": latent.get("alpha_bucket"),
            "is_held_out_cell": latent.get("is_held_out_cell", False),
        })
        stats.kept += 1
    return out, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-file", default="data/structured/train.json",
                   help="Structured train split with teacher-filled scenario_text.")
    p.add_argument("--output-file", default="data/sft/train_pairs.jsonl",
                   help="Destination JSONL. One SFT pair per line.")
    p.add_argument("--seed", type=int, default=20260421)
    p.add_argument("--require-scenario", action="store_true",
                   help="If set, fail if ANY record lacks scenario_text (default: skip them).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed + 10000)
    payload = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    records = payload.get("records", [])

    if args.require_scenario:
        missing = [r["record_id"] for r in records if not r.get("scenario_text")]
        if missing:
            raise ValueError(f"{len(missing)} records have no scenario_text; example: {missing[:3]}")

    pairs, stats = build_pairs(records, rng)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for p in pairs:
            handle.write(json.dumps(p, ensure_ascii=False) + "\n")

    summary = {
        "status": "sft_written",
        "input": args.input_file,
        "output": str(out_path),
        "total_in": stats.total_in,
        "kept": stats.kept,
        "dropped_no_scenario": stats.dropped_no_scenario,
        "dropped_cot_mcq_leak": stats.dropped_cot_mcq_leak,
        "dropped_cot_forbidden_term": stats.dropped_cot_forbidden_term,
        "dropped_scenario_leak": stats.dropped_scenario_leak,
        "dropped_cot_too_short": stats.dropped_cot_too_short,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
