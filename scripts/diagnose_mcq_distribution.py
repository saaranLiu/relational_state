"""Quick diagnostic: verify MCQ distribution and identification margins for
eval_A / placebo_test. Reports:
  - total records, degraded flags
  - identification_tau, identification_margin (min/p5/median) and tau=0.10 compliance
  - pair-wise margin compliance (min pair-wise margin vs TAU_PAIR_*)
  - gold_rule distribution
  - gold_letter distribution (A/B/C/D balance)
  - distractor rule usage counts (per non-gold rule)
  - margin tier distribution (hard/medium/easy)
  - scenario_text fill rate (how many still need teacher)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config.rule_label_templates import RULE_IDS  # noqa: E402
from data.generation.identification_filter import (  # noqa: E402
    TAU_PAIR_PLACEBO,
    TAU_PAIR_POSITIONAL,
    TAU_PLACEBO,
    TAU_POSITIONAL,
)


def _pair_min_margin(all_rule_x: Dict[str, float], option_rules: List[str], F: float) -> float:
    denom = max(F, 1e-9)
    mins = []
    for a, b in combinations(option_rules, 2):
        mins.append(abs(all_rule_x[a] - all_rule_x[b]) / denom)
    return min(mins) if mins else float("inf")


def diagnose(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = payload.get("split") or path.stem
    records: List[Dict[str, Any]] = payload.get("records", [])
    n = len(records)

    is_placebo = split == "placebo_test"
    tau_ref = TAU_PLACEBO if is_placebo else TAU_POSITIONAL
    tau_pair_ref = TAU_PAIR_PLACEBO if is_placebo else TAU_PAIR_POSITIONAL

    gold_rule_counts: Counter[str] = Counter()
    gold_letter_counts: Counter[str] = Counter()
    distractor_rule_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    selection_degraded = 0
    identification_degraded = 0
    tau_violations: List[str] = []
    pair_violations: List[str] = []
    margins_all: List[float] = []
    pair_mins_all: List[float] = []
    scenario_filled = 0
    scenario_missing = 0

    for rec in records:
        ora = rec.get("oracle") or {}
        mcq = ora.get("mcq") or {}
        latent = ora.get("latent") or {}
        F = float(latent.get("F") or 0.0)

        gold_rule = mcq.get("gold_rule_id") or ""
        gold_letter = mcq.get("gold_letter") or ""
        gold_rule_counts[gold_rule] += 1
        gold_letter_counts[gold_letter] += 1
        tier = mcq.get("identification_margin_tier")
        if tier:
            tier_counts[tier] += 1

        if mcq.get("selection_degraded"):
            selection_degraded += 1
        if mcq.get("identification_degraded"):
            identification_degraded += 1

        options_oracle = mcq.get("options_oracle") or []
        rule_ids_in_question = [o["rule_id"] for o in options_oracle]
        for rid in rule_ids_in_question:
            if rid != gold_rule:
                distractor_rule_counts[rid] += 1

        ident_margin = float(mcq.get("identification_margin") or 0.0)
        margins_all.append(ident_margin)
        if ident_margin + 1e-9 < tau_ref:
            tau_violations.append(rec["record_id"])

        all_rule_x = mcq.get("all_rule_x_values") or {}
        if all_rule_x and rule_ids_in_question:
            pair_min = _pair_min_margin(all_rule_x, rule_ids_in_question, F)
            pair_mins_all.append(pair_min)
            if pair_min + 1e-9 < tau_pair_ref:
                pair_violations.append(rec["record_id"])

        if rec.get("scenario_text"):
            scenario_filled += 1
        else:
            scenario_missing += 1

    def _stats(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {}
        xs2 = sorted(xs)
        p5 = xs2[max(0, int(0.05 * len(xs2)) - 1)]
        p95 = xs2[min(len(xs2) - 1, int(0.95 * len(xs2)))]
        return {
            "min": round(min(xs2), 4),
            "p5": round(p5, 4),
            "median": round(median(xs2), 4),
            "p95": round(p95, 4),
            "max": round(max(xs2), 4),
        }

    non_gold_rules = [r for r in RULE_IDS if r != next(iter(gold_rule_counts))]
    expected_per_rule = (n * 3) // len(non_gold_rules) if non_gold_rules else 0

    return {
        "split": split,
        "input": str(path),
        "total_records": n,
        "scenario_filled": scenario_filled,
        "scenario_missing": scenario_missing,
        "gold_rule_counts": dict(gold_rule_counts),
        "gold_letter_counts": dict(gold_letter_counts),
        "identification": {
            "tau_ref": tau_ref,
            "records_below_tau": len(tau_violations),
            "identification_degraded_flag": identification_degraded,
            "margin_stats": _stats(margins_all),
            "sample_violations": tau_violations[:5],
        },
        "pairwise": {
            "tau_pair_ref": tau_pair_ref,
            "records_below_tau_pair": len(pair_violations),
            "pair_min_stats": _stats(pair_mins_all),
            "sample_violations": pair_violations[:5],
        },
        "selection_degraded_flag": selection_degraded,
        "distractor_rule_counts": dict(distractor_rule_counts),
        "distractor_expected_per_rule": expected_per_rule,
        "distractor_rule_max_minus_min": (
            (max(distractor_rule_counts.values()) - min(distractor_rule_counts.values()))
            if distractor_rule_counts else 0
        ),
        "margin_tier_counts": dict(tier_counts),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-file", action="append", required=True,
                   help="can be repeated; e.g. eval_A.json, placebo_test.json")
    args = p.parse_args()

    reports = [diagnose(Path(f)) for f in args.input_file]
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
