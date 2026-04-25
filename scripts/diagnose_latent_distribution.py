"""Summarise the latent-parameter distribution of one or more structured splits.

For each file we report:
  - Sample count / domain / scene coverage
  - alpha_i histogram (continuous + bucket assignment)
  - F histogram (per scene and overall)
  - Cell distribution (alpha-bucket × dispersion-bucket × skew-bucket)
  - N (peer count) distribution
  - x_j / g_ij summary stats (mean, std, min, max)
  - ref_sum and x_i_star quantiles
  - held-out cell rate
  - identification_margin / tier distribution (from oracle.mcq)
  - gold_letter balance
  - scenario_text fill rate

Usage:
    python scripts/diagnose_latent_distribution.py \\
        --input-file data/structured/eval_A.json \\
        --input-file data/structured/placebo_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config.langtry_parameters import iter_all_cells  # noqa: E402


ALPHA_BUCKET_EDGES = {  # mirrors data/config/latent_specs (public thresholds)
    "low": (0.0, 0.15),
    "mid": (0.15, 0.45),
    "high": (0.45, 0.85),
    "placebo": (0.0, 0.02),
}


def _alpha_bucket_for_value(alpha: float) -> str:
    if alpha < 0.02:
        return "placebo"
    if alpha < 0.15:
        return "low"
    if alpha < 0.45:
        return "mid"
    return "high"


def _stats(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {}
    xs2 = sorted(xs)

    def _q(frac: float) -> float:
        if not xs2:
            return 0.0
        idx = min(len(xs2) - 1, max(0, int(round(frac * (len(xs2) - 1)))))
        return xs2[idx]

    return {
        "n": len(xs2),
        "min": round(min(xs2), 4),
        "p5": round(_q(0.05), 4),
        "p25": round(_q(0.25), 4),
        "median": round(median(xs2), 4),
        "mean": round(mean(xs2), 4),
        "p75": round(_q(0.75), 4),
        "p95": round(_q(0.95), 4),
        "max": round(max(xs2), 4),
        "std": round(pstdev(xs2), 4) if len(xs2) > 1 else 0.0,
    }


def _eval_a_margin_tier(m: float) -> str:
    if m < 0.20:
        return "hard"
    if m < 0.50:
        return "medium"
    return "easy"


def diagnose(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = payload.get("split") or path.stem
    records: List[Dict[str, Any]] = payload.get("records", [])
    n_records = len(records)

    alpha_vals: List[float] = []
    F_vals: List[float] = []
    c_vals: List[float] = []
    N_vals: List[int] = []
    x_j_all: List[float] = []
    g_ij_all: List[float] = []
    ref_sum_vals: List[float] = []
    x_star_vals: List[float] = []
    alpha_bucket_counts: Counter[str] = Counter()
    cell_counts: Counter[str] = Counter()
    alpha_dim_counts: Counter[str] = Counter()
    disp_counts: Counter[str] = Counter()
    skew_counts: Counter[str] = Counter()
    heldout_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    gold_letter_counts: Counter[str] = Counter()
    gold_rule_counts: Counter[str] = Counter()
    margin_tier_counts: Counter[str] = Counter()
    margins: List[float] = []
    F_by_scene: Dict[str, List[float]] = {}
    alpha_by_cell: Dict[str, List[float]] = {}
    N_by_scene: Dict[str, List[int]] = {}
    scenario_missing = 0
    scenario_filled = 0

    known_cells = set(iter_all_cells())
    known_cell_ids = {
        f"alpha_{a}__disp_{d}__skew_{s}" for (a, d, s) in known_cells
    }

    for rec in records:
        ora = rec.get("oracle") or {}
        latent = ora.get("latent") or {}
        alpha = float(latent.get("alpha_i") or 0.0)
        F = float(latent.get("F") or 0.0)
        c = float(latent.get("c") or 0.0)
        x_j = [float(v) for v in (latent.get("x_j") or [])]
        g_ij = [float(v) for v in (latent.get("g_ij") or [])]
        N = int(latent.get("N") or len(x_j))
        ref_sum = float(latent.get("ref_sum") or 0.0)
        x_star = float(latent.get("x_i_star") or 0.0)
        cell_id = latent.get("cell_id") or ""

        alpha_vals.append(alpha)
        F_vals.append(F)
        c_vals.append(c)
        N_vals.append(N)
        x_j_all.extend(x_j)
        g_ij_all.extend(g_ij)
        ref_sum_vals.append(ref_sum)
        x_star_vals.append(x_star)

        bucket = _alpha_bucket_for_value(alpha)
        alpha_bucket_counts[bucket] += 1
        cell_counts[cell_id or "(unknown)"] += 1
        if "__" in (cell_id or ""):
            parts = cell_id.split("__")
            if len(parts) == 3:
                a_part, d_part, s_part = parts
                alpha_dim_counts[a_part.replace("alpha_", "")] += 1
                disp_counts[d_part.replace("disp_", "")] += 1
                skew_counts[s_part.replace("skew_", "")] += 1
        is_heldout = cell_id not in known_cell_ids if cell_id else False
        heldout_counts["heldout" if is_heldout else "main"] += 1

        scene_obj = rec.get("scene") or {}
        scene_id = scene_obj.get("scene_id") or ""
        family = scene_obj.get("family") or ""
        domain = scene_obj.get("domain_key") or ""
        scene_counts[scene_id] += 1
        family_counts[family] += 1
        domain_counts[domain] += 1
        F_by_scene.setdefault(scene_id, []).append(F)
        alpha_by_cell.setdefault(cell_id or "(unknown)", []).append(alpha)
        N_by_scene.setdefault(scene_id, []).append(N)

        mcq = ora.get("mcq") or {}
        gl = mcq.get("gold_letter") or ""
        gr = mcq.get("gold_rule_id") or ""
        gold_letter_counts[gl] += 1
        gold_rule_counts[gr] += 1
        ident_margin = mcq.get("identification_margin")
        if ident_margin is not None:
            margins.append(float(ident_margin))
            tier = mcq.get("identification_margin_tier") or _eval_a_margin_tier(float(ident_margin))
            margin_tier_counts[tier] += 1

        if rec.get("scenario_text"):
            scenario_filled += 1
        else:
            scenario_missing += 1

    return {
        "split": split,
        "input": str(path),
        "total_records": n_records,
        "scenario_filled": scenario_filled,
        "scenario_missing": scenario_missing,
        "domain_counts": dict(domain_counts),
        "family_counts": dict(family_counts),
        "scene_counts": dict(scene_counts),
        "gold_rule_counts": dict(gold_rule_counts),
        "gold_letter_counts": dict(gold_letter_counts),
        "alpha_bucket_counts": dict(alpha_bucket_counts),
        "cell_counts": dict(cell_counts),
        "cell_margin_dim_counts": {
            "alpha": dict(alpha_dim_counts),
            "dispersion": dict(disp_counts),
            "skew": dict(skew_counts),
        },
        "heldout_counts": dict(heldout_counts),
        "alpha_i_stats": _stats(alpha_vals),
        "F_stats": _stats(F_vals),
        "c_stats": _stats(c_vals),
        "N_stats": _stats([float(v) for v in N_vals]),
        "x_j_stats": _stats(x_j_all),
        "g_ij_stats": _stats(g_ij_all),
        "ref_sum_stats": _stats(ref_sum_vals),
        "x_star_stats": _stats(x_star_vals),
        "identification_margin_stats": _stats(margins),
        "margin_tier_counts": dict(margin_tier_counts),
        "F_by_scene_mean": {s: round(mean(v), 2) for s, v in F_by_scene.items()},
        "alpha_by_cell_mean": {s: round(mean(v), 3) for s, v in alpha_by_cell.items()},
        "N_by_scene_mean": {s: round(mean(v), 2) for s, v in N_by_scene.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-file", action="append", required=True,
                   help="Can be repeated; e.g. eval_A.json placebo_test.json")
    p.add_argument("--output-file", default=None, help="Optional: also write full JSON report to this path.")
    args = p.parse_args()

    reports = [diagnose(Path(f)) for f in args.input_file]
    text = json.dumps(reports, ensure_ascii=False, indent=2)
    print(text)
    if args.output_file:
        Path(args.output_file).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
