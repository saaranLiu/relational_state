"""Print a structured distribution report over data/structured/*.json.

Usage:
    python3 scripts/distribution_report.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "structured"


def _pct(c: Counter, total: int) -> str:
    if total == 0:
        return ""
    return "  " + "  ".join(f"{k}={v} ({v/total*100:.1f}%)" for k, v in sorted(c.items()))


def _load(name: str) -> dict:
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        return {"records": [], "record_count": 0}
    return json.loads(path.read_text())


def _latent(r: dict) -> dict:
    return (r.get("oracle") or {}).get("latent") or {}


def _mcq_oracle(r: dict) -> dict:
    return (r.get("oracle") or {}).get("mcq") or {}


def _pair_oracle(r: dict) -> dict:
    return (r.get("oracle") or {}).get("pair") or {}


def _ood_social_oracle(r: dict) -> dict:
    return (r.get("oracle") or {}).get("ood_social") or {}


def _ood_career_oracle(r: dict) -> dict:
    return (r.get("oracle") or {}).get("ood_career") or {}


def _gold_rule(r: dict) -> str:
    return ((r.get("oracle") or {}).get("gold") or {}).get("rule_id", "")


def report_train() -> None:
    d = _load("train")
    recs = d["records"]
    n = len(recs)
    if n == 0:
        print("[train]  (empty)")
        return
    print(f"[train]  total = {n}")
    by_dom = Counter(r["scene"]["domain_key"] for r in recs)
    by_cell = Counter(_latent(r).get("cell_id", "") for r in recs)
    pos_recs = [r for r in recs if not _latent(r).get("is_placebo", False)]
    placebo_recs = [r for r in recs if _latent(r).get("is_placebo", False)]
    print(" domain :", _pct(by_dom, n))
    print(f" positional vs placebo : pos={len(pos_recs)} ({len(pos_recs)/n*100:.1f}%) "
          f"placebo={len(placebo_recs)} ({len(placebo_recs)/n*100:.1f}%)")
    pn = len(pos_recs)
    if pn:
        print(f" positional.alpha:", _pct(Counter(_latent(r).get("alpha_bucket") for r in pos_recs), pn))
        print(f" positional.disp :", _pct(Counter(_latent(r).get("dispersion_bucket") for r in pos_recs), pn))
        print(f" positional.skew :", _pct(Counter(_latent(r).get("skew_bucket") for r in pos_recs), pn))
        print(f" positional.held_out:",
              _pct(Counter("held_out" if _latent(r).get("is_held_out_cell") else "main" for r in pos_recs), pn))
    print(f" distinct cells : {len(by_cell)}   (expected: 21 main + 1 'placebo' sentinel)")
    if by_cell:
        print(f"   cell count  min={min(by_cell.values())}  max={max(by_cell.values())}  "
              f"mean={sum(by_cell.values())/len(by_cell):.1f}")
    print(f" scenario_text filled : {sum(1 for r in recs if r.get('scenario_text'))}/{n} "
          f"(teacher LLM not yet run)")
    print()


def report_eval_a() -> None:
    d = _load("eval_A")
    recs = d["records"]
    n = len(recs)
    if n == 0:
        print("[eval_A] (empty)")
        return
    print(f"[eval_A] total = {n}")
    by_dom = Counter(r["scene"]["domain_key"] for r in recs)
    by_alpha = Counter(_latent(r).get("alpha_bucket") for r in recs)
    by_disp = Counter(_latent(r).get("dispersion_bucket") for r in recs)
    by_skew = Counter(_latent(r).get("skew_bucket") for r in recs)
    by_cell = Counter(_latent(r).get("cell_id") for r in recs)
    by_ho = Counter("held_out" if _latent(r).get("is_held_out_cell") else "main" for r in recs)
    by_gold_letter = Counter(_mcq_oracle(r).get("gold_letter") for r in recs)
    margins = [_mcq_oracle(r).get("identification_margin", 0.0) for r in recs]
    print(" domain:", _pct(by_dom, n))
    print(" alpha :", _pct(by_alpha, n))
    print(" disp  :", _pct(by_disp, n))
    print(" skew  :", _pct(by_skew, n))
    print(" cell type:", _pct(by_ho, n))
    print(" gold_letter (A/B/C/D):", _pct(by_gold_letter, n))
    print(f" distinct cells: {len(by_cell)}   (expected: 27)")
    if margins:
        m_min = min(margins); m_max = max(margins); m_mean = sum(margins)/len(margins)
        print(f" identification_margin: min={m_min:.3f}  mean={m_mean:.3f}  max={m_max:.3f}")
    print()


def report_eval_b() -> None:
    d = _load("eval_B")
    recs = d["records"]
    n = len(recs)
    if n == 0:
        print("[eval_B] (empty)")
        return
    print(f"[eval_B] total = {n} records ({n//2} pairs)")
    by_pert = Counter(_pair_oracle(r).get("perturbation") for r in recs)
    by_dir = Counter(_pair_oracle(r).get("direction") for r in recs)
    by_role = Counter(_pair_oracle(r).get("perturbed_role", "NA") for r in recs)
    by_actual_role = Counter((r.get("pair") or {}).get("role") for r in recs)
    gaps = [_pair_oracle(r).get("gap_over_F", 0.0) for r in recs]
    ho_A = Counter("held_out" if _pair_oracle(r).get("is_held_out_cell_A") else "main" for r in recs)
    print(" perturbation:", _pct(by_pert, n))
    print(" direction (must be ~50/50):", _pct(by_dir, n))
    print(" perturbed_role (must be ~50/50):", _pct(by_role, n))
    print(" role A vs B (must be 50/50 exactly):", _pct(by_actual_role, n))
    print(" cell_A type (expect all 'main', held-out tested by Eval-A):", _pct(ho_A, n))
    if gaps:
        print(f" gap_over_F: min={min(gaps):.3f}  mean={sum(gaps)/len(gaps):.3f}  max={max(gaps):.3f}")
    print()


def report_placebo_test() -> None:
    d = _load("placebo_test")
    recs = d["records"]
    n = len(recs)
    if n == 0:
        print("[placebo_test] (empty)")
        return
    print(f"[placebo_test] total = {n}")
    by_dom = Counter(r["scene"]["domain_key"] for r in recs)
    by_gold = Counter(_mcq_oracle(r).get("gold_letter") for r in recs)
    by_gold_rule = Counter(_gold_rule(r) for r in recs)
    print(" domain:", _pct(by_dom, n))
    print(" gold_rule (must be 100% D_pure_private):", _pct(by_gold_rule, n))
    print(" gold_letter:", _pct(by_gold, n))
    print()


def report_ood_social() -> None:
    d = _load("ood_social")
    recs = d["records"]
    n = len(recs)
    if n == 0:
        print("[ood_social] (empty)")
        return
    print(f"[ood_social] total = {n}")
    by_bkt = Counter(_latent(r).get("match_distance_bucket") for r in recs)
    by_gold_letter = Counter(_ood_social_oracle(r).get("gold_letter") for r in recs)
    by_gold_cand = Counter(_latent(r).get("gold_candidate_index") for r in recs)
    print(" match_distance_bucket:", _pct(by_bkt, n))
    print(" gold_letter:", _pct(by_gold_letter, n))
    print(" gold_candidate_index (0=P1,1=P2):", _pct(by_gold_cand, n))
    print()


def report_ood_career() -> None:
    d = _load("ood_career")
    recs = d["records"]
    n = len(recs)
    if n == 0:
        print("[ood_career] (empty)")
        return
    print(f"[ood_career] total = {n}")
    by_bkt = Counter(_latent(r).get("alpha_2i_bucket") for r in recs)
    by_firm = Counter(_latent(r).get("gold_firm") for r in recs)
    by_letter = Counter(_ood_career_oracle(r).get("gold_letter") for r in recs)
    by_firm_bkt = Counter((_latent(r).get("alpha_2i_bucket"), _latent(r).get("gold_firm")) for r in recs)
    print(" alpha_2i bucket:", _pct(by_bkt, n))
    print(" gold_firm:", _pct(by_firm, n))
    print(" gold_letter:", _pct(by_letter, n))
    print(" gold_firm per bucket:")
    for (bkt, firm), c in sorted(by_firm_bkt.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        print(f"   {str(bkt):>4} / {firm}: {c}")
    print()


def main() -> None:
    print("=" * 72)
    print("Distribution report (data/structured/*.json)")
    print("=" * 72)
    report_train()
    report_eval_a()
    report_eval_b()
    report_placebo_test()
    report_ood_social()
    report_ood_career()


if __name__ == "__main__":
    main()
