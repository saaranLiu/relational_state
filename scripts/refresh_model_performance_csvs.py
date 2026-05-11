#!/usr/bin/env python3
"""
Rebuild notebook-derived CSVs under analysis/model_performance_error_analysis/
from evaluation/outputs/*/*_predictions.jsonl.

Skips experimental dirs (e.g. gpt-5.5, temp) so the six-model suite stays stable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "evaluation" / "outputs"
EXPORT_DIR = ROOT / "analysis" / "model_performance_error_analysis"

TASK_ORDER = ["eval_A", "eval_B", "placebo_test", "ood_social", "ood_career"]

# Do not include in cross-model / accuracy tables (user may keep experimental runs on disk).
SKIP_OUTPUT_PARENTS = frozenset({"gpt-5.5", "temp"})


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON in {path}:{line_no}") from exc
    return rows


def flatten_prediction(row: dict, model_dir: str, prediction_file: Path) -> dict:
    meta = row.get("meta") or {}
    flat: Dict[str, Any] = {
        "model_dir": model_dir,
        "model": row.get("model") or model_dir,
        "task_id": row.get("task_id"),
        "dataset_split": row.get("dataset_split"),
        "gold_letter": row.get("gold_letter"),
        "parsed_letter": row.get("parsed_letter"),
        "parsed_rule_id": row.get("parsed_rule_id"),
        "is_correct": bool(row.get("is_correct")),
        "raw_response": row.get("raw_response") or "",
        "prompt": row.get("prompt") or "",
        "prediction_file": str(prediction_file.relative_to(ROOT)),
    }
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            flat[key] = value
        elif key == "letter_to_rule" and isinstance(value, dict):
            flat[key] = value
            gl = flat.get("gold_letter")
            pl = flat.get("parsed_letter")
            flat["gold_rule_id"] = flat.get("gold_rule_id") or value.get(str(gl))
            flat["parsed_rule_id"] = flat.get("parsed_rule_id") or value.get(str(pl))
        else:
            flat[key] = json.dumps(value, ensure_ascii=False)
    return flat


def infer_error_cause(row: pd.Series) -> str:
    split = str(row.get("dataset_split"))
    parsed = row.get("parsed_rule_id")
    gold = row.get("gold_rule_id")
    raw = (row.get("raw_response") or "").lower()

    if row.get("is_correct"):
        return "correct"
    if not row.get("is_parsed", True):
        return "format / parse failure"

    if split == "eval_A":
        if parsed == "C_uniform_avg":
            return "uses flat average instead of closeness weights"
        if parsed == "B_top_anchor":
            return "over-anchors on closest/highest peer"
        if parsed == "E_closest_mimicry":
            return "copies closest peer directly"
        if parsed == "F_median_anchor":
            return "uses median/central peer shortcut"
        if parsed == "D_pure_private":
            return "ignores social reference in positional task"
        return f"other rule confusion: {gold} -> {parsed}"

    if split == "placebo_test":
        if parsed == "A_peer_weighted":
            return "hallucinates peer-weighted pull in placebo"
        if parsed in {"B_top_anchor", "C_uniform_avg", "E_closest_mimicry", "F_median_anchor"}:
            return "uses social shortcut although gold is private baseline"
        return f"other placebo confusion: {gold} -> {parsed}"

    if split == "eval_B":
        pert = row.get("perturbation")
        if pert == "alpha_i_up":
            return "misses social-sensitivity alpha change"
        if pert == "F_up":
            return "misses private-baseline F change"
        if pert == "ref_sum_up":
            return "misses weighted reference aggregate change"
        if pert == "peer_action_up":
            return "misses peer-action level change"
        if pert == "top_weight_up":
            return "misreads closeness-weight redistribution"
        return "comparative-static direction error"

    if split == "ood_social":
        return "misses b/c matching stability criterion"

    if split == "ood_career":
        if "top of" in raw or "status" in raw or "relative" in raw:
            return "overweights relative status versus Langtry threshold"
        if "salary" in raw or "paycheck" in raw or "higher" in raw:
            return "overweights absolute salary / prestige"
        return "career threshold tradeoff error"

    return "other"


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_files = sorted(OUTPUT_ROOT.glob("*/*_predictions.jsonl"))
    prediction_files = [p for p in prediction_files if p.parent.name not in SKIP_OUTPUT_PARENTS]
    if not prediction_files:
        print(f"No prediction files under {OUTPUT_ROOT} (after skip).", file=sys.stderr)
        sys.exit(1)

    all_rows: List[dict] = []
    for path in prediction_files:
        model_dir = path.parent.name
        for row in read_jsonl(path):
            all_rows.append(flatten_prediction(row, model_dir, path))

    pred = pd.DataFrame(all_rows)
    pred["model_dir"] = pred["model_dir"].replace({"gpt-4": "gpt4"})
    pred["dataset_split"] = pd.Categorical(pred["dataset_split"], TASK_ORDER, ordered=True)
    pred["correct_int"] = pred["is_correct"].astype(int)
    pred["is_parsed"] = pred["parsed_letter"].notna()

    # --- accuracy_by_model_task.csv ---
    perf = (
        pred.groupby(["model_dir", "dataset_split"], observed=True)
        .agg(
            total=("task_id", "count"),
            correct=("correct_int", "sum"),
            accuracy=("correct_int", "mean"),
            parse_rate=("is_parsed", "mean"),
        )
        .reset_index()
    )
    perf["accuracy_pct"] = 100 * perf["accuracy"]
    acc_table = (
        perf.pivot(index="model_dir", columns="dataset_split", values="accuracy_pct")
        .reindex(columns=TASK_ORDER)
        .sort_values(by=list(TASK_ORDER), ascending=False, na_position="last")
    )
    acc_table.reset_index().to_csv(EXPORT_DIR / "accuracy_by_model_task.csv", index=False)

    # --- model_gap_by_task.csv ---
    task_gap = (
        perf.groupby("dataset_split", observed=True)
        .agg(
            best_acc=("accuracy_pct", "max"),
            worst_acc=("accuracy_pct", "min"),
            mean_acc=("accuracy_pct", "mean"),
            n_models=("model_dir", "nunique"),
        )
        .reset_index()
    )
    task_gap["spread"] = task_gap["best_acc"] - task_gap["worst_acc"]

    def best_models_for_task(split: str) -> str:
        sub = perf[perf["dataset_split"].astype(str) == split]
        if sub.empty:
            return ""
        best = sub["accuracy_pct"].max()
        return ", ".join(sub.loc[sub["accuracy_pct"].eq(best), "model_dir"].sort_values())

    task_gap["best_model"] = task_gap["dataset_split"].astype(str).map(best_models_for_task)
    task_gap.to_csv(EXPORT_DIR / "model_gap_by_task.csv", index=False)

    # --- error_cause_summary.csv ---
    pred["error_cause"] = pred.apply(infer_error_cause, axis=1)
    wrong = pred[~pred["is_correct"]].copy()
    cause_summary = (
        wrong.groupby(["dataset_split", "model_dir", "error_cause"], observed=True)
        .size()
        .reset_index(name="n")
    )
    cause_summary["task_errors"] = cause_summary.groupby(
        ["dataset_split", "model_dir"], observed=True
    )["n"].transform("sum")
    cause_summary["share_of_model_task_errors"] = cause_summary["n"] / cause_summary["task_errors"]
    cause_summary.to_csv(EXPORT_DIR / "error_cause_summary.csv", index=False)

    # --- eval_A 27-cell tables ---
    eval_a = pred[pred["dataset_split"].astype(str).eq("eval_A")].copy()
    required_cols = ["alpha_bucket", "dispersion_bucket", "skew_bucket", "cell_id", "model_dir", "correct_int"]
    missing = [c for c in required_cols if c not in eval_a.columns]
    if missing:
        print(f"Warning: Eval-A cell analysis skipped; missing columns: {missing}", file=sys.stderr)
    else:
        bucket_order = ["low", "mid", "high"]
        for c in ["alpha_bucket", "dispersion_bucket", "skew_bucket"]:
            eval_a[c] = pd.Categorical(eval_a[c], categories=bucket_order, ordered=True)

        cell_summary = (
            eval_a.groupby(
                ["model_dir", "cell_id", "alpha_bucket", "dispersion_bucket", "skew_bucket"],
                observed=True,
            )
            .agg(total=("task_id", "count"), correct=("correct_int", "sum"))
            .reset_index()
        )
        cell_summary["accuracy_pct"] = 100 * cell_summary["correct"] / cell_summary["total"]
        cell_summary.to_csv(EXPORT_DIR / "evalA_by_27_cells.csv", index=False)

        cell_common = (
            cell_summary.groupby(
                ["cell_id", "alpha_bucket", "dispersion_bucket", "skew_bucket"], observed=True
            )
            .agg(
                mean_acc=("accuracy_pct", "mean"),
                std_acc=("accuracy_pct", "std"),
                min_acc=("accuracy_pct", "min"),
                max_acc=("accuracy_pct", "max"),
            )
            .reset_index()
        )
        cell_common["spread"] = cell_common["max_acc"] - cell_common["min_acc"]
        cell_common.to_csv(EXPORT_DIR / "evalA_cell_commonality.csv", index=False)

        # Hard-vote counts (README §3.1.2): cell is "hard" for a model if acc <= Q1 of that model's 27 cells.
        hard_vote: Dict[str, int] = {}
        for mdir, g in cell_summary.groupby("model_dir", observed=True):
            thr = float(g["accuracy_pct"].quantile(0.25))
            for _, r in g.iterrows():
                if float(r["accuracy_pct"]) <= thr:
                    cid = str(r["cell_id"])
                    hard_vote[cid] = hard_vote.get(cid, 0) + 1

        cell_readme = cell_common.copy()
        cell_readme["hard_votes"] = cell_readme["cell_id"].map(lambda x: hard_vote.get(str(x), 0))
        cell_readme["common_issue"] = cell_readme["hard_votes"] >= 4
        cell_readme = cell_readme.sort_values(
            ["hard_votes", "mean_acc"], ascending=[False, True]
        )
        cell_readme.to_csv(EXPORT_DIR / "evalA_cell_readme_order.csv", index=False)

    print(f"Loaded {len(pred):,} predictions from {len(prediction_files)} files.")
    print("Models:", ", ".join(sorted(pred["model_dir"].unique())))
    print("Wrote CSVs under", EXPORT_DIR)


if __name__ == "__main__":
    main()
