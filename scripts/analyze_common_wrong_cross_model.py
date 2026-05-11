#!/usr/bin/env python3
"""
Cross-model common wrong-answer analysis + lightweight reasoning text mining.

Reads all `evaluation/outputs/*/*_predictions.jsonl`, groups by `task_id` and
`dataset_split`, and finds items where every model (or k-of-n) answered wrong.

Reasoning source: the `raw_response` field. For the standard eval prompt format,
the script extracts the segment between `Reasoning:` and `Choice:` when present;
otherwise it uses the full `raw_response` (e.g. long CoT models).

Outputs under `analysis/model_performance_error_analysis/`:
  - common_wrong_by_split.csv
  - common_wrong_eval_A_universal.csv          (all models wrong on eval_A)
  - common_wrong_eval_A_reasoning_keywords.csv  (keyword counts on universal set)
  - common_wrong_small3_by_split.csv           (ollama llama×2 + qwen2-7b all wrong)
  - common_wrong_eval_A_small3_overlap.csv     (eval_A: 3-small vs 6-all; large-tier acc on small-fail tasks)
  - common_wrong_eval_A_small_vs_large_rules.csv (parsed_rule histograms by tier; includes wrong-only large)
  - small_vs_large_accuracy_by_model_split.csv / small_vs_large_mean_accuracy_by_split.csv
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "evaluation" / "outputs"
EXPORT_DIR = ROOT / "analysis" / "model_performance_error_analysis"

# Local Llama + Qwen runs (small-parameter tier in this benchmark).
SMALL_LLAMA_QWEN = frozenset(
    {"ollama-llama3-1-latest", "ollama-llama3-8b", "ollama-qwen2-7b"}
)
LARGE_TIER = frozenset({"gpt4", "DeepSeek-R1-671B", "Qwen"})


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_reasoning_snippet(raw: str) -> str:
    if not raw:
        return ""
    m = re.search(
        r"Reasoning:\s*(.*?)(?=\n\s*Choice:|\nChoice:|\Z)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return raw.strip()[:2000]


REASONING_KEYWORDS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("mentions_closest_or_intimate", re.compile(r"closest|intimate|tight|confidant|roommate|mentor|one\s+peer|single\s+peer", re.I)),
    ("mentions_average_or_mean", re.compile(r"\baverage\b|\bmean\b|mid-?pack|typical|median", re.I)),
    ("mentions_private_or_baseline", re.compile(r"private|baseline|alone|personal\s+floor|stick\s+to\s+my", re.I)),
    ("mentions_weight_or_closeness", re.compile(r"weight|weighted|closeness|proximity|bond|relationship", re.I)),
    ("mentions_match_or_copy", re.compile(r"match|mirror|copy|align\s+with|same\s+as|imitate", re.I)),
    ("mentions_social_pressure", re.compile(r"peer|crowd|visible|pressure|impress|norm|everyone", re.I)),
)


def keyword_hits(text: str) -> Dict[str, int]:
    t = text or ""
    return {name: int(bool(pat.search(t))) for name, pat in REASONING_KEYWORDS}


def _task_all_wrong_for_subset(
    sub: pd.DataFrame, model_subset: frozenset, n_expected: int
) -> bool:
    sub_m = sub[sub["model_dir"].isin(model_subset)]
    by_m = sub_m.groupby("model_dir")["wrong"].max()
    if len(by_m) != n_expected:
        return False
    return bool(by_m.all())


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_files = sorted(OUTPUT_ROOT.glob("*/*_predictions.jsonl"))
    if not prediction_files:
        raise FileNotFoundError(f"No predictions under {OUTPUT_ROOT}")

    rows: List[Dict[str, Any]] = []
    for path in prediction_files:
        model_dir = "gpt4" if path.parent.name == "gpt-4" else path.parent.name
        for r in read_jsonl(path):
            meta = r.get("meta") or {}
            rows.append(
                {
                    "model_dir": model_dir,
                    "task_id": r.get("task_id"),
                    "dataset_split": r.get("dataset_split") or meta.get("dataset_split"),
                    "is_correct": r.get("is_correct"),
                    "gold_letter": r.get("gold_letter"),
                    "parsed_letter": r.get("parsed_letter"),
                    "parsed_rule_id": r.get("parsed_rule_id"),
                    "raw_response": r.get("raw_response") or "",
                    "scene_id": meta.get("scene_id"),
                    "cell_id": meta.get("cell_id"),
                    "gold_rule_id": meta.get("gold_rule_id"),
                }
            )

    df = pd.DataFrame(rows)
    df = df[df["task_id"].notna() & df["dataset_split"].notna()]
    df["wrong"] = ~df["is_correct"].fillna(False)

    models = sorted(df["model_dir"].unique())
    n_models = len(models)

    summary_rows: List[Dict[str, Any]] = []
    small3_summary_rows: List[Dict[str, Any]] = []
    universal_eval_a: List[Dict[str, Any]] = []

    for split, g in df.groupby("dataset_split"):
        by_task = g.groupby("task_id")
        n_wrong = by_task["wrong"].sum()
        n_seen = by_task["wrong"].count()
        universal = n_wrong[n_wrong == n_seen].index.tolist()

        small3_all_wrong_tasks: List[Any] = []
        for tid, tg in g.groupby("task_id"):
            if _task_all_wrong_for_subset(tg, SMALL_LLAMA_QWEN, len(SMALL_LLAMA_QWEN)):
                small3_all_wrong_tasks.append(tid)

        summary_rows.append(
            {
                "dataset_split": split,
                "n_tasks": int(by_task.ngroups),
                "n_models": n_models,
                "universal_wrong_count": len(universal),
                "universal_wrong_rate": len(universal) / max(by_task.ngroups, 1),
            }
        )
        small3_summary_rows.append(
            {
                "dataset_split": split,
                "n_tasks": int(by_task.ngroups),
                "n_small_models": len(SMALL_LLAMA_QWEN),
                "small3_all_wrong_count": len(small3_all_wrong_tasks),
                "small3_all_wrong_rate": len(small3_all_wrong_tasks) / max(by_task.ngroups, 1),
            }
        )

        if split == "eval_A" and universal:
            for tid in universal:
                sub = g[g["task_id"] == tid]
                first = sub.iloc[0]
                reasons = []
                kws = Counter()
                for _, rr in sub.iterrows():
                    snip = extract_reasoning_snippet(str(rr["raw_response"]))
                    reasons.append(f"[{rr['model_dir']}] {snip[:400]}")
                    for k, v in keyword_hits(snip).items():
                        kws[k] += v
                parsed_rules = sub["parsed_rule_id"].dropna().astype(str).tolist()
                rule_ct = Counter(parsed_rules)
                universal_eval_a.append(
                    {
                        "task_id": tid,
                        "scene_id": first.get("scene_id"),
                        "cell_id": first.get("cell_id"),
                        "gold_letter": first.get("gold_letter"),
                        "gold_rule_id": first.get("gold_rule_id"),
                        "parsed_rule_histogram": json.dumps(dict(rule_ct), ensure_ascii=False),
                        "reasoning_snippets_concat": "\n---\n".join(reasons),
                        "keyword_hits_across_models": json.dumps(dict(kws), ensure_ascii=False),
                    }
                )

    pd.DataFrame(summary_rows).to_csv(EXPORT_DIR / "common_wrong_by_split.csv", index=False)
    pd.DataFrame(small3_summary_rows).to_csv(
        EXPORT_DIR / "common_wrong_small3_by_split.csv", index=False
    )

    # --- eval_A: 3-small vs 6-all overlap and rule histograms (small vs large on same items) ---
    eval_a = df[df["dataset_split"] == "eval_A"]
    eval_tasks = sorted(eval_a["task_id"].unique())
    six_wrong_set: set = set()
    small3_wrong_set: set = set()
    for tid in eval_tasks:
        tg = eval_a[eval_a["task_id"] == tid]
        if len(tg["model_dir"].unique()) < n_models:
            continue
        if tg["wrong"].all():
            six_wrong_set.add(tid)
        if _task_all_wrong_for_subset(tg, SMALL_LLAMA_QWEN, len(SMALL_LLAMA_QWEN)):
            small3_wrong_set.add(tid)

    n_s3 = len(small3_wrong_set)
    n_6 = len(six_wrong_set)
    small3_not_six = small3_wrong_set - six_wrong_set
    # On tasks where all 3 small wrong but some large correct: large "rescues"
    large_rescue_tasks = list(small3_not_six)

    def _rule_counts(
        frame: pd.DataFrame, tids: List[Any], model_set: frozenset, wrong_only: bool
    ) -> Counter:
        ct: Counter = Counter()
        mask = frame["task_id"].isin(tids) & frame["model_dir"].isin(model_set)
        if wrong_only:
            mask = mask & (~frame["is_correct"].fillna(False))
        sub = frame[mask]
        for pr in sub["parsed_rule_id"].dropna().astype(str):
            ct[pr] += 1
        return ct

    ct_small_on_s3 = _rule_counts(eval_a, list(small3_wrong_set), SMALL_LLAMA_QWEN, False)
    ct_large_on_s3 = _rule_counts(eval_a, list(small3_wrong_set), LARGE_TIER, False)
    ct_large_wrong_on_s3 = _rule_counts(eval_a, list(small3_wrong_set), LARGE_TIER, True)
    ct_small_on_6 = _rule_counts(eval_a, list(six_wrong_set), SMALL_LLAMA_QWEN, False)
    ct_large_on_6 = _rule_counts(eval_a, list(six_wrong_set), LARGE_TIER, False)
    ct_large_wrong_on_6 = _rule_counts(eval_a, list(six_wrong_set), LARGE_TIER, True)

    sub_s3_large = eval_a[
        eval_a["task_id"].isin(small3_wrong_set) & eval_a["model_dir"].isin(LARGE_TIER)
    ]
    large_mean_acc_on_small3_fail = (
        float(sub_s3_large["is_correct"].fillna(False).mean()) if len(sub_s3_large) else 0.0
    )
    sub_s36_large = eval_a[
        eval_a["task_id"].isin(six_wrong_set) & eval_a["model_dir"].isin(LARGE_TIER)
    ]
    large_mean_acc_on_six_fail = (
        float(sub_s36_large["is_correct"].fillna(False).mean()) if len(sub_s36_large) else 0.0
    )

    def counter_to_rows(ct: Counter, label: str) -> List[Dict[str, Any]]:
        tot = sum(ct.values()) or 1
        return [
            {"tier_slice": label, "parsed_rule_id": k, "count": v, "share": round(v / tot, 5)}
            for k, v in ct.most_common()
        ]

    overlap_rows = [
        {
            "metric": "eval_A_tasks_three_small_all_wrong",
            "count": n_s3,
            "rate_of_1512": round(n_s3 / 1512, 5),
        },
        {
            "metric": "eval_A_tasks_six_all_wrong",
            "count": n_6,
            "rate_of_1512": round(n_6 / 1512, 5),
        },
        {
            "metric": "eval_A_tasks_three_small_wrong_but_not_six_all_wrong",
            "count": len(small3_not_six),
            "rate_of_1512": round(len(small3_not_six) / 1512, 5),
        },
        {
            "metric": "six_all_wrong_is_subset_of_three_small_all_wrong",
            "count": int(six_wrong_set <= small3_wrong_set),
            "rate_of_1512": None,
        },
        {
            "metric": "large_tier_mean_accuracy_on_eval_A_tasks_where_small3_all_wrong",
            "count": None,
            "rate_of_1512": round(large_mean_acc_on_small3_fail, 5),
        },
        {
            "metric": "large_tier_mean_accuracy_on_eval_A_tasks_where_six_all_wrong",
            "count": None,
            "rate_of_1512": round(large_mean_acc_on_six_fail, 5),
        },
    ]
    pd.DataFrame(overlap_rows).to_csv(
        EXPORT_DIR / "common_wrong_eval_A_small3_overlap.csv", index=False
    )

    rule_compare = (
        counter_to_rows(ct_small_on_s3, "three_small_on_small3_wrong_tasks")
        + counter_to_rows(ct_large_on_s3, "three_large_all_preds_on_small3_wrong_tasks")
        + counter_to_rows(
            ct_large_wrong_on_s3, "three_large_wrong_only_on_small3_wrong_tasks"
        )
        + counter_to_rows(ct_small_on_6, "three_small_on_six_wrong_tasks")
        + counter_to_rows(ct_large_on_6, "three_large_all_preds_on_six_wrong_tasks")
        + counter_to_rows(ct_large_wrong_on_6, "three_large_wrong_only_on_six_wrong_tasks")
    )
    pd.DataFrame(rule_compare).to_csv(
        EXPORT_DIR / "common_wrong_eval_A_small_vs_large_rules.csv", index=False
    )

    # Mean accuracy by split: small Ollama trio vs large-tier trio (same task counts per model).
    acc_rows: List[Dict[str, Any]] = []
    for split, g in df.groupby("dataset_split"):
        for m in sorted(SMALL_LLAMA_QWEN | LARGE_TIER):
            mg = g[g["model_dir"] == m]
            tier = "small_llama_qwen_ollama" if m in SMALL_LLAMA_QWEN else "large_gpt4_qwen_deepseek"
            tot = len(mg)
            acc = float(mg["is_correct"].fillna(False).mean()) if tot else 0.0
            acc_rows.append(
                {
                    "dataset_split": split,
                    "tier": tier,
                    "model_dir": m,
                    "n": tot,
                    "accuracy": round(acc, 5),
                }
            )
    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(EXPORT_DIR / "small_vs_large_accuracy_by_model_split.csv", index=False)
    tier_mean = (
        acc_df.groupby(["dataset_split", "tier"], as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "mean_accuracy_across_models_in_tier"})
    )
    tier_mean.to_csv(EXPORT_DIR / "small_vs_large_mean_accuracy_by_split.csv", index=False)

    uni_df = pd.DataFrame(universal_eval_a)
    if not uni_df.empty:
        uni_df.to_csv(EXPORT_DIR / "common_wrong_eval_A_universal.csv", index=False)

        # Aggregate keyword counts over all snippets (each model counts once per item).
        total_kw = Counter()
        for _, row in uni_df.iterrows():
            kobj = json.loads(row["keyword_hits_across_models"])
            for k, v in kobj.items():
                total_kw[k] += int(v)
        pd.DataFrame(
            [{"keyword": k, "hit_count_models_times_items": v} for k, v in total_kw.most_common()]
        ).to_csv(EXPORT_DIR / "common_wrong_eval_A_reasoning_keywords.csv", index=False)

    print(f"Models: {models} (n={n_models})")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nWrote CSVs under {EXPORT_DIR}")


if __name__ == "__main__":
    main()
