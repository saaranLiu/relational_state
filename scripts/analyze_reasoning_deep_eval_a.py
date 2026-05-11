#!/usr/bin/env python3
"""
Deeper lexical / structural analysis of Eval-A model reasoning (Reasoning: block).

Reads evaluation/outputs/*/*_predictions.jsonl, filters dataset_split == eval_A,
extracts reasoning text, adds hand-crafted linguistic features, aggregates by
model, by wrong parsed_rule_id, and compares small Ollama trio vs large tier
on the six-model-universal-wrong task subset.

Outputs (analysis/model_performance_error_analysis/):
  - reasoning_deep_eval_a_per_row.csv
  - reasoning_deep_eval_a_by_model.csv
  - reasoning_deep_eval_a_by_parsed_rule_wrong.csv
  - reasoning_deep_eval_a_small_vs_large_six_wrong.csv
  - reasoning_deep_eval_a_tail_letter_mismatch.csv
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "evaluation" / "outputs"
EXPORT_DIR = ROOT / "analysis" / "model_performance_error_analysis"

SMALL3 = frozenset({"ollama-llama3-1-latest", "ollama-llama3-8b", "ollama-qwen2-7b"})
LARGE3 = frozenset({"gpt4", "Qwen", "DeepSeek-R1-671B"})


def read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def extract_reasoning(raw: str) -> str:
    if not raw:
        return ""
    m = re.search(
        r"Reasoning:\s*(.*?)(?=\n\s*Choice:|\nChoice:|\Z)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return raw.strip()


def extract_choice_letter(raw: str) -> Optional[str]:
    if not raw:
        return None
    m = re.search(r"Choice:\s*([A-E])\b", raw, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def last_reasoning_stated_letter(text: str) -> Optional[str]:
    """Heuristic: last explicit A–E tied to option/answer language in tail of reasoning."""
    if not text:
        return None
    tail = text[-650:]
    patterns = [
        r"(?:answer|choice|pick|select|choose|letter)\s*(?:is|:)?\s*\*?\*?\s*([A-E])\b",
        r"(?:option|rule)\s*\(?([A-E])\)?(?:\s*is|\s*would|\s*seems|\s*fits)?",
        r"\b(?:therefore|thus|so),?\s+(?:the\s+)?(?:answer|choice|letter)\s+is\s*\*?\*?\s*([A-E])\b",
        r"(?:go(?:es)? with|land(?:s)? on)\s+(?:option\s*)?\(?([A-E])\)?",
        r"(?:I(?:'d| would)\s+)?(?:pick|choose|select)\s+(?:option\s*)?\(?([A-E])\b",
    ]
    found: List[Tuple[int, str]] = []
    for pat in patterns:
        for m in re.finditer(pat, tail, flags=re.IGNORECASE):
            found.append((m.end(), m.group(1).upper()))
    if not found:
        return None
    found.sort(key=lambda x: x[0])
    return found[-1][1]


def featurize_reasoning(text: str) -> Dict[str, Any]:
    t = text or ""
    wc = len(t.split())
    sents = re.split(r"(?<=[.!?])\s+", t)
    n_sents = len([s for s in sents if s.strip()])
    return {
        "reasoning_char_len": len(t),
        "reasoning_word_len": wc,
        "reasoning_sentence_count": n_sents,
        "flag_single_peer_or_copy": bool(
            re.search(
                r"single\s+closest|closest\s+(?:parent|friend|peer)|one\s+peer|"
                r"copy\s+whatever|mirror|emulate|match\s+(?:her|his|their|the)\s+",
                t,
                re.I,
            )
        ),
        "flag_multi_peer_or_aggregate": bool(
            re.search(
                r"each\s+of\s+the\s+parents|every\s+parent|all\s+parents|"
                r"weighted\s+(?:peer|aggregate|average)|"
                r"closeness[- ]weighted|relationship\s+weight|"
                r"multiple\s+peers|group\s+of\s+parents",
                t,
                re.I,
            )
        ),
        "flag_numeric_deliberation": bool(
            re.search(
                r"\$\d|divided\s+by|/\s*5|average\s+is|median\s+is|sum\s+of|total\s+\$",
                t,
                re.I,
            )
        ),
        "flag_explicit_compromise": bool(
            re.search(
                r"compromise|balance|half\s+and\s+half|1:1|torn\s+between|"
                r"mix(?:ture|ed)?\s+(?:of\s+)?(?:private|peer|social)",
                t,
                re.I,
            )
        ),
        "flag_stick_private": bool(
            re.search(
                r"stick\s+to\s+(?:my|the)\s+(?:original|own|quiet|simple|private)|"
                r"ignore\s+peer|without\s+any\s+lift|private\s+baseline\s+only|"
                r"anchor\s+entirely\s+on\s+the\s+private",
                t,
                re.I,
            )
        ),
        "flag_invert_or_counter": bool(
            re.search(
                r"invert|counter[- ]conform|go\s+lower|against\s+the\s+grain|opposite",
                t,
                re.I,
            )
        ),
    }


def load_eval_a_frame() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(OUTPUT_ROOT.glob("*/*_predictions.jsonl")):
        split = path.stem.replace("_predictions", "")
        if split != "eval_A":
            continue
        model_dir = "gpt4" if path.parent.name == "gpt-4" else path.parent.name
        for r in read_jsonl(path):
            meta = r.get("meta") or {}
            ds = r.get("dataset_split") or meta.get("dataset_split")
            if ds != "eval_A":
                continue
            raw = r.get("raw_response") or ""
            reason = extract_reasoning(raw)
            ch = extract_choice_letter(raw)
            parsed = r.get("parsed_letter")
            if isinstance(parsed, str):
                parsed = parsed.upper()
            tail_letter = last_reasoning_stated_letter(reason)
            feats = featurize_reasoning(reason)
            rows.append(
                {
                    "model_dir": model_dir,
                    "task_id": r.get("task_id"),
                    "is_correct": r.get("is_correct"),
                    "gold_letter": r.get("gold_letter"),
                    "parsed_letter": parsed,
                    "parsed_rule_id": r.get("parsed_rule_id"),
                    "choice_line_letter": ch,
                    "tail_stated_letter": tail_letter,
                    "tail_matches_choice_line": (
                        (tail_letter == ch) if (tail_letter and ch) else None
                    ),
                    "reasoning_text": reason[:4000],
                    **feats,
                }
            )
    return pd.DataFrame(rows)


def six_wrong_task_ids(df: pd.DataFrame) -> set:
    df = df.copy()
    df["wrong"] = ~df["is_correct"].fillna(False)
    out: set = set()
    for tid, g in df.groupby("task_id"):
        if g["wrong"].all() and len(g["model_dir"].unique()) == 6:
            out.add(tid)
    return out


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_eval_a_frame()
    df["wrong"] = ~df["is_correct"].fillna(False)
    df["tier"] = df["model_dir"].apply(
        lambda m: "small_ollama" if m in SMALL3 else ("large_api" if m in LARGE3 else "?")
    )
    uni = six_wrong_task_ids(df)
    df["six_universal_wrong_task"] = df["task_id"].isin(uni)

    per_path = EXPORT_DIR / "reasoning_deep_eval_a_per_row.csv"
    df.drop(columns=["reasoning_text"]).to_csv(per_path, index=False)

    flag_cols = [
        c
        for c in df.columns
        if c.startswith("flag_") or c in ("reasoning_char_len", "reasoning_word_len", "reasoning_sentence_count")
    ]

    by_model = (
        df.groupby("model_dir", as_index=False)
        .agg(
            n=("task_id", "count"),
            acc=("is_correct", lambda s: float(s.fillna(False).mean())),
            mean_chars_wrong=("reasoning_char_len", lambda s: float(s[df.loc[s.index, "wrong"]].mean())),
            mean_chars_correct=("reasoning_char_len", lambda s: float(s[~df.loc[s.index, "wrong"]].mean())),
            p_single_when_wrong=("flag_single_peer_or_copy", lambda s: float(s[df.loc[s.index, "wrong"]].mean())),
            p_multi_when_wrong=("flag_multi_peer_or_aggregate", lambda s: float(s[df.loc[s.index, "wrong"]].mean())),
            p_numeric_when_wrong=("flag_numeric_deliberation", lambda s: float(s[df.loc[s.index, "wrong"]].mean())),
            p_compromise_when_wrong=("flag_explicit_compromise", lambda s: float(s[df.loc[s.index, "wrong"]].mean())),
            p_stick_private_when_wrong=("flag_stick_private", lambda s: float(s[df.loc[s.index, "wrong"]].mean())),
        )
    )
    # Fix aggregation: groupby apply per group properly
    def agg_group(g: pd.DataFrame) -> pd.Series:
        w = g["wrong"]
        return pd.Series(
            {
                "n": len(g),
                "acc": float(g["is_correct"].fillna(False).mean()),
                "mean_chars_wrong": float(g.loc[w, "reasoning_char_len"].mean()),
                "mean_chars_correct": float(g.loc[~w, "reasoning_char_len"].mean()),
                "p_single_when_wrong": float(g.loc[w, "flag_single_peer_or_copy"].mean()) if w.any() else 0.0,
                "p_multi_when_wrong": float(g.loc[w, "flag_multi_peer_or_aggregate"].mean()) if w.any() else 0.0,
                "p_numeric_when_wrong": float(g.loc[w, "flag_numeric_deliberation"].mean()) if w.any() else 0.0,
                "p_compromise_when_wrong": float(g.loc[w, "flag_explicit_compromise"].mean()) if w.any() else 0.0,
                "p_stick_private_when_wrong": float(g.loc[w, "flag_stick_private"].mean()) if w.any() else 0.0,
            }
        )

    by_model = df.groupby("model_dir", group_keys=False).apply(agg_group).reset_index()
    by_model.to_csv(EXPORT_DIR / "reasoning_deep_eval_a_by_model.csv", index=False)

    wrong = df[df["wrong"] & df["parsed_rule_id"].notna()].copy()
    by_rule = wrong.groupby("parsed_rule_id", group_keys=False).apply(agg_group).reset_index()
    by_rule.to_csv(EXPORT_DIR / "reasoning_deep_eval_a_by_parsed_rule_wrong.csv", index=False)

    sub = df[df["six_universal_wrong_task"]].copy()
    sv = (
        sub.groupby("tier", group_keys=False)
        .apply(agg_group)
        .reset_index()
        .rename(columns={"tier": "model_dir"})
    )
    sv.to_csv(EXPORT_DIR / "reasoning_deep_eval_a_small_vs_large_six_wrong.csv", index=False)

    mm = df[df["tail_matches_choice_line"].notna()].copy()
    mm_summ = (
        mm.groupby("model_dir", as_index=False)
        .agg(
            n_tail_both=("tail_matches_choice_line", "count"),
            frac_tail_matches_choice=("tail_matches_choice_line", "mean"),
        )
    )
    mm_summ.to_csv(EXPORT_DIR / "reasoning_deep_eval_a_tail_letter_mismatch.csv", index=False)

    # Co-occurrence: wrong only, P(flag | parsed_rule_id) top rules
    top_rules = wrong["parsed_rule_id"].value_counts().head(6).index.tolist()
    co_rows: List[Dict[str, Any]] = []
    for rule in top_rules:
        g = wrong[wrong["parsed_rule_id"] == rule]
        n = len(g)
        for fc in [
            "flag_single_peer_or_copy",
            "flag_multi_peer_or_aggregate",
            "flag_numeric_deliberation",
            "flag_explicit_compromise",
            "flag_stick_private",
        ]:
            co_rows.append(
                {
                    "parsed_rule_id": rule,
                    "n_wrong": n,
                    "feature": fc,
                    "p_feature_given_rule": float(g[fc].mean()) if n else 0.0,
                }
            )
    pd.DataFrame(co_rows).to_csv(
        EXPORT_DIR / "reasoning_deep_eval_a_flags_by_top_wrong_rule.csv", index=False
    )

    print(f"Eval-A rows: {len(df)} | six-universal tasks: {len(uni)}")
    print(by_model.to_string(index=False))
    print("\nWrote:", EXPORT_DIR)


if __name__ == "__main__":
    main()
