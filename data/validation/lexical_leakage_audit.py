"""
Lexical Leakage Audit (LLA).

Verifies the vocabulary-separation invariants across the full dataset:

    R-CoT   : SFT CoT responses must NOT contain any TEST_MCQ_RULE_VOCAB phrase.
    R-CoT-F : SFT CoT responses must NOT contain any FORBIDDEN_RESEARCH_TERM.
    R-Scen-F: Teacher-generated scenario narratives must NOT contain any
              FORBIDDEN_RESEARCH_TERM.
    R-Scen-M: Teacher-generated scenario narratives must NOT contain any
              TEST_MCQ_RULE_VOCAB phrase.
    R-MCQ-F : MCQ option text must NOT contain any FORBIDDEN_RESEARCH_TERM.
    R-MCQ-T : (Informational only.) Reports MCQ option text that reuses
              TRAIN_COT_VOCAB phrases. This is not a hard violation because
              concept-level overlap (e.g. "private baseline") is natural in
              both CoT and MCQ, but it is useful to monitor.

Inputs (any subset can be audited in one run):
  --train-structured  data/structured/train.json
  --eval-structured   one or more of: eval_A.json, eval_B.json, placebo_test.json,
                      ood_social.json, ood_career.json
  --sft-jsonl         data/sft/train_pairs.jsonl
  --dpo-jsonl         data/dpo/train_pairs.jsonl

Output:
  * Prints a JSON summary with per-invariant violation counts.
  * Optional --report-file writes a full JSON report with sample record ids
    and the phrases that leaked.
  * Exits 1 if any violation is found, 0 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config.vocabulary_pools import (  # noqa: E402
    contains_any_phrase,
    TEST_MCQ_RULE_VOCAB,
    TRAIN_COT_VOCAB,
    FORBIDDEN_RESEARCH_TERMS,
)


# ---------------------------------------------------------------------------
# Audit primitives
# ---------------------------------------------------------------------------

def _audit_text(text: Optional[str], phrases: Iterable[str]) -> List[str]:
    if not text:
        return []
    return contains_any_phrase(text, phrases)


def _collect_mcq_option_texts(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return [(option_letter, option_text)] for any record that carries MCQ."""
    out: List[Tuple[str, str]] = []
    mcq = record.get("mcq")
    if isinstance(mcq, dict):
        for opt in mcq.get("options", []) or []:
            letter = opt.get("letter", "?")
            text = opt.get("text") or ""
            if text:
                out.append((letter, text))
    return out


# ---------------------------------------------------------------------------
# Per-file auditors
# ---------------------------------------------------------------------------

def audit_structured_file(
    path: Path,
    eval_mode: bool,
    informational_mcq_train_overlap: bool = True,
) -> Dict[str, Any]:
    """Audit a structured JSON file (train or eval split).

    For training records: primarily audits scenario_text only (MCQ absent).
    For eval records: audits scenario_text + MCQ options.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    violations: List[Dict[str, Any]] = []
    informational: List[Dict[str, Any]] = []
    for rec in records:
        rid = rec.get("record_id", "<unknown>")
        scenario_text = rec.get("scenario_text")

        # R-Scen-F: forbidden terms in narrative.
        hits = _audit_text(scenario_text, FORBIDDEN_RESEARCH_TERMS)
        if hits:
            violations.append({"record_id": rid, "rule": "R-Scen-F", "phrases": hits})
        # R-Scen-M: MCQ rule vocab in narrative.
        hits = _audit_text(scenario_text, TEST_MCQ_RULE_VOCAB)
        if hits:
            violations.append({"record_id": rid, "rule": "R-Scen-M", "phrases": hits})

        if eval_mode:
            for letter, text in _collect_mcq_option_texts(rec):
                # R-MCQ-F: forbidden terms in option text.
                hits = _audit_text(text, FORBIDDEN_RESEARCH_TERMS)
                if hits:
                    violations.append({
                        "record_id": rid, "option": letter,
                        "rule": "R-MCQ-F", "phrases": hits,
                    })
                # R-MCQ-T: TRAIN_COT_VOCAB in option text (informational).
                hits = _audit_text(text, TRAIN_COT_VOCAB)
                if hits:
                    entry = {
                        "record_id": rid, "option": letter,
                        "rule": "R-MCQ-T", "phrases": hits,
                    }
                    if informational_mcq_train_overlap:
                        informational.append(entry)
                    else:
                        violations.append(entry)

    return {
        "path": str(path),
        "record_count": len(records),
        "violation_count": len(violations),
        "informational_count": len(informational),
        "violations": violations,
        "informational": informational,
    }


def audit_sft_jsonl(path: Path) -> Dict[str, Any]:
    """Audit an SFT JSONL for CoT leakage."""
    violations: List[Dict[str, Any]] = []
    n = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            n += 1
            rec = json.loads(line)
            rid = rec.get("sft_id") or rec.get("source_record_id") or "<unknown>"
            resp = rec.get("response") or ""
            hits = _audit_text(resp, TEST_MCQ_RULE_VOCAB)
            if hits:
                violations.append({"sft_id": rid, "rule": "R-CoT", "phrases": hits})
            hits = _audit_text(resp, FORBIDDEN_RESEARCH_TERMS)
            if hits:
                violations.append({"sft_id": rid, "rule": "R-CoT-F", "phrases": hits})
            # Also check the user_prompt for scenario leaks.
            user_prompt = rec.get("user_prompt") or ""
            hits = _audit_text(user_prompt, FORBIDDEN_RESEARCH_TERMS)
            if hits:
                violations.append({"sft_id": rid, "rule": "R-Scen-F", "phrases": hits})
            hits = _audit_text(user_prompt, TEST_MCQ_RULE_VOCAB)
            if hits:
                violations.append({"sft_id": rid, "rule": "R-Scen-M", "phrases": hits})
    return {
        "path": str(path),
        "record_count": n,
        "violation_count": len(violations),
        "violations": violations,
    }


def audit_dpo_jsonl(path: Path) -> Dict[str, Any]:
    """Audit a DPO JSONL for CoT leakage in both chosen and rejected."""
    violations: List[Dict[str, Any]] = []
    n = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            n += 1
            rec = json.loads(line)
            rid = rec.get("dpo_id") or rec.get("source_record_id") or "<unknown>"
            for side in ("chosen", "rejected"):
                text = rec.get(side) or ""
                hits = _audit_text(text, TEST_MCQ_RULE_VOCAB)
                if hits:
                    violations.append({
                        "dpo_id": rid, "side": side,
                        "rule": "R-CoT", "phrases": hits,
                    })
                hits = _audit_text(text, FORBIDDEN_RESEARCH_TERMS)
                if hits:
                    violations.append({
                        "dpo_id": rid, "side": side,
                        "rule": "R-CoT-F", "phrases": hits,
                    })
    return {
        "path": str(path),
        "record_count": n,
        "violation_count": len(violations),
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-structured", default=None,
                   help="Path to structured/train.json (audited as narrative-only).")
    p.add_argument("--eval-structured", nargs="*", default=[],
                   help="Paths to eval/placebo/OOD structured JSON files (MCQ audited).")
    p.add_argument("--sft-jsonl", default=None, help="SFT train_pairs.jsonl")
    p.add_argument("--dpo-jsonl", default=None, help="DPO train_pairs.jsonl")
    p.add_argument("--report-file", default=None,
                   help="If set, write a full JSON report including per-record violations.")
    p.add_argument("--fail-on-violation", action="store_true",
                   help="Exit with code 1 if any violation is found.")
    p.add_argument("--strict-mcq-train-overlap", action="store_true",
                   help=("Treat TRAIN_COT_VOCAB appearing in MCQ options as a HARD violation "
                         "rather than informational."))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report: Dict[str, Any] = {"audits": []}
    total_violations = 0

    informational = not args.strict_mcq_train_overlap

    if args.train_structured:
        r = audit_structured_file(
            Path(args.train_structured), eval_mode=False,
            informational_mcq_train_overlap=informational,
        )
        total_violations += r["violation_count"]
        report["audits"].append({"role": "train_structured", **r})

    for ev in args.eval_structured:
        r = audit_structured_file(
            Path(ev), eval_mode=True,
            informational_mcq_train_overlap=informational,
        )
        total_violations += r["violation_count"]
        report["audits"].append({"role": "eval_structured", **r})

    if args.sft_jsonl:
        r = audit_sft_jsonl(Path(args.sft_jsonl))
        total_violations += r["violation_count"]
        report["audits"].append({"role": "sft_jsonl", **r})

    if args.dpo_jsonl:
        r = audit_dpo_jsonl(Path(args.dpo_jsonl))
        total_violations += r["violation_count"]
        report["audits"].append({"role": "dpo_jsonl", **r})

    # Terse summary
    summary = {
        "total_violations": total_violations,
        "by_source": [
            {
                "role": a["role"],
                "path": a["path"],
                "record_count": a["record_count"],
                "violation_count": a["violation_count"],
                "informational_count": a.get("informational_count", 0),
            }
            for a in report["audits"]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.report_file:
        Path(args.report_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_file).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if args.fail_on_violation and total_violations > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
