"""
Three disjoint vocabulary pools enforced across the pipeline.

Purpose
-------
Prevent shortcut learning where the model matches a training CoT phrase to a
test MCQ rule label via surface word overlap. We maintain three disjoint
lexical pools:

  * TRAIN_COT_VOCAB
    Phrases allowed (and encouraged) inside the SFT Chain-of-Thought. These
    anchor concepts like "baseline floor", "closeness-weighted mix", etc.

  * TEST_MCQ_RULE_VOCAB
    Phrases used INSIDE MCQ option paraphrases (see rule_label_templates.py).
    Must NOT appear in any SFT CoT; if they do, the audit script flags it.

  * TEST_SCENARIO_VOCAB
    Phrases that the teacher LLM is ALLOWED to use when writing the test
    scenario narrative (Eval-A / -B / -C scenes). Overlap with TRAIN_COT_VOCAB
    is tolerated as long as TEST_MCQ_RULE_VOCAB is kept out of the scenario.

The auditor (data/validation/lexical_leakage_audit.py, shipped later)
performs a case-insensitive substring match for each pool and raises if any
pair violates the disjointness invariant it targets.

This module exports:
  - The three frozensets of phrases.
  - A helper `contains_any_phrase` used by both the teacher prompt validators
    and the downstream auditor.
"""
from __future__ import annotations

import re
from typing import FrozenSet, Iterable, List, Tuple


# -----------------------------------------------------------------------------
# Training CoT vocabulary (SFT target language)
# -----------------------------------------------------------------------------
# This set anchors the CANONICAL phrases the CoT generator can emit inside
# reasoning. Each phrase is a short lowercase substring; audits are
# case-insensitive. These SHOULD be distinct from MCQ paraphrases.
TRAIN_COT_VOCAB: FrozenSet[str] = frozenset({
    "private baseline",
    "no-comparison floor",
    "personal reservation level",
    "closeness-weighted reference",
    "weighted peer reference",
    "reference aggregate",
    "comparison sensitivity",
    "peer-weight profile",
    "positional consumption cue",
    "positional investment cue",
    "non-positional cue",
    "langtry best-response form",
    "counterfactual check",
    "scene-type taxonomy",
    "positional scene type",
    "non-positional scene type",
    "scene-type flag",
})


# -----------------------------------------------------------------------------
# Test MCQ rule-label vocabulary (paraphrases in rule_label_templates.py)
# -----------------------------------------------------------------------------
# These phrases are allowed ONLY inside MCQ options. If a training CoT leaks
# any of them verbatim, the audit flags the training sample.
TEST_MCQ_RULE_VOCAB: FrozenSet[str] = frozenset({
    # Rule A (peer-weighted)
    "scale the",
    "how close each",
    "closeness-weighted blend",
    "in proportion to how close",
    "relationship-weighted peer pull",
    "proximity-weighted aggregate",
    "proximity-graded blend",
    "weighted peer aggregate",
    "relational weight",
    # Rule B (top anchor)
    "single closest",
    "tightest",
    "one nearest",
    "most tight-knit",
    "single most intimate",
    "single inner-circle",
    "closest-in-life",
    "single nearest",
    "tightest-bond",
    # Rule C (uniform avg)
    "democratic vote",
    "pool the",
    "flat average",
    "weigh every",
    "unweighted mean",
    "plain mean",
    "ungraded average",
    "strictly egalitarian average",
    "strictly uniform",
    # Rule D (pure private)
    "stay on the private floor",
    "ignore the",
    "stick with what the private baseline",
    "hold the personal floor",
    "private utility only",
    "pegged to the private baseline",
    "not a domain where",
    "visible peer amounts as information",
    "private situation alone would justify",
    # Rule E (closest mimicry)
    "copy whatever",
    "mirror the closest",
    "match the one nearest",
    "just replicate",
    "replace the private baseline",
    "clone the closest-in-life",
    "skip the baseline math",
    "adopt the nearest",
    "lean fully on the closest",
    # Rule F (median anchor)
    "anchor on the middle",
    "median of what",
    "middle-of-the-road",
    "mid-pack",
    "middle value",
    "half of the",
    "sort the",
    "representative central",
    "center-pick",
    # Rule G (counter-conformist)
    "deliberately push below",
    "cut the",
    "contrarian signal",
    "refuse to match",
    "invert the peer signal",
    "flip its sign",
    "rebellious stance",
    "act as a dampener",
    "deliberate retreat",
    # Rule H (equal mix)
    "split the difference evenly",
    "take the midpoint",
    "blend half the",
    "average the personal baseline",
    "fifty-fifty",
    "half-baseline half-peer",
    "1:1 ratio",
    "persona-agnostic balance",
    "arithmetic mean of the private",
})


# -----------------------------------------------------------------------------
# Test scenario narrative vocabulary (teacher-writable)
# -----------------------------------------------------------------------------
# The teacher LLM is ALLOWED to use these in the scenario narrative. Overlap
# with TRAIN_COT_VOCAB is fine (scenarios are shared between train and test,
# only the CoT preamble enforces vocabulary separation). The auditor checks
# ONLY that TEST_MCQ_RULE_VOCAB does not appear in the scenario narrative.
TEST_SCENARIO_VOCAB: FrozenSet[str] = frozenset({
    "group chat",
    "shared trip-ledger",
    "wedding gift pool",
    "story feed",
    "peer circle",
    "closest friends",
    "inner circle",
    "acquaintances",
    "visible spending",
    "norm",
    "pressure",
    "bill",
    "hours",
    "study group",
    "cohort channel",
})


# -----------------------------------------------------------------------------
# Forbidden research-leak phrases (must appear nowhere)
# -----------------------------------------------------------------------------
# Hard guardrails: the teacher LLM and the CoT generator must never emit any
# of these, because they would reveal the latent design to the model.
FORBIDDEN_RESEARCH_TERMS: FrozenSet[str] = frozenset({
    r"\balpha\b",
    r"\bgamma\b",
    r"\bbeta\b",
    r"\bg_ij\b",
    r"\bx_j\b",
    r"\bx_star\b",
    r"\bx_i\*\b",
    r"\bsample_type\b",
    r"\bbucket\b",
    r"\blatent parameter\b",
    r"\bbest response\b",
    r"\bdispersion bucket\b",
    r"\bskew bucket\b",
    r"\blangtry\b",
})


# -----------------------------------------------------------------------------
# Simple lookup API (used by the scenario validator + lexical audit)
# -----------------------------------------------------------------------------

def _compile_patterns(phrases: Iterable[str]) -> List[re.Pattern[str]]:
    out: List[re.Pattern[str]] = []
    for p in phrases:
        # If the phrase looks like a \b...\b regex, compile it as-is; else
        # treat it as a literal substring wrapped in word-boundary logic that
        # is relaxed (we just lowercase-match the substring).
        if p.startswith("\\b") and p.endswith("\\b"):
            out.append(re.compile(p, re.IGNORECASE))
        else:
            out.append(re.compile(re.escape(p), re.IGNORECASE))
    return out


_TRAIN_COT_PATTERNS = _compile_patterns(TRAIN_COT_VOCAB)
_TEST_MCQ_PATTERNS = _compile_patterns(TEST_MCQ_RULE_VOCAB)
_TEST_SCENARIO_PATTERNS = _compile_patterns(TEST_SCENARIO_VOCAB)
_FORBIDDEN_PATTERNS = _compile_patterns(FORBIDDEN_RESEARCH_TERMS)


def contains_any_phrase(text: str, phrases: Iterable[str]) -> List[str]:
    """Return the list of phrases present in `text` (case-insensitive).

    Phrases that look like pre-compiled regex (\\b...\\b) are handled the same
    way as literal substrings. Used by the audit and by teacher-prompt guards.
    """
    hits: List[str] = []
    for p in phrases:
        if p.startswith("\\b") and p.endswith("\\b"):
            if re.search(p, text, flags=re.IGNORECASE):
                hits.append(p)
        else:
            if re.search(re.escape(p), text, flags=re.IGNORECASE):
                hits.append(p)
    return hits


def find_mcq_rule_leaks_in_cot(cot_text: str) -> List[str]:
    """Phrases from TEST_MCQ_RULE_VOCAB that leaked into an SFT CoT.

    Each hit is a pipeline bug -- either rewrite the CoT or relax the rule
    vocabulary (not recommended).
    """
    return contains_any_phrase(cot_text, TEST_MCQ_RULE_VOCAB)


def find_forbidden_research_terms(text: str) -> List[str]:
    """Research-leak terms that appeared anywhere in a training or test string."""
    return contains_any_phrase(text, FORBIDDEN_RESEARCH_TERMS)


def describe_pools() -> dict:
    return {
        "train_cot_phrase_count": len(TRAIN_COT_VOCAB),
        "test_mcq_rule_phrase_count": len(TEST_MCQ_RULE_VOCAB),
        "test_scenario_phrase_count": len(TEST_SCENARIO_VOCAB),
        "forbidden_research_term_count": len(FORBIDDEN_RESEARCH_TERMS),
        "disjointness_invariants": [
            "train_cot ∩ test_mcq_rule == ∅  (enforced by audit)",
            "test_scenario ∩ test_mcq_rule == ∅  (enforced by audit)",
            "forbidden_research ∩ any-output == ∅  (hard fail)",
        ],
    }
