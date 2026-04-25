"""
Unified evaluation runner for the relational-state benchmark suite.

Supported splits (auto-detected from `dataset_split`):

    eval_A           4-choice rule-identification MCQ on d1/d3 scenes.
    placebo_test     4-choice MCQ on d2 (placebo, gold = D_pure_private).
    eval_B           Pairwise comparative-statics. Two structured records
                     share a `pair.partner_record_id`; the runner groups them
                     into a single 2-scenario prompt and asks the model to
                     judge the direction (which side would commit more).
    ood_social       2-candidate MCQ (pick the friendship partner whose b/c
                     ratio matches the protagonist's).
    ood_career       2-firm MCQ (pick the firm where the big-pond vs small-
                     pond tradeoff favours sorting).

For each task the runner:
    1. Loads structured records (teacher scenario_text REQUIRED for eval-A/B/
       placebo; OOD accepts raw structured cues if scenario_text is empty).
    2. Builds a prompt message list appropriate for the task.
    3. Calls an OpenAI-compatible chat endpoint with retries / concurrency.
    4. Parses the model's answer letter from the response.
    5. Writes JSONL predictions with all the metadata needed for downstream
       stratified summarisation (by alpha bucket, cell, held-out flag,
       perturbation type, match-distance bucket, alpha_2i bucket, ...).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, local
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from openai import OpenAI
except ImportError as exc:
    raise ImportError("Missing dependency `openai`. `pip install openai`.") from exc

from data.config.record_access import (  # noqa: E402
    oracle_latent,
    oracle_mcq,
    oracle_pair,
    oracle_ood_social,
    oracle_ood_career,
)


_LOCK = Lock()
_THREAD = local()


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_EVAL_A = (
    "You are a careful reasoner about everyday relational decisions. "
    "Read the scenario and the four candidate reasoning rules. "
    "Pick the ONE rule that best explains what the protagonist would commit to. "
    "Reason briefly, then answer with a single letter."
)

SYSTEM_PROMPT_EVAL_B = (
    "You are a careful reasoner about everyday relational decisions. "
    "You will see two scenarios (A and B) that describe the SAME kind of commitment "
    "but differ in some detail (social sensitivity, private baseline, or what peers are doing). "
    "Judge which side would settle on a HIGHER commitment. Reason briefly, then answer with a letter."
)

SYSTEM_PROMPT_OOD_SOCIAL = (
    "You are a careful reasoner about friendship formation. "
    "A person is deciding which of two people to match with. A stable friendship requires both sides to "
    "have a similar benefit-cost ratio b/c. The prompt describes b as a benefit tier and c as a "
    "financial-constraint cost tier. Pick the partner whose implied ratio most closely matches the "
    "protagonist's. "
    "Reason briefly, then answer with a letter."
)

SYSTEM_PROMPT_OOD_CAREER = (
    "You are a careful reasoner about career sorting between firms. "
    "A worker compares two firms that differ in the average coworker wage and the wage they themselves would earn. "
    "They also care about how their wage compares to the coworker average (relative status). "
    "Pick the firm that best fits the worker's status-sensitivity described in the scenario. "
    "Reason briefly, then answer with a letter."
)

FORMAT_INSTRUCTION_ABCD = (
    "Output format (STRICT — do not add extra lines):\n"
    "Reasoning: <exactly 1 to 3 sentences, no more>\n"
    "Choice: <one letter: A, B, C, or D>"
)
FORMAT_INSTRUCTION_AB = (
    "Output format (STRICT — do not add extra lines):\n"
    "Reasoning: <exactly 1 to 3 sentences, no more>\n"
    "Choice: <one letter: A or B>"
)


def _scenario_text_or_fallback(rec: Dict[str, Any]) -> str:
    txt = rec.get("scenario_text")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    # Lightweight fallback for OOD or records missing teacher text.
    scene = rec["scene"]
    return f"[{scene['title']}] {scene['summary']}"


def _render_eval_a_user(rec: Dict[str, Any]) -> str:
    scene = rec["scene"]
    options_block = "\n".join(
        f"({opt['letter']}) {opt['text']}"
        for opt in rec["mcq"]["options"]
    )
    action_label = scene.get("action_label", "the commitment")
    return (
        f"Scenario:\n{_scenario_text_or_fallback(rec)}\n\n"
        f"Candidate rules for {action_label}:\n{options_block}\n\n"
        f"{FORMAT_INSTRUCTION_ABCD}"
    )


def _render_eval_b_user(rec_A: Dict[str, Any], rec_B: Dict[str, Any]) -> str:
    scene = rec_A["scene"]
    action_label = scene.get("action_label", "the commitment")
    return (
        f"Scenario A:\n{_scenario_text_or_fallback(rec_A)}\n\n"
        f"Scenario B:\n{_scenario_text_or_fallback(rec_B)}\n\n"
        f"Which side would settle on a higher {action_label}?\n"
        f"(A) Scenario A settles higher.\n"
        f"(B) Scenario B settles higher.\n\n"
        f"{FORMAT_INSTRUCTION_AB}"
    )


def _render_ood_social_user(rec: Dict[str, Any]) -> str:
    scene = rec["scene"]
    ood = rec["ood_social"]
    action_label = scene.get("action_label", "shared commitment")
    lines = []
    for opt in ood["options"]:
        if "b_hint" in opt and "c_hint" in opt:
            lines.append(
                f"({opt['letter']}) Candidate whose benefit is described as: {opt['b_hint']}; "
                f"financial constraint is described as: {opt['c_hint']}."
            )
        else:
            # Backward compatibility for older structured files.
            if "b_display" in opt and "c_display" in opt:
                lines.append(
                    f"({opt['letter']}) Candidate with benefit score {opt['b_display']} "
                    f"and financial-constraint cost {opt['c_display']} for {action_label}."
                )
            else:
                lines.append(
                    f"({opt['letter']}) A candidate whose preferred {action_label} is {opt['x_display']}"
                )
    options_block = "\n".join(lines)
    return (
        f"Scenario:\n{_scenario_text_or_fallback(rec)}\n\n"
        f"Candidates:\n{options_block}\n\n"
        f"{FORMAT_INSTRUCTION_AB}"
    )


def _render_ood_career_user(rec: Dict[str, Any]) -> str:
    ood = rec["ood_career"]
    lines = []
    for opt in ood["options"]:
        lines.append(
            f"({opt['letter']}) Firm where the worker would earn {opt['x_S_display']} "
            f"and the coworker average is {opt['x_bar_display']}; "
            f"this means the worker sits at the {opt['relative_rank']}."
        )
    options_block = "\n".join(lines)
    return (
        f"Scenario:\n{_scenario_text_or_fallback(rec)}\n\n"
        f"Firm options:\n{options_block}\n\n"
        f"{FORMAT_INSTRUCTION_AB}"
    )


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

_CHOICE_RE_4 = re.compile(r"choice\s*:\s*([abcd])\b", re.IGNORECASE)
_CHOICE_RE_2 = re.compile(r"choice\s*:\s*([ab])\b", re.IGNORECASE)


def _parse_letter(raw: str, choice_set: str) -> Optional[str]:
    text = raw.replace("*", "")
    regex = _CHOICE_RE_4 if choice_set == "ABCD" else _CHOICE_RE_2
    m = regex.search(text)
    if m:
        return m.group(1).upper()
    # Fallback: last standalone letter token in the response.
    candidates = re.findall(r"\b([ABCD])\b", text.upper() if choice_set == "ABCD" else text)
    if choice_set == "AB":
        candidates = re.findall(r"\b([AB])\b", text.upper())
    if candidates:
        return candidates[-1]
    return None


# ---------------------------------------------------------------------------
# Task assembly
# ---------------------------------------------------------------------------

def _pair_eval_b_records(records: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    by_id: Dict[str, Dict[str, Any]] = {r["record_id"]: r for r in records}
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    seen: set = set()
    for rec in records:
        pair_info = rec.get("pair") or {}
        if pair_info.get("role") != "A":
            continue
        partner = by_id.get(pair_info.get("partner_record_id"))
        if partner is None:
            continue
        key = tuple(sorted([rec["record_id"], partner["record_id"]]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((rec, partner))
    return pairs


def _iter_tasks(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand input records into per-task dicts with
    {task_id, messages, gold_letter, choice_set, meta}.
    """
    if not records:
        return []
    split = records[0].get("dataset_split", "")
    # Ensure all records share the same split (runner is single-split).
    splits = {r.get("dataset_split") for r in records}
    if len(splits) > 1:
        raise ValueError(f"Mixed splits in input: {splits}. Run per-split evaluation.")

    tasks: List[Dict[str, Any]] = []
    # NOTE: The prompt-builder helpers (_render_*_user) MUST read only public
    # record fields (scene, scenario_text, *.options). The scorer / metadata
    # here is allowed to pull from record["oracle"] via the oracle_* helpers.
    if split == "eval_A":
        for rec in records:
            if "mcq" not in rec:
                continue
            lat = oracle_latent(rec)
            mcq_o = oracle_mcq(rec)
            user_prompt = _render_eval_a_user(rec)
            tasks.append({
                "task_id": rec["record_id"],
                "split": split,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_EVAL_A},
                    {"role": "user", "content": user_prompt},
                ],
                "prompt": user_prompt,
                "choice_set": "ABCD",
                "gold_letter": mcq_o["gold_letter"],
                "meta": {
                    "dataset_split": split,
                    "scene_id": rec["scene"]["scene_id"],
                    "domain_key": rec["scene"]["domain_key"],
                    "cell_id": lat.get("cell_id"),
                    "alpha_bucket": lat.get("alpha_bucket"),
                    "dispersion_bucket": lat.get("dispersion_bucket"),
                    "skew_bucket": lat.get("skew_bucket"),
                    "is_held_out_cell": lat.get("is_held_out_cell"),
                    "gold_rule_id": mcq_o.get("gold_rule_id"),
                    "identification_margin": mcq_o.get("identification_margin"),
                    "identification_degraded": mcq_o.get("identification_degraded"),
                    "letter_to_rule": mcq_o.get("letter_to_rule", {}),
                },
            })
    elif split == "placebo_test":
        for rec in records:
            if "mcq" not in rec:
                continue
            mcq_o = oracle_mcq(rec)
            user_prompt = _render_eval_a_user(rec)
            tasks.append({
                "task_id": rec["record_id"],
                "split": split,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_EVAL_A},
                    {"role": "user", "content": user_prompt},
                ],
                "prompt": user_prompt,
                "choice_set": "ABCD",
                "gold_letter": mcq_o["gold_letter"],
                "meta": {
                    "dataset_split": split,
                    "scene_id": rec["scene"]["scene_id"],
                    "domain_key": rec["scene"]["domain_key"],
                    "gold_rule_id": mcq_o.get("gold_rule_id"),
                    "trial_alpha_used_for_distractors": mcq_o.get("trial_alpha_used_for_distractors"),
                    "letter_to_rule": mcq_o.get("letter_to_rule", {}),
                },
            })
    elif split == "eval_B":
        for rec_A, rec_B in _pair_eval_b_records(records):
            pair_o = oracle_pair(rec_A)
            direction = pair_o["direction"]
            if direction == "B>A":
                gold = "B"
            elif direction == "A>B":
                gold = "A"
            else:
                continue  # tied pairs skipped
            user_prompt = _render_eval_b_user(rec_A, rec_B)
            tasks.append({
                "task_id": f"{rec_A['record_id']}||{rec_B['record_id']}",
                "split": split,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_EVAL_B},
                    {"role": "user", "content": user_prompt},
                ],
                "prompt": user_prompt,
                "choice_set": "AB",
                "gold_letter": gold,
                "meta": {
                    "dataset_split": split,
                    "scene_id": rec_A["scene"]["scene_id"],
                    "domain_key": rec_A["scene"]["domain_key"],
                    "perturbation": pair_o.get("perturbation"),
                    "perturbed_role": pair_o.get("perturbed_role"),
                    "gap_over_F": pair_o.get("gap_over_F"),
                    "cell_id_A": pair_o.get("cell_id_A"),
                    "cell_id_B": pair_o.get("cell_id_B"),
                    "is_held_out_cell_A": pair_o.get("is_held_out_cell_A"),
                    "is_held_out_cell_B": pair_o.get("is_held_out_cell_B"),
                    "alpha_bucket_A": oracle_latent(rec_A).get("alpha_bucket"),
                    "alpha_bucket_B": oracle_latent(rec_B).get("alpha_bucket"),
                    "record_id_A": rec_A["record_id"],
                    "record_id_B": rec_B["record_id"],
                },
            })
    elif split == "ood_social":
        for rec in records:
            if "ood_social" not in rec:
                continue
            o = oracle_ood_social(rec)
            user_prompt = _render_ood_social_user(rec)
            tasks.append({
                "task_id": rec["record_id"],
                "split": split,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_OOD_SOCIAL},
                    {"role": "user", "content": user_prompt},
                ],
                "prompt": user_prompt,
                "choice_set": "AB",
                "gold_letter": o["gold_letter"],
                "meta": {
                    "dataset_split": split,
                    "scene_id": rec["scene"]["scene_id"],
                    "domain_key": rec["scene"]["domain_key"],
                    "match_distance_bucket": o.get("match_distance_bucket"),
                },
            })
    elif split == "ood_career":
        for rec in records:
            if "ood_career" not in rec:
                continue
            o = oracle_ood_career(rec)
            user_prompt = _render_ood_career_user(rec)
            tasks.append({
                "task_id": rec["record_id"],
                "split": split,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_OOD_CAREER},
                    {"role": "user", "content": user_prompt},
                ],
                "prompt": user_prompt,
                "choice_set": "AB",
                "gold_letter": o["gold_letter"],
                "meta": {
                    "dataset_split": split,
                    "scene_id": rec["scene"]["scene_id"],
                    "domain_key": rec["scene"]["domain_key"],
                    "alpha_2i_bucket": o.get("alpha_2i_bucket"),
                    "gold_firm": o.get("gold_firm"),
                },
            })
    else:
        raise ValueError(f"Unsupported dataset_split `{split}`.")
    return tasks


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def _get_client(api_key: str, api_base: str) -> OpenAI:
    cur = getattr(_THREAD, "client", None)
    sig = getattr(_THREAD, "sig", None)
    if cur is None or sig != (api_key, api_base):
        cur = OpenAI(api_key=api_key, base_url=api_base)
        _THREAD.client = cur
        _THREAD.sig = (api_key, api_base)
    return cur


def _normalize_api_base(api_base: str) -> str:
    clean = api_base.rstrip("/")
    if clean.endswith("/chat/completions"):
        clean = clean[:-len("/chat/completions")]
    if clean.endswith("/v1"):
        return clean
    return f"{clean}/v1"


def _call(client: OpenAI, model: str, messages: List[Dict[str, str]],
          retries: int, timeout: int, temperature: float,
          max_tokens: Optional[int] = None) -> str:
    t: Optional[int] = timeout if timeout > 0 else None
    create_kwargs: Dict[str, Any] = dict(
        model=model, messages=messages, temperature=temperature, timeout=t,
    )
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(**create_kwargs)
            if not resp.choices:
                print(f"[warn attempt {attempt}] empty choices list.", flush=True)
                time.sleep(1.0 + attempt)
                continue
            msg = resp.choices[0].message
            # Standard content field (final answer for thinking models).
            content = (getattr(msg, "content", None) or "").strip()
            # Fallback: DeepSeek-R1 / OpenAI o-series thinking models expose
            # reasoning_content as a non-standard field stored in model_extra
            # by the OpenAI Python SDK (not accessible via plain getattr).
            if not content:
                extra = getattr(msg, "model_extra", None) or {}
                content = (extra.get("reasoning_content") or "").strip()
            if content:
                return content
            finish = getattr(resp.choices[0], "finish_reason", "?")
            extra_keys = list((getattr(msg, "model_extra", None) or {}).keys())
            print(
                f"[warn attempt {attempt}] empty content. "
                f"finish_reason={finish} model_extra_keys={extra_keys}",
                flush=True,
            )
        except Exception as exc:
            sleep = min(2 ** attempt, 20) + random.uniform(0, 1.0)
            print(f"[api err attempt {attempt}] {type(exc).__name__}: {exc}", flush=True)
            time.sleep(sleep)
    return ""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _load_completed_ids(path: Path) -> set:
    done: set = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                tid = r.get("task_id")
                if isinstance(tid, str):
                    done.add(tid)
            except Exception:
                continue
    return done


def _process(
    task: Dict[str, Any],
    model: str, api_key: str, api_base: str,
    retries: int, timeout: int, temperature: float,
    output_file: Path,
    max_tokens: Optional[int] = None,
) -> None:
    client = _get_client(api_key, api_base)
    raw = _call(client, model, task["messages"], retries, timeout, temperature, max_tokens)
    parsed = _parse_letter(raw, task["choice_set"]) if raw else None
    is_correct = (parsed == task["gold_letter"]) if parsed is not None else None
    record_out = {
        "task_id": task["task_id"],
        "dataset_split": task["split"],
        "model": model,
        "gold_letter": task["gold_letter"],
        "parsed_letter": parsed,
        "is_correct": is_correct,
        "raw_response": raw,
        "prompt": task.get("prompt", ""),
        "meta": task["meta"],
    }
    # Resolve parsed rule id for eval_A / placebo.
    letter_to_rule = task["meta"].get("letter_to_rule")
    if isinstance(letter_to_rule, dict) and parsed:
        record_out["parsed_rule_id"] = letter_to_rule.get(parsed)
    with _LOCK:
        with output_file.open("a", encoding="utf-8") as h:
            h.write(json.dumps(record_out, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summarize(output_file: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    with output_file.open("r", encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    def _acc(bucket: List[Dict[str, Any]]) -> Dict[str, float]:
        total = len(bucket)
        correct = sum(1 for r in bucket if r.get("is_correct") is True)
        return {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
        }

    out: Dict[str, Any] = {"overall": _acc(rows), "by_split": {}, "slices": {}}
    # By split (should be single split per file, but just in case).
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r.get("dataset_split", "?"), []).append(r)
    for k, v in groups.items():
        out["by_split"][k] = _acc(v)

    # Slice by key meta fields.
    slice_keys = (
        "alpha_bucket", "cell_id", "is_held_out_cell", "dispersion_bucket",
        "skew_bucket", "perturbation", "match_distance_bucket",
        "alpha_2i_bucket", "domain_key", "scene_id", "gold_rule_id",
    )
    for key in slice_keys:
        groups2: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            v = r.get("meta", {}).get(key)
            if v is None:
                continue
            groups2.setdefault(str(v), []).append(r)
        if groups2:
            out["slices"][key] = {k: _acc(v) for k, v in groups2.items()}

    # Confusion matrix for eval_A / placebo (gold_rule -> parsed_rule).
    confusion: Dict[str, Dict[str, int]] = {}
    for r in rows:
        gold_rule = r.get("meta", {}).get("gold_rule_id")
        parsed_rule = r.get("parsed_rule_id")
        if gold_rule and parsed_rule:
            confusion.setdefault(gold_rule, {}).setdefault(parsed_rule, 0)
            confusion[gold_rule][parsed_rule] += 1
    if confusion:
        out["confusion_gold_to_parsed_rule"] = confusion

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-file", required=True,
                   help="Structured eval split JSON (records list).")
    p.add_argument("--output-file", required=True,
                   help="JSONL file to append model predictions to.")
    p.add_argument("--summary-file", default=None,
                   help="Optional JSON file to write the aggregated metrics report.")
    p.add_argument("--model", default=os.getenv("EVAL_MODEL", ""),
                   help="Model name exposed by the endpoint.")
    p.add_argument("--api-key", default=os.getenv("EVAL_API_KEY", ""),
                   help="OpenAI-compatible API key.")
    p.add_argument("--api-base", default=os.getenv("EVAL_API_BASE", "http://127.0.0.1:8000/v1"),
                   help="OpenAI-compatible endpoint base URL.")
    p.add_argument("--max-workers", type=int,
                   default=int(os.getenv("EVAL_PERSONA_WORKERS", "8")))
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--request-timeout", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Cap total generated tokens per request (limits thinking length "
                        "for reasoning models like DeepSeek-R1). Recommended: 512-1024.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model:
        raise ValueError("Missing --model (or EVAL_MODEL).")
    if not args.api_key:
        raise ValueError("Missing --api-key (or EVAL_API_KEY).")

    payload = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Expected {args.input_file} to contain a `records` list.")

    api_base = _normalize_api_base(args.api_base)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_completed_ids(out_path)

    tasks = _iter_tasks(records)
    pending = [t for t in tasks if t["task_id"] not in done]
    print(json.dumps({
        "input_file": args.input_file,
        "output_file": str(out_path),
        "total_tasks": len(tasks),
        "already_done": len(tasks) - len(pending),
        "pending": len(pending),
        "model": args.model,
        "api_base": api_base,
        "workers": args.max_workers,
    }, ensure_ascii=False), flush=True)

    start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = {
            ex.submit(
                _process, t, args.model, args.api_key, api_base,
                args.max_retries, args.request_timeout, args.temperature, out_path,
                args.max_tokens,
            ): t["task_id"] for t in pending
        }
        done_count = 0
        total = len(futures)
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                print(f"[thread err] {tid}: {exc}", flush=True)
            done_count += 1
            elapsed = max(time.time() - start, 1e-9)
            rate = done_count / elapsed
            eta = int(round((total - done_count) / rate)) if rate > 0 else 0
            print(
                f"progress {done_count}/{total} ({done_count/total:.1%}) "
                f"elapsed={elapsed:.1f}s eta={eta}s",
                flush=True,
            )

    summary = _summarize(out_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.summary_file:
        Path(args.summary_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_file).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
