"""
Call the teacher LLM to fill `scenario_text` for every structured record.

Input:  one structured split file (data/structured/<split>.json) produced by
        data/generation/build_structured_dataset.py.
Output: the same split file, in-place updated, where each record's
        `scenario_text` is replaced by a teacher-generated narrative plus
        metadata (`scene_kind`, `style_tags`, `quality_notes`, `teacher_model`,
        `teacher_api_base`).

All concurrency / retry / checkpointing logic lives here. Prompt design lives
in data/config/scenario_writer_prompts.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config.scenario_writer_prompts import (  # noqa: E402
    EXPECTED_FIELDS,
    build_user_prompt,
    get_system_prompt_for_domain,
    scenario_embeds_required_anchors,
)

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise ImportError("Missing dependency: openai. Install it with `pip install openai`.") from exc


DEFAULT_API_KEY = os.getenv("DEEPSEEK_API_KEY", os.getenv("CHATANYWHERE_API_KEY", ""))
DEFAULT_API_BASE = "https://gpt-api.hkust-gz.edu.cn/v1"
DEFAULT_MODEL = "DeepSeek-R1-671B"
JSON_CODE_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-file", required=True,
                   help="Structured split JSON from build_structured_dataset.py.")
    p.add_argument("--output-file", default=None,
                   help="Where to write the enriched JSON. Defaults to overwriting --input-file.")
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    p.add_argument("--api-base", default=DEFAULT_API_BASE)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--request-timeout", type=int, default=180)
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--save-every", type=int, default=50,
                   help="Flush the enriched file after every N successful records.")
    p.add_argument("--json-indent", type=int, default=2,
                   help="Indent spaces for writing output JSON.")
    p.add_argument("--limit", type=int, default=0, help="Cap the number of records (0 = all).")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate even if `scenario_text` is already non-null.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _extract_json_candidates(content: str) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()

    def add(v: str) -> None:
        cleaned = v.strip().replace("\ufeff", "")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)

    add(content)
    for m in JSON_CODE_BLOCK_PATTERN.finditer(content):
        add(m.group(1))
    depth = 0
    start: Optional[int] = None
    in_str = False
    esc = False
    for i, ch in enumerate(content):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                add(content[start:i + 1])
                start = None
    return out


def _parse_json_payload(content: str) -> Dict[str, Any]:
    last: Optional[Exception] = None
    for cand in _extract_json_candidates(content):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError as exc:
            last = exc
            continue
        if isinstance(obj, dict):
            return obj
        last = ValueError("Parsed JSON payload was not an object.")
    if last is None:
        raise ValueError("Model did not return valid JSON content.")
    raise ValueError(f"Model returned invalid JSON content: {last}") from last


_REQUIRED_NON_EMPTY_FIELDS = {"scenario", "scene_kind"}


def _normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce the teacher's JSON into the expected shape.

    Only `scenario` and `scene_kind` are strictly required (hard-fail).
    `style_tags` and `quality_notes` are cosmetic metadata — we coerce them
    to sane defaults instead of failing the whole record, because R1 sometimes
    omits them or returns the wrong shape. Anchor checks on `scenario` still
    run in `_call_teacher` and remain strict.
    """
    out: Dict[str, Any] = {}
    for field in EXPECTED_FIELDS:
        v = payload.get(field)
        if field == "style_tags":
            if isinstance(v, list):
                out[field] = [x.strip() for x in v if isinstance(x, str) and x.strip()]
            elif isinstance(v, str) and v.strip():
                out[field] = [v.strip()]
            else:
                out[field] = []
            continue
        if field in _REQUIRED_NON_EMPTY_FIELDS:
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"Field `{field}` must be a non-empty string.")
            out[field] = v.strip()
        else:
            out[field] = v.strip() if isinstance(v, str) else ""
    return out


# ---------------------------------------------------------------------------
# Teacher call
# ---------------------------------------------------------------------------

def _call_teacher(
    client: OpenAI,
    model: str,
    record: Dict[str, Any],
    temperature: float,
    max_retries: int,
    request_timeout: int,
) -> Dict[str, Any]:
    domain = record["scene"]["domain_key"]
    system_prompt = get_system_prompt_for_domain(domain)
    user_prompt = build_user_prompt(record)

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                timeout=request_timeout,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _retry_user_prompt(user_prompt, attempt, record)},
                ],
            )
            if not getattr(resp, "choices", None):
                raise ValueError("Empty `choices` from teacher.")
            msg = resp.choices[0].message
            content = getattr(msg, "content", None) or getattr(msg, "reasoning_content", None)
            if not content:
                raise ValueError("Empty content.")
            payload = _normalize_payload(_parse_json_payload(content))
            ok, missing = scenario_embeds_required_anchors(payload["scenario"], record)
            if not ok:
                raise ValueError(f"scenario missing anchors: {missing}")
            return payload
        except Exception as exc:
            last_err = exc
            print(json.dumps({
                "status": "teacher_retry",
                "record_id": record["record_id"],
                "attempt": attempt,
                "error": str(exc),
            }, ensure_ascii=False), flush=True)
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Teacher failed for {record['record_id']}: {last_err}")


def _retry_user_prompt(user_prompt: str, attempt: int, record: Dict[str, Any]) -> str:
    if attempt <= 1:
        return user_prompt
    # Minimal retry suffix that re-states the hard constraints.
    return (
        user_prompt
        + "\n\nRetry notes (previous draft failed checks):\n"
        + "- Return exactly one valid JSON object with keys scene_kind, scenario, style_tags, "
        + "quality_notes. No markdown fences.\n"
        + "- Every numeric anchor from the brief MUST appear verbatim in the narrative.\n"
        + "- Do not invent an extra 'final answer' number.\n"
        + "- No research jargon.\n"
    )


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing API key. Set --api-key or DEEPSEEK_API_KEY / CHATANYWHERE_API_KEY.")

    src = Path(args.input_file)
    dst = Path(args.output_file) if args.output_file else src
    payload = json.loads(src.read_text(encoding="utf-8"))
    records = payload["records"]
    if args.limit > 0:
        records = records[: args.limit]

    pending = [r for r in records if args.overwrite or not r.get("scenario_text")]
    if not pending:
        print(json.dumps({"status": "all_records_already_have_scenario", "total": len(records)},
                         ensure_ascii=False, indent=2), flush=True)
        if args.output_file and dst != src:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(
                json.dumps(payload, ensure_ascii=False, indent=args.json_indent),
                encoding="utf-8",
            )
        return

    def worker(rec: Dict[str, Any]) -> Dict[str, Any]:
        client = OpenAI(api_key=args.api_key, base_url=args.api_base)
        p = _call_teacher(client, args.model, rec, args.temperature,
                          args.max_retries, args.request_timeout)
        return {
            "record_id": rec["record_id"],
            "payload": p,
        }

    id_to_index: Dict[str, int] = {r["record_id"]: i for i, r in enumerate(records)}
    processed = 0
    failed = 0
    failed_ids: List[str] = []
    pending_iter = iter(pending)

    dst.parent.mkdir(parents=True, exist_ok=True)

    def _flush() -> None:
        payload["teacher_last_write"] = datetime.now(timezone.utc).isoformat()
        payload["teacher_failed_record_ids"] = list(failed_ids)
        dst.write_text(
            json.dumps(payload, ensure_ascii=False, indent=args.json_indent),
            encoding="utf-8",
        )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures: Dict[Future[Dict[str, Any]], str] = {}
        for _ in range(min(args.max_workers, len(pending))):
            try:
                rec = next(pending_iter)
            except StopIteration:
                break
            futures[executor.submit(worker, rec)] = rec["record_id"]

        while futures:
            done, _ = wait(tuple(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                rid = futures.pop(fut)
                try:
                    res = fut.result()
                except Exception as exc:
                    # Single-record failure: log, leave scenario_text=null so a
                    # later resume picks it up, and keep the job alive.
                    failed += 1
                    if rid not in failed_ids:
                        failed_ids.append(rid)
                    print(json.dumps({
                        "status": "scenario_failed",
                        "record_id": rid,
                        "failed": failed,
                        "error": str(exc),
                    }, ensure_ascii=False), flush=True)
                else:
                    idx = id_to_index[rid]
                    payload_r = res["payload"]
                    records[idx]["scenario_text"] = payload_r["scenario"]
                    records[idx]["teacher_meta"] = {
                        "scene_kind": payload_r.get("scene_kind", ""),
                        "style_tags": payload_r.get("style_tags", []),
                        "quality_notes": payload_r.get("quality_notes", ""),
                        "model": args.model,
                        "api_base": args.api_base,
                    }
                    processed += 1
                    print(json.dumps({
                        "status": "scenario_generated",
                        "record_id": rid,
                        "processed": processed,
                    }, ensure_ascii=False), flush=True)

                    if processed % args.save_every == 0:
                        payload["records"] = records
                        _flush()
                        print(json.dumps({
                            "status": "checkpoint",
                            "processed": processed,
                            "failed_so_far": failed,
                        }, ensure_ascii=False), flush=True)

                try:
                    nxt = next(pending_iter)
                except StopIteration:
                    continue
                futures[executor.submit(worker, nxt)] = nxt["record_id"]

    payload["records"] = records
    _flush()
    print(json.dumps({
        "status": "done",
        "total_processed": processed,
        "total_failed": failed,
        "failed_record_ids_sample": failed_ids[:20],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
