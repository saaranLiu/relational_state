#!/usr/bin/env python3
"""
Regenerate `analysis/evalA_reasoning.md` (machine-generated body):
5 universal-wrong eval_A tasks (greedy distinct dominant wrong rule) + all
configured models' full raw reasoning up to Choice (no truncation; synthetic
Choice line when missing).

警告：运行本脚本会整文件覆盖 `evalA_reasoning.md`。若仓库中该文件含人工「简析」
或小节，请先提交 git 或备份后再运行。

Inputs: common_wrong_eval_A_universal.csv, per-model eval_A_predictions.jsonl,
gpt-4 row for letter_to_rule (and optional prompt reference in prose only).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "analysis/model_performance_error_analysis/common_wrong_eval_A_universal.csv"
OUT_MD = ROOT / "analysis/evalA_reasoning.md"
REF_PATH = ROOT / "evaluation/outputs/gpt-4/eval_A_predictions.jsonl"

MODEL_FILES: List[Tuple[str, Path]] = [
    ("DeepSeek-V4-Flash", ROOT / "evaluation/outputs/DeepSeek-V4-Flash/eval_A_predictions.jsonl"),
    ("Qwen", ROOT / "evaluation/outputs/Qwen/eval_A_predictions.jsonl"),
    ("gpt-4", ROOT / "evaluation/outputs/gpt-4/eval_A_predictions.jsonl"),
    ("ollama-llama3-1-latest", ROOT / "evaluation/outputs/ollama-llama3-1-latest/eval_A_predictions.jsonl"),
    ("ollama-llama3-8b", ROOT / "evaluation/outputs/ollama-llama3-8b/eval_A_predictions.jsonl"),
    ("ollama-qwen2-7b", ROOT / "evaluation/outputs/ollama-qwen2-7b/eval_A_predictions.jsonl"),
    ("ollama-qwen2.5-14b", ROOT / "evaluation/outputs/ollama-qwen2.5-14b/eval_A_predictions.jsonl"),
]

N_MODELS = len(MODEL_FILES)


def extract_reasoning(raw: str) -> str:
    if not raw:
        return ""
    text = raw.strip()
    m = re.search(
        r"Reasoning:\s*(.*?)(?=\n\s*Choice:|\nChoice:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    last_start: Optional[int] = None
    for m2 in re.finditer(r"(?i)\bChoice\s*:\s*[A-E]\b", text):
        last_start = m2.start()
    if last_start is not None:
        return text[:last_start].strip()
    return text


def extract_choice_line(raw: str) -> str:
    if not raw:
        return ""
    matches = list(re.finditer(r"(?i)\bChoice\s*:\s*([A-E])\b", raw))
    if not matches:
        return ""
    return f"Choice: {matches[-1].group(1).upper()}"


def build_full_reasoning_choice_block(raw: str, parsed_letter: Optional[str]) -> str:
    reason = extract_reasoning(raw).strip() or "(empty)"
    ch = extract_choice_line(raw)
    if ch:
        return f"{reason}\n\n{ch}"
    if parsed_letter:
        pl = str(parsed_letter).strip().upper()
        return (
            f"{reason}\n\n"
            f"# --- 评测补写：raw 无 Choice 行，按 parsed_letter 补 ---\n"
            f"Choice: {pl}"
        )
    return f"{reason}\n\n# --- Choice 缺失且无法补全 ---\n"


def fence_body(body: str) -> str:
    if "```" in body:
        return "~~~text\n" + body + "\n~~~\n"
    return "```text\n" + body + "\n```\n"


def dominant_rule(hist_json: str) -> str:
    h = json.loads(hist_json)
    return max(h.items(), key=lambda kv: kv[1])[0]


def pick_five_tasks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = df.sort_values("task_id")
    picked: List[Dict[str, Any]] = []
    seen_dom: set[str] = set()
    for _, r in rows.iterrows():
        dom = dominant_rule(r["parsed_rule_histogram"])
        if dom not in seen_dom and len(picked) < 5:
            picked.append(dict(r))
            seen_dom.add(dom)
    if len(picked) < 5:
        for _, r in rows.iterrows():
            if len(picked) >= 5:
                break
            tid = r["task_id"]
            if any(p["task_id"] == tid for p in picked):
                continue
            picked.append(dict(r))
    return picked


def load_record(path: Path, task_id: str) -> Optional[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("task_id") == task_id:
                return rec
    return None


def letter_rule_table(l2r: Any) -> str:
    if not isinstance(l2r, dict) or not l2r:
        return "_（无 `letter_to_rule`）_\n\n"
    rows = ["| 字母 | `parsed_rule_id` |", "|---|---|"]
    for letter in sorted(l2r.keys()):
        rows.append(f"| `{letter}` | `{l2r[letter]}` |")
    return "\n".join(rows) + "\n\n"


def _reasoning_for_analysis(raw: str) -> str:
    """Same window as shown in the doc (before final Choice:)."""
    return extract_reasoning(raw or "")


def auto_reasoning_comment(
    raw: str, parsed_rule_id: Optional[str], had_synthetic_choice: bool
) -> str:
    """
    Short fluent Chinese note (heuristic, not LLM): narrative cues vs chosen rule label.
    """
    body = _reasoning_for_analysis(raw)
    pr_id = (parsed_rule_id or "").strip()
    low = body.lower()
    low_cn = body  # keep some Chinese if models mix
    bits: List[str] = []
    if had_synthetic_choice:
        bits.append(
            "原始 `raw_response` 不含 `Choice:` 行，文末选项字母为按评测结果补写，请勿与模型自行输出的结尾混为一谈。"
        )

    nch = len(body.strip())
    if nch < 50:
        bits.append("论述较短，对「多名同伴与亲疏权重」展开不足。")
    elif nch > 2200:
        bits.append("篇幅较长，但收束到选项时论证仍显跳跃。")

    # bilingual light cues
    closest = bool(
        re.search(
            r"closest|confidant|roommate|intimate|tight\s+friend|最亲|闺蜜|室友|死党",
            low + low_cn,
            re.I,
        )
    )
    avg = bool(re.search(r"\baverage\b|\bmean\b|\bmedian\b|平均|中位|大伙|大家", low + low_cn, re.I))
    priv = bool(re.search(r"private|baseline|alone|quiet|stick\s+to|私域|基线|独处|低调", low + low_cn, re.I))
    weight = bool(re.search(r"weight|weighted|closeness|aggregate|亲疏|加权|权重", low + low_cn, re.I))
    mimic = bool(re.search(r"mirror|match|same\s+as|copy|仿|对齐|跟着", low + low_cn, re.I))

    if closest and mimic:
        bits.append("侧重与最亲密者对齐或效仿，社会比较被压缩为单一参照点。")
    elif closest and not mimic:
        bits.append("多次提及最亲密者，但收束论证未必与「仅模仿一人」的规则一致。")

    if avg and not weight:
        bits.append("中间易滑向「平均/众人」式表述，与「按 g_ij 对多名同伴加权」并不相同。")

    if priv and pr_id != "D_pure_private":
        bits.append("私域、基线等用语较多，但最终选项并非纯私域，叙述与规则标签存在一定张力。")
    elif priv and pr_id == "D_pure_private":
        bits.append("强调按个人内部标准行事，与选择纯私域规则基本一致。")

    if weight and pr_id and pr_id != "A_peer_weighted":
        bits.append("出现加权、亲疏等表述，但最终选项未回到 gold 对应规则，叙述与标签不一致。")

    rule_hint = {
        "E_closest_mimicry": "该规则强调仅效仿最亲近者的取值；可对照正文是否将论证始终压在这一单一参照上。",
        "B_top_anchor": "该规则将社会项交给最亲或最强锚点；可对照正文是否将其他同伴边缘化。",
        "D_pure_private": "该规则相当于忽略同伴、仅保留私域；可对照正文是否将社交信息视为可忽略。",
        "C_uniform_avg": "该规则近似同伴均匀平均；可对照正文是否抹平了亲疏梯度。",
        "F_median_anchor": "该规则偏向中位数锚定；可对照正文是否以中位替代个体加权。",
        "H_equal_mix": "该规则常为私域与社会项的折中混合；可对照正文是否采取简单折中。",
        "G_counter_conformist": "该规则可能走向反从众；可对照正文是否刻意反向立身。",
    }
    if pr_id in rule_hint and len(bits) < 4:
        bits.append(rule_hint[pr_id])

    if not bits:
        bits.append("关键词线索不明显，结论更像依赖场景语感选定选项。")

    # 2–3 句，通顺为主
    out = "".join(bits[:3])
    if len(out) > 320:
        out = out[:317] + "…"
    return out


def main() -> None:
    df = pd.read_csv(CSV)
    picked = pick_five_tasks(df)
    n_uni = len(df)

    cn_n = {5: "五", 6: "六", 7: "七", 8: "八"}.get(N_MODELS, str(N_MODELS))
    lines: list[str] = []
    lines.append(f"# eval_A {cn_n}路模型全错 · Top 5 题\n\n")
    lines.append(
        f"**数据**：`common_wrong_eval_A_universal.csv`（当前 **{n_uni}** 道在 `eval_A` 上 **{N_MODELS}** 路模型预测均错误的题目）。\n\n"
        f"**选题**：按 `task_id` **升序**遍历，**贪心**选取 **5** 题，使各题错误答案众数对应的 **`parsed_rule_id` 尽可能互不重复**。\n\n"
        "**Gold**：每题标准答案均为 **`A_peer_weighted`**（按同伴亲疏权重聚合多名同伴，而非仅模仿最亲一人或纯私域基线）。\n\n"
        "**正文**：摘自各模型 **`raw_response`** 的推理片段：若存在 `Reasoning:` 块，则取其内容至其后**首个** `Choice:` 之前；否则为全文至**最后一个** `Choice:` 之前。\n\n"
        "若某条响应未包含 `Choice:`，以下按评测解析的 `parsed_letter` **补写**一行并加注说明；**`parsed_rule_id`** 由 `parsed_letter` 经本题 `letter_to_rule` 映射得到。\n\n"
    )

    lines.append("## 题目\n\n")
    lines.append(f"| # | `task_id` | `cell_id` | gold | 众数错选 | 错选（{N_MODELS} 次） |\n")
    lines.append("|---:|---|---|---|---|---|\n")
    for i, meta in enumerate(picked, 1):
        tid = meta["task_id"]
        short = tid if len(tid) <= 56 else tid[:53] + "…"
        dom = dominant_rule(meta["parsed_rule_histogram"])
        hist = json.loads(meta["parsed_rule_histogram"])
        hist_s = ", ".join(f"`{k}`×{v}" for k, v in sorted(hist.items(), key=lambda kv: (-kv[1], kv[0])))
        gl, gr = meta.get("gold_letter", ""), meta.get("gold_rule_id", "")
        lines.append(
            f"| {i} | `{short}` | `{meta.get('cell_id', '')}` | `{gl}`→`{gr}` | `{dom}` | {hist_s} |\n"
        )
    lines.append("\n---\n\n")

    for i, meta in enumerate(picked, 1):
        tid = meta["task_id"]
        hist_s = ", ".join(
            f"`{k}`×{v}"
            for k, v in sorted(
                json.loads(meta["parsed_rule_histogram"]).items(),
                key=lambda kv: (-kv[1], kv[0]),
            )
        )
        lines.append(f"## {i}. `{tid}`\n\n")
        lines.append(
            f"`scene_id`：**{meta.get('scene_id', '')}** · `cell_id`：**{meta.get('cell_id', '')}** · "
            f"错选直方图：{hist_s} · 众数：**`{dominant_rule(meta['parsed_rule_histogram'])}`**\n\n"
        )

        ref = load_record(REF_PATH, tid)
        if ref:
            mref = ref.get("meta") or {}
            l2r = mref.get("letter_to_rule")
            gl = ref.get("gold_letter")
            gold_rule = (
                l2r.get(str(gl), meta.get("gold_rule_id"))
                if isinstance(l2r, dict)
                else meta.get("gold_rule_id")
            )
            lines.append(f"**Gold**：`{gl}` → `{gold_rule}`。\n\n")
            lines.append(letter_rule_table(l2r))
            prompt_text = (ref.get("prompt") or mref.get("prompt") or "").strip()
            if prompt_text:
                lines.append(
                    "**题干**（摘自 `evaluation/outputs/gpt-4/eval_A_predictions.jsonl` 中该 `task_id` 的 `prompt` 字段，与评测时模型所见一致）：\n\n"
                )
                lines.append(fence_body(prompt_text) + "\n")
            else:
                lines.append(
                    f"**题干**：`prompt` 字段为空；请核对 `evaluation/outputs/gpt-4/eval_A_predictions.jsonl` 中 `task_id = {tid}`。\n\n"
                )
        else:
            lines.append("_（无 gpt-4 参考行：缺 `letter_to_rule` / 题干路径）_\n\n")

        lines.append(f"### 各模型输出（`raw_response`，未截断，共 {N_MODELS} 路）\n\n")
        lines.append(
            "_以下「简析」由脚本根据关键词与规则标签启发式生成，仅供快速浏览；请以原文及 `letter_to_rule` 映射为准。_\n\n"
        )

        for name, path in MODEL_FILES:
            lines.append(f"#### {name}\n\n")
            rec = load_record(path, tid)
            if not rec:
                lines.append("_（无该行）_\n\n")
                continue
            raw = rec.get("raw_response") or ""
            pr, pl = rec.get("parsed_rule_id"), rec.get("parsed_letter")
            pl_norm = str(pl).strip().upper() if pl else None
            lines.append(f"`parsed_letter` → `parsed_rule_id`：**{pl}** → **{pr}**\n\n")
            lines.append(fence_body(build_full_reasoning_choice_block(raw, pl_norm)) + "\n")
            had_syn = (extract_choice_line(raw) == "") and (pl_norm is not None)
            blurb = auto_reasoning_comment(raw, pr if isinstance(pr, str) else None, had_syn)
            lines.append(f"**简析**：{blurb}\n\n")

        lines.append("\n---\n\n")

    body = "".join(lines)
    appendix = (
        "## 附录\n\n"
        "生成：`python3 scripts/export_top5_universal_evalA_reasoning.py` → "
        f"`{OUT_MD.relative_to(ROOT)}`（会覆盖全文；各段「简析」为脚本启发式，"
        "若需人工分析请在生成后自行编辑或从版本库恢复）。\n"
    )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    full = body + appendix
    OUT_MD.write_text(full, encoding="utf-8")
    print(f"Wrote {OUT_MD} ({len(full)} chars)")


if __name__ == "__main__":
    main()
