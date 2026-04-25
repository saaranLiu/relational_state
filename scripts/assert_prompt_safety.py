"""
Static audit: ensure student-facing prompt builders never touch oracle data.

We define STUDENT_PROMPT_BUILDERS = { (file, function_name) } — these are the
functions whose return value is passed into the student model's `messages`.
For each, we parse the function body and assert:

    A. No string literal "oracle" appears.
    B. No subscript access to keys in FORBIDDEN_ORACLE_KEYS (direct or nested).
    C. No call to any oracle_* helper from data.config.record_access.

Additionally we audit the builders' TRANSITIVE call graph within the same
file, so a helper called by a prompt builder also gets checked.

This is a best-effort AST lint — a determined author could still route
oracle data through a string concatenation chain. But it catches the common
accidents (e.g. `latent["alpha_i"]`, `record["oracle"]["gold"]["x_value"]`,
`oracle_mcq(rec)`).

Exit code 0 on success, 1 on violation.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]

# (relative-path, function-name) pairs that emit student-visible prompts.
STUDENT_PROMPT_BUILDERS: List[Tuple[str, str]] = [
    ("evaluation/eval_runner.py", "_render_eval_a_user"),
    ("evaluation/eval_runner.py", "_render_eval_b_user"),
    ("evaluation/eval_runner.py", "_render_ood_social_user"),
    ("evaluation/eval_runner.py", "_render_ood_career_user"),
    ("evaluation/eval_runner.py", "_scenario_text_or_fallback"),
    ("training/build_sft_data.py", "_compose_student_user_prompt"),
]

FORBIDDEN_ORACLE_KEYS: Set[str] = {
    "oracle",
    "latent",
    "gold",
    "peer_cards",
}

FORBIDDEN_ORACLE_HELPERS: Set[str] = {
    "oracle_latent", "oracle_gold", "oracle_peer_cards", "oracle_mcq",
    "oracle_pair", "oracle_ood_social", "oracle_ood_career",
}


def _collect_funcs(tree: ast.AST) -> Dict[str, ast.FunctionDef]:
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _violations_in_node(node: ast.AST) -> List[str]:
    viol: List[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if sub.value in FORBIDDEN_ORACLE_KEYS:
                viol.append(f"string literal '{sub.value}' @ line {sub.lineno}")
        elif isinstance(sub, ast.Subscript):
            slc = sub.slice
            if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
                if slc.value in FORBIDDEN_ORACLE_KEYS:
                    viol.append(f"subscript ['{slc.value}'] @ line {sub.lineno}")
        elif isinstance(sub, ast.Call):
            fn = sub.func
            fname = None
            if isinstance(fn, ast.Name):
                fname = fn.id
            elif isinstance(fn, ast.Attribute):
                fname = fn.attr
            if fname in FORBIDDEN_ORACLE_HELPERS:
                viol.append(f"call to oracle helper `{fname}()` @ line {sub.lineno}")
    return viol


def _called_funcs(node: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Name):
                names.add(sub.func.id)
    return names


def audit_builder(file_rel: str, func_name: str) -> List[str]:
    path = ROOT / file_rel
    if not path.exists():
        return [f"{file_rel} not found"]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs = _collect_funcs(tree)
    if func_name not in funcs:
        return [f"{file_rel}::{func_name} not found"]

    visited: Set[str] = set()
    frontier = [func_name]
    all_viol: List[str] = []
    while frontier:
        fn = frontier.pop()
        if fn in visited or fn not in funcs:
            continue
        visited.add(fn)
        vv = _violations_in_node(funcs[fn])
        for v in vv:
            all_viol.append(f"{file_rel}::{fn} -> {v}")
        # follow same-file callees
        for callee in _called_funcs(funcs[fn]):
            if callee in funcs and callee not in visited:
                frontier.append(callee)
    return all_viol


def main() -> int:
    any_bad = False
    for file_rel, func in STUDENT_PROMPT_BUILDERS:
        viols = audit_builder(file_rel, func)
        if viols:
            any_bad = True
            print(f"[FAIL] {file_rel}::{func}")
            for v in viols:
                print(f"   - {v}")
        else:
            print(f"[OK]   {file_rel}::{func}")
    if any_bad:
        print("\nPrompt-safety audit FAILED. Student-facing prompt builders must "
              "only read public record fields (scene, scenario_text, *.options).")
        return 1
    print("\nPrompt-safety audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
