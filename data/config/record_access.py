"""
Uniform, opinionated accessors for a structured dataset record.

A record is partitioned into two top-level namespaces:

    record
    ├── <public fields>     student prompt builders MAY read these
    │   ├── record_id, dataset_split, schema_version, generator, seed
    │   ├── scene              (scene_id, title, summary, action_label, unit, …)
    │   ├── scenario_text      (narrative from teacher LLM)
    │   ├── mcq.options        (letter, text, x_display, rule_short, …)
    │   ├── pair.role, pair.partner_record_id
    │   ├── ood_social.options (letter, option_id, b_hint, c_hint)
    │   └── ood_career.options (letter, firm_tag, x_S_display, x_bar_display, …)
    │
    └── oracle                  student prompt builders MUST NOT read these
        ├── latent              Langtry parameters (alpha_i, F, c, x_j, g_ij, …)
        ├── gold                rule_id, x_value, x_display
        ├── peer_cards          full cards with g_ij + closeness_hint (teacher only)
        ├── mcq                 gold_letter, identification_margin, all_rule_x_values, …
        ├── pair                direction, gap_over_F, perturbation, cell_ids, …
        ├── ood_social          gold_letter, gold_candidate_index, match_distance_bucket, …
        └── ood_career          gold_letter, gold_firm, alpha_2i_bucket, …

This module is the **one** place that knows the split between public and oracle.
Callers should import `get_*` helpers below rather than dereferencing raw keys,
so the (public, oracle) boundary is grep-able.

Hard rules (enforced by scripts/assert_prompt_safety.py):

    R1. Any function whose output is serialized into a student-facing prompt
        (e.g. `_render_eval_a_user`, `_compose_student_user_prompt`) MUST NOT
        mention the string `"oracle"` or call any `oracle_*` helper from this
        module.
    R2. Teacher-LLM prompt builders MAY read oracle helpers — the teacher is
        an oracle-side actor by design.
    R3. Scoring / metrics / filtering code MAY read oracle helpers.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

ORACLE_KEY = "oracle"


# ---------------------------------------------------------------------------
# Public accessors (safe for student prompt builders)
# ---------------------------------------------------------------------------

def get_record_id(rec: Mapping[str, Any]) -> str:
    return rec["record_id"]


def get_scene(rec: Mapping[str, Any]) -> Dict[str, Any]:
    return rec["scene"]


def get_scenario_text(rec: Mapping[str, Any]) -> Optional[str]:
    return rec.get("scenario_text")


def get_mcq_options(rec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return rec.get("mcq", {}).get("options", [])


def get_pair_role(rec: Mapping[str, Any]) -> Optional[str]:
    return rec.get("pair", {}).get("role")


def get_pair_partner_id(rec: Mapping[str, Any]) -> Optional[str]:
    return rec.get("pair", {}).get("partner_record_id")


def get_ood_social_options(rec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return rec.get("ood_social", {}).get("options", [])


def get_ood_career_options(rec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return rec.get("ood_career", {}).get("options", [])


# ---------------------------------------------------------------------------
# Oracle accessors (teacher / scorer / metrics only)
# ---------------------------------------------------------------------------

def _oracle(rec: Mapping[str, Any]) -> Dict[str, Any]:
    if ORACLE_KEY not in rec:
        raise KeyError(
            "Record has no 'oracle' namespace — it may be a public view. "
            "Oracle data was intentionally stripped; callers that need it must "
            "load the original record file."
        )
    return rec[ORACLE_KEY]


def oracle_latent(rec: Mapping[str, Any]) -> Dict[str, Any]:
    return _oracle(rec)["latent"]


def oracle_gold(rec: Mapping[str, Any]) -> Dict[str, Any]:
    return _oracle(rec)["gold"]


def oracle_peer_cards(rec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return _oracle(rec).get("peer_cards", [])


def oracle_mcq(rec: Mapping[str, Any]) -> Dict[str, Any]:
    return _oracle(rec).get("mcq", {})


def oracle_pair(rec: Mapping[str, Any]) -> Dict[str, Any]:
    return _oracle(rec).get("pair", {})


def oracle_ood_social(rec: Mapping[str, Any]) -> Dict[str, Any]:
    return _oracle(rec).get("ood_social", {})


def oracle_ood_career(rec: Mapping[str, Any]) -> Dict[str, Any]:
    return _oracle(rec).get("ood_career", {})


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def public_view(rec: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of `rec` with the oracle namespace removed.

    Useful as a safety barrier: pass the result to debug / logging code that
    might serialize the record and you cannot accidentally leak oracle data.
    """
    copy_rec = copy.deepcopy(dict(rec))
    copy_rec.pop(ORACLE_KEY, None)
    return copy_rec


def assert_no_oracle(obj: Any, *, where: str = "public view") -> None:
    """Assert that no serialized string under `obj` contains the oracle
    namespace key. Intended for runtime defense in prompt-builders."""
    import json as _json
    blob = _json.dumps(obj, ensure_ascii=False, default=str)
    if f'"{ORACLE_KEY}"' in blob:
        raise AssertionError(
            f"Oracle namespace leaked into {where}. This is a programming bug; "
            f"student-facing code must not serialize `record['{ORACLE_KEY}']`."
        )


# ---------------------------------------------------------------------------
# Small constructors used by build_structured_dataset.py
# ---------------------------------------------------------------------------

def ensure_oracle(rec: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Ensure `rec[ORACLE_KEY]` exists and return a mutable reference to it."""
    if ORACLE_KEY not in rec:
        rec[ORACLE_KEY] = {}
    return rec[ORACLE_KEY]
