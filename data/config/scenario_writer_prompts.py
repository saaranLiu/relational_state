"""
Teacher-LLM scenario writer prompts.

The teacher is given ONE structured record (produced by
build_structured_dataset.py) and must return JSON with:
    scene_kind      short lowercase tag (e.g. "gift_shortlist")
    scenario        natural-language passage (~140-220 words)
    style_tags      list of short descriptive tags
    quality_notes   one short sentence on craft / voice

Strict design constraints
-------------------------
* Every peer card (id, action_display, closeness hint) MUST be rendered as an
  intelligible peer/partner inside the story. Tight cards get warm language;
  peripheral cards get distant language. We do NOT tell the teacher the
  closeness weights numerically.
* Each peer's action value MUST appear verbatim somewhere in the narrative
  (label, story beat, dialogue, spreadsheet row, etc.). This is how the
  student model can recover x_j at eval time.
* The private baseline F MUST also appear as a concrete sentence/beat in the
  narrative (e.g. "the version that would feel fine without anyone around
  costs $420/month").
* The teacher NEVER reveals the gold x_i_star, the closeness weights g_ij,
  alpha_i, or any Langtry terminology. The student must infer the rule.
* Scene text is first-person-adjacent (using the protagonist's pronoun /
  name) but keeps a light narrator voice. No research jargon.

Domain-specific templates
-------------------------
* d1 + d3: peer-comparison scene with N peer cards (3-5).
* d2 (placebo): same format but WITHOUT peer cards (omit list) and with
  an explicit "no audience" cue; x_i_star == F.
* d4 OOD (social): protagonist and two candidate partners each have a three-level
  benefit tier and a three-level marginal-cost tier. The marginal cost c is
  narrated as financial constraint / budget pressure around taking the action.
  No numeric b, c, or b/c ratio is rendered; the student must infer which
  semantic ratio matches best. No closeness weights.
* d5 OOD (career): two firm options (H / L), each with crowd mean and the
  protagonist's own wage. No peer list; the comparison is between crowd
  averages and own rank.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


# -----------------------------------------------------------------------------
# System prompts (domain-aware)
# -----------------------------------------------------------------------------

_SYSTEM_BASE = (
    "You are a careful narrative-writer. You turn a small structured brief into one natural passage "
    "(the `scenario`) that a downstream model will read. You return strictly valid JSON with the keys "
    "listed at the end; no code fences, no extra commentary.\n"
    "\n"
    "Craft rules:\n"
    "- Natural prose (140-220 words). No bullet lists recycled into the narrative.\n"
    "- Surface EVERY named peer / candidate / firm. Use short, believable names (e.g. Mia, Jake).\n"
    "- Every numeric value the user gives you must be embedded verbatim (formatted exactly as shown) "
    "inside the narrative: treat them as labels, quotes, napkin math, sticker prices, chat-screenshot lines.\n"
    "- Do not invent an additional 'final answer' number beyond the values the user provided.\n"
    "- No research jargon. Do NOT print: alpha, beta, gamma, bucket, latent, weight, coefficient, "
    "scalar, reference aggregate, baseline parameter, best response, sample type, social-comparison "
    "intensity.\n"
    "- You MAY use plain words like friend, crew, group chat, feed, cohort, office, etc.\n"
    "- End on tension / choice framing, not on a resolution.\n"
)


_SYSTEM_POSITIONAL = _SYSTEM_BASE + (
    "\n"
    "Domain-specific (positional consumption / human capital):\n"
    "- Render the peer group around the protagonist as a real social circle. Map each peer's "
    "closeness_hint to the narrative voice: `tight` = close friend / roommate / mentor; `mid` = regular "
    "teammate / study-buddy / neighbour; `peripheral` = seen-on-feed acquaintance / Slack lurker.\n"
    "- Show, don't numerically state, the difference in closeness (e.g. 'my sister,' vs. 'someone from "
    "the welcome Slack').\n"
    "- Make the public-ness of the commitment legible (group chat, shared ledger, feed, wrist watch, "
    "driveway, leaderboard). But NEVER say how much the protagonist weighs peer influence — that is "
    "exactly what the student must infer.\n"
    "- Embed the protagonist's private baseline once (e.g. 'what I'd spend if nobody was watching') and "
    "make clear it sits below the peer values.\n"
)


_SYSTEM_PLACEBO = _SYSTEM_BASE + (
    "\n"
    "Domain-specific (non-positional investment — peers may be visible, but the decision is "
    "private-utility-only):\n"
    "- The commitment SHOULD be made in a realistic everyday context, and that context naturally includes "
    "other people making their own versions of the same class of decision (a sibling who just paid their "
    "dentist, a co-worker whose roof was resealed, a friend who bought a new water heater). Render those "
    "peers as real, specific people — same warm/distant voice rules as the positional template.\n"
    "- CRITICAL FRAME: the protagonist's outcome depends ONLY on their own private situation (their own "
    "tooth, their own child, their own house, their own body). Others' choices are visible but they do "
    "not set a comparison standard for this decision — there is no social ranking, no feed leaderboard, "
    "no group-chat commitment bar, no status from over-spending. Make that asymmetry feel natural in the "
    "narrative (e.g. 'the roofer priced my house, not my neighbour's'; 'the dentist's plan is for my "
    "tooth, not a competition').\n"
    "- Each rendered peer's action value MUST appear verbatim in the narrative so the student can see "
    "the peer amounts and must consciously choose to ignore them.\n"
    "- Embed the protagonist's private baseline as the quote from the professional / the figure on the "
    "bill / the number on the form. It must be legible as 'what MY case actually costs'.\n"
    "- End on the pull the protagonist feels FROM the visible peers vs the self-knowledge that "
    "this is a private-value call — but do not resolve it in the narrative. The student must resolve it "
    "by domain reasoning.\n"
)


_SYSTEM_OOD_SOCIAL = _SYSTEM_BASE + (
    "\n"
    "Domain-specific (social-network formation, friendship choice):\n"
    "- The protagonist is weighing two candidate people (call them by natural names; you'll see "
    "option_id=P1 and P2 as their internal labels).\n"
    "- For the protagonist and each candidate, render TWO semantic tiers: how much they value the "
    "relationship/shared habit, and how financially constrained the action feels for them.\n"
    "- Do NOT provide numeric b, numeric c, the benefit-cost ratio, preferred action level, or match "
    "distance. The student must infer which candidate has the most similar benefit-cost ratio from the "
    "semantic tiers.\n"
    "- Make the tiers intuitive for the scene: benefit is how much they value the relationship or "
    "shared habit; marginal cost is the action cost term in the same sense as the main decision utility.\n"
    "- SEMANTICIZE c as financial constraint / economic pressure around spending on the shared action. "
    "Use the provided c_hint to write it in layers: comfortable budget, budget-conscious tradeoffs, or "
    "meaningful financial constraint. Avoid literal technical or clinical labels; write it as ordinary "
    "life circumstances, not as a named parameter.\n"
    "- Do not literally print bucket labels such as low, mid, or high; paraphrase the hints as natural "
    "character details.\n"
    "- DO NOT say which candidate is the 'match' — the student must decide.\n"
    "- End on the protagonist's hesitation over who to build the regular habit with.\n"
)


_SYSTEM_OOD_CAREER = _SYSTEM_BASE + (
    "\n"
    "Domain-specific (career sorting, firm choice):\n"
    "- The protagonist has two concrete offers (a high-pay firm H and a lower-pay firm L). Give each "
    "firm a believable name and context appropriate to the scene family.\n"
    "- For each firm, render BOTH numbers: the protagonist's own wage (x_S) AND the firm's average "
    "coworker wage (x_bar). Use natural phrasing (e.g. 'the offer comes in at $X while the team average "
    "is around $Y').\n"
    "- Include the rank intuition (e.g. 'near the bottom of the pack' for H, 'comfortably above average' "
    "for L).\n"
    "- Do NOT say which firm is 'the right answer'. End on the dilemma between earning more and ranking "
    "higher.\n"
)


SCENARIO_WRITER_SYSTEM_PROMPT: Dict[str, str] = {
    "domain1_positional_consumption": _SYSTEM_POSITIONAL,
    "domain3_effort_and_human_capital": _SYSTEM_POSITIONAL,
    "domain2_non_positional_investment": _SYSTEM_PLACEBO,
    "domain4_social_network_formation": _SYSTEM_OOD_SOCIAL,
    "domain5_career_sorting": _SYSTEM_OOD_CAREER,
}


EXPECTED_FIELDS: Tuple[str, ...] = ("scene_kind", "scenario", "style_tags", "quality_notes")


# -----------------------------------------------------------------------------
# User prompt builders (per split)
# -----------------------------------------------------------------------------

def _render_peer_lines(peer_cards: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for c in peer_cards:
        lines.append(
            f"- {c['id']}: closeness_hint={c['closeness_hint']}, "
            f"action_value_display={c['action_display']}"
        )
    return "\n".join(lines) if lines else "(no peer list — placebo scene)"


def _baseline_display(latent: Dict[str, Any], scene: Dict[str, Any]) -> str:
    from data.config.rule_label_templates import format_value
    return format_value(latent.get("F", 0.0), scene.get("value_format", "currency"), scene.get("action_unit", ""))


def build_user_prompt_positional(record: Dict[str, Any]) -> str:
    scene = record["scene"]
    latent = record["oracle"]["latent"]
    peer_lines = _render_peer_lines(record["oracle"].get("peer_cards", []))
    f_display = _baseline_display(latent, scene)

    peer_singular = scene.get("peer_noun_singular") or "peer"
    peer_plural = scene.get("peer_noun_plural") or "peers"

    return (
        f"Scene brief:\n"
        f"- title: {scene['title']}\n"
        f"- summary: {scene['summary']}\n"
        f"- action_label: {scene['action_label']}\n"
        f"- action_unit: {scene['action_unit']}\n"
        f"- peer noun: {peer_singular} (plural: {peer_plural})\n"
        f"\n"
        f"Protagonist's private baseline (must appear in the narrative exactly as formatted): "
        f"{f_display}\n"
        f"\n"
        f"Peer cards (must ALL appear; map closeness_hint to voice, render action_value_display verbatim):\n"
        f"{peer_lines}\n"
        f"\n"
        f"Return JSON with keys: scene_kind, scenario, style_tags, quality_notes.\n"
        f"No code fences. No jargon. End on tension, not resolution."
    )


def build_user_prompt_placebo(record: Dict[str, Any]) -> str:
    scene = record["scene"]
    latent = record["oracle"]["latent"]
    peer_lines = _render_peer_lines(record["oracle"].get("peer_cards", []))
    f_display = _baseline_display(latent, scene)

    peer_singular = scene.get("peer_noun_singular") or "person"
    peer_plural = scene.get("peer_noun_plural") or "people"

    return (
        f"Scene brief (non-positional investment — peers are visible but the decision is private):\n"
        f"- title: {scene['title']}\n"
        f"- summary: {scene['summary']}\n"
        f"- action_label: {scene['action_label']}\n"
        f"- action_unit: {scene['action_unit']}\n"
        f"- peer noun: {peer_singular} (plural: {peer_plural})\n"
        f"\n"
        f"Protagonist's private baseline (must appear verbatim — it is what THIS person's own case "
        f"actually warrants, quoted by a professional or written on the bill/form): {f_display}\n"
        f"\n"
        f"Peer cards — render each peer as a real specific person handling THEIR own version of the "
        f"same class of decision. Map closeness_hint to voice. Action values MUST appear verbatim. "
        f"Do NOT frame them as a standard to match or beat — they are simply what those other people "
        f"paid for their own situations:\n"
        f"{peer_lines}\n"
        f"\n"
        f"Return JSON with keys: scene_kind, scenario, style_tags, quality_notes.\n"
        f"End on the tension between the visible peer amounts and the fact that this protagonist's "
        f"outcome depends only on their own private situation — do not resolve which way they go."
    )


def build_user_prompt_ood_social(record: Dict[str, Any]) -> str:
    scene = record["scene"]
    latent = record["oracle"]["latent"]
    ood = record["ood_social"]
    candidate_lines = []
    for opt in ood["options"]:
        candidate_lines.append(
            f"- letter={opt['letter']} option_id={opt['option_id']} "
            f"benefit_bucket={opt['b_bucket']} benefit_hint=\"{opt['b_hint']}\" "
            f"cost_bucket={opt['c_bucket']} "
            f"cost_hint=\"{opt.get('c_hint', '')}\""
        )
    candidates_block = "\n".join(candidate_lines)
    friendship_ctx = scene.get("extras", {}).get("friendship_context", "")
    return (
        f"Scene brief (friendship formation OOD):\n"
        f"- title: {scene['title']}\n"
        f"- summary: {scene['summary']}\n"
        f"- shared activity / action_label: {scene['action_label']} ({scene['action_unit']})\n"
        f"- friendship_context: {friendship_ctx}\n"
        f"\n"
        f"Protagonist's benefit and marginal-cost tiers (write these as natural prose; "
        f"do NOT introduce numeric b/c values or compute their ratio):\n"
        f"- benefit_bucket={latent['b_i_bucket']} benefit_hint=\"{latent['b_i_hint']}\" "
        f"cost_bucket={latent['c_i_bucket']} "
        f"cost_hint=\"{latent.get('c_i_hint', '')}\"\n"
        f"\n"
        f"Candidate partners (both must appear; use benefit_hint and cost_hint naturally; "
        f"do NOT introduce numeric b/c values, compute ratios, or reveal which "
        f"one is the gold match):\n"
        f"{candidates_block}\n"
        f"\n"
        f"Return JSON with keys: scene_kind, scenario, style_tags, quality_notes."
    )


def build_user_prompt_ood_career(record: Dict[str, Any]) -> str:
    scene = record["scene"]
    latent = record["oracle"]["latent"]  # noqa: F841  (unused but documents schema)
    ood = record["ood_career"]
    lines = []
    for opt in ood["options"]:
        lines.append(
            f"- letter={opt['letter']} firm_tag={opt['firm_tag']} own_wage={opt['x_S_display']} "
            f"crowd_average={opt['x_bar_display']} rank_cue=\"{opt['relative_rank']}\""
        )
    options_block = "\n".join(lines)
    return (
        f"Scene brief (career sorting OOD):\n"
        f"- title: {scene['title']}\n"
        f"- summary: {scene['summary']}\n"
        f"- scene family: {scene['family']}\n"
        f"\n"
        f"Two firm offers (both must appear; render BOTH own_wage and crowd_average verbatim):\n"
        f"{options_block}\n"
        f"\n"
        f"Do NOT reveal which firm is the gold choice. End on the tension between earning more at H "
        f"vs ranking higher at L.\n"
        f"\n"
        f"Return JSON with keys: scene_kind, scenario, style_tags, quality_notes."
    )


def build_user_prompt(record: Dict[str, Any]) -> str:
    domain = record["scene"]["domain_key"]
    if domain == "domain2_non_positional_investment":
        return build_user_prompt_placebo(record)
    if domain == "domain4_social_network_formation":
        return build_user_prompt_ood_social(record)
    if domain == "domain5_career_sorting":
        return build_user_prompt_ood_career(record)
    return build_user_prompt_positional(record)


def get_system_prompt_for_domain(domain_key: str) -> str:
    return SCENARIO_WRITER_SYSTEM_PROMPT.get(domain_key, _SYSTEM_POSITIONAL)


# -----------------------------------------------------------------------------
# Value-anchor verification
# -----------------------------------------------------------------------------

_VALUE_TOKEN_PATTERN = re.compile(r"(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?)")


def scenario_embeds_value_display(scenario: str, value_display: str) -> bool:
    """Return True if the formatted value_display appears (allowing comma/$/unit variance)."""
    scenario = scenario.strip()
    vd = value_display.strip()
    if not vd:
        return True
    if vd in scenario:
        return True
    match = _VALUE_TOKEN_PATTERN.search(vd)
    if not match:
        return vd in scenario
    token = match.group(1)
    if token in scenario:
        return True
    bare = token.lstrip("$")
    if bare in scenario:
        return True
    compact = re.sub(r"[^\d.]", "", token)
    if not compact:
        return False
    sn = scenario.replace(",", "")
    return bool(re.search(rf"(?<![\d.])({re.escape(compact)})(?![\d.])", sn))


def scenario_embeds_required_anchors(scenario: str, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check that the narrative surfaces every required numeric anchor."""
    missing: List[str] = []
    scene = record["scene"]
    oracle = record.get("oracle", {})
    latent = oracle.get("latent", {})
    from data.config.rule_label_templates import format_value
    vf = scene.get("value_format", "currency")
    unit = scene.get("action_unit", "")

    domain = scene["domain_key"]
    if domain in ("domain1_positional_consumption", "domain3_effort_and_human_capital"):
        F_disp = format_value(latent.get("F", 0.0), vf, unit)
        if not scenario_embeds_value_display(scenario, F_disp):
            missing.append(f"F={F_disp}")
        for card in oracle.get("peer_cards", []):
            if not scenario_embeds_value_display(scenario, card["action_display"]):
                missing.append(f"peer_{card['id']}={card['action_display']}")
    elif domain == "domain2_non_positional_investment":
        F_disp = format_value(latent.get("F", 0.0), vf, unit)
        if not scenario_embeds_value_display(scenario, F_disp):
            missing.append(f"F={F_disp}")
        # v3: d2 narratives now include peer cards; enforce them too.
        for card in oracle.get("peer_cards", []):
            if not scenario_embeds_value_display(scenario, card["action_display"]):
                missing.append(f"peer_{card['id']}={card['action_display']}")
    elif domain == "domain4_social_network_formation":
        # d4 intentionally does not expose numeric b/c ingredients to the
        # teacher or student. Semantic tier coverage is checked qualitatively
        # by prompt instructions rather than numeric-anchor matching.
        pass
    elif domain == "domain5_career_sorting":
        for opt in record.get("ood_career", {}).get("options", []):
            if not scenario_embeds_value_display(scenario, opt["x_S_display"]):
                missing.append(f"career_{opt['firm_tag']}_xS={opt['x_S_display']}")
            if not scenario_embeds_value_display(scenario, opt["x_bar_display"]):
                missing.append(f"career_{opt['firm_tag']}_xbar={opt['x_bar_display']}")
    return (len(missing) == 0), missing


# -----------------------------------------------------------------------------
# Back-compat alias for existing callers / audit tooling
# -----------------------------------------------------------------------------

def scenario_embeds_prompt_anchors(scenario: str, record_or_fields: Dict[str, Any]) -> bool:
    """Legacy alias. Accepts either a full record or an older prompt_fields dict."""
    if "scene" in record_or_fields and "oracle" in record_or_fields:
        ok, _ = scenario_embeds_required_anchors(scenario, record_or_fields)
        return ok
    baseline = str(record_or_fields.get("baseline_value_display", "") or "")
    reference = str(record_or_fields.get("reference_value_display", "") or "")
    return scenario_embeds_value_display(scenario, baseline) and (
        not reference or scenario_embeds_value_display(scenario, reference)
    )
