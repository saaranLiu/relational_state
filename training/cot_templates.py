"""
Chain-of-Thought (CoT) templates used as SFT targets.

Each generated CoT has three mandatory sections:

  (1) Taxonomy preamble
      Classifies the scene into one of three Langtry scene types and runs an
      explicit counterfactual check. Four skeleton variants ensure that the
      preamble cannot be memorised as a single fixed prefix.

  (2) Reasoning body
      Walks through the Langtry best-response in qualitative terms:
        * Which peer matters most and why (via closeness, not action magnitude).
        * How the peer field combines into a weighted reference aggregate.
        * How comparison sensitivity modulates the lift above the private
          floor.
      NO explicit arithmetic is emitted -- the body never mentions
      "= 345" or "0.5 * 290"; numeric anchors appear only as final displays.

  (3) Conclusion
      States the qualitative gold rule and commits to the target action
      level. For placebo scenes it states "hold the private floor" explicitly.

Leakage-prevention properties (R1-R5 from the design doc):
  R1: All four skeleton variants rotate randomly, and counterfactual sentence
      order is also randomised across the two placebo / non-positional
      alternatives.
  R2: No phrase from TEST_MCQ_RULE_VOCAB is present in the preamble or body.
      This is checked at audit time via
      `vocabulary_pools.find_mcq_rule_leaks_in_cot`.
  R3: No FORBIDDEN_RESEARCH_TERMS (alpha, gamma, x_star, ...).
  R4: Numeric slots are only used for final anchor displays in the conclusion,
      NOT inside reasoning steps.
  R5: A "tricky-positional" subset explicitly uses the counterfactual arm to
      rule out the non-positional interpretation even when the scene
      superficially reads as private (e.g. solo weekend that turns into a
      photo-posting dinner).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Scene-type taxonomy tags (used as the self-tag the model must produce)
# -----------------------------------------------------------------------------

SCENE_TYPE_POSITIONAL_CONSUMPTION = "positional-consumption"
SCENE_TYPE_POSITIONAL_HUMAN_CAPITAL = "positional-human-capital"
SCENE_TYPE_NON_POSITIONAL = "non-positional"
SCENE_TYPE_TRICKY_POSITIONAL = "tricky-positional"   # training subset only; tag still resolves to one of the three above at inference.

ALL_SCENE_TYPES = (
    SCENE_TYPE_POSITIONAL_CONSUMPTION,
    SCENE_TYPE_POSITIONAL_HUMAN_CAPITAL,
    SCENE_TYPE_NON_POSITIONAL,
    SCENE_TYPE_TRICKY_POSITIONAL,
)


def domain_to_scene_type(domain_key: str) -> str:
    if domain_key == "domain1_positional_consumption":
        return SCENE_TYPE_POSITIONAL_CONSUMPTION
    if domain_key == "domain3_effort_and_human_capital":
        return SCENE_TYPE_POSITIONAL_HUMAN_CAPITAL
    if domain_key == "domain2_non_positional_investment":
        return SCENE_TYPE_NON_POSITIONAL
    raise ValueError(f"Domain `{domain_key}` is not a training domain.")


def counterfactual_alternatives(scene_type: str) -> Tuple[str, str]:
    """The two alternatives the counterfactual check must rule out."""
    if scene_type == SCENE_TYPE_POSITIONAL_CONSUMPTION:
        return (SCENE_TYPE_POSITIONAL_HUMAN_CAPITAL, SCENE_TYPE_NON_POSITIONAL)
    if scene_type == SCENE_TYPE_POSITIONAL_HUMAN_CAPITAL:
        return (SCENE_TYPE_POSITIONAL_CONSUMPTION, SCENE_TYPE_NON_POSITIONAL)
    if scene_type == SCENE_TYPE_NON_POSITIONAL:
        return (SCENE_TYPE_POSITIONAL_CONSUMPTION, SCENE_TYPE_POSITIONAL_HUMAN_CAPITAL)
    raise ValueError(f"Unknown scene_type `{scene_type}`.")


# -----------------------------------------------------------------------------
# Preamble skeletons
# -----------------------------------------------------------------------------
# Four hand-written skeletons that produce a multi-sentence preamble. Each
# skeleton receives the same set of slots:
#   {scene_type}             resolved scene-type tag
#   {protagonist}            name or pronoun phrase
#   {counterfactual_1}       alt scene type #1
#   {counterfactual_2}       alt scene type #2
#   {cf_rule_out_1}          why alt #1 is not the case
#   {cf_rule_out_2}          why alt #2 is not the case

_PREAMBLE_SKELETONS: Tuple[str, ...] = (
    # Skeleton 1 — front-loaded classification, then counterfactuals in order.
    (
        "Scene-type flag: {scene_type}. "
        "The observable cues around {protagonist} place this inside the {scene_type} category. "
        "Counterfactual check: this is not {counterfactual_1} because {cf_rule_out_1}; it is also not "
        "{counterfactual_2} because {cf_rule_out_2}."
    ),
    # Skeleton 2 — counterfactuals first, classification last.
    (
        "Before tagging, rule out the alternatives: this scene is not {counterfactual_1} — {cf_rule_out_1}. "
        "It is also not {counterfactual_2} — {cf_rule_out_2}. "
        "What remains and fits the observable cues around {protagonist} is {scene_type}, so the scene-type flag is "
        "{scene_type}."
    ),
    # Skeleton 3 — inline classification followed by a reason-based counterfactual block.
    (
        "This reads as a {scene_type} scene for {protagonist}: the observable cues in the setting match that "
        "category. "
        "If it were {counterfactual_1}, then {cf_rule_out_1}, which is not what the scene actually shows. "
        "If it were {counterfactual_2}, then {cf_rule_out_2}, which likewise does not match the scene."
    ),
    # Skeleton 4 — concise classification plus compact side-by-side counterfactuals.
    (
        "Scene-type taxonomy: {scene_type}. "
        "Contrast A (not {counterfactual_1}): {cf_rule_out_1}. "
        "Contrast B (not {counterfactual_2}): {cf_rule_out_2}. "
        "The cues around {protagonist} line up with {scene_type} and not the alternatives."
    ),
)


# -----------------------------------------------------------------------------
# Default rule-out clauses (per scene-type pair)
# -----------------------------------------------------------------------------
# These are hand-written, short, and do NOT use any phrase from
# TEST_MCQ_RULE_VOCAB. They capture the distinguishing cue between scene types
# in plain prose.

_CF_RULE_OUT: dict = {
    # Given current scene-type is X, rule-out clause for "not Y".
    (SCENE_TYPE_POSITIONAL_CONSUMPTION, SCENE_TYPE_POSITIONAL_HUMAN_CAPITAL): (
        "there is no study-hour or skill-signal race at stake, only a spend whose size is read socially"
    ),
    (SCENE_TYPE_POSITIONAL_CONSUMPTION, SCENE_TYPE_NON_POSITIONAL): (
        "the commitment is read off a shared social standard — peers' amounts are treated as a bar to "
        "meet, not just as other people's private situations"
    ),
    (SCENE_TYPE_POSITIONAL_HUMAN_CAPITAL, SCENE_TYPE_POSITIONAL_CONSUMPTION): (
        "the visible signal is hours-of-work or skill marks, not a consumption bill"
    ),
    (SCENE_TYPE_POSITIONAL_HUMAN_CAPITAL, SCENE_TYPE_NON_POSITIONAL): (
        "the hours and output are tracked against a cohort standard, not sized to the protagonist's own "
        "private training target"
    ),
    (SCENE_TYPE_NON_POSITIONAL, SCENE_TYPE_POSITIONAL_CONSUMPTION): (
        "even though other people's amounts are visible, the payoff here is bounded by the protagonist's "
        "own private situation — their own tooth, their own roof, their own child — so peer amounts do "
        "not set a standard to match"
    ),
    (SCENE_TYPE_NON_POSITIONAL, SCENE_TYPE_POSITIONAL_HUMAN_CAPITAL): (
        "the relevant benefit is intrinsic to the protagonist's own case (their own health, their own "
        "property) and is not graded against a cohort, so peer hours or output are not a comparison bar"
    ),
}


def _default_rule_out(scene_type: str, counterfactual: str) -> str:
    try:
        return _CF_RULE_OUT[(scene_type, counterfactual)]
    except KeyError:
        # Conservative fallback (never used in practice but keeps the code total).
        return "the scene's observable cues do not fit that alternative"


# -----------------------------------------------------------------------------
# Public preamble builder
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PreambleContext:
    scene_type: str                      # one of ALL_SCENE_TYPES (tricky-positional resolves to pos-consumption|pos-human-capital for the tag)
    protagonist: str                     # name or phrase ("she", "the protagonist")
    resolved_scene_type: Optional[str] = None  # if scene_type == tricky-positional, caller sets this
    cf_rule_out_override: Optional[Tuple[str, str]] = None   # override (alt_1 clause, alt_2 clause)


def build_taxonomy_preamble(
    ctx: PreambleContext,
    rng: random.Random,
) -> str:
    """Return a single-string taxonomy preamble using a random skeleton."""
    resolved = ctx.resolved_scene_type or ctx.scene_type
    if resolved == SCENE_TYPE_TRICKY_POSITIONAL:
        raise ValueError("tricky-positional must resolve to a concrete scene-type via PreambleContext.resolved_scene_type")
    if resolved not in (SCENE_TYPE_POSITIONAL_CONSUMPTION, SCENE_TYPE_POSITIONAL_HUMAN_CAPITAL, SCENE_TYPE_NON_POSITIONAL):
        raise ValueError(f"Unsupported resolved scene_type `{resolved}`")

    alt_1, alt_2 = counterfactual_alternatives(resolved)
    if rng.random() < 0.5:
        alt_1, alt_2 = alt_2, alt_1

    if ctx.cf_rule_out_override is not None:
        cf_rule_out_1, cf_rule_out_2 = ctx.cf_rule_out_override
    else:
        cf_rule_out_1 = _default_rule_out(resolved, alt_1)
        cf_rule_out_2 = _default_rule_out(resolved, alt_2)

    skeleton = rng.choice(_PREAMBLE_SKELETONS)
    return skeleton.format(
        scene_type=resolved,
        protagonist=ctx.protagonist or "this person",
        counterfactual_1=alt_1,
        counterfactual_2=alt_2,
        cf_rule_out_1=cf_rule_out_1,
        cf_rule_out_2=cf_rule_out_2,
    )


# -----------------------------------------------------------------------------
# Reasoning-body templates (qualitative, no arithmetic)
# -----------------------------------------------------------------------------
# The body is a sequence of short sentences that walk through the Langtry
# closed-form reasoning in plain prose. Two body variants per scene-type
# family keep the surface varied.

_BODY_POSITIONAL: Tuple[str, ...] = (
    (
        "Within this category, the commitment is lifted above the private baseline by a closeness-weighted "
        "reference drawn from the {peer_noun_plural}. The close-in-life relationships carry most of the weight; "
        "the more peripheral ones barely register. The comparison-sensitivity of {protagonist} scales that "
        "aggregate pull before it stacks on top of the baseline floor."
    ),
    (
        "Because the audience is real and the commitment is read socially, the chosen level sits somewhere above "
        "the private floor. The lift is not the same share per {peer_noun_singular}; it is a mix where "
        "close-in-life {peer_noun_plural} count much more than peripheral ones. How strong the lift is depends on "
        "how much {protagonist} actually weighs visible peer cues overall."
    ),
)

_BODY_NON_POSITIONAL: Tuple[str, ...] = (
    (
        "Other {peer_noun_plural} may be visibly handling their own versions of this kind of decision, and "
        "their amounts are legible to {protagonist}. But the payoff here is bounded by {protagonist}'s own "
        "private situation — it does not rise when the chosen commitment matches or exceeds what others "
        "did on theirs. So even with those amounts in view, the closeness-weighted aggregate from the "
        "{peer_noun_plural} does not pull the chosen level above the private baseline. The right level is "
        "whatever {protagonist}'s own case warrants."
    ),
    (
        "The scene does surface what close-in-life and peripheral {peer_noun_plural} are paying on their "
        "own versions of this decision, so {protagonist} can see those numbers. What the scene does NOT "
        "do is tie {protagonist}'s outcome to matching or beating them — the benefit is intrinsic to "
        "{protagonist}'s own situation, not relative. So the peer amounts are information, not a "
        "comparison target, and there is no reason to lift above the private baseline."
    ),
)


def _body_for_scene_type(scene_type: str, rng: random.Random) -> str:
    if scene_type in (SCENE_TYPE_POSITIONAL_CONSUMPTION, SCENE_TYPE_POSITIONAL_HUMAN_CAPITAL):
        return rng.choice(_BODY_POSITIONAL)
    if scene_type == SCENE_TYPE_NON_POSITIONAL:
        return rng.choice(_BODY_NON_POSITIONAL)
    raise ValueError(f"Unsupported scene_type `{scene_type}`.")


# -----------------------------------------------------------------------------
# Conclusion templates
# -----------------------------------------------------------------------------

_CONCLUSION_POSITIONAL: Tuple[str, ...] = (
    "Gold rule: anchor on the baseline and lift by the closeness-weighted reference, giving a target {action_label} near {x_display}.",
    "The implied commitment is the private floor plus the closeness-weighted peer reference, landing near {x_display}.",
)

_CONCLUSION_NON_POSITIONAL: Tuple[str, ...] = (
    "Gold rule: hold the private floor; the target {action_label} equals the baseline, near {x_display}.",
    "No peer lift applies; the commitment stays at the private baseline, about {x_display}.",
)


def _conclusion_for_scene_type(
    scene_type: str,
    rng: random.Random,
    action_label: str,
    x_display: str,
) -> str:
    if scene_type in (SCENE_TYPE_POSITIONAL_CONSUMPTION, SCENE_TYPE_POSITIONAL_HUMAN_CAPITAL):
        template = rng.choice(_CONCLUSION_POSITIONAL)
    elif scene_type == SCENE_TYPE_NON_POSITIONAL:
        template = rng.choice(_CONCLUSION_NON_POSITIONAL)
    else:
        raise ValueError(f"Unsupported scene_type `{scene_type}`.")
    return template.format(action_label=action_label or "commitment", x_display=x_display)


# -----------------------------------------------------------------------------
# Full CoT builder
# -----------------------------------------------------------------------------

@dataclass
class CoTContext:
    domain_key: str
    protagonist: str
    peer_noun_singular: str
    peer_noun_plural: str
    action_label: str
    x_display: str
    # Optional overrides
    tricky_positional: bool = False
    cf_rule_out_override: Optional[Tuple[str, str]] = None


def build_full_cot(ctx: CoTContext, rng: random.Random) -> str:
    scene_type_resolved = domain_to_scene_type(ctx.domain_key)
    preamble_scene_type = (
        SCENE_TYPE_TRICKY_POSITIONAL if ctx.tricky_positional else scene_type_resolved
    )
    preamble_ctx = PreambleContext(
        scene_type=preamble_scene_type,
        protagonist=ctx.protagonist,
        resolved_scene_type=scene_type_resolved,
        cf_rule_out_override=ctx.cf_rule_out_override,
    )
    preamble = build_taxonomy_preamble(preamble_ctx, rng=rng)
    body = _body_for_scene_type(scene_type_resolved, rng).format(
        protagonist=ctx.protagonist or "this person",
        peer_noun_singular=ctx.peer_noun_singular or "peer",
        peer_noun_plural=ctx.peer_noun_plural or "peers",
    )
    conclusion = _conclusion_for_scene_type(
        scene_type_resolved, rng, ctx.action_label, ctx.x_display
    )
    return " ".join(["Reasoning:", preamble, body, conclusion])
