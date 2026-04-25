"""
Code-templated MCQ rule labels for Eval-A (Rule Identification) and SFT /
DPO contrastive options.

Why code-templated (vs teacher-written):
  * Stability: every evaluator sees literally the same wording for the same
    rule family, so metric variance collapses to model behaviour.
  * No teacher bias: a teacher LLM paraphrases tend to leak which option is
    "correct" via tone; hand-written paraphrases eliminate that leak.
  * Easier audit: the Lexical Leakage Audit (LLA) can match exact phrases to
    rule ids.

The 8 rules are an exhaustive taxonomy of plausible behavioural aggregation
strategies an LLM might imagine when reading a peer-influence scenario. Each
rule predicts a specific x-value given the Langtry latent draw (F, alpha_i,
x_j, g_ij). Rule A is Langtry's closed-form gold for positional domains;
Rule D is the placebo gold.

Each paraphrase is a slot template with:
  {action_label}   e.g. "monthly restaurant spend"
  {action_verb}    e.g. "spend"  (conjugated per scene; default "commit to")
  {peer_noun_plural}  e.g. "friends"
  {x_display}      e.g. "$345/month"  (formatted from the rule's predicted x)

Rules:
  A_peer_weighted       x = F + alpha * Sum g_j x_j    (Langtry gold, d1/d3)
  B_top_anchor          x = F + alpha * x_of_closest_peer
  C_uniform_avg         x = F + alpha * mean(x_j)
  D_pure_private        x = F                         (placebo gold, d2)
  E_closest_mimicry     x = x_of_closest_peer         (no F anchor)
  F_median_anchor       x = F + alpha * median(x_j)
  G_counter_conformist  x = F - alpha * Sum g_j x_j    (anti-conformity)
  H_equal_mix           x = 0.5 * F + 0.5 * Sum g_j x_j   (no alpha)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Rule registry
# -----------------------------------------------------------------------------

RULE_IDS: Tuple[str, ...] = (
    "A_peer_weighted",
    "B_top_anchor",
    "C_uniform_avg",
    "D_pure_private",
    "E_closest_mimicry",
    "F_median_anchor",
    "G_counter_conformist",
    "H_equal_mix",
)


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    short_name: str          # for logs / confusion matrix (e.g. "peer_weighted")
    compute_fn: Callable[["RuleInputs"], float]
    paraphrases: Tuple[str, ...]   # 4 paraphrase templates


@dataclass(frozen=True)
class RuleInputs:
    F: float
    alpha_i: float
    x_j: Sequence[float]
    g_ij: Sequence[float]
    closest_peer_index: int   # argmax_j g_ij

    @property
    def ref_sum(self) -> float:
        return sum(g * x for g, x in zip(self.g_ij, self.x_j))

    @property
    def mean_x(self) -> float:
        return sum(self.x_j) / max(len(self.x_j), 1)

    @property
    def median_x(self) -> float:
        xs = sorted(self.x_j)
        n = len(xs)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return xs[n // 2]
        return 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    @property
    def x_top(self) -> float:
        return self.x_j[self.closest_peer_index]


# ---- rule compute functions -------------------------------------------------

def _compute_A(inp: RuleInputs) -> float:
    return inp.F + inp.alpha_i * inp.ref_sum


def _compute_B(inp: RuleInputs) -> float:
    return inp.F + inp.alpha_i * inp.x_top


def _compute_C(inp: RuleInputs) -> float:
    return inp.F + inp.alpha_i * inp.mean_x


def _compute_D(inp: RuleInputs) -> float:
    return inp.F


def _compute_E(inp: RuleInputs) -> float:
    return inp.x_top


def _compute_F_med(inp: RuleInputs) -> float:
    return inp.F + inp.alpha_i * inp.median_x


def _compute_G(inp: RuleInputs) -> float:
    # Anti-conformist: pull AWAY from peers. Floor at 0 so we don't emit
    # negative action values (we clip rather than guarding the whole bucket).
    return max(0.0, inp.F - inp.alpha_i * inp.ref_sum)


def _compute_H(inp: RuleInputs) -> float:
    return 0.5 * inp.F + 0.5 * inp.ref_sum


# ---- hand-crafted paraphrase templates ---------------------------------------
# Style rules:
#   * NO numerics inside the descriptive clause (the "{x_display}" is the only
#     numeric slot).
#   * NO words forbidden in the vocabulary_pools.TEST_MCQ_RULE_VOCAB audit.
#   * 4 paraphrases per rule; chosen uniformly at random at MCQ build time.
#   * Rule D (private) is phrased to clearly distinguish "ignores peers" from
#     "follows no peers because peers don't exist"; this is crucial for the
#     placebo contrast.

_PARAPHRASES_A: Tuple[str, ...] = (
    ("Scale the {action_label} by how close each of the {peer_noun_plural} feels in daily life, and "
     "add that social pull on top of the private floor — lands around {x_display}."),
    ("Let each of the {peer_noun_plural} tug the {action_label} in proportion to how much weight that "
     "relationship carries, layered above the baseline need — near {x_display}."),
    ("Combine the baseline commitment with a closeness-weighted blend of what the {peer_noun_plural} do, "
     "so familiar voices count more than distant ones — ends up near {x_display}."),
    ("Anchor on the private floor, then shift toward the {peer_noun_plural} in proportion to how close "
     "each one really is — the resulting {action_label} sits around {x_display}."),
    ("Stack the baseline need first, then layer a peer lift where each {peer_noun_singular}'s share "
     "tracks their standing in the life — arrives near {x_display}."),
    ("Add to the personal floor a relationship-weighted peer pull so intimate {peer_noun_plural} count "
     "heavier than peripheral ones — settles around {x_display}."),
    ("Start from what the private situation warrants, then tilt toward a proximity-weighted aggregate "
     "of the {peer_noun_plural} — the {action_label} ends near {x_display}."),
    ("Build on the baseline with a social contribution in which close ties weigh heavier than "
     "background acquaintances — lands near {x_display}."),
    ("Keep the private floor as the anchor and add each {peer_noun_singular}'s influence in "
     "proportion to relational weight — arriving at {x_display}."),
    ("Use a proximity-graded blend of what the {peer_noun_plural} are doing as a lift above the "
     "personal need — the {action_label} sits near {x_display}."),
    ("Elevate the baseline by a weighted peer aggregate where tighter bonds contribute more share than "
     "looser ones — around {x_display}."),
    ("Lift the private floor by a peer term that amplifies voices close in daily life and dampens "
     "distant ones — the {action_label} ends near {x_display}."),
)

_PARAPHRASES_B: Tuple[str, ...] = (
    ("Let the single closest {peer_noun_singular} set the tone on top of the private floor; the rest of "
     "the circle fades into background — around {x_display}."),
    ("Pin the {action_label} to whatever the one nearest {peer_noun_singular} is doing, stacked on the "
     "baseline comfort level — near {x_display}."),
    ("Track only the tightest {peer_noun_singular}, ignore the rest of the group noise, and add that to "
     "the private floor — lands near {x_display}."),
    ("Use the most tight-knit {peer_noun_singular}'s move as the only relevant pull above the baseline "
     "need — the {action_label} ends up around {x_display}."),
    ("Peg the peer term to the nearest {peer_noun_singular}'s move alone, stacked on top of the private "
     "floor, and treat the broader circle as static — near {x_display}."),
    ("Lock the social pull to the single most intimate {peer_noun_singular} and add it above the "
     "baseline commitment — lands around {x_display}."),
    ("Set aside the wider group and pin the social lift above the private baseline to what the "
     "closest-in-life {peer_noun_singular} is doing — about {x_display}."),
    ("Route the peer component entirely through the most relationally close {peer_noun_singular} and "
     "layer it on top of the personal floor — around {x_display}."),
    ("Let the single inner-circle {peer_noun_singular} carry the entire social lift above the private "
     "baseline — the {action_label} ends near {x_display}."),
    ("Mute the group as a whole and route the peer term through the sole nearest-in-life "
     "{peer_noun_singular}'s move above the baseline — near {x_display}."),
    ("Treat only the tightest-bond {peer_noun_singular} as the relevant social pull on top of the "
     "private floor — about {x_display}."),
    ("Collapse the social pull to the one closest-in-life {peer_noun_singular} and stack that lift on "
     "the personal baseline — lands near {x_display}."),
)

_PARAPHRASES_C: Tuple[str, ...] = (
    ("Average what every {peer_noun_singular} in the circle is doing equally and add that average pull on "
     "top of the private floor — arrives around {x_display}."),
    ("Treat each {peer_noun_singular} as one democratic vote and scale by the group's flat average "
     "above the baseline — near {x_display}."),
    ("Pool the {peer_noun_plural}' visible choices with identical weights, then stack that on the "
     "private floor — lands near {x_display}."),
    ("Weigh every {peer_noun_singular}'s move the same way and layer the blended result atop the "
     "baseline commitment — about {x_display}."),
    ("Take the unweighted mean of what the {peer_noun_plural} are doing and stack that social "
     "term above the private floor — arrives near {x_display}."),
    ("Give every {peer_noun_singular} an identical share of the peer aggregate, then layer that "
     "flat pull on the baseline commitment — around {x_display}."),
    ("Flatten the social term so each {peer_noun_singular} counts the same, and add it on top of the "
     "personal baseline — lands near {x_display}."),
    ("Disregard relational closeness and use a uniform group mean as the peer lift above the private "
     "floor — around {x_display}."),
    ("Treat the {peer_noun_plural} as an undifferentiated group and add their plain mean on top of "
     "the baseline need — the {action_label} ends near {x_display}."),
    ("Sum the {peer_noun_plural} choices and divide by head count, then put that on top of the private "
     "floor with no closeness weighting — about {x_display}."),
    ("Use the ungraded average of the {peer_noun_plural}' moves as the sole social lift above the "
     "baseline — near {x_display}."),
    ("Combine the private floor with a strictly egalitarian average over the {peer_noun_plural} — "
     "no bond gets a stronger say — the {action_label} lands around {x_display}."),
)

_PARAPHRASES_D: Tuple[str, ...] = (
    ("Stay on the private floor: the {action_label} reflects only the personal need and the practical "
     "cost, not anything the {peer_noun_plural} are doing — around {x_display}."),
    ("Ignore the {peer_noun_plural} altogether and commit to exactly the level that the private cost-"
     "and-need calculation would justify — about {x_display}."),
    ("Stick with what the private baseline dictates without any lift from what the circle is doing — the "
     "{action_label} sits at {x_display}."),
    ("Hold the personal floor for the {action_label} and let nothing from the {peer_noun_plural} bend "
     "it up or down — lands at {x_display}."),
    ("Commit to the level the private situation alone would justify, treating whatever the "
     "{peer_noun_plural} are doing as irrelevant to this choice — around {x_display}."),
    ("Set the {action_label} purely by the personal cost-and-need balance and decline to adjust for "
     "anything the {peer_noun_plural} pick — at {x_display}."),
    ("Anchor entirely on the private baseline — the {peer_noun_plural}' visible choices are noise "
     "for this category of decision — the {action_label} ends at {x_display}."),
    ("Act on private utility only, refusing to let the {peer_noun_plural}' moves nudge the "
     "{action_label} up or down — around {x_display}."),
    ("Let the personal floor be the whole answer and treat visible peer amounts as information "
     "without pull — the {action_label} sits at {x_display}."),
    ("Keep the {action_label} pegged to the private baseline alone, because the benefit here does "
     "not scale with what the {peer_noun_plural} commit — at {x_display}."),
    ("Pick whatever the private situation warrants and leave the {peer_noun_plural}' visible choices "
     "out of the calculus — around {x_display}."),
    ("Hold the personal-need level unchanged, since this is not a domain where the {peer_noun_plural} "
     "set a bar — the {action_label} lands at {x_display}."),
)

_PARAPHRASES_E: Tuple[str, ...] = (
    ("Copy whatever the single closest {peer_noun_singular} is doing outright, letting the private "
     "floor recede entirely — around {x_display}."),
    ("Match the one nearest {peer_noun_singular}'s move directly, as if their level were the only "
     "anchor — lands at {x_display}."),
    ("Mirror the closest {peer_noun_singular}'s level straight up, with no separate private-baseline "
     "term — near {x_display}."),
    ("Just replicate the tightest {peer_noun_singular}'s visible choice, setting aside whatever the "
     "private floor would have said — about {x_display}."),
    ("Adopt the nearest {peer_noun_singular}'s number as-is for the {action_label}, collapsing the "
     "personal floor entirely into their choice — near {x_display}."),
    ("Replace the private baseline with the single most intimate {peer_noun_singular}'s move and "
     "stop there — ends at {x_display}."),
    ("Clone the closest-in-life {peer_noun_singular}'s commitment directly, treating it as the whole "
     "answer — around {x_display}."),
    ("Take the sole closest {peer_noun_singular}'s level at face value and set the {action_label} to "
     "exactly that — lands at {x_display}."),
    ("Skip the baseline math entirely and imitate the one tight-bond {peer_noun_singular} one-to-one — "
     "near {x_display}."),
    ("Set the {action_label} equal to whatever the most intimate {peer_noun_singular} picked, without "
     "any private floor showing up in the answer — about {x_display}."),
    ("Mirror the nearest {peer_noun_singular}'s figure exactly, as if the private situation played no "
     "role — around {x_display}."),
    ("Lean fully on the closest-in-life {peer_noun_singular}'s commitment and make that the sole "
     "number for the {action_label} — at {x_display}."),
)

_PARAPHRASES_F: Tuple[str, ...] = (
    ("Anchor on the middle {peer_noun_singular} of the group and scale up the private floor by that "
     "median pull — arrives near {x_display}."),
    ("Use the median of what the {peer_noun_plural} are doing as the social pull above the private "
     "baseline — around {x_display}."),
    ("Let the middle-of-the-road {peer_noun_singular} set the comparison level, then layer that on the "
     "private floor — lands near {x_display}."),
    ("Target the typical mid-pack {peer_noun_singular}'s move as the social lift above the personal "
     "baseline need — about {x_display}."),
    ("Pick the center {peer_noun_singular} of the distribution as the peer reference and layer that "
     "on top of the baseline commitment — near {x_display}."),
    ("Use the middle value from the {peer_noun_plural}' choices as the social term above the "
     "private floor — lands near {x_display}."),
    ("Step past both the extreme high and the extreme low of the {peer_noun_plural} and stack the "
     "center of them on the personal baseline — about {x_display}."),
    ("Let the typical-case {peer_noun_singular} set the social component and add that lift to the "
     "private floor — around {x_display}."),
    ("Take the point where half of the {peer_noun_plural} land above and half below, then lift the "
     "baseline by that central value — near {x_display}."),
    ("Sort the {peer_noun_plural}' commitments and use the halfway {peer_noun_singular}'s level on "
     "top of the private floor — about {x_display}."),
    ("Skip both tails of what the {peer_noun_plural} are doing and use the center-pick as the peer "
     "pull above the private baseline — around {x_display}."),
    ("Route the peer pull through the representative central {peer_noun_singular} and stack it on the "
     "private floor — the {action_label} lands near {x_display}."),
)

_PARAPHRASES_G: Tuple[str, ...] = (
    ("Deliberately push below where the circle of {peer_noun_plural} is heading, signalling independence "
     "from the group — around {x_display}."),
    ("Cut the {action_label} in the opposite direction of the visible {peer_noun_plural} trend, treating "
     "peer pull as a reason to pull back — near {x_display}."),
    ("Treat what the {peer_noun_plural} do as a contrarian signal and dial the {action_label} down from "
     "the private floor — about {x_display}."),
    ("Refuse to match the circle and actively pull the {action_label} under the private baseline in "
     "proportion to peer pressure — lands near {x_display}."),
    ("React against the {peer_noun_plural}' visible trend by pulling the {action_label} strictly "
     "below the private floor — around {x_display}."),
    ("Invert the peer signal: the harder the {peer_noun_plural} push up, the further the "
     "{action_label} dips beneath the personal baseline — near {x_display}."),
    ("Use a rebellious stance where the weighted peer aggregate subtracts from the private floor "
     "rather than adding to it — about {x_display}."),
    ("Take the closeness-weighted peer pull and flip its sign, so the {action_label} ends below the "
     "baseline commitment — around {x_display}."),
    ("Signal non-conformity by moving the {action_label} down from the private floor by an amount "
     "proportional to what the {peer_noun_plural} are pushing — near {x_display}."),
    ("Let the peer aggregate act as a dampener on the baseline rather than a lift, dragging the "
     "{action_label} below the private floor — about {x_display}."),
    ("Oppose the visible {peer_noun_plural} drift by subtracting the weighted peer pull from the "
     "private floor — the {action_label} lands near {x_display}."),
    ("Make the {action_label} a deliberate retreat from wherever the {peer_noun_plural} are heading, "
     "below the private baseline — around {x_display}."),
)

_PARAPHRASES_H: Tuple[str, ...] = (
    ("Split the difference evenly between the private floor and the closeness-weighted social pull, "
     "without scaling by how much peers matter — around {x_display}."),
    ("Take the midpoint of the baseline commitment and the group's weighted average, ignoring how "
     "sensitive this person is to peers — near {x_display}."),
    ("Blend half the private floor with half of what the weighted {peer_noun_plural} are doing, "
     "regardless of peer-sensitivity — arrives near {x_display}."),
    ("Average the personal baseline and the closeness-weighted peer aggregate one-to-one, bypassing any "
     "persona weighting — about {x_display}."),
    ("Combine fifty-fifty the private floor and the weighted peer aggregate, disregarding any "
     "individual peer-sensitivity parameter — around {x_display}."),
    ("Set the {action_label} as an unweighted halfway point between the private baseline and the "
     "relationship-weighted peer component — near {x_display}."),
    ("Fix equal shares for the personal floor and the closeness-weighted peer pull, without letting "
     "peer sensitivity scale the mix — about {x_display}."),
    ("Use a fixed half-baseline half-peer-aggregate recipe that ignores how responsive this person "
     "is to the circle — lands around {x_display}."),
    ("Mix the private floor and the weighted social pull at a flat 1:1 ratio, paying no attention to "
     "the individual's peer-sensitivity — near {x_display}."),
    ("Set the {action_label} exactly between the baseline commitment and the weighted peer aggregate, "
     "ignoring the persona's sensitivity dial — about {x_display}."),
    ("Apply a persona-agnostic balance: half private floor, half closeness-weighted peer aggregate, "
     "no modulation — around {x_display}."),
    ("Fix the {action_label} at the arithmetic mean of the private baseline and the weighted peer "
     "component, with no sensitivity scaling — near {x_display}."),
)


RULE_SPECS: Dict[str, RuleSpec] = {
    "A_peer_weighted":      RuleSpec("A_peer_weighted",     "peer_weighted",      _compute_A,     _PARAPHRASES_A),
    "B_top_anchor":         RuleSpec("B_top_anchor",        "top_anchor",         _compute_B,     _PARAPHRASES_B),
    "C_uniform_avg":        RuleSpec("C_uniform_avg",       "uniform_avg",        _compute_C,     _PARAPHRASES_C),
    "D_pure_private":       RuleSpec("D_pure_private",      "pure_private",       _compute_D,     _PARAPHRASES_D),
    "E_closest_mimicry":    RuleSpec("E_closest_mimicry",   "closest_mimicry",    _compute_E,     _PARAPHRASES_E),
    "F_median_anchor":      RuleSpec("F_median_anchor",     "median_anchor",      _compute_F_med, _PARAPHRASES_F),
    "G_counter_conformist": RuleSpec("G_counter_conformist","counter_conformist", _compute_G,     _PARAPHRASES_G),
    "H_equal_mix":          RuleSpec("H_equal_mix",         "equal_mix",          _compute_H,     _PARAPHRASES_H),
}


# -----------------------------------------------------------------------------
# Numerical formatting helper
# -----------------------------------------------------------------------------

def format_value(value: float, value_format: str, unit: str) -> str:
    """Format an x-value for display inside a rule paraphrase.

    Matches the style previously used by data/generation/latent_sampler.py so
    the numeric anchor visible in MCQ options matches the numbers visible in
    the scenario narrative.
    """
    value = max(value, 0.0)  # never emit negatives; G-rule clamps at 0 already
    if value_format == "currency":
        if unit == "USD":
            return f"${int(round(value)):,}"
        return f"{unit} {int(round(value)):,}"
    if value_format == "hours":
        return f"{value:.1f} {unit}"
    if value_format == "index":
        return f"{value:.1f} {unit}" if unit else f"{value:.1f}"
    if value_format == "score":
        return f"{value:.2f}"
    return f"{value:.2f} {unit}".strip()


# -----------------------------------------------------------------------------
# Option builder
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MCQOption:
    letter: str         # "A" | "B" | "C" | "D"
    rule_id: str        # from RULE_IDS
    rule_short: str
    text: str
    x_value: float      # the numeric anchor this option reports
    x_display: str      # formatted copy shown in `text`


def compute_rule_x_values(inp: RuleInputs) -> Dict[str, float]:
    """Return {rule_id: predicted x-value} for all 8 rules."""
    return {rid: spec.compute_fn(inp) for rid, spec in RULE_SPECS.items()}


def build_option_text(
    rule_id: str,
    rng: random.Random,
    x_value: float,
    action_label: str,
    action_verb: str,
    peer_noun_singular: str,
    peer_noun_plural: str,
    value_format: str,
    unit: str,
) -> Tuple[str, str]:
    spec = RULE_SPECS[rule_id]
    template = rng.choice(spec.paraphrases)
    x_display = format_value(x_value, value_format, unit)
    text = template.format(
        action_label=action_label or "commitment",
        action_verb=action_verb or "commit to",
        peer_noun_singular=peer_noun_singular or "peer",
        peer_noun_plural=peer_noun_plural or "peers",
        x_display=x_display,
    )
    return text, x_display


def build_mcq_options(
    rng: random.Random,
    gold_rule_id: str,
    candidate_distractor_ids: Sequence[str],
    rule_x_values: Dict[str, float],
    action_label: str,
    peer_noun_singular: str,
    peer_noun_plural: str,
    value_format: str,
    unit: str,
    action_verb: str = "commit to",
    num_distractors: int = 3,
    forced_gold_letter: Optional[str] = None,
) -> List[MCQOption]:
    """Pick `num_distractors` distractors from `candidate_distractor_ids`, shuffle with gold, assign A-D.

    Caller is responsible for feeding candidate distractor ids that have
    ALREADY passed the identification filter (numeric separability guarantee).

    If `forced_gold_letter` is one of "A", "B", "C", "D", the gold rule is
    deterministically placed at that letter (distractors fill the remaining
    three positions in random order). Use this to enforce exact A/B/C/D
    uniformity via round-robin cycling from the caller.
    """
    if gold_rule_id not in RULE_SPECS:
        raise ValueError(f"Unknown gold_rule_id `{gold_rule_id}`.")
    if gold_rule_id in candidate_distractor_ids:
        raise ValueError("gold_rule_id must not appear in the distractor pool.")

    if len(candidate_distractor_ids) < num_distractors:
        raise ValueError(
            f"Need >= {num_distractors} distractors, got {len(candidate_distractor_ids)}: "
            f"{list(candidate_distractor_ids)}"
        )
    picks = rng.sample(list(candidate_distractor_ids), num_distractors)
    letters = ("A", "B", "C", "D")
    if forced_gold_letter in letters:
        distractor_order = list(picks)
        rng.shuffle(distractor_order)
        ordered_ids: List[str] = []
        d_iter = iter(distractor_order)
        for letter in letters:
            if letter == forced_gold_letter:
                ordered_ids.append(gold_rule_id)
            else:
                ordered_ids.append(next(d_iter))
    else:
        ordered_ids = [gold_rule_id, *picks]
        rng.shuffle(ordered_ids)

    options: List[MCQOption] = []
    for letter, rid in zip(letters, ordered_ids):
        x_val = rule_x_values[rid]
        text, disp = build_option_text(
            rule_id=rid,
            rng=rng,
            x_value=x_val,
            action_label=action_label,
            action_verb=action_verb,
            peer_noun_singular=peer_noun_singular,
            peer_noun_plural=peer_noun_plural,
            value_format=value_format,
            unit=unit,
        )
        options.append(MCQOption(
            letter=letter,
            rule_id=rid,
            rule_short=RULE_SPECS[rid].short_name,
            text=text,
            x_value=x_val,
            x_display=disp,
        ))
    return options


def gold_rule_for_domain(domain_key: str) -> str:
    """For d1/d3 the Langtry gold is A_peer_weighted; for d2 placebo it is D_pure_private."""
    if domain_key == "domain2_non_positional_investment":
        return "D_pure_private"
    return "A_peer_weighted"
