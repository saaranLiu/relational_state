"""
Scene pool for the relational-state dataset.

Each Scene binds a concrete everyday decision setting to:
    * a Langtry domain_key  (d1 / d2 / d3 / d4 / d5)
    * a template family     (a named cluster of similar decisions)
    * action label / unit / value_format / F range
    * peer noun (friend, coworker, classmate, ...)
    * split tag             (train | test | both)

The pool is deliberately disjoint across train and test so that the model
cannot memorise surface words from a scene it has seen during SFT.

Split conventions
-----------------
* Main positional domains (d1, d3):   family-level 3:1 train:test split.
  Every family contributes at least one held-out test scene.
* Placebo domain (d2):                families mixed; each has 1 train + 2 test
  scenes so the taxonomy-distinction behaviour is learnable AND testable
  on unseen placebo scenes.
* OOD domains (d4, d5):               all scenes are `split="test"` (Eval-D /
  Eval-E). They are NEVER used for training.

The pool currently contains ~70 scenes which is enough to support the
5600-example SFT set (with ~80 latent cells per scene average). Adding more
scenes is a drop-in change and does not require touching any other module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from data.config.langtry_parameters import (
    MAIN_POSITIONAL_DOMAINS,
    OOD_DOMAIN_CAREER,
    OOD_DOMAIN_SOCIAL,
    PLACEBO_DOMAIN,
)


# -----------------------------------------------------------------------------
# Dataclass
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Scene:
    scene_id: str
    domain_key: str
    family: str
    title: str
    summary: str
    action_label: str
    action_unit: str
    value_format: str  # "currency" | "hours" | "index" | "score"
    F_range: Tuple[float, float]
    peer_noun_singular: str
    peer_noun_plural: str
    split: str  # "train" | "test" | "both"
    # Domain-specific extras (e.g. wage_anchor for d5, friendship_context for d4)
    extras: Dict[str, object] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# d1 positional consumption scenes
# -----------------------------------------------------------------------------
# Eight families x ~3 scenes, split roughly 2 train : 1 test per family.
_D1_SCENES: Tuple[Scene, ...] = (
    # Family: visible weekend restaurants / brunch
    Scene("d1_dining_01", "domain1_positional_consumption", "dining_out",
          "Saturday brunch at a Michelin-adjacent place",
          "A weekend brunch line-up with a tight friend group where menu photos end up on stories.",
          "monthly restaurant spend", "USD", "currency", (180.0, 560.0),
          "friend", "friends", "train"),
    Scene("d1_dining_02", "domain1_positional_consumption", "dining_out",
          "Birthday dinner in the chef's-counter seats",
          "A birthday dinner where the host picked a chef's-counter place that the crew will talk about.",
          "monthly restaurant spend", "USD", "currency", (200.0, 620.0),
          "friend", "friends", "train"),
    Scene("d1_dining_03", "domain1_positional_consumption", "dining_out",
          "Tasting-menu night after a promotion",
          "A crew of new-promoted consultants reservation-hunting a twelve-course tasting spot.",
          "monthly restaurant spend", "USD", "currency", (220.0, 700.0),
          "friend", "friends", "test"),

    # Family: gifting
    Scene("d1_gift_01", "domain1_positional_consumption", "gifting",
          "Baby-shower gift among new-parent WhatsApp group",
          "A WhatsApp group of new parents sending baby-shower gifts that everyone will unwrap together.",
          "gift spend", "USD", "currency", (50.0, 260.0),
          "friend", "friends", "train"),
    Scene("d1_gift_02", "domain1_positional_consumption", "gifting",
          "Wedding gift pool among college friends",
          "A college friend group coordinating wedding gifts on a shared Splitwise.",
          "gift spend", "USD", "currency", (80.0, 360.0),
          "friend", "friends", "train"),
    Scene("d1_gift_03", "domain1_positional_consumption", "gifting",
          "Teacher-appreciation bundle in a parent circle",
          "A parent group chat pooling teacher-appreciation gifts that parents will compare at the end-of-year coffee.",
          "gift spend", "USD", "currency", (40.0, 200.0),
          "friend", "friends", "test"),

    # Family: travel / short trips
    Scene("d1_travel_01", "domain1_positional_consumption", "travel",
          "Long-weekend group trip to a mid-tier resort",
          "A group trip where a shared trip-ledger shows everyone's flights and rooms.",
          "trip budget", "USD", "currency", (600.0, 2600.0),
          "friend", "friends", "train"),
    Scene("d1_travel_02", "domain1_positional_consumption", "travel",
          "Bachelor trip to a boutique lodge",
          "A bachelor trip where the choice between two lodges is visible on the shared Notion page.",
          "trip budget", "USD", "currency", (800.0, 3200.0),
          "friend", "friends", "test"),
    Scene("d1_travel_03", "domain1_positional_consumption", "travel",
          "Anniversary mini-getaway",
          "A couple picking an anniversary getaway whose photos will land on mutual feeds.",
          "trip budget", "USD", "currency", (500.0, 2400.0),
          "friend", "friends", "train"),

    # Family: wearables / fashion
    Scene("d1_wearable_01", "domain1_positional_consumption", "wearable",
          "New watch for a milestone birthday",
          "A milestone-birthday watch purchase visible on a wrist during Friday drinks.",
          "watch purchase", "USD", "currency", (500.0, 4200.0),
          "friend", "friends", "train"),
    Scene("d1_wearable_02", "domain1_positional_consumption", "wearable",
          "Winter outerwear tier decision",
          "A shoppable-list of coats within a fashion-aware peer group before winter.",
          "coat purchase", "USD", "currency", (300.0, 1800.0),
          "friend", "friends", "train"),
    Scene("d1_wearable_03", "domain1_positional_consumption", "wearable",
          "Designer handbag entry purchase",
          "A first-designer-bag purchase among peers who post unboxings.",
          "handbag purchase", "USD", "currency", (700.0, 3600.0),
          "friend", "friends", "test"),

    # Family: wheels (car / ebike)
    Scene("d1_wheels_01", "domain1_positional_consumption", "wheels",
          "Mid-size SUV upgrade decision",
          "An upgrade from a reliable sedan to a mid-size SUV in a peer circle with visible driveways.",
          "vehicle price", "USD", "currency", (18000.0, 55000.0),
          "coworker", "coworkers", "train"),
    Scene("d1_wheels_02", "domain1_positional_consumption", "wheels",
          "Premium e-bike choice in a cycling club",
          "An e-bike pick inside a club where bike brands are immediately obvious at weekend rides.",
          "e-bike price", "USD", "currency", (1400.0, 6800.0),
          "friend", "friends", "train"),
    Scene("d1_wheels_03", "domain1_positional_consumption", "wheels",
          "EV purchase within a new-parent circle",
          "An EV choice in a new-parent circle where who drives what is noted at the nursery pickup line.",
          "EV price", "USD", "currency", (28000.0, 78000.0),
          "friend", "friends", "test"),

    # Family: home staging / renovation accents
    Scene("d1_home_01", "domain1_positional_consumption", "home_staging",
          "Housewarming main room refresh",
          "A housewarming where the main room will be photographed by guests and posted.",
          "refresh budget", "USD", "currency", (400.0, 3000.0),
          "friend", "friends", "train"),
    Scene("d1_home_02", "domain1_positional_consumption", "home_staging",
          "Rented-flat upgrade before hosting",
          "A rented flat getting an upgrade before hosting a dinner party whose table will be filmed.",
          "refresh budget", "USD", "currency", (300.0, 2200.0),
          "friend", "friends", "test"),

    # Family: kids' extracurriculars (positional school signal)
    Scene("d1_kids_01", "domain1_positional_consumption", "kids_extras",
          "Summer camp tier among school parents",
          "A summer-camp tier decision among school-parents whose kids will compare camp stories in September.",
          "camp fee", "USD", "currency", (800.0, 4200.0),
          "parent", "parents", "train"),
    Scene("d1_kids_02", "domain1_positional_consumption", "kids_extras",
          "Private-tutor hours tier",
          "A private-tutor-hours decision among a parent WeChat group where marks get compared.",
          "monthly tutor fees", "USD", "currency", (300.0, 2400.0),
          "parent", "parents", "test"),

    # Family: wedding / celebration
    Scene("d1_cele_01", "domain1_positional_consumption", "celebration",
          "Engagement-party venue tier",
          "An engagement-party venue chosen in front of a family WhatsApp group where other parties set the bar.",
          "venue fee", "USD", "currency", (1200.0, 9000.0),
          "family member", "family", "train"),
    Scene("d1_cele_02", "domain1_positional_consumption", "celebration",
          "Kids' first birthday bash",
          "A first-birthday bash planned in a young-parent group chat where prior bashes live on Instagram.",
          "party budget", "USD", "currency", (500.0, 3600.0),
          "parent", "parents", "test"),
)

# -----------------------------------------------------------------------------
# d3 human-capital investment scenes (positional -- because education and
# skill signals are compared on visible rankings / offers / marks).
# -----------------------------------------------------------------------------
_D3_SCENES: Tuple[Scene, ...] = (
    # Family: standardised test prep
    Scene("d3_testprep_01", "domain3_effort_and_human_capital", "test_prep",
          "GRE/GMAT hours/week in an MBA-applicant Slack",
          "An MBA-applicant Slack where weekly study hours and mock-scores are routinely dropped.",
          "weekly study hours", "hours", "hours", (8.0, 30.0),
          "classmate", "classmates", "train"),
    Scene("d3_testprep_02", "domain3_effort_and_human_capital", "test_prep",
          "LSAT hours/week before the February sitting",
          "An LSAT cohort in a Discord where people post daily block totals before the February sitting.",
          "weekly study hours", "hours", "hours", (10.0, 34.0),
          "classmate", "classmates", "train"),
    Scene("d3_testprep_03", "domain3_effort_and_human_capital", "test_prep",
          "CFA Level II hours/week in a WeChat study group",
          "A CFA Level II WeChat study group where every Sunday everyone reports the week's hours.",
          "weekly study hours", "hours", "hours", (10.0, 28.0),
          "classmate", "classmates", "test"),

    # Family: leetcode / technical grind
    Scene("d3_leet_01", "domain3_effort_and_human_capital", "technical_grind",
          "LeetCode problems/week before on-sites",
          "A pre-internship WeChat group where daily LeetCode solved counts are pinned at midnight.",
          "weekly leetcode problems", "problems", "index", (15.0, 90.0),
          "classmate", "classmates", "train"),
    Scene("d3_leet_02", "domain3_effort_and_human_capital", "technical_grind",
          "System-design mock sessions/month",
          "A senior-engineer peer group running weekly mock system-design sessions with a shared calendar.",
          "monthly mock sessions", "sessions", "index", (3.0, 16.0),
          "coworker", "coworkers", "test"),

    # Family: graduate fellowship prep
    Scene("d3_fellow_01", "domain3_effort_and_human_capital", "fellowship_prep",
          "Fulbright essay drafts in a cohort channel",
          "A Fulbright-applicant cohort channel where essay-draft counts and peer-review rounds are visible.",
          "total draft revisions", "drafts", "index", (4.0, 18.0),
          "classmate", "classmates", "train"),
    Scene("d3_fellow_02", "domain3_effort_and_human_capital", "fellowship_prep",
          "Rhodes interview prep hours before campus round",
          "A Rhodes-candidate study pair logging practice-interview hours in a shared Google sheet.",
          "weekly practice hours", "hours", "hours", (4.0, 20.0),
          "classmate", "classmates", "test"),

    # Family: PhD / research effort
    Scene("d3_phd_01", "domain3_effort_and_human_capital", "research_effort",
          "Weekly writing hours among co-authors",
          "A shared writing-sprint channel among co-authors where weekly writing hours get Slack-reacted.",
          "weekly writing hours", "hours", "hours", (5.0, 26.0),
          "classmate", "classmates", "train"),
    Scene("d3_phd_02", "domain3_effort_and_human_capital", "research_effort",
          "ICML submission push overtime",
          "An ICML-submission pre-deadline channel where everyone posts their hour tallies at the 72-hour mark.",
          "weekly writing hours", "hours", "hours", (10.0, 38.0),
          "classmate", "classmates", "train"),
    Scene("d3_phd_03", "domain3_effort_and_human_capital", "research_effort",
          "First-year prelim prep among cohort",
          "A first-year PhD cohort compiling a shared study schedule two months before prelims.",
          "weekly study hours", "hours", "hours", (6.0, 28.0),
          "classmate", "classmates", "test"),

    # Family: language-learning streak
    Scene("d3_lang_01", "domain3_effort_and_human_capital", "language",
          "IELTS speaking hours before the March sitting",
          "An IELTS study group where speaking-prep hours are auto-tracked and leaderboarded.",
          "weekly study hours", "hours", "hours", (3.0, 18.0),
          "classmate", "classmates", "train"),
    Scene("d3_lang_02", "domain3_effort_and_human_capital", "language",
          "JLPT N2 vocabulary daily streak",
          "A JLPT N2 cohort using a shared Anki deck where daily new-words count is visible.",
          "daily new vocabulary", "words", "index", (30.0, 140.0),
          "classmate", "classmates", "test"),

    # Family: recruiting grind
    Scene("d3_recruit_01", "domain3_effort_and_human_capital", "recruiting",
          "Referrals/week before the fall recruiting cycle",
          "A consulting-recruiting club where weekly coffee-chats and referral DMs are logged.",
          "weekly coffee chats", "chats", "index", (4.0, 22.0),
          "classmate", "classmates", "train"),
    Scene("d3_recruit_02", "domain3_effort_and_human_capital", "recruiting",
          "Investment-banking SA interview prep",
          "An SA-interview prep circle where mock-interview counts are tracked in a live Airtable.",
          "weekly mock interviews", "interviews", "index", (2.0, 12.0),
          "classmate", "classmates", "test"),
)

# -----------------------------------------------------------------------------
# d2 placebo: non-positional private investment.
# Gold = F. α_i ~ 0. Scenes must not carry social-comparison cues.
# -----------------------------------------------------------------------------
_D2_SCENES: Tuple[Scene, ...] = (
    # Family: private health decisions
    Scene("d2_health_01", "domain2_non_positional_investment", "health",
          "Private dental-crown timing",
          "A private dental-crown timing decision with no peer group involved; the dentist's advice drives the choice.",
          "dental cost", "USD", "currency", (400.0, 2200.0),
          "", "", "train"),
    Scene("d2_health_02", "domain2_non_positional_investment", "health",
          "Solo eye-exam and glasses budget",
          "An eye-exam plus glasses decision where nobody in their life comments on frames.",
          "glasses cost", "USD", "currency", (120.0, 780.0),
          "", "", "test"),
    Scene("d2_health_03", "domain2_non_positional_investment", "health",
          "Home physiotherapy sessions after a minor injury",
          "A private home-physio plan prescribed after a minor running injury; no visibility to others.",
          "physio sessions", "sessions", "index", (4.0, 22.0),
          "", "", "test"),

    # Family: private admin / paperwork
    Scene("d2_admin_01", "domain2_non_positional_investment", "admin",
          "Filing an amended tax return",
          "A private amended-tax-return filing where effort affects only the refund, not anyone's opinion.",
          "hours spent on amendment", "hours", "hours", (2.0, 14.0),
          "", "", "train"),
    Scene("d2_admin_02", "domain2_non_positional_investment", "admin",
          "Estate-planning paperwork for a single person",
          "An estate-planning paperwork session done privately; nobody will see the final document.",
          "hours spent on paperwork", "hours", "hours", (3.0, 20.0),
          "", "", "test"),

    # Family: private home maintenance
    Scene("d2_home_01", "domain2_non_positional_investment", "maintenance",
          "Boiler service timing",
          "A boiler-service timing decision; nobody visits the utility cupboard and the outcome is purely private comfort.",
          "maintenance cost", "USD", "currency", (120.0, 680.0),
          "", "", "train"),
    Scene("d2_home_02", "domain2_non_positional_investment", "maintenance",
          "Gutter cleaning before rainy season",
          "A gutter-cleaning decision in a detached house whose gutters are not visible to anyone.",
          "maintenance cost", "USD", "currency", (80.0, 420.0),
          "", "", "test"),

    # Family: mundane private savings
    Scene("d2_save_01", "domain2_non_positional_investment", "private_saving",
          "Choosing a 12-month term deposit tenor",
          "A term-deposit tenor decision made privately; there is no peer audience for this choice.",
          "planned monthly deposit", "USD", "currency", (200.0, 1200.0),
          "", "", "train"),
    Scene("d2_save_02", "domain2_non_positional_investment", "private_saving",
          "Rainy-day fund target size",
          "A rainy-day fund target size decided by one person alone; nobody is watching the balance.",
          "target reserve size", "USD", "currency", (1000.0, 9000.0),
          "", "", "test"),

    # Family: solitary hobby investment
    Scene("d2_hobby_01", "domain2_non_positional_investment", "solitary_hobby",
          "New strings for a practice guitar nobody sees",
          "A set of strings for a guitar that lives in a private study and is never played for others.",
          "annual maintenance", "USD", "currency", (30.0, 220.0),
          "", "", "train"),
    Scene("d2_hobby_02", "domain2_non_positional_investment", "solitary_hobby",
          "Solo reading list — book purchases this quarter",
          "A quarterly personal reading list where books are read alone, with no book-club audience.",
          "quarterly book spend", "USD", "currency", (30.0, 260.0),
          "", "", "test"),
)

# -----------------------------------------------------------------------------
# d4 OOD -- social-network formation.  All scenes are split="test".
# extras.friendship_context describes the social mechanism the teacher must
# preserve (friendship only stabilises when x_i matches x_j).
# -----------------------------------------------------------------------------
_D4_SCENES: Tuple[Scene, ...] = (
    Scene("d4_new_city_01", OOD_DOMAIN_SOCIAL, "new_city_friendship",
          "Picking a new close-friend after a move to Boston",
          "A young professional who just moved to Boston is weighing two candidates from a Friday co-working meetup.",
          "monthly leisure spend", "USD", "currency", (200.0, 1200.0),
          "acquaintance", "acquaintances", "test",
          extras={"friendship_context": "Two candidates visibly differ in leisure-spend profile; only one matches."}),
    Scene("d4_grad_cohort_01", OOD_DOMAIN_SOCIAL, "new_city_friendship",
          "Picking a study buddy in a new grad cohort",
          "A first-year grad student sizing up two cohort-mates for a long-term study pair after a shared seminar.",
          "weekly study hours", "hours", "hours", (4.0, 24.0),
          "classmate", "classmates", "test",
          extras={"friendship_context": "Study-hour rhythm is the observable trait; matched rhythm is required."}),
    Scene("d4_parent_group_01", OOD_DOMAIN_SOCIAL, "neighborhood_friendship",
          "Picking a playdate family in a new neighbourhood",
          "A new-parent family deciding which of two neighbourhood families to deepen ties with.",
          "weekly family outings", "outings", "index", (1.0, 8.0),
          "neighbor", "neighbors", "test",
          extras={"friendship_context": "Observable weekly-outing frequency matches the benefit-cost ratio."}),
    Scene("d4_runclub_01", OOD_DOMAIN_SOCIAL, "new_city_friendship",
          "Picking a regular running partner",
          "A runner new to a Saturday run club deciding who to pair up with for midweek runs.",
          "weekly running mileage", "miles", "index", (8.0, 42.0),
          "acquaintance", "acquaintances", "test",
          extras={"friendship_context": "Running mileage is the visible trait; sustainable friendship needs matching weekly load."}),
    Scene("d4_church_01", OOD_DOMAIN_SOCIAL, "community_friendship",
          "Picking a small-group partner at a new church",
          "A newly-joined member of a church small-group weighing two potential regulars.",
          "weekly volunteer hours", "hours", "hours", (1.0, 10.0),
          "acquaintance", "acquaintances", "test",
          extras={"friendship_context": "Volunteer-hour commitment level is observable and determines matching stability."}),
    Scene("d4_founder_01", OOD_DOMAIN_SOCIAL, "career_friendship",
          "Picking a peer-founder accountability buddy",
          "A solo founder at a Friday founder dinner choosing who to add as a weekly accountability buddy.",
          "weekly deep-work hours", "hours", "hours", (8.0, 38.0),
          "acquaintance", "acquaintances", "test",
          extras={"friendship_context": "Deep-work cadence is observable; only matching cadences sustain the pairing."}),
    Scene("d4_hobby_01", OOD_DOMAIN_SOCIAL, "hobby_friendship",
          "Picking a tennis hitting partner",
          "A club tennis player deciding which of two mid-level partners to book regular courts with.",
          "weekly court hours", "hours", "hours", (1.0, 8.0),
          "acquaintance", "acquaintances", "test",
          extras={"friendship_context": "Weekly court time is observable; committed partners must match hours."}),
    Scene("d4_diaspora_01", OOD_DOMAIN_SOCIAL, "community_friendship",
          "Picking a language-exchange buddy",
          "A diaspora young adult picking a regular language-exchange buddy at a monthly community social.",
          "weekly exchange hours", "hours", "hours", (1.0, 6.0),
          "acquaintance", "acquaintances", "test",
          extras={"friendship_context": "Commitment hours are visible; mutual stability requires matching load."}),
)

# -----------------------------------------------------------------------------
# d5 OOD -- career sorting (big fish / small pond). All scenes are split="test".
# extras.wage_anchor sets roughly where the high-pay firm's crowd mean sits.
# -----------------------------------------------------------------------------
_D5_SCENES: Tuple[Scene, ...] = (
    Scene("d5_consulting_01", OOD_DOMAIN_CAREER, "elite_vs_regional_firm",
          "MBB offer vs boutique regional consulting",
          "A fresh MBA with two offers: MBB as a middle-of-pack associate vs a boutique regional firm as a star hire.",
          "firm choice", "USD", "currency", (110000.0, 260000.0),
          "coworker", "coworkers", "test",
          extras={"wage_anchor": 180000.0}),
    Scene("d5_law_01", OOD_DOMAIN_CAREER, "elite_vs_regional_firm",
          "Big-Law associate vs mid-market firm partner track",
          "A law grad with a Big-Law NYC associate offer (bottom of the pack) and a mid-market Chicago partner-track offer.",
          "firm choice", "USD", "currency", (150000.0, 310000.0),
          "coworker", "coworkers", "test",
          extras={"wage_anchor": 225000.0}),
    Scene("d5_tech_01", OOD_DOMAIN_CAREER, "faang_vs_unicorn",
          "FAANG L4 vs unicorn Series-C senior",
          "A senior engineer choosing between an L4 FAANG offer (near the bottom of a high-paid team) and a senior role at a profitable Series-C startup.",
          "firm choice", "USD", "currency", (170000.0, 320000.0),
          "coworker", "coworkers", "test",
          extras={"wage_anchor": 245000.0}),
    Scene("d5_phd_01", OOD_DOMAIN_CAREER, "academic_job_market",
          "R1 assistant prof vs R2 tenured fast-track",
          "A freshly-defended PhD weighing an R1 assistant-professor seat in a stellar department vs an R2 university that offers a fast-track tenured role.",
          "firm choice", "USD", "currency", (90000.0, 170000.0),
          "coworker", "coworkers", "test",
          extras={"wage_anchor": 135000.0}),
    Scene("d5_finance_01", OOD_DOMAIN_CAREER, "hedge_vs_regional_bank",
          "Global hedge-fund analyst vs regional bank AVP",
          "A finance MBA with two offers: global hedge-fund analyst (lowest seat at the pod) vs regional-bank AVP (top of the credit team).",
          "firm choice", "USD", "currency", (140000.0, 280000.0),
          "coworker", "coworkers", "test",
          extras={"wage_anchor": 210000.0}),
    Scene("d5_design_01", OOD_DOMAIN_CAREER, "elite_agency_vs_boutique",
          "Global design agency vs hometown boutique",
          "A designer torn between a junior seat at a global IDEO-like agency and a lead seat at a profitable hometown boutique.",
          "firm choice", "USD", "currency", (70000.0, 180000.0),
          "coworker", "coworkers", "test",
          extras={"wage_anchor": 135000.0}),
    Scene("d5_medicine_01", OOD_DOMAIN_CAREER, "academic_hospital_vs_community",
          "Academic hospital junior attending vs community hospital senior attending",
          "A new-attending physician deciding between an academic hospital (bottom of the clinical-research pack) and a community hospital (top earner on staff).",
          "firm choice", "USD", "currency", (220000.0, 480000.0),
          "coworker", "coworkers", "test",
          extras={"wage_anchor": 350000.0}),
    Scene("d5_edtech_01", OOD_DOMAIN_CAREER, "edtech_vs_k12",
          "Edtech unicorn PM vs district-level edu-director",
          "A mid-career PM weighing an edtech unicorn PM role vs a district-level director-of-curriculum seat in their hometown.",
          "firm choice", "USD", "currency", (95000.0, 210000.0),
          "coworker", "coworkers", "test",
          extras={"wage_anchor": 155000.0}),
)


# -----------------------------------------------------------------------------
# Registry API
# -----------------------------------------------------------------------------

ALL_SCENES: Tuple[Scene, ...] = (*_D1_SCENES, *_D3_SCENES, *_D2_SCENES, *_D4_SCENES, *_D5_SCENES)

_BY_ID: Dict[str, Scene] = {s.scene_id: s for s in ALL_SCENES}


def get_scene(scene_id: str) -> Scene:
    if scene_id not in _BY_ID:
        raise KeyError(f"Unknown scene_id `{scene_id}`.")
    return _BY_ID[scene_id]


def list_scenes(
    domain_key: Optional[str] = None,
    split: Optional[str] = None,
    family: Optional[str] = None,
) -> List[Scene]:
    out = list(ALL_SCENES)
    if domain_key is not None:
        out = [s for s in out if s.domain_key == domain_key]
    if split is not None:
        out = [s for s in out if s.split == split or s.split == "both"]
    if family is not None:
        out = [s for s in out if s.family == family]
    return out


def list_training_scenes() -> List[Scene]:
    """Scenes eligible for SFT / DPO training (d1, d3, d2 with split in {train, both})."""
    eligible = [s for s in ALL_SCENES if s.domain_key in (*MAIN_POSITIONAL_DOMAINS, PLACEBO_DOMAIN)]
    return [s for s in eligible if s.split in ("train", "both")]


def list_test_scenes(include_ood: bool = True) -> List[Scene]:
    """Scenes that may appear in any Eval-* bundle.

    By default, this includes d1/d3/d2 test scenes plus all d4/d5 scenes.
    """
    out = [s for s in ALL_SCENES if s.split in ("test", "both")]
    if not include_ood:
        out = [s for s in out if s.domain_key not in (OOD_DOMAIN_SOCIAL, OOD_DOMAIN_CAREER)]
    return out


def summarize_pool() -> Dict[str, object]:
    counts: Dict[str, Dict[str, int]] = {}
    for s in ALL_SCENES:
        d = counts.setdefault(s.domain_key, {"train": 0, "test": 0, "both": 0, "total": 0})
        d[s.split] = d.get(s.split, 0) + 1
        d["total"] += 1
    return {
        "total_scenes": len(ALL_SCENES),
        "by_domain": counts,
        "families_by_domain": {
            dk: sorted({s.family for s in ALL_SCENES if s.domain_key == dk})
            for dk in sorted({s.domain_key for s in ALL_SCENES})
        },
    }
