# Project Proposal: Relational-State Alignment for Structured Social Decision Making

## Working Title
Relational-State Alignment for Large Language Models via Theory-Grounded Scenario Synthesis and Structured Choice Evaluation

## 1. Executive Summary
This project studies whether large language models can be made to reason and act in ways that reflect relational-state dynamics rather than only surface-level preference cues. We focus on decisions where private needs, social reference points, and practical costs jointly shape behavior. Instead of hard-coding a small set of fixed scenes, we build a theory-guided generation pipeline that preserves latent behavioral structure while allowing a teacher model to instantiate diverse realistic scenarios. We then evaluate downstream models with a unified three-way action-choice benchmark that measures whether their decisions align with the intended relational-state structure across domains, templates, and prompting conditions.

## 2. Motivation
Many real-world decisions are neither purely utility-driven nor reducible to simple preference labels. Spending, effort, self-investment, and path-selection decisions are often shaped by:

- a private baseline of what would feel sufficient in isolation,
- a social reference environment that changes what feels normal or acceptable,
- and a practical cost term that constrains escalation.

Current LLM alignment and behavior evaluation pipelines often miss this structure. They either use simple helpfulness/safety settings, or they rely on narrow scenario collections that encourage memorization of scene-specific heuristics. Our project aims to model a richer decision process in which relative standing and reference dependence matter, and to test whether prompting or alignment methods can recover those structured behaviors.

## 3. Core Research Question
Can we induce and evaluate relational-state-consistent decision behavior in large language models across heterogeneous decision domains, while preserving realistic scene diversity and scale fidelity?

## 4. Main Hypotheses

### H1: Structural Learnability
LLMs can be guided to produce decisions that track relational-state variables such as private baseline, social reference pressure, and cost pressure.

### H2: Cross-Scene Generalization
If the data pipeline captures domain-level structure rather than a fixed inventory of scenarios, then models can generalize relational-state behavior across previously unseen scene instantiations within the same domain.

### H3: Prompting / Alignment Sensitivity
Different prompting conditions will produce measurably different levels of relational-state consistency, revealing which interventions improve structured social reasoning rather than merely changing response style.

## 5. Scope of the Project
The project currently covers four decision domains:

1. positional consumption
2. non-positional investment
3. effort and human capital
4. market sorting and matching

Each domain is represented by abstract templates rather than a fixed list of hard-coded scenes. Templates define:

- action mode,
- unit and value scale,
- baseline/cost/reference semantics,
- scale constraints,
- allowed and forbidden scene families,
- and domain-appropriate social/friction cues.

This makes the benchmark more realistic and more robust to memorization, while retaining a clean latent structure for evaluation.

## 6. Method Overview

### 6.1 Theory-Grounded Latent Sampling
For each sample we draw:

- `alpha`: sensitivity to social/reference pressure,
- `gamma`: intrinsic pull or internal drive,
- `F`: private baseline level,
- `M`: reference multiplier,
- `R`: social reference point,
- `c`: practical cost term,
- `x_star`: latent target action.

The sampler also computes a normalized action target used for downstream evaluation.

### 6.2 Structured Record Construction
The structured dataset builder combines:

- domain/template configuration,
- latent samples,
- narrative semantics,
- prompt-ready cues,
- and unified evaluation targets.

Each structured record includes a natural-language scaffold, metadata about the latent decision structure, and a three-way evaluation target:

- `A`: stay close to the private baseline,
- `B`: make a moderate upward move,
- `C`: make a strong stretch toward the socially salient level.

This provides a common action interface across domains with different units and scales.

### 6.3 Teacher-Model Scenario Instantiation
The teacher model receives:

- the domain and template,
- numeric anchors for baseline/reference/target,
- scale guardrails,
- allowed and forbidden scene families,
- and social/cost cues.

Its job is to instantiate a realistic scene that matches the latent scale. The scene generation stage is deliberately separated from the action-evaluation stage so that the generated scenario remains natural and does not leak the correct label.

### 6.4 Benchmark Bundle Construction
After scenario generation, a benchmark builder merges structured records with generated scenes and creates a unified action-choice benchmark bundle. Each benchmark item contains a single prompt payload aligned with the current evaluation task, with optional in-context examples when few-shot mode is explicitly requested at build time.

Each benchmark item contains:

- the scenario,
- A/B/C action options,
- the gold choice,
- metadata about domain/template/sample type,
- and prompt payloads for each evaluation condition.

## 7. Why We Use A/B/C Instead of Raw Continuous Actions
The domains in this project differ in meaning, unit, and numerical scale. Some decisions are measured in USD, some in hours, and some are discrete path choices. Asking the model to output a raw numeric target would produce unstable parsing and poor comparability. A single binary label would be too coarse and would collapse moderate and strong escalation into one category.

The three-way action design is a practical compromise:

- it preserves intensity information,
- it can be applied to both continuous and discrete domains,
- it is easy to parse,
- and it supports both exact-match and ordinal evaluation.

## 8. Evaluation Design

### 8.1 Main Task
Given one scenario and three action options, the model must output exactly one of:

- `Choice: A`
- `Choice: B`
- `Choice: C`

The model may provide short reasoning, but the output format is tightly controlled for reliable parsing.

### 8.2 Prompt Conditions
We currently support four prompt conditions per benchmark family:

- persona-style structural reasoning,
- action-selection framing without explicit latent notation,
- OOD structural reasoning,
- OOD action selection with stricter evidence grounding.

The benchmark construction pipeline produces these prompt variants automatically.

### 8.3 Metrics
Primary metrics:

- exact-match accuracy,
- accuracy by prompt key,
- accuracy by sample type,
- accuracy by domain,
- accuracy by template.

Secondary metrics:

- within-one accuracy for ordinal proximity,
- mean absolute error over A/B/C indices.

These metrics allow us to distinguish severe structural failures from mild near-misses.

## 9. Expected Contributions

### Contribution 1: A Structured Social-Decision Data Pipeline
We provide a reusable pipeline for generating theory-constrained yet scene-diverse decision scenarios.

### Contribution 2: A Unified Cross-Domain Action Benchmark
We introduce a common A/B/C action interface that bridges heterogeneous domains without forcing all tasks into a single numeric regression format.

### Contribution 3: A Prompt-Sensitive Evaluation Framework
We provide a benchmark that can compare multiple prompting or alignment conditions under a shared latent decision structure.

### Contribution 4: A Generalization Test for Relational-State Reasoning
Because scenes are instantiated from abstract templates rather than fixed archetypes, the benchmark can test structural generalization within a domain.

## 10. Current Implementation Status
The current repository already supports the core pipeline:

1. sample theory-grounded latent records,
2. build structured scenarios with template guardrails,
3. generate natural scenarios with a teacher model,
4. validate generated scenario quality,
5. build benchmark bundles,
6. evaluate downstream models with A/B/C action outputs.

In particular, the repository now includes:

- structured evaluation targets in `build_structured_dataset.py`,
- benchmark bundle construction in `evaluation/build_action_benchmark.py`,
- A/B/C choice parsing and summary metrics in `evaluation/eval_persona.py`,
- and end-to-end automation through `run_llm_data_generation.sh`.

## 11. Risks and Open Issues

### Risk 1: Scene-Scale Mismatch
Teacher-generated scenes may still occasionally mismatch numeric anchors. This risk is partially controlled by template guardrails, but could be reduced further with a dedicated scene-scale validator.

### Risk 2: Prompt Leakage
Structured scenario scaffolds may still expose too much latent information if the generated scenes are not sufficiently natural. This should be monitored through manual review and validation.

### Risk 3: Weak Few-Shot Diversity
When few-shot examples are enabled, the current builder selects examples by domain proximity and fallback order. This is functional but could be improved with more principled coverage selection.

### Risk 4: Evaluation Granularity
The current three-way action mapping is robust and simple, but future versions may explore alternative gold-label mappings based on relative uplift from the private baseline rather than only normalized target bins.

## 12. Near-Term Next Steps

1. Add a scene-scale consistency validator for generated scenarios.
2. Improve few-shot example selection when ICL is enabled, so template coverage is more diverse.
3. Run pilot evaluations on a small set of base and aligned models.
4. Compare prompt conditions on exact-match and ordinal metrics.
5. Analyze which domains and templates remain most difficult.

## 13. Deliverables

- a domain-template-based structured scenario dataset,
- teacher-generated scenario corpora,
- a unified action-choice benchmark bundle with optional few-shot augmentation,
- A/B/C action evaluation scripts and summaries,
- and an empirical study of relational-state-consistent decision behavior in LLMs.

## 14. Bottom Line
This project aims to move beyond narrow scenario engineering and toward a more principled framework for studying social-reference-sensitive decision behavior in language models. By separating latent theory, scenario instantiation, and action evaluation, the pipeline provides both realism and analytical control. The resulting benchmark can support prompt-based studies now and alignment-method comparisons later.
