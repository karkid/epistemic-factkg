# Project Plan: Epistemic Fact Verification

## Domain

Explainable AI / Fact Verification / Graph Reasoning

## Problem

Existing fact verification systems do not explicitly explain how evidence is obtained, resulting in limited explainability and transparency in AI-based fact verification.

## Approach

Develop an epistemic fact verification framework combining AI2-THOR and AVeriTeC claims with Pramana-inspired reasoning and GNN-based graph reasoning.

## Expected Outcome

An epistemically annotated multimodal claim dataset and a reasoning-aware verification model capable of incorporating different epistemic evidence types and confidence priors during claim verification.

## Risks & Challenges

- Ambiguity in epistemic labeling — some claims may plausibly fit multiple Pramana categories
- Overlap between reasoning categories — the boundary between inference and postulation can be unclear
- Balancing multimodal evidence — AI2-THOR (triples) and AVeriTeC (text) have fundamentally different evidence structures
- Limited generalisation from simulation environments — perception-grounded claims from AI2-THOR may not transfer well to real-world settings

## Success Metrics

- Verification accuracy improvement (epistemic-aware vs. baseline)
- Confidence calibration quality (ECE, reliability diagrams)
- Explainability evaluation (qualitative + quantitative)
- Impact of epistemic priors on reasoning performance (ablation study)

---

## Phase 1 — Literature Review & Problem Definition

**Duration:** 1–2 weeks

**Tasks:**
- Study existing fact verification approaches (FEVER, AVeriTeC, FactKG, explainable AI, GNN reasoning)
- Finalise research gap
- Define epistemic categories and heuristic labeling rules

**Deliverables:**
- Literature summary
- Research objectives
- Initial framework design

---

## Phase 2 — Dataset Construction

**Duration:** 2–3 weeks

**Tasks:**
- Generate AI2-THOR perception claims (supported and refuted)
- Extract and preprocess AVeriTeC claims
- Normalise dataset to unified schema v2.0

**Deliverables:**
- `data/raw/ai2thor/claims_all.jsonl` — AI2-THOR raw claims
- `data/raw/averitec/{train,dev,test}.json` — AVeriTeC splits
- `data/schema/unified_schema.json` — schema definition

---

## Phase 3 — Epistemic Annotation Framework

**Duration:** 1–2 weeks

**Tasks:**
- Design heuristic labeling rules per data source
- Assign Pramana-inspired categories using adapter-level `infer_pramana()` logic
- Define heuristic epistemic confidence priors
- Validate annotation consistency

**Deliverables:**
- `data/processed/` — unified JSONL files with epistemic annotations
- [ADR-007](adr/007-heuristic-epistemic-labeling.md) — labeling strategy
- [ADR-008](adr/008-heuristic-prior-weight-values.md) — prior weight values

---

## Phase 4 — Graph Construction & Model Development

**Duration:** 3–4 weeks

**Tasks:**
- Construct claim-evidence graphs using unified JSONL output
- Define GNN node types (ClaimNode, EvidenceNode, EpistemicNode) and edge relations
- Implement GNN architecture
- Integrate epistemic confidence weights as graph edge priors

**Deliverables:**
- Graph construction pipeline
- Baseline GNN model
- Epistemic-aware reasoning module

---

## Phase 5 — Training & Experimentation

**Duration:** 2–3 weeks

**Tasks:**
- Train baseline verification model (without epistemic priors)
- Train epistemic-aware model (with Pramana confidence weights)
- Perform ablation studies:
  - Without epistemic reasoning
  - With epistemic reasoning
  - With heuristic priors only
  - With learned priors

**Deliverables:**
- Experimental results
- Accuracy and confidence calibration analysis
- Performance comparison tables

---

## Phase 6 — Evaluation & Analysis

**Duration:** 1–2 weeks

**Tasks:**
- Analyse reasoning-type influence on model decisions
- Evaluate explainability (human evaluation + automatic metrics)
- Study confidence calibration
- Interpret model behaviour per Pramana category

**Deliverables:**
- Findings report
- Error analysis
- Visualisations (per-Pramana performance breakdown)

---

## Phase 7 — Documentation & Final Report

**Duration:** 2–3 weeks

**Tasks:**
- Write methodology section
- Prepare architecture diagrams
- Summarise findings
- Finalise research report or paper

**Deliverables:**
- Final research report / paper draft
- Presentation slides
- Architecture diagrams
