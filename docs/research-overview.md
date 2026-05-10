# Research Overview: Epistemic Fact Verification

## Introduction

Fact verification has emerged as a critical challenge due to the rapid proliferation of misinformation on digital platforms. Most existing systems attempt to classify claims as true or false by gathering evidence and determining the claim type. While these systems perform well on benchmark datasets, their decision-making processes often lack transparency. More importantly, existing approaches do not explicitly model how knowledge is obtained. Whether a claim is verified through perception, inference, or external testimony is typically implicit, even though this distinction is important for understanding reliability and trust.

This work investigates the idea that fact verification should not only determine *what* is true, but also *how we know* it is true. To address this, we introduce an approach that incorporates explicit epistemic reasoning into the verification process, combining both simulated and real-world data sources.

## Inspiration: The Pramana System

This work is inspired by epistemic frameworks in Sanatana Dharma, particularly the **Pramana system**. The Pramana system provides foundational principles for validating knowledge based on personal experience and self-realisation, rather than relying on blind belief. It categorises sources of knowledge into six types:

| Pramana (Sanskrit) | English Term | Role in this work |
|---|---|---|
| Pratyakṣa | Perception | Claims verified via direct sensory data (simulation state, sensor observations) |
| Anupalabdhi | Non-apprehension | Claims verified by confirmed absence of an object or state |
| Upamāna | Comparison | Claims verified through analogy or numerical comparison |
| Śabda | Testimony | Claims verified via reliable textual data or external documentation |
| Anumāna | Inference | Claims verified through multi-hop logic using multiple data points |
| Arthāpatti | Postulation | Claims verified by assuming a necessary fact to explain a circumstance |

These categories provide a principled basis for understanding how different types of evidence contribute to fact verification, motivating our approach to explicitly model epistemic reasoning as a first-class component.

## Related Work

| Approach | Representative Work | Limitation |
|---|---|---|
| Text-based | FEVER, AVeriTeC | Evidence retrieved from text; no explicit modelling of knowledge source type — no distinction between inference and testimony |
| Knowledge Graph | FactKG | Claims represented as triples, reasoning via graph paths; does not distinguish evidence types or model epistemic reliability |
| Explainable / Reasoning | Multi-hop and rationale generation models | Explanations are typically unstructured; don't represent a specific reasoning type or knowledge source |
| Simulation | AI2-THOR-based approaches | Enables direct perception in controlled environments; limited to simulation, no integration of real-world epistemic reasoning |

## Research Gap

All existing approaches address different aspects of fact verification in isolation — they do not explicitly model *how knowledge is obtained* across different source types. This highlights the need for a unified framework that:

1. Integrates epistemic reasoning into the verification process as a structural component
2. Unifies simulated (perception-grounded) and real-world (testimony/inference-grounded) data under a shared epistemic schema
3. Provides principled confidence priors per evidence type, rather than treating all evidence as equivalent

## Approach

In Phase 1, we use the AI2-THOR simulation environment to generate perception-grounded claims in a controlled setting. The simulator allows direct observation of object properties, spatial relations, and interaction states, enabling creation of both supported and refuted claims with transparent perceptual evidence. These claims are tagged as `perception` Pramana, as verification is concluded from direct sensor observations.

In parallel, we incorporate the AVeriTeC dataset to add real-world textual fact verification examples. Both datasets are unified under a single schema using heuristic Pramana labels — see [ADR-007](adr/007-heuristic-epistemic-labeling.md) for the labeling strategy and [ADR-008](adr/008-heuristic-prior-weight-values.md) for the confidence weight rationale.

After constructing the epistemically annotated dataset, we plan to train a GNN-based verification model. Claims, evidence items, and epistemic categories are represented as interconnected graph nodes, enabling the GNN to model relational dependencies during verification. Epistemic confidence weights serve as edge priors in the graph reasoning process — see [ADR-004](adr/004-gnn-unification-at-epistemic-layer.md) for the unification architecture.

## Expected Outcomes

1. **Epistemically annotated dataset** — a unified dataset integrating AI2-THOR perceptual claims and AVeriTeC real-world claims, categorised with heuristic Pramana labels and confidence priors

2. **Reasoning-aware verification model** — a GNN model capable of incorporating different epistemic evidence types during claim verification, using Pramana weights as principled edge priors

3. **Epistemic influence analysis** — insights into whether certain evidence types (perception vs. testimony vs. inference) contribute more reliably to verification decisions, and whether epistemic-aware reasoning improves transparency and confidence calibration

4. **Contribution toward future research** — a reusable framework for explainable AI, graph-based reasoning, and multi-modal fact verification that could be extended to additional datasets and Pramana categories
