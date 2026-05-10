# ADR-004: GNN Unification at the Epistemic Layer

## Status

Accepted

## Context

The two data sources produce structurally incompatible evidence representations:

- **AI2-THOR** produces **EntityNodes** connected by predicate edges (a knowledge graph native format). Evidence is a set of triples. Reasoning is explicit and typed.
- **AVeriTeC** produces **TextNodes** — sentence-level evidence from web documents. There are no triples. Reasoning is implicit.

A GNN model must process both. Options:

1. **Unify at the data layer** — force both sources into the same node representation before the GNN. This means either flattening AI2-THOR triples to text (losing graph structure) or running NLP extraction on AVeriTeC text to get triples (noisy, future work)
2. **Separate GNN heads per source** — one sub-model for AI2-THOR, one for AVeriTeC; combine predictions at inference time. Doubles model complexity and prevents cross-source learning
3. **Unify at the epistemic layer** — let source-specific encoders process their native formats up to an intermediate `EpistemicNode` embedding; share all downstream components from that point

## Decision

**Unify at the epistemic layer** via `pramana_primary`:

```
AI2-THOR:  EntityNodes + predicate edges  →  EpistemicNode (perception / non_apprehension)
AVeriTeC:  TextNode (evidence sentence)  →  EpistemicNode (testimony / inference)
                                                    ↓
                                        Shared verdict classifier
```

The `pramana_primary` field in every unified record declares the EpistemicNode type. After the epistemic embedding layer, source identity (`provenance.dataset`) stops mattering to the model. The verdict classifier operates on a source-agnostic epistemic representation.

## Consequences

**Positive:**
- Source-specific encoders can be optimised for their native format (graph vs. text) without affecting shared downstream components
- Future datasets only need to produce a valid `pramana_primary` to participate in the shared classifier
- The Pramana confidence weight becomes a natural edge prior for the EpistemicNode → ClaimNode edge

**Negative:**
- The epistemic embedding layer must bridge very different input types — this is a non-trivial modelling challenge
- If source-specific patterns are lost at the EpistemicNode boundary, cross-source generalisation may not improve over source-specific baselines
- The independence assumption (both sources contribute independently to the same epistemic claim) may not hold for all claims

**Future work:**
Phase 5 ablation will measure whether shared-classifier performance exceeds source-specific baselines, validating this unification choice.
