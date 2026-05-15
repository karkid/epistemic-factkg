# ADR-012: Dataset Composition and Generation Strategy

## Status

Accepted (revised after pipeline calibration; updated in v3.0 to add synthetic third source — see ADR-022, ADR-023)

## Context

Before Phase 4 (model building), the training dataset composition must be fixed. Three sources are available:

- **AI2THOR** — synthetic, generated on demand, perception-grounded, closed-world. Claim count is fully controllable.
- **AVeriTeC** — real-world, pre-existing, text-based, web evidence. Fixed pool: train=3,068 + dev=500 = **3,568 total** (original ~4,250 estimate was optimistic).
- **Synthetic (LLM/template-based)** — fictional shortcut-breaking claims generated to break the stance→verdict shortcut. See ADR-022 and ADR-023.

Three decisions needed:

1. **Total dataset size** — how many claims are sufficient for a GNN-based research study?
2. **Source split** — what ratio of AI2THOR to AVeriTeC?
3. **AI2THOR generation distribution** — which claim types to generate, and in what quantities?

### Problem with naive generation

If AI2THOR claims are generated without a controlled distribution, `one_hop` and `conjunction` observation claims dominate — both map to `perception`. This creates two problems:

- **Pramana imbalance**: `perception` becomes the second-largest label after `testimony`, while `comparison_analogy` and `inference` from AI2THOR are near-absent
- **Structural confound**: `perception` always co-occurs with populated `claim_triples` and `simulation_state` modality. The GNN cannot determine whether it is learning from the Pramana label or from the data structure

### Problem with equal source split (50/50)

If AI2THOR and AVeriTeC contribute equally (~3,000 each), the model trains on as many synthetic claims as real ones. The practical evaluation target is real-world claims (AVeriTeC). Over-representing synthetic data risks the model learning scene-specific object layout patterns from AI2THOR floorplans that do not transfer.

## Decision

**Total size:** ~6,400 claims — sufficient for GNN training with 5 evidence types and 4 verdict classes, while keeping graph construction tractable.

**Source split: ~28% AI2THOR / ~56% AVeriTeC / ~16% synthetic**

| Source | Count | Rationale |
|---|---|---|
| AI2THOR | ~1,800 | Epistemic anchor: provides high-confidence perception and non_apprehension ground truth; real RDF triples |
| AVeriTeC | 3,568 | Real-world target domain; full dataset (train=3,068 + dev=500) — ceiling hit |
| Synthetic | ~1,000 | Shortcut-breaking: same stance → different verdict based on epistemic reliability (ADR-022) |

**AI2THOR generation targets (10 scenes, per-context counts in `config.yaml`):**

| Claim type | Pramana | Per-context | × 2 (corruption) | × 10 scenes | Net yield | Rationale |
|---|---|---|---|---|---|---|
| `one_hop` | `perception` | 10 | × 2 | 10 | 200 | Corruption doubles output — 10 per context yields 200 not 100 |
| `conjunction` | `perception` | 10 | × 2 | 10 | 200 | — |
| `negation` | `perception` | 10 | × 2 | 10 | 200 | — |
| `absence` | `non_apprehension` | 150 | × 0.5* | 10 | ~750 | *Only 50% are supported (sensor-confirmed absent); rest are corrupted/refuted |
| **Total AI2THOR** | | | | | **~1,350 + ~450 corrupted** | |

**Key calibration note — corruption doubling**: `add_corruption=True` generates one refuted variant for each supported claim. For `one_hop`/`conjunction`/`negation`, both the supported and refuted variants are labeled `perception`. For `absence`, only the supported variant (object genuinely absent) becomes `non_apprehension`; the corrupted variant (object present) becomes `perception`. This was not accounted for in the initial per-context estimates, causing `perception` to overshoot significantly.

**Resulting full dataset Pramana distribution (post-calibration estimates):**

| Pramana | AI2THOR | AVeriTeC | Total | % |
|---|---|---|---|---|
| `testimony` | — | ~2,100 | ~2,100 | ~39% |
| `perception` | ~600* | ~340 | ~940 | ~17% |
| `non_apprehension` | ~750 | — | ~750 | ~14% |
| `inference` | — | ~550–650** | ~600 | ~11% |
| `comparison_analogy` | — | ~627 | ~627 | ~12% |
| `postulation_derivation` | — | — | 0 | excluded (see ADR-011) |

*AI2THOR perception = 600 (200 per type × 3 types, including corruption pairs). AVeriTeC image/video/audio fact-checks contribute ~340 additional perception records — empirically observed, not initially anticipated.

**Inference threshold was relaxed from `n_abstractive >= 2` to `n_abstractive >= 1` (keeping `src_urls >= 2`). Rationale: one abstractive synthesis answer drawn from multiple URLs constitutes genuine inference. The URL guard still prevents single-source lookups from being mislabelled. The original threshold was too strict and excluded too many real inference cases.

**Verdict distribution note:** AVeriTeC's verdict distribution is not uniform (`refuted` is over-represented at ~56%). Class balancing strategy (weighted loss or oversampling) is a separate training decision, not resolved here.

## Consequences

**Positive:**
- All 5 active Pramana types have ≥ 500 samples — confidently learnable for GNN representation learning
- No class dominates above 40% — `testimony` (39%) is the largest but not overwhelming
- `non_apprehension` at ~750 (14%) is the unique AI2THOR contribution — large enough to demonstrate the value of simulation-grounded absence reasoning
- `inference` threshold relaxation is epistemically honest: one synthesised answer from multiple sources is real inference
- AVeriTeC contributes ~66% of the data — model evaluation on real-world claims is not diluted by synthetic data
- AI2THOR generation is deterministic and reproducible (seeded, config-controlled)

**Negative:**
- `inference` relaxation (≥1 abstractive vs ≥2) may admit some single-abstractive records that are closer to testimony with light synthesis — a known trade-off, stated in the paper
- The 34/66 AI2THOR/AVeriTeC split is further from the original 30/70 target than planned — acceptable given AVeriTeC ceiling
- `comparison_analogy` and `non_apprehension` have no AVeriTeC contribution — they are entirely AI2THOR-sourced, making the model dependent on simulation quality for these classes

**Implementation note:**
Generation targets are configured in `configs/config.yaml` under `ai2thor.generation` (per-type counts: `n_one_hop`, `n_conjunction`, `n_negation`, `n_absence`). The `just build` command reads these values automatically; CLI overrides are available. `just filter` applies ADR-011 exclusion. `just validate-training` checks the resulting distribution. The corruption doubling effect must be accounted for when setting `n_one_hop`/`n_conjunction`/`n_negation` — effective perception yield = `n × 2 × n_scenes`.
