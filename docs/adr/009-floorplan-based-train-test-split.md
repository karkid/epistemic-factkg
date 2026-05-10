# ADR-009: Floorplan-Based Train/Dev/Test Split

## Status

Accepted

## Context

AI2-THOR claims must be split into train/dev/test sets for model evaluation. Each claim has a `provenance.context_id` field identifying the floorplan (scene) it came from. Multiple claims share each floorplan (e.g., `FloorPlan15` generates hundreds of claims about objects in that scene).

Two split strategies were considered:

| Strategy | Description | Problem |
|---|---|---|
| **Random by claim** | Shuffle all claims and split by index | Claims from the same scene can appear in both train and test; the model may memorise scene-specific object layouts and achieve inflated test accuracy |
| **By floorplan (scene)** | Assign entire floorplans to one split; all claims from that scene stay together | No scene appears in more than one split — no leakage of scene priors |

## Decision

Split by **`provenance.context_id` (floorplan)**. All claims from a given floorplan go to exactly one split.

Implementation: `src/cli/split_ai2thor.py` supports three modes:
- `pct` (default) — assign floorplans by percentage (80/10/10)
- `counts` — specify exact floorplan counts per split
- `lists` — specify exact floorplan names per split

A manifest file (`ai2thor_splits_manifest.json`) records which floorplans went to which split and the exact claim counts, enabling reproducibility.

The split uses a configurable seed (default: 13) so the same split is produced deterministically for any given input file.

## Consequences

**Positive:**
- No scene-level data leakage between train and test — the model cannot overfit to specific room layouts or object co-occurrences from a scene it saw during training
- Split is deterministic and auditable (manifest records exact floorplan assignments)
- Supports all three modes (pct, counts, explicit lists) for different experimental needs

**Negative:**
- Split sizes are *approximate* — each floorplan has a variable number of claims (some scenes have more objects than others), so the actual claim count per split may differ from the percentage target
- With very few floorplans (< 3), a meaningful three-way split is not possible; the implementation handles this as a degenerate case

**Note on AVeriTeC:**
AVeriTeC is pre-split by the original dataset authors into train/dev/test — no re-splitting is needed or appropriate, since the original split is already public and reproducible.
