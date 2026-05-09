# Epistemic FactKG

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A fact verification dataset and pipeline grounded in Indian Pramana epistemology. Combines AI2-THOR simulation-state claims and AVeriTeC web-evidence claims into a single unified schema for training epistemic graph neural networks.

## Overview

Traditional fact-checking treats all evidence as equivalent. This project assigns each claim an **epistemic category** from the Pramana system — the classical Indian framework for valid knowledge sources — giving GNN training a principled confidence prior per evidence type.

| Pramana | Category | Confidence | Used by |
|---|---|---|---|
| Pratyakṣa | `perception` | 0.90 | AI2-THOR simulation state |
| Śabda | `testimony` | 0.85 | AVeriTeC web text / PDFs |
| Anupalabdhi | `non_apprehension` | 0.80 | Absence of evidence |
| Upamāna | `comparison_analogy` | 0.75 | Numeric claims |
| Anumāna | `inference` | 0.70 | Multi-source synthesis |
| Arthāpatti | `postulation_derivation` | 0.60 | Implicit derivation |

## Quick Start

```bash
# Install dependencies
just init

# Full pipeline: build KG → generate claims → convert → validate → split
just pipeline-all

# Or step by step:
just build-kg          # RDF knowledge graph from AI2-THOR
just build-claims      # Generate claims from KG
just convert-unified   # Convert all datasets to v2.0 JSONL
just validate-unified  # Validate outputs
just split-ai2thor     # Train/dev/test split by floorplan
```

## Installation

**Requirements:** Python 3.14+, [uv](https://github.com/astral-sh/uv), [just](https://github.com/casey/just)

```bash
# macOS
brew install uv just

# then:
git clone <repo-url> && cd epistemic-factkg
just init
```

## Data Setup

### AVeriTeC

Download from [fever.ai/task.html](https://fever.ai/task.html) and place under `data/raw/averitec/`:
```
data/raw/averitec/
├── train.json
├── dev.json
└── test.json
```

### AI2-THOR

Claims are generated from the simulation — no external download needed:
```bash
just build-kg       # builds out/knowledge_graph.ttl
just build-claims   # writes data/raw/ai2thor/claims_all.jsonl
```

## Unified Schema v2.0

All outputs conform to a single JSON schema (`data/schema/unified_schema.json`). Example record:

```json
{
  "schema_version": "2.0",
  "id": "claim-FloorPlan15-onehop-sup-000000",
  "claim": "The stove knob is at room temperature.",
  "verdict": {
    "label": "supported",
    "justification": "Sensor shows temperature=RoomTemp, matching claim."
  },
  "epistemic": {
    "pramana_primary": "perception",
    "pramana_all": ["perception"],
    "confidence_weight": 0.9,
    "assignment_method": "rule_based"
  },
  "claim_triples": [
    ["http://epistemicfactkg.org/entities/StoveKnob|-03.60|+01.11|+02.02",
     "temperature", "RoomTemp"]
  ],
  "reasoning": {"structural": "one_hop", "strategy": "direct_observation"},
  "evidence": [{
    "evidence_id": "claim-FloorPlan15-onehop-sup-000000-e0",
    "text": "The stove knob is at room temperature.",
    "triples": [["http://epistemicfactkg.org/entities/StoveKnob|-03.60|+01.11|+02.02",
                 "temperature", "RoomTemp"]],
    "triple_source": "ground_truth",
    "modality": "simulation_state",
    "stance": "supports",
    "source_url": null
  }],
  "provenance": {"dataset": "ai2thor", "split": null, "context_id": "FloorPlan15"},
  "meta": {"schema_version": "2.0", "created_utc": "2026-02-11T18:19:16Z"}
}
```

Key fields per dataset:

| Field | AI2-THOR | AVeriTeC |
|---|---|---|
| `claim_triples` | populated (graph triples) | `null` |
| `reasoning` | populated (`one_hop`, `conjunction`, …) | `null` |
| `evidence[].modality` | `simulation_state` | `web_text`, `pdf`, … |
| `evidence[].triple_source` | `ground_truth` | `null` |
| `epistemic.pramana_primary` | `perception` / `non_apprehension` | `testimony` / `inference` / … |

## Project Structure

```
epistemic-factkg/
├── configs/
│   └── ai2thor_default.yaml      # Scene + claim generation config
├── data/
│   ├── raw/                      # Source datasets (not generated)
│   │   ├── ai2thor/claims_all.jsonl
│   │   └── averitec/{train,dev,test}.json
│   ├── processed/                # Unified v2.0 JSONL (gitignored, reproducible)
│   └── schema/
│       ├── unified_schema.json   # JSON Schema (Draft-07)
│       └── unified_example.json  # 3 annotated example records
├── src/
│   ├── adapters/                 # One subpackage per dataset
│   │   ├── ai2thor/
│   │   │   ├── converter.py      # AI2ThorConverter (DatasetConverter)
│   │   │   ├── validator.py      # AI2ThorValidator (DatasetValidator)
│   │   │   ├── data_source.py    # AI2-THOR scene graph reader
│   │   │   └── …                 # config, NLG, ontology, registry, semantics
│   │   └── averitec/
│   │       ├── converter.py      # AveritecConverter (DatasetConverter)
│   │       └── validator.py      # AveritecValidator (DatasetValidator)
│   ├── core/
│   │   ├── claims/
│   │   │   ├── labels.py         # PramanaLabel, ReasoningLabels, CONFIDENCE_WEIGHTS
│   │   │   ├── claim_schema.py   # CLAIM_SCHEMA (v2.0 inline)
│   │   │   ├── claim_validator.py
│   │   │   ├── claim_generator.py
│   │   │   └── types.py          # ClaimInstance, ClaimCorpus, Evidence, …
│   │   ├── graph/types.py        # Triple, TripleList
│   │   ├── ports/dataset/
│   │   │   ├── converter.py      # DatasetConverter ABC
│   │   │   └── validator.py      # DatasetValidator ABC
│   │   ├── ontology/             # Base ontology + mappings
│   │   ├── nlg/                  # Natural language generation
│   │   └── registry/             # Entity + relation registries
│   ├── pipelines/
│   │   ├── convert_to_unified.py # Entry point: CONVERTERS dispatch dict
│   │   ├── build_claims.py       # AI2-THOR claim generation pipeline
│   │   └── build_rdf.py          # RDF graph construction pipeline
│   ├── cli/                      # Thin argparse wrappers (one per command)
│   │   ├── convert_to_unified.py
│   │   ├── validate_unified_dataset.py
│   │   ├── split_ai2thor.py
│   │   ├── build_claims.py
│   │   ├── build_rdf.py
│   │   ├── build_viz.py
│   │   └── …
│   ├── infra/rdf/                # RDF/TTL I/O, SPARQL engine
│   ├── visualizer/               # Interactive HTML graph viewer
│   └── utils/                    # time, io, logger, exceptions
├── tests/
├── Justfile                      # Task automation (just --list)
└── pyproject.toml
```

## Key Commands

```bash
# Development
just dev       # ruff format + lint
just test      # pytest

# Knowledge graph
just build-kg  # out/knowledge_graph.ttl
just viz-kg    # out/visualizer/knowledge_graph.html
just open-viz

# Dataset pipeline
just convert-unified          # all datasets → data/processed/
just validate-unified         # schema + logic checks
just split-ai2thor            # 80/10/10 pct split
just split-ai2thor-counts n_train=6 n_dev=1 n_test=1
just validate-ai2thor-split

# Full pipelines (timestamped logs → runs/<RUN_ID>/)
just pipeline-all             # end-to-end
just pipeline-data            # convert + validate only
just pipeline-split           # split + validate only

# Analysis
just analyze-averitec         # raw data profile
just report RUN_ID=<id>       # dataset report (md + plots)
```

## Architecture

The project follows a **Ports & Adapters** (hexagonal) pattern:

- **`src/core/ports/`** defines abstract interfaces (`DatasetConverter`, `DatasetValidator`).
- **`src/adapters/{dataset}/`** implements them — one `converter.py` + `validator.py` per dataset.
- **`src/pipelines/convert_to_unified.py`** registers all converters in a `CONVERTERS` dict and dispatches by dataset name.
- Adding a new dataset means implementing the two ABCs and adding one line to `CONVERTERS` — no core code changes.

## Adding a New Dataset

1. Create `src/adapters/<name>/converter.py` implementing `DatasetConverter`:
   - `dataset_name` → short lowercase string
   - `infer_pramana(raw)` → `(primary, all_labels, confidence_weight)`
   - `convert_one(raw, rec_id)` → unified v2.0 dict

2. Create `src/adapters/<name>/validator.py` implementing `DatasetValidator`.

3. Register in `src/pipelines/convert_to_unified.py`:
   ```python
   from src.adapters.<name>.converter import MyConverter
   CONVERTERS["<name>"] = MyConverter()
   ```

4. Register validator in `src/cli/validate_unified_dataset.py`:
   ```python
   from src.adapters.<name>.validator import MyValidator
   _DATASET_VALIDATORS["<name>"] = MyValidator()
   ```

## Configuration

Edit `configs/ai2thor_default.yaml` to control scene types, object randomization, and claim generation parameters.

## Dependencies

Key packages: `ai2thor`, `rdflib`, `pyvis`, `jsonschema`, `sentence-transformers`. See [pyproject.toml](pyproject.toml) for the full list.

## Testing

```bash
just test
# or
uv run pytest tests/ -v --cov=src --cov-report=html
```

## Citation

If you use this work in research, please cite:

```bibtex
@software{epistemic_factkg_2026,
  title  = {Epistemic FactKG: Pramana-grounded Fact Verification},
  author = {Karki, Dheeraj},
  year   = {2026},
  url    = {https://github.com/yourusername/epistemic-factkg}
}
```

Also cite the datasets used:

```bibtex
@inproceedings{schlichtkrull2023averitec,
  title  = {AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web},
  author = {Schlichtkrull, Michael and others},
  booktitle = {NeurIPS Datasets and Benchmarks},
  year   = {2023}
}

@article{kolve2017ai2thor,
  title  = {AI2-THOR: An Interactive 3D Environment for Visual AI},
  author = {Kolve, Eric and others},
  journal = {arXiv:1712.05474},
  year   = {2017}
}
```

## License

MIT — see [LICENSE](LICENSE).
