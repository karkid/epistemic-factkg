# Epistemic FactKG

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A fact verification dataset and pipeline grounded in Indian Pramana epistemology. Combines AI2-THOR simulation-state claims and AVeriTeC web-evidence claims into a unified schema for training epistemic graph neural networks. Unlike existing systems that treat all evidence as equivalent, this framework assigns each claim an explicit epistemic category — encoding *how knowledge is obtained*, not just whether the claim is true.

## Pramana Framework

Confidence weights derived from the classical Indian knowledge-source taxonomy:

| Pramana | Category | Confidence | Source |
|---|---|---|---|
| Pratyakṣa | `perception` | 0.95 | AI2-THOR simulation state (closed-world ground truth) |
| Śabda | `testimony` | 0.80 | AVeriTeC web text / PDFs |
| Anupalabdhi | `non_apprehension` | 0.75 | Sensor-confirmed absence of object or state |
| Upamāna | `comparison_analogy` | 0.65 | Numeric or analogy-based claims |
| Anumāna | `inference` | 0.55 | Multi-hop or synthesised reasoning |
| Arthāpatti | `postulation_derivation` | 0.40 | Hypothetical derivation |

When multiple Pramana apply, confidence is combined via diminishing returns:
`combined = 1 − Π(1 − wᵢ)` — implemented in `src/core/claims/labels.py:combine_pramana_weights()`.

## Quick Start

```bash
just init      # Install dependencies

just run       # Full pipeline: build → validate → report (logs → runs/<RUN_ID>/)

# Or step by step:
just build     # KG + AI2-THOR claims + convert all datasets to unified JSONL
just validate  # Schema + semantic + Pramana checks
just report    # Dataset report (markdown + charts)
```

## Installation

**Requirements:** Python 3.14+, [uv](https://github.com/astral-sh/uv), [just](https://github.com/casey/just)

```bash
# macOS
brew install uv just

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

Claims are generated from simulation — no external download needed:
```bash
just build   # runs KG generation + claim generation + conversion in sequence
```
Outputs: `out/knowledge_graph.ttl` (RDF graph) and `data/raw/ai2thor/claims_all.jsonl` (raw claims).

## Architecture

**Ports & Adapters** (hexagonal) pattern — see [ADR-003](docs/adr/003-ports-and-adapters-architecture.md):

- `src/core/ports/` — abstract interfaces (`DatasetConverter`, `DatasetValidator`)
- `src/adapters/{dataset}/` — one `converter.py` + `validator.py` per dataset
- Adding a new dataset = implement the two ABCs + add one line to the `CONVERTERS` dict

GNN unification happens at the epistemic layer — see [ADR-004](docs/adr/004-gnn-unification-at-epistemic-layer.md).

## Project Structure

```
epistemic-factkg/
├── configs/                          # Scene + claim generation config
├── data/
│   ├── raw/                          # Source datasets
│   ├── processed/                    # Unified v2.0 JSONL (gitignored, reproducible)
│   └── schema/unified_schema.json    # JSON Schema Draft-07
├── docs/
│   ├── adr/                          # Architecture Decision Records
│   ├── research-overview.md
│   ├── project-plan.md
│   └── data-flow.md
├── src/
│   ├── adapters/                     # One subpackage per dataset
│   ├── core/                         # Domain logic, ABCs, claims, graph types
│   ├── cli/                          # Thin argparse entry points
│   ├── infra/rdf/                    # RDF/TTL I/O, SPARQL engine
│   └── utils/                        # time, io, logger, exceptions
├── tests/
├── Justfile
└── pyproject.toml
```

## Key Commands

```bash
just init      # Install dependencies
just build     # KG + claims + convert all datasets to unified JSONL
just validate  # Schema + semantic + Pramana checks
just report    # Dataset report (markdown + charts)
just test      # ruff format + lint + pytest
just run       # Full pipeline with timestamped logs → runs/<RUN_ID>/
just clean     # Delete all generated outputs (out/, data/processed/, runs/)
```

## Documentation

| Document | Description |
|---|---|
| [Research Overview](docs/research-overview.md) | Problem motivation, Pramana inspiration, related work, research gap |
| [Project Plan](docs/project-plan.md) | Phases 1–7, risks, success metrics |
| [Data Flow](docs/data-flow.md) | Pipeline: raw → KG → claims → unified JSONL → split |
| [ADR-001](docs/adr/001-pramana-epistemic-framework.md) | Why Pramana as epistemic framework |
| [ADR-002](docs/adr/002-unified-schema-v2-null-tolerant.md) | Unified null-tolerant schema design |
| [ADR-003](docs/adr/003-ports-and-adapters-architecture.md) | Hexagonal architecture |
| [ADR-004](docs/adr/004-gnn-unification-at-epistemic-layer.md) | GNN unification strategy |
| [ADR-005](docs/adr/005-anupalabdhi-distinct-from-not-enough-evidence.md) | non_apprehension vs not_enough_evidence |
| [ADR-006](docs/adr/006-diminishing-returns-combination-formula.md) | Multi-Pramana confidence formula |
| [ADR-007](docs/adr/007-heuristic-epistemic-labeling.md) | Heuristic labeling strategy |
| [ADR-008](docs/adr/008-heuristic-prior-weight-values.md) | Confidence weight values and rationale |
| [ADR-009](docs/adr/009-floorplan-based-train-test-split.md) | Floorplan-based split strategy |
| [ADR-010](docs/adr/010-rdf-as-kg-intermediate-format.md) | RDF/Turtle as KG format |

## Testing

```bash
just test
# or separately:
uv run ruff format . && uv run ruff check .
uv run pytest tests/ -v --tb=short
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

Also cite the datasets:

```bibtex
@inproceedings{schlichtkrull2023averitec,
  title     = {AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web},
  author    = {Schlichtkrull, Michael and others},
  booktitle = {NeurIPS Datasets and Benchmarks},
  year      = {2023}
}

@article{kolve2017ai2thor,
  title   = {AI2-THOR: An Interactive 3D Environment for Visual AI},
  author  = {Kolve, Eric and others},
  journal = {arXiv:1712.05474},
  year    = {2017}
}
```

## License

MIT — see [LICENSE](LICENSE).
