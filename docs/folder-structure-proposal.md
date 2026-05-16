# Proposed Folder Structure — Separation of Concerns

Current state works but has room for clarity. This proposal groups code by responsibility layer and data flow stage.

---

## Current Issues

1. **`src/core/gnn/`** has 10+ files at one level — hard to see which are architecture (encoder, heads) vs. training infrastructure (train, dataset) vs. config (config, types)
2. **`src/cli/`** mixes data pipeline (generate, build, filter), model pipeline (train, eval), and utilities (validate, report) with no subdivision
3. **Naming issue:** `gnn/` is technology-specific (what if V2 isn't a GNN?). Should be named by purpose, like `model/`.
4. **Adapters** own their strategy maps locally (good!) but the logic differs per adapter — no shared abstraction
5. **`src/infra/`** houses RDF but is unclear if it's production critical or research-only
6. Hard to answer "which files do I touch to add V2 node types?" or "where's the training logic?"

---

## Proposed Structure

```
src/
├── epistemic/                          # ← Core epistemic framework (never changes)
│   ├── __init__.py
│   ├── enums.py                        # Verdict, EvidenceStance, EvidenceType, ReasoningStrategy
│   ├── formula.py                      # EC formula + aggregation (combine_evidence_weights, compute_evidence_confidence, aggregate_scores, derive_verdict)
│   ├── registry.py                     # Source trust registry loading + resolution
│   ├── schema.py                       # v3.0 unified record shape (Pydantic if used)
│   └── validator.py                    # Record validation against schema
│
├── adapters/                           # Data source → v3.0 JSONL converters
│   ├── __init__.py
│   ├── base.py                         # Abstract converter + strategy mapping interface (new)
│   ├── ai2thor/
│   │   ├── __init__.py
│   │   ├── converter.py                # Main entry point; includes local _STRATEGY_MAP
│   │   ├── template.py
│   │   ├── validator.py
│   │   ├── data_source.py
│   │   ├── ontology.py
│   │   ├── registry.py
│   │   ├── config_loader.py
│   │   ├── scene_randomizer.py
│   │   ├── ids/
│   │   │   ├── __init__.py
│   │   │   ├── object_types.py
│   │   │   └── predicates.py
│   │   └── semantics/
│   │       ├── __init__.py
│   │       ├── entity_lexicon.py
│   │       ├── predicate_lexicon.py
│   │       └── semantic_rules.py
│   ├── averitec/
│   │   ├── __init__.py
│   │   ├── converter.py                # Includes local _STRATEGY_MAP + _STRATEGY_PRIORITY
│   │   └── validator.py
│   └── synthetic/
│       ├── __init__.py
│       ├── fictional_generator.py      # Includes local _STRATEGY_MAP (15 templates)
│       ├── validator.py
│       └── client/
│           ├── __init__.py
│           ├── base.py
│           ├── local_client.py
│           ├── grounded_client.py
│           └── llm/
│               ├── __init__.py
│               ├── llm_client.py
│               └── prompt_builder.py
│
├── pipeline/                           # Data and model pipelines (orchestration)
│   ├── __init__.py
│   ├── data/                           # Data pipeline stages (CLI entry points)
│   │   ├── __init__.py
│   │   ├── generate_synthetic.py       # Generate synthetic data
│   │   ├── build_claims.py             # AI2THOR simulator claims
│   │   ├── convert_to_unified.py       # Run all adapters → unified JSONL
│   │   ├── filter_training.py          # Remove non-training types
│   │   ├── split_dataset.py            # Train/val/test split
│   │   └── validate.py                 # Validate each stage
│   │
│   ├── model/                          # Model pipeline stages (CLI entry points)
│   │   ├── __init__.py
│   │   ├── build_graphs.py             # Build HeteroData from JSONL
│   │   ├── train.py                    # Train EpistemicHGNN
│   │   ├── evaluate.py                 # Evaluate on test split
│   │   └── report.py                   # Generate summary statistics
│   │
│   └── reporting.py                    # Shared reporting utilities (validate output dir, write JSON, etc.)
│
├── model/                              # Neuro-symbolic fact verification (architecture-agnostic)
│   ├── __init__.py
│   │
│   ├── config.py                       # GraphConfig (single source of truth)
│   │
│   ├── data/                           # Data preparation
│   │   ├── __init__.py
│   │   ├── featurizer.py               # Text → embeddings + categorical → one-hot
│   │   ├── builder.py                  # v3.0 record → HeteroData (was graph_builder.py)
│   │   ├── dataset.py                  # PyG InMemoryDataset (optional, currently unused)
│   │   └── types.py                    # Node/edge/verdict type constants + ClaimGraph dataclass
│   │
│   ├── architecture/                   # Model components (V1: HeteroConv-based; never trained separately)
│   │   ├── __init__.py
│   │   ├── encoder.py                  # Shared HeteroConv encoder
│   │   ├── heads.py                    # StanceHead (H1), ISHead (H2), VerdictHead
│   │   └── aggregator.py               # Stateless EC formula (SymbolicAggregator)
│   │
│   ├── epistemichgnn.py                # EpistemicHGNN — assembles all components (was model.py)
│   │
│   ├── training/                       # Training logic
│   │   ├── __init__.py
│   │   ├── config.py                   # TrainConfig (hyperparameters)
│   │   └── trainer.py                  # Trainer class (loss computation, epoch loop, checkpoints)
│   │
│   ├── evaluation/                     # Evaluation logic
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Pure metric functions (accuracy, F1, RMSE, etc.)
│   │   └── inference.py                # Single-graph inference wrapper (single predict call)
│   │
│   └── v1/                             # ← V1-specific code (future: v2/ directory for new arch)
│       ├── __init__.py
│       └── schema.md                   # V1 architecture doc (node dims, edge types, feature composition)
│
├── nlg/                                # Natural language generation (AI2THOR-specific)
│   ├── __init__.py
│   ├── triple_realizer.py              # Convert (s, p, o) to English sentence
│   └── sentence_template.py
│
├── ontology/                           # Ontologies and lexicons (AI2THOR-specific)
│   ├── __init__.py
│   ├── core.py                         # Core ontology classes
│   ├── registry.py                     # Global object/predicate registry
│   └── semantics/
│       ├── __init__.py
│       ├── entity_lexicon.py
│       ├── predicate_lexicon.py
│       └── formatter.py
│
├── infra/                              # Infrastructure (research-only, optional)
│   ├── __init__.py
│   ├── rdf/
│   │   ├── __init__.py
│   │   ├── builder.py                  # RDF graph assembly
│   │   ├── engine.py                   # SPARQL query engine
│   │   ├── formatter.py                # RDF formatting
│   │   ├── ttl.py                      # Turtle serialization
│   │   ├── namespaces.py
│   │   └── result.py
│   └── graph/
│       ├── __init__.py
│       └── types.py                    # RDF graph type definitions
│
├── utils/                              # Utilities (no domain logic)
│   ├── __init__.py
│   ├── io.py                           # JSONL read/write helpers
│   ├── logger.py                       # Logging setup
│   ├── exceptions.py                   # Project exception classes
│   ├── time.py                         # ISO 8601 timestamp utilities
│   └── config_loader.py                # YAML config loading
│
└── ports/                              # Abstract interfaces for adapters (optional)
    ├── __init__.py
    └── converter.py                    # DatasetConverter, DatasetValidator protocols
```

---

## What Changed and Why

### 1. `src/epistemic/` — New top-level layer

**Why:** Everything else depends on epistemic definitions (enums, EC formula, source trust). By extracting this into its own package, we make it crystal clear what the ground truth is.

**Files moved:**
- `src/core/claims/labels.py` → split into:
  - `src/epistemic/enums.py` (Verdict, EvidenceStance, EvidenceType, ReasoningStrategy, ClaimStructure)
  - `src/epistemic/formula.py` (EC formula functions: combine_evidence_weights, compute_evidence_confidence, aggregate_scores, derive_verdict)
  - `src/epistemic/registry.py` (source trust registry functions)
- `src/core/claims/claim_schema.py` → `src/epistemic/schema.py`
- `src/core/claims/claim_validator.py` → `src/epistemic/validator.py`

**Benefits:**
- Single source of truth for epistemic math — no one should re-implement the EC formula
- Adapters and GNN both import from `epistemic`, not `core.claims`
- Clear that epistemic definitions never change (V1, V2, V10 all use the same enums)

---

### 2. `src/pipeline/` — New orchestration layer

**Why:** Currently CLI scripts are scattered. By grouping them, we make the data flow explicit: `pipeline/data/` → stage 1/2/3/4/5, then `pipeline/model/` → train/eval.

**Subdirectories:**
- `data/` — all CLI entry points for data prep (generate, build, filter, split, validate)
- `model/` — all CLI entry points for model (build_graphs, train, evaluate, report)
- `reporting.py` — shared utilities for all pipelines (ensure output dirs exist, JSON write patterns, etc.)

**Benefits:**
- New user immediately sees `pipeline/data/` and knows "run these first"
- Easier to add new pipeline stages (e.g. `data/augment.py`)
- Model pipeline code can import from `model/`, data pipeline code imports from `adapters/` + `epistemic/`

---

### 3. `src/model/` — Reorganised into subpackages by responsibility

**Old structure:** 10 files at one level in `src/core/gnn/` (`config.py`, `encoder.py`, `heads.py`, `aggregator.py`, `model.py`, `train.py`, `featurizer.py`, `graph_builder.py`, `dataset.py`, `types.py`, `metrics.py`).

**New structure:**
- `data/` — everything for converting JSONL → PyG graphs (featurizer, builder, types, dataset)
- `architecture/` — model components (encoder, heads, aggregator) that are never trained/updated independently
- `training/` — training logic (config, trainer)
- `evaluation/` — evaluation logic (metrics, inference wrapper)
- `v1/` — V1-specific docs; future `v2/` for V2 architecture if needed
- `epistemichgnn.py` — top-level module assembler (was `model.py`)

**Why:**
- Clear separation: "data prep" (featurizer, builder) vs. "architecture" (encoder, heads) vs. "training" (trainer.py) vs. "evaluation" (metrics)
- Purpose-driven naming: `model/` describes what we do (verify claims), not how (GNN). Future-proof if we switch architectures.
- Easy to find where to add a new evaluation metric (`model/evaluation/metrics.py`) or a new head (`model/architecture/heads.py`)
- Extensibility: V2 architecture adds `model/v2/config.py` and adjusts imports; nothing breaks

**New interdependencies:**
- `model/data/types.py` → `epistemic/enums.py` (verdict, stance mappings)
- `model/architecture/encoder.py` → `model/config.py` (reads config, no hardcoded names)
- `model/epistemichgnn.py` → all architecture components
- `model/training/trainer.py` → `model/architecture/heads.py`, `model/data/types.py` (for loss, no forward pass logic)
- `model/evaluation/metrics.py` → pure torch (no model-specific imports)

---

### 4. `src/adapters/base.py` — New abstract base (optional)

**Why:** Each adapter has `converter.py` with a local `_STRATEGY_MAP` and `_to_strategy()` function. By defining a `StrategyMapper` interface in `base.py`, we make it explicit that all adapters follow the same pattern.

```python
# src/adapters/base.py
class StrategyMapper(ABC):
    @abstractmethod
    def map_strategy(self, raw_value: Any) -> str:
        """Map source-specific strategy value to canonical ReasoningStrategy."""
        pass

# src/adapters/ai2thor/converter.py
class AI2ThorStrategyMapper(StrategyMapper):
    def map_strategy(self, raw: str | None) -> str:
        return _STRATEGY_MAP.get(raw or "", ...)
```

**Benefits:**
- Documents that this is a required step for any new adapter
- Makes it easy to add a debug utility like `print_all_strategy_mappings()` that validates all adapters

---

### 5. `src/ontology/` — Extracted from `ai2thor/`

**Why:** AI2THOR ontology (object types, predicates, lexicons) is specific to AI2THOR but lives in the adapter. By moving it to a sibling package, we make it clear it's independent infrastructure, not part of the conversion logic.

**Who imports it:**
- `adapters/ai2thor/converter.py` — to resolve object IDs to entity names
- `nlg/` — to generate natural language
- Optional future uses: Knowledge graph export, visualization, etc.

---

### 6. `src/nlg/` — Extracted from `ai2thor/`

**Why:** NLG (triple → English) is AI2THOR-specific but could be reused elsewhere. By separating it, we document that.

---

### 7. `src/infra/` — Marked as research-only

**Why:** RDF export is not needed for the training pipeline. By keeping it in `infra/`, we signal "this is optional / research use only". Future: consider moving to a separate `tools/` package if it grows.

---

## Migration Path (low-risk)

**Phase 1: Create new structure alongside old (no deletions)**
```bash
mkdir -p src/epistemic src/pipeline/{data,model} src/model/{data,architecture,training,evaluation}
```

**Phase 2: Copy and update imports**
```python
# New location: src/epistemic/formula.py
from src.epistemic.enums import EvidenceType, Verdict

# Old location still works (backward compat):
# src/core/claims/labels.py imports from src/epistemic and re-exports
from src.epistemic.formula import compute_evidence_confidence
__all__ = ["compute_evidence_confidence", ...]
```

**Phase 3: Update CLI scripts**
```python
# src/pipeline/data/convert_to_unified.py (was src/cli/convert_to_unified.py)
from src.adapters.ai2thor.converter import Converter as AI2ThorConverter
from src.adapters.averitec.converter import Converter as AVeriTecConverter
```

**Phase 4: Delete old files**
- `src/core/claims/` → contents moved to `src/epistemic/`
- `src/cli/` → contents moved to `src/pipeline/`
- `src/core/gnn/` → reorganised into `src/model/`

---

## Import Guidelines (Post-Restructure)

**Data pipeline:**
```python
from src.epistemic import Verdict, EvidenceType
from src.adapters.ai2thor.converter import Converter
from src.pipeline.reporting import write_jsonl
```

**Model pipeline:**
```python
from src.epistemic import Verdict
from src.model.data import ClaimGraphBuilder
from src.model.epistemichgnn import EpistemicHGNN
from src.model.training import Trainer, TrainConfig
from src.model.evaluation import compute_accuracy
```

**Adapters (internal only):**
```python
from src.epistemic import ReasoningStrategy, EvidenceType, EvidenceStance, Verdict
```

**Forbidden (creates circular dependency):**
```python
# ❌ Don't do this from model/ code
from src.adapters.ai2thor import ...

# ❌ Don't do this from adapters/ code
from src.model import ...
```

---

## Justfile Updates

Current `just` targets map directly to `src/cli/` scripts. After restructure:

```makefile
# Before
just build  →  src/cli/convert_to_unified.py

# After
just build  →  src/pipeline/data/convert_to_unified.py
```

Update Justfile target `script` paths:

```makefile
build:
    @cd $(ROOT) && uv run python -m src.pipeline.data.convert_to_unified ...

train:
    @cd $(ROOT) && uv run python -m src.pipeline.model.train ...

eval:
    @cd $(ROOT) && uv run python -m src.pipeline.model.evaluate ...
```

---

## Benefits Summary

| Concern | Old | New | Benefit |
|---|---|---|---|
| **Where are enums?** | `src/core/claims/labels.py` (90 lines, mixed with formula) | `src/epistemic/enums.py` | Clear, focused file |
| **Where does EC math live?** | `src/core/claims/labels.py` (scattered) | `src/epistemic/formula.py` | Centralized formula logic |
| **How do I add a new CLI stage?** | Create `src/cli/foo.py` | Add to `src/pipeline/data/foo.py` or `src/pipeline/model/foo.py` | Clear ownership |
| **How do I add a new model metric?** | Add to `src/core/gnn/metrics.py` (10+ files at one level) | Add to `src/model/evaluation/metrics.py` | Clear location, purpose-driven naming |
| **How do I extend to V2?** | Unclear; need to trace imports | Create `src/model/v2/config.py`, update `epistemichgnn.py` to read config version | Explicit versioning, technology-agnostic naming |
| **Can data/model pipelines share code?** | Yes but unclear what | `src/pipeline/reporting.py` | Shared utilities documented |
| **Is RDF export needed for training?** | Unclear | Lives in `src/infra/` marked as optional | Clear critical vs. optional path |
| **Is this folder specific to GNNs?** | Yes, `src/core/gnn/` signals "GNN implementation" | No, `src/model/` signals "fact verification model" | Future-proof if architecture changes

---

## Non-Breaking Alternatives

If a full restructure feels risky, a lighter refactor:

1. **Keep file locations**, reorganise imports only:
   - Create `src/epistemic.py` as a facade that re-exports all epistemic symbols from `src/core/claims/`
   - All new code imports from `src.epistemic`
   - Old code can keep using `src.core.claims` until migrated

2. **Mark pipeline stages**, reorganise folders later:
   - Add a `src/cli/STAGE.txt` file documenting which scripts are "data" vs. "model"
   - When time permits, move physical folders

3. **Document in README**, no restructure:
   - Add a section to `README.md` or `docs/folder-structure.md` explaining current layout
   - Use this as a reference until a full migration is feasible

Choose based on how much refactoring risk your team wants to take on right now.
