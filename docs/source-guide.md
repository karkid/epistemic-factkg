# EpistemicFactKG — Source Code Guide

A layer-by-layer walkthrough of every source file. Read top-to-bottom the first time, then use it as a reference index when refactoring.

---

## How the layers fit together

```
Raw data (AI2THOR simulator / AVeriTeC JSON / LLM generation)
       ↓  src/adapters/*/
  v3.0 unified JSONL (out/unified/epistemic_factkg.jsonl)
       ↓  src/cli/filter_for_training.py + split_dataset.py
  Filtered + split JSONL
       ↓  src/core/gnn/graph_builder.py  (via train_gnn.py)
  PyG HeteroData graphs
       ↓  src/core/gnn/model.py  (EpistemicHGNN)
  Predictions (stance, IS, verdict)
       ↓  src/cli/evaluate_gnn.py
  Metric JSON files (out/evaluation/)
```

The `src/core/claims/` package contains the epistemic math that everything else depends on — start there.

---

## Layer 1 — Epistemic Foundation (`src/core/claims/`)

### `labels.py` — the single most important file

Everything epistemic lives here. No other file replicates this logic.

**Enums (read first):**
- `Verdict` — four verdict strings: `supported`, `refuted`, `conflicting_evidence`, `not_enough_evidence`
- `EvidenceStance` — five stance strings an evidence item can hold; `absent` means sensor-confirmed absence (not "missing data")
- `EvidenceType` (alias `Pramana`) — six epistemic categories: `perception`, `testimony`, `non_apprehension`, `comparison_analogy`, `inference`, `postulation_derivation`
- `ReasoningStrategy` — six canonical strategies assigned at the *claim* level: `direct_observation`, `absence_detection`, `spatial_comparison`, `testimonial_lookup`, `multi_hop_inference`, `conflicting_evidence`
- `ClaimStructure` — structural shape of the claim (one_hop, multi_hop, etc.)

**Key constants:**
- `CONFIDENCE_WEIGHTS` — EW per `EvidenceType`; `perception=0.95` is highest, `postulation_derivation=0.40` lowest
- `TRAINING_EVIDENCE_TYPES` — the 5 types used for training (excludes `postulation_derivation`)
- `SUPPORT_THRESHOLD=0.75`, `REFUTE_THRESHOLD=0.75`, `CONFLICT_FLOOR=0.40` — ADR-014 thresholds

**Core formula functions (the EC formula chain):**
1. `combine_evidence_weights(evidence_types)` — multi-label EW: `1 - Π(1 - w_i)`. Multiple types combine with diminishing returns.
2. `compute_evidence_confidence(st, ew, is_)` — per-evidence EC_i: `1 - (1 - ST)^(EW × IS)`
3. `aggregate_scores(evidence_items, registry)` — full SupportScore + RefuteScore over a claim's evidence list using EC_i
4. `derive_verdict(support_score, refute_score)` — rule-based threshold comparison → verdict string

**Source trust registry:**
- `load_source_trust_registry(path)` — loads JSONL to `{source_id: record}` dict
- `get_source_trust(source_id, registry)` — returns `source_trust` float or `DEFAULT_SOURCE_TRUST=0.40`
- `resolve_source_id(domain, modality, registry)` — 7-step lookup cascade: exact → parent → archive → TLD → social media → modality default → `unknown_web`
- `make_source_id(domain, modality)` — builds candidate key without registry lookup

**Refactoring notes:**
- `Pramana` and `combine_pramana_weights` are backward-compat aliases — safe to remove once adapters are fully updated
- `_SOCIAL_MEDIA_DOMAINS`, `_COMPOUND_TLDS`, the private helpers at the bottom — these are stable, no need to touch
- `is_training_record()` uses `epistemic.evidence_types_all` — verify this field still exists in current v3.0 output

---

### `claim_schema.py` — JSON schema for v3.0 records

Defines the Pydantic/dataclass shapes for the unified JSONL format. If a field appears in the schema here but is missing in actual records, that's a data pipeline bug.

**Refactoring notes:**
- Cross-reference with `claim_validator.py` — both describe the schema; they should be consistent
- The `epistemic` block at the claim level aggregates per-evidence types into `evidence_types_all` — this drives `is_training_record()`

---

### `claim_validator.py` — validates individual records

Called by `validate_unified_dataset.py` and `validate_training_dataset.py`. Walks every field and returns a list of validation errors.

---

### `claim_generator.py` — AI2THOR claim generation

Generates claim text from simulator state. Used upstream in the AI2THOR pipeline before the converter. This is **not** called at training time — it runs once during data collection.

---

### `types.py` — shared type aliases

Simple Python type aliases for the claims domain. Keep small.

---

### `result.py` — result dataclass

Holds `(claim, evidence_items, verdict, metadata)` as an intermediate representation before serialization to JSONL.

---

## Layer 2 — Adapters (`src/adapters/`)

Each adapter converts source-specific raw data to the v3.0 unified JSONL format. They are independent — changing one does not affect the others.

### AI2THOR adapter (`src/adapters/ai2thor/`)

The most complex adapter; converts simulator state (triples, actions, observations) to claims.

**`converter.py` — the entry point:**
- `_STRATEGY_MAP` + `_to_strategy(raw)` — maps AI2THOR action types (`spatial_reasoning → spatial_comparison`, `action_testing → multi_hop_inference`) to canonical `ReasoningStrategy` values. This map is adapter-local (not in core).
- `_LABEL_MAP` — normalises `"support"/"supported"` and `"refute"/"refuted"` to `Verdict` enum
- `_classify_strategy(predicate, ev_triples)` — infers strategy from predicate type when not explicitly set
- Key output fields: `reasoning.strategy`, per-evidence `evidence_types`, `source_id="ai2thor_simulation"`, `inference_strength`

**`template.py`** — template configs for different claim types (spatial, affordance, etc.)

**`registry.py`** — maps AI2THOR object types to the entity/predicate ontology

**`scene_randomizer.py`** — randomises scene states to generate diverse claims

**`validator.py`** — `AI2ThorValidator`: validates converted records against v3.0 schema

**`data_source.py`** — loads raw `claims_all.jsonl` and drives the converter

**`ontology.py`** — object type → natural-language entity mapping

**`ids/`** — object type and predicate string constants

**`semantics/`** — lexicons and rules for natural-language generation from triples

**Refactoring notes:**
- `converter.py` is the main file to understand — the others are support infrastructure
- The `reasoning` field is now always emitted (removed the old `if structural else None` condition) — verify all downstream consumers handle this
- `inference_strength` is hardcoded per action type in the template definitions — review if these values are still correct for your rubric

---

### AVeriTeC adapter (`src/adapters/averitec/`)

**`converter.py`:**
- `_STRATEGY_MAP` — maps AVeriTeC `fact_checking_strategies` strings to canonical `ReasoningStrategy` values
- `_STRATEGY_PRIORITY` — when a claim has multiple strategies, the most informative wins: `multi_hop_inference > spatial_comparison > testimonial_lookup`
- `_to_strategy(strategies: list[str])` — takes the **full** `fact_checking_strategies` list (not just `[0]`); 44% of AVeriTeC records have more than one
- `_LABEL_MAP` — normalises AVeriTeC verdict strings including `"conflicting evidence/cherrypicking"`
- `_STRATEGY_EVIDENCE_TYPE_MAP` — enriches evidence_types based on strategy (e.g. `consultation → inference`)
- `resolve_source_id()` from labels.py is used here for source trust lookup

**`validator.py`** — validates converted AVeriTeC records

**Refactoring notes:**
- AVeriTeC has no per-evidence stance ground truth — the claim-level verdict is cloned to all evidence items as an approximation. This is a known limitation documented in ADR-014.
- The `evidence_types` assignment logic in converter.py is non-trivial (perceptual vs textual vs table path) — read carefully before changing

---

### Synthetic adapter (`src/adapters/synthetic/`)

**`fictional_generator.py` — the core:**
- `_STRATEGY_MAP` + `_to_strategy(template_name)` — maps all 15 template names to canonical strategies
- `SyntheticFictionalGenerator.generate()` — drives template-based claim generation with shortcut-breaking guarantees
- The docstring at the top has the full template matrix — read it first
- Shortcut-breaking is guaranteed by construction: the template determines ST and IS values, which determine the verdict through EC formula regardless of text content

**`client/`:**
- `base.py` — `SyntheticTextClient` abstract base + `EvidenceSpec` dataclass
- `local_client.py` — `LocalTextClient`: offline, template-fill-in text (no API key)
- `grounded_client.py` — `GroundedClient`: draws from a seed pool for more natural text
- `llm/llm_client.py` — `LLMClient`: calls Anthropic API for maximum diversity

**`llm/prompt_builder.py`** — builds prompts for the LLM client

**`validator.py`** — validates generated synthetic records

**Refactoring notes:**
- `MIN_SHORTCUT_FRACTION = 0.35` — at least 35% of generated records must be shortcut-breaking. Verify this threshold is still met after any template changes.
- The three clients are interchangeable — `LocalTextClient` is what you want for tests and CI

---

## Layer 3 — GNN Stack (`src/core/gnn/`)

Eight modules, one per concern. Read them in this order.

### `types.py` — constants and container

All integer mappings and the `ClaimGraph` dataclass. Never has business logic.

**Key mappings:**
- `VERDICT_TO_INT` — `{supported:0, refuted:1, not_enough_evidence:2}`; 3-class, no `conflicting_evidence`
- `STANCE_TO_INT` — `{supports:0, absent:0, refutes:1, not_enough_evidence:2, conflicting_evidence:2}`; note `absent=supports` (sensor-confirmed absence verifies a claim about a missing object)
- `EVIDENCE_TYPE_TO_INT` — 5-class multi-hot index (excludes `postulation_derivation`)
- `REASONING_STRATEGY_TO_INT` — 6-class one-hot index for claim nodes
- `MODALITY_TO_INT` — 5-class one-hot for evidence nodes
- `SOURCE_TYPE_TO_INT` — 6-class one-hot for evidence nodes
- `get_source_category(source_id, registry)` — resolves registry `source_type` to the 6 encoder categories

**Node feature dimensions (computed from the mappings above):**
- `CLAIM_DIM = 390` (384 text + 6 strategy)
- `EVIDENCE_DIM = 400` (384 text + 5 modality + 5 evidence_types + 6 source_type)
- `TRIPLE_DIM = 384`

**`ClaimGraph` dataclass fields:**
- `data` — `HeteroData` with nodes/edges + `data["evidence"].stance_y`, `is_y`, `ew`, `st`
- `label` — verdict integer (for evaluation)
- `dataset` — provenance string (`"ai2thor"`, `"averitec"`, `"synthetic"`) for per-source metrics

**Refactoring notes:**
- `NUM_VERDICT = 3` not 4 — `conflicting_evidence` is dropped (not enough samples). If you want to add it back, update `VERDICT_TO_INT`, `NUM_VERDICT`, and all loss/metric code.
- The `Pramana` alias in labels.py → `EvidenceType` here. Make sure you're using `EvidenceType` everywhere in new code.

---

### `config.py` — graph schema (single source of truth)

`GraphConfig` dataclass with `node_dims`, `edge_types`, `target_node`, `symbolic_fields`.

`GraphConfig.v1()` defines the V1 schema:
- Node dims: `{claim:390, evidence:400, triple:384}`
- Edge types: `has_evidence`, `connected_to`, `co_evidence`, `has_triple`, `from_triple`

**Why this exists:** `EpistemicEncoder` reads `node_dims` and `edge_types` at construction and never hardcodes names. To add a V2 source node, you add one line to `node_dims` and one to `edge_types`; the encoder adapts automatically.

**Refactoring notes:**
- `symbolic_fields = ["ew", "st"]` is declared but not currently used by the encoder — it's a placeholder for a future version that might route these fields automatically

---

### `featurizer.py` — text + categorical encoding

`Featurizer` loads `all-MiniLM-L6-v2` (384-d) once and caches embeddings to a `.pkl` file keyed by SHA256 hash.

**Methods:**
- `encode_texts(texts)` → `[N, 384]` — batch-embeds, hits cache for repeats
- `encode_modality(modality)` → `[5]` one-hot
- `encode_evidence_types(evidence_types)` → `[5]` **multi-hot** (not one-hot; an item can have multiple types)
- `encode_reasoning_strategy(strategy)` → `[6]` one-hot (claim node only)
- `encode_source_type(category)` → `[6]` one-hot
- `save_cache()` — call after building graphs to persist the embedding cache

**Refactoring notes:**
- The `_hash` function uses only the first 16 hex chars of SHA256 — collision probability is negligible (2^-64) but non-zero. Acceptable for a cache.
- The model is lazy-loaded on first `encode_texts()` call — the constructor is fast. If you need eager loading (e.g. for timing), call `self._model_()` explicitly after construction.
- Cache path is optional — if `None`, embeddings are recomputed every run. Always pass a cache path in production.

---

### `graph_builder.py` — JSONL record → HeteroData

`ClaimGraphBuilder` converts a single v3.0 record dict into a `ClaimGraph`.

**`build(record)` does five things:**
1. Claim node: `encode_texts([claim_text])` + `encode_reasoning_strategy(strategy)` → `[1, 390]`
2. Evidence nodes: for each item, builds the 400-d feature vector + stores `stance_y`, `is_y`, `ew`, `st` as separate tensors
3. Triple nodes (AI2THOR only): embeds `"s p o"` strings → `[N_tr, 384]`
4. Edges: `has_evidence` (claim→ev), `connected_to` (ev→claim), `co_evidence` (N×(N-1) all pairs), plus triple edges
5. Stores `data["claim"].y` = verdict integer for DataLoader batching

**Important design decisions:**
- `ew` and `st` are **not** in the evidence `x` feature vector — they are stored as separate tensors. This prevents the encoder from shortcutting "high ST → supported".
- `co_evidence` is fully connected within each claim: if N=5 evidence items, that's 20 directed edges. GATConv attention learns which pairs matter.
- Degenerate case (0 evidence items): creates a dummy neutral evidence node so the graph is always valid.

**`_extract_strategy(record)`** — one-liner: `record.get("reasoning", {}).get("strategy") or "testimonial_lookup"`. No source-specific branching.

**Refactoring notes:**
- The `from_paths(registry_path, embed_cache_path)` classmethod is a convenience constructor for CLI usage
- The `zip(*[(i,j) for ...])` pattern for co_evidence edge construction is correct but dense — consider extracting to a helper for readability

---

### `encoder.py` — shared HeteroConv encoder

`EpistemicEncoder` builds a two-layer `HeteroConv` from `GraphConfig`. Never hardcodes node or edge names.

**Architecture:**
1. `input_proj`: one `nn.Linear(dim → hidden_dim)` per node type, applied before message passing
2. `conv1` and `conv2`: `HeteroConv` wrapping `GATConv((-1,-1), hidden_dim//heads, heads=heads, concat=True)` for every edge type in the config
3. ELU activation + dropout between layers; ELU after layer 2 (no dropout)

**`(-1, -1)` input dims in GATConv:** PyTorch Geometric lazy initialization — parameters are created on the first forward pass, not at construction. This is why calling `.numel()` before the first forward pass returns 0.

**`forward(data)`:**
- Projects node features to `hidden_dim`
- Runs two-layer message passing
- Returns `x_dict` mapping node type → enriched embeddings
- Only `x_dict[NodeType.EVIDENCE]` is used downstream by H1 and H2

**Refactoring notes:**
- Edge types not present in `data.edge_types` are silently skipped (`if (src, rel, dst) in data.edge_types`). AI2THOR graphs have triple edges; AVeriTeC and synthetic do not. This is correct behaviour.
- If you add a V2 source node: add it to `GraphConfig.v1()` and `graph_builder.py`. The encoder automatically picks it up.

---

### `heads.py` — H1, H2, VerdictHead

Three lightweight `nn.Module` classes. All are graph-structure-agnostic.

**`StanceHead` (H1):**
- `Linear(hidden_dim → hidden_dim//2) → ReLU → Linear(hidden_dim//2 → 3)`
- Input: evidence embeddings `[N_ev, hidden_dim]`
- Output: logits `[N_ev, 3]`; at inference, `argmax` gives stance integer (0=supports, 1=refutes, 2=neutral)

**`ISHead` (H2):**
- `Linear(hidden_dim → hidden_dim//2) → ReLU → Linear(hidden_dim//2 → 1) → Sigmoid`
- Output: IS scalars `[N_ev, 1]` in `[0, 1]`

**`VerdictHead`** (ADR-014):
- `Linear(2 → 3)` — maps `(support_score, refute_score)` to 3-class verdict logits
- Only 6 trainable parameters (2×3 weights + 3 biases)
- Replaces the hard-coded 0.75/0.40 thresholds that failed on AVeriTeC (25% accuracy)
- Trained end-to-end with `CrossEntropyLoss` against claim-level verdict labels

**Refactoring notes:**
- H1 and H2 both receive the same evidence embedding from the encoder — they share the encoder's output but have independent parameters
- `VerdictHead` is tiny by design — interpretability is preserved because `support_score` and `refute_score` are still computed and reported

---

### `aggregator.py` — symbolic EC formula (no parameters)

`SymbolicAggregator` is a plain class (not `nn.Module`) — it has no trainable parameters and never changes.

**Two modes:**

`compute_soft_scores(stance_probs, is_pred, ew, st)` — **used during training:**
- Takes softmax probabilities from H1 (not argmax) so gradients flow back through H1 and H2 into the encoder
- `soft_support = 1 - Π(1 - EC_i × p_support_i)` — weighted by how much H1 thinks this item supports
- `soft_refute  = 1 - Π(1 - EC_i × p_refute_i)`
- Returns `(support_score [1], refute_score [1])` — one scalar each

`compute_scores(stance_pred, is_pred, ew, st)` — **used at inference:**
- Takes argmax stance predictions (hard assignment)
- Partitions evidence into supporters and refuters
- Returns Python `float` pair

`get_verdict(support_score, refute_score)` — hard threshold rules (kept as ablation baseline with `λ₂=0`).

**Refactoring notes:**
- `_aggregate(ec, mask)` at the bottom handles the empty-set case (returns `0.0` when no supporters/refuters exist)
- During training, the soft scores feed into `VerdictHead` — gradients flow: `verdict_CE → VerdictHead → soft_scores → H1/H2 → encoder`
- At inference, `model.predict()` calls `compute_scores()` (hard), not `compute_soft_scores()`

---

### `model.py` — EpistemicHGNN (assembler)

The top-level `nn.Module`. Assembles encoder + H1 + H2 + VerdictHead + SymbolicAggregator. Contains no novel logic — just wires the components together.

**`forward(data)` (training):**
1. Encoder → `ev_emb [N_ev, hidden_dim]`
2. H1 → `stance_logits [N_ev, 3]`
3. H2 → `is_pred [N_ev, 1]`
4. `_soft_verdict_logits()` → `verdict_logits [N_claims, 3]`

**`predict(data)` (inference, `@torch.no_grad`):**
1. Calls `forward()` to get `stance_logits`
2. `argmax` → `stance_pred`
3. `aggregator.compute_scores()` (hard) → `support_score`, `refute_score`
4. `verdict_logits.argmax()` → verdict string via `_INT_TO_VERDICT`
5. Returns all outputs for interpretability

**`_soft_verdict_logits(data, stance_logits, is_pred)`:**
- Uses `data["evidence"].batch` pointer to group evidence items by claim index in a batched DataLoader
- Falls back to `zeros` batch pointer for single-graph inference
- Loops over each claim: calls `aggregator.compute_soft_scores()`, stacks results, feeds to `VerdictHead`

**Refactoring notes:**
- The loop over claims in `_soft_verdict_logits` is O(N_claims) — this is fine for typical batch sizes (32 claims). If batches grow very large, consider vectorising.
- `predict()` calls `forward()` and then re-runs the aggregation with hard argmax separately — there's a minor redundancy (H1/H2 run twice). Acceptable for interpretability.

---

### `train.py` — training loop

`TrainConfig` and `Trainer`. No external dependencies beyond PyTorch and PyG.

**`TrainConfig` fields:**
- `is_loss_weight` (λ₁ = 0.5) — scales IS regression loss relative to stance CE
- `verdict_loss_weight` (λ₂ = 1.0) — scales verdict CE loss
- `patience = 10` — early stopping on validation total loss

**`Trainer._run_epoch(loader, train)`:**
- Total loss: `stance_CE + λ₁ × IS_MSE + λ₂ × verdict_CE`
- Stance loss: `CrossEntropyLoss(weight=stance_class_weights)` — inverse-frequency weights for class imbalance
- IS loss: `MSELoss`
- Verdict loss: `CrossEntropyLoss` — no class weights here yet (known issue: NEE class never predicted)
- Tracks `EpochResult`: per-epoch losses + `stance_acc` + `verdict_acc`

**`Trainer.fit(train_loader, val_loader)`:**
- Early stopping saves best checkpoint to `out/checkpoints/best_model.pt`
- Returns history list for `training_history.json`
- `ReduceLROnPlateau` scheduler on `val.loss` with `patience=5, factor=0.5`

**Refactoring notes:**
- The total loss normalization uses `n_ev` (evidence count) as denominator — this is consistent for stance and IS but slightly off for verdict loss (which should use `n_claims`). The current code already corrects for this: `total_verdict / dc` where `dc = max(n_claims, 1)`.
- **Known issue:** `verdict_criterion = CrossEntropyLoss()` has no class weights. The NEE class (class 2) is under-represented — add inverse-frequency weights the same way stance weights are computed. One-line fix.

---

### `dataset.py` — PyG InMemoryDataset (currently bypassed)

`EpistemicFactDataset` wraps `InMemoryDataset` to cache graphs to a `.pt` file.

**Known issue:** PyG `InMemoryDataset` saves to `root/processed/<filename>` but the code tries to save to an explicit `self._pt_cache` path. On a fresh run, the processed check passes, `.process()` runs, but the cache is not written to the expected location, causing a rebuild on every run.

The training CLI (`train_gnn.py`) **bypasses this class entirely** — it builds graphs directly from JSONL. This class exists for potential future use with a proper PyG dataset pipeline.

**Refactoring notes (if you want to fix it):**
- Override `processed_dir` property to point to `self._pt_cache.parent`
- Or simplify: remove `InMemoryDataset` inheritance and just use `torch.save` / `torch.load` directly
- The `get_class_weights()` method is unused (train_gnn.py recomputes this inline)

---

### `metrics.py` — pure metric functions

No model dependencies. Every function takes `torch.Tensor` arguments.

| Function | Purpose |
|---|---|
| `compute_accuracy(preds, labels)` | fraction correct |
| `compute_macro_f1(preds, labels, n_classes)` | unweighted average F1 |
| `compute_weighted_f1(preds, labels, n_classes)` | support-weighted F1 |
| `compute_per_class_metrics(preds, labels, n_classes)` | P/R/F1/support per class |
| `compute_confusion_matrix(preds, labels, n_classes)` | N×N confusion matrix |
| `compute_ece(logits, labels, n_bins=10)` | Expected Calibration Error |
| `compute_rmse(preds, targets)` | H2 IS regression error |
| `compute_pearson_r(preds, targets)` | H2 IS prediction correlation |
| `compute_per_group_accuracy(preds, labels, groups)` | per-source accuracy breakdown |

**Refactoring notes:**
- All functions are pure — they have no side effects and can be tested in isolation
- `compute_ece` uses `max(softmax)` confidence — this is the standard implementation

---

## Layer 4 — CLI Entry Points (`src/cli/`)

Each script is a standalone `main()` with an `argparse` parser. They are invoked via `just` targets in the Justfile.

### Data pipeline (run in order)

| Script | `just` target | What it does |
|---|---|---|
| `generate_synthetic.py` | `just synthetic` | Calls `SyntheticFictionalGenerator`, writes `out/synthetic/` JSONL |
| `build_claims.py` | `just claims` | AI2THOR: runs templates against simulator, writes `data/raw/ai2thor/claims_all.jsonl` |
| `convert_to_unified.py` | `just build` | Runs all three adapters, merges into `out/unified/epistemic_factkg.jsonl` |
| `validate_unified_dataset.py` | `just validate` | Validates unified JSONL against v3.0 schema |
| `filter_for_training.py` | `just filter` | Removes non-training evidence types; writes `out/training/` JSONL |
| `split_dataset.py` | `just split` | Writes train/val/test index JSON files to `out/splits/` |

### Model pipeline

| Script | `just` target | What it does |
|---|---|---|
| `build_graph_dataset.py` | `just graph` | Builds HeteroData graphs, writes to `out/graphs/graph_dataset.pt` (uses old `EpistemicFactDataset` — consider updating or removing) |
| `train_gnn.py` | `just train` | Trains EpistemicHGNN; writes `out/checkpoints/best_model.pt` + `training_history.json` |
| `evaluate_gnn.py` | `just eval` | Evaluates on test split; writes `out/evaluation/<run>/` JSON files |

### Other

| Script | `just` target | What it does |
|---|---|---|
| `validate_synthetic.py` | `just validate-synthetic` | Validates synthetic-only records |
| `validate_training_dataset.py` | `just check-train` | Validates the filtered training JSONL |
| `build_dataset_report.py` | `just report` | Generates summary statistics report |
| `build_rdf.py` | `just rdf` | Exports to RDF/Turtle format via `src/infra/rdf/` |

**`train_gnn.py` key flags:**
- `--jsonl` (required) — filtered training JSONL
- `--is-loss-weight` (default 0.5) — λ₁
- `--verdict-loss-weight` (default 1.0) — λ₂; set to 0.0 for ablation baseline (hard thresholds only)
- `--no-class-weights` — disables stance inverse-frequency weighting
- `--embed-cache` — path to `.pkl` embedding cache (always use this in production)

**`evaluate_gnn.py` key outputs:**
- `stance_metrics.json` — H1 accuracy, macro F1, ECE, per-class P/R/F1
- `is_metrics.json` — H2 RMSE, Pearson r, pred/true mean
- `verdict_metrics.json` — verdict accuracy, macro F1, per-class, confusion matrix, per-source

---

## Layer 5 — Infrastructure (`src/infra/`, `src/core/ports/`, `src/utils/`)

### `src/core/ports/`

Abstract base classes (interfaces) for adapters. Defines `DatasetConverter` and `DatasetValidator` protocols. Each adapter's `converter.py` and `validator.py` implement these.

### `src/core/graph/types.py`

Graph type definitions for the RDF/knowledge-graph layer. Independent of the PyG GNN layer.

### `src/core/nlg/`

Natural language generation from triples. Used by AI2THOR adapter to produce claim text from `(subject, predicate, object)` triples.
- `triple_realizer.py` — converts `(s, p, o)` to an English sentence
- `sentence_template.py` — template patterns for NLG

### `src/core/ontology.py` and `src/core/registry.py`

Ontology definitions and global object registry for the AI2THOR domain. Read by `ai2thor/ontology.py` and `ai2thor/registry.py`.

### `src/core/semantics/`

Lexicons (entity and predicate → natural language) used in NLG. AI2THOR-specific.

### `src/infra/rdf/`

RDF export infrastructure. `builder.py` assembles an RDF graph from claim records; `engine.py` runs SPARQL queries; `ttl.py` serialises to Turtle format. Not needed for GNN training.

### `src/utils/`

| File | Purpose |
|---|---|
| `io.py` | JSONL read/write helpers |
| `logger.py` | Configured `logging` instance |
| `config_loader.py` | YAML config loader |
| `exceptions.py` | Project-specific exception classes |
| `time.py` | `utc_now_iso()` — ISO 8601 timestamp string |

---

## Known Issues and Refactoring Priorities

### High priority (affects training quality)

1. **Verdict class weights missing** — `Trainer.verdict_criterion = CrossEntropyLoss()` has no class weights. The `not_enough_evidence` class is under-represented → NEE is never predicted. Fix: compute inverse-frequency weights from `batch[NodeType.CLAIM].y` the same way `stance_weights` is computed. One-line change in `train.py` + one new arg in `train_gnn.py`.

2. **AVeriTeC stance approximation** — AVeriTeC has no per-evidence stance ground truth; the claim-level verdict is cloned to all evidence items. This is a known data limitation (ADR-014). No clean fix without manual annotation.

### Medium priority (code quality)

3. **`dataset.py` cache path mismatch** — `EpistemicFactDataset` saves to `self._pt_cache` but PyG checks `root/processed/`. The class is currently bypassed by `train_gnn.py` so this is harmless but should be fixed or the class removed entirely.

4. **`Pramana` / `combine_pramana_weights` aliases** — backward-compat aliases in `labels.py`. Safe to remove once you confirm no adapter still imports them by the old name.

5. **`build_graph_dataset.py` / `just graph`** — uses the stale `EpistemicFactDataset`. Either update to use `ClaimGraphBuilder` directly (same pattern as `train_gnn.py`) or remove the target.

### Low priority (nice to have)

6. **`_soft_verdict_logits` loop** — O(N_claims) Python loop in the forward pass. Vectorisable with `scatter` operations from PyG if batches become large.

7. **Synthetic accuracy regression** — went from ~97% (V0) to ~60% (V1) after adding VerdictHead. Investigate whether the VerdictHead is correctly learning synthetic patterns or whether the shared head over-optimises for AVeriTeC.

8. **Ablation study** — run `--verdict-loss-weight 0.0` to reproduce the hard-threshold baseline and quantify the VerdictHead contribution for the paper.

---

## Data flow summary (one-liner per file)

```
labels.py         → EC formula math, enums, source trust resolution
claim_schema.py   → v3.0 record shape definition
claim_validator.py → validates individual records
ai2thor/converter.py → simulator state → v3.0 record (reasoning.strategy, evidence_types, IS)
averitec/converter.py → AVeriTeC JSON → v3.0 record (all strategies, source_id via resolve_source_id)
synthetic/fictional_generator.py → template → v3.0 record (EC-verified verdict by construction)
types.py (gnn)    → integer lookup tables + CLAIM/EVIDENCE/TRIPLE dims + ClaimGraph dataclass
config.py         → GraphConfig V1: node dims + edge types (read by encoder)
featurizer.py     → text → 384-d embeddings (cached) + categorical → one-hot/multi-hot
graph_builder.py  → v3.0 record → HeteroData with stance_y/is_y/ew/st tensors
encoder.py        → HeteroData → context-enriched evidence embeddings [N_ev, hidden_dim]
heads.py          → H1 stance logits, H2 IS scalars, VerdictHead logits
aggregator.py     → EC formula tensor ops (soft for training, hard for inference)
model.py          → wires encoder + H1 + H2 + aggregator + VerdictHead
train.py          → multi-task loss loop (stance_CE + λ₁*IS_MSE + λ₂*verdict_CE)
metrics.py        → pure metric functions (accuracy, F1, RMSE, Pearson r, ECE)
evaluate_gnn.py   → test split → stance/IS/verdict JSON output files
train_gnn.py      → CLI: load JSONL → build graphs → train → save checkpoint
```
