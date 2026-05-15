# Epistemic FactKG
# Run `just` to list all commands.

[default]
[doc("List available commands")]
default:
    @just --list

# ── Variables ────────────────────────────────────────────────────────────────
CONFIG              := "configs/config.yaml"
REGISTRY            := "data/registry/source_trust_registry.jsonl"
AI2THOR_RAW_DIR     := "data/raw/ai2thor"
AI2THOR_CLAIMS      := AI2THOR_RAW_DIR + "/claims_all.jsonl"
RAW_AVERITEC_TRAIN  := "data/raw/averitec/train.json"
RAW_AVERITEC_DEV    := "data/raw/averitec/dev.json"
SYNTHETIC_RAW_DIR   := "data/raw/synthetic"
SYNTHETIC_JSONL     := SYNTHETIC_RAW_DIR + "/synthetic_current.jsonl"
SEED_POOL           := "data/registry/seed_pool.jsonl"
KG_TTL              := "out/knowledge_graph.ttl"
UNIFIED_JSONL       := "out/unified/epistemic_factkg.jsonl"
VALIDATION_JSON     := "out/report/validation.json"
REPORT_DIR          := "out/report"
TRAINING_JSONL      := "out/training/epistemic_factkg_training.jsonl"
TRAINING_VALIDATION := "out/report/training_validation.json"
GRAPH_DATASET       := "out/graphs/graph_dataset.pt"
SPLITS_DIR          := "out/splits"
CHECKPOINTS_DIR     := "out/checkpoints"
RESULTS_DIR         := "out/results"


# ── Setup ────────────────────────────────────────────────────────────────────
[doc("Install dependencies and set up the environment")]
init:
    uv venv && uv sync && uv pip install -e ".[dev,notebook]"


# ── Build ────────────────────────────────────────────────────────────────────
[doc("Convert frozen AI2THOR claims + AVeriTeC + any synthetic batches to unified v3.0 JSONL")]
build:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -f "{{AI2THOR_CLAIMS}}" ]; then
        echo "Error: {{AI2THOR_CLAIMS}} not found. Run 'just rebuild' to generate it." >&2
        exit 1
    fi
    SYNTHETIC_ARG=""
    if [ -f "{{SYNTHETIC_JSONL}}" ]; then
        SYNTHETIC_ARG="--synthetic {{SYNTHETIC_JSONL}}"
    fi
    uv run python -m src.cli.convert_to_unified \
        --registry {{REGISTRY}} \
        --averitec {{RAW_AVERITEC_TRAIN}} {{RAW_AVERITEC_DEV}} \
        --ai2thor {{AI2THOR_CLAIMS}} \
        $SYNTHETIC_ARG \
        --output {{UNIFIED_JSONL}} \
        --intermediate_dir out/intermediate


[doc("Re-generate AI2THOR claims from scratch via simulator, then generate synthetic batch, then convert all datasets to unified v3.0 JSONL")]
rebuild max_contexts="10":
    uv run python -m src.cli.build_rdf \
        --config {{CONFIG}} --out {{KG_TTL}} --verbose
    uv run python -m src.cli.build_claims {{KG_TTL}} \
        --output-dir {{AI2THOR_RAW_DIR}} \
        --config {{CONFIG}} \
        --max-contexts {{max_contexts}} \
        --verbose
    just synthetic
    just build


# ── Synthetic generation ──────────────────────────────────────────────────────
[doc("Generate synthetic shortcut-breaking claims (grounded by default; use --client llm for API)")]
synthetic n_records="1000":
    mkdir -p {{SYNTHETIC_RAW_DIR}}
    uv run python -m src.cli.generate_synthetic \
        --config {{CONFIG}} \
        --registry {{REGISTRY}} \
        --seed-pool {{SEED_POOL}} \
        --ai2thor-claims {{AI2THOR_CLAIMS}} \
        --n-records {{n_records}} \
        --output {{SYNTHETIC_JSONL}}
    echo "Generated: {{SYNTHETIC_JSONL}}"
    uv run python -m src.cli.validate_synthetic \
        --input {{SYNTHETIC_JSONL}} \
        --registry {{REGISTRY}}


# ── Validate ─────────────────────────────────────────────────────────────────
[doc("Validate the unified JSONL (schema + semantic + pramana checks for all datasets)")]
validate:
    mkdir -p {{REPORT_DIR}}
    uv run python -m src.cli.validate_unified_dataset \
        --files {{UNIFIED_JSONL}} \
        --out {{VALIDATION_JSON}}


# ── Filter ───────────────────────────────────────────────────────────────────
[doc("Filter unified JSONL to GNN training records — excludes postulation_derivation (ADR-011)")]
filter:
    mkdir -p out/training
    uv run python -m src.cli.filter_for_training \
        --input {{UNIFIED_JSONL}} \
        --output {{TRAINING_JSONL}} \
        --verbose


# ── Validate training ────────────────────────────────────────────────────────
[doc("Validate training JSONL against ADR-012 Pramana distribution targets")]
check-train:
    mkdir -p {{REPORT_DIR}}
    uv run python -m src.cli.validate_training_dataset \
        --input {{TRAINING_JSONL}} \
        --config {{CONFIG}} \
        --out {{TRAINING_VALIDATION}}


# ── Report ───────────────────────────────────────────────────────────────────
[doc("Generate dataset report (markdown + charts) from validation output")]
report:
    uv run python -m src.cli.build_dataset_report \
        --summary {{VALIDATION_JSON}} \
        --out_dir {{REPORT_DIR}} \
        --training-summary {{TRAINING_VALIDATION}} \
        --title "Epistemic FactKG Dataset Report"


# ── GNN pipeline ─────────────────────────────────────────────────────────────
[doc("Build PyG HeteroData graph dataset from filtered training JSONL")]
graph:
    mkdir -p out/graphs
    uv run python -m src.cli.build_graph_dataset \
        --input {{TRAINING_JSONL}} \
        --output {{GRAPH_DATASET}} \
        --embed-cache out/graphs/embed_cache.pkl \
        --verbose


[doc("Generate deterministic train/val/test split index files (ADR-009)")]
split:
    mkdir -p {{SPLITS_DIR}}
    uv run python -m src.cli.split_dataset \
        --input {{TRAINING_JSONL}} \
        --output-dir {{SPLITS_DIR}} \
        --seed 42 \
        --verbose


[doc("Train EpistemicHGNN (Pathway A — heuristic Pramana prior)")]
train:
    mkdir -p {{CHECKPOINTS_DIR}}
    uv run python -m src.cli.train_gnn \
        --dataset {{GRAPH_DATASET}} \
        --jsonl {{TRAINING_JSONL}} \
        --splits-dir {{SPLITS_DIR}} \
        --checkpoint-dir {{CHECKPOINTS_DIR}} \
        --epochs 50 \
        --lr 1e-3 \
        --batch-size 32 \
        --device cpu \
        --verbose


[doc("Ablation training — Runs A, B, D (no-stance variants; Run C=full baseline)")]
ablation:
    mkdir -p {{CHECKPOINTS_DIR}}
    @echo "=== Run B: no-stance edges, epistemic present (primary test) ==="
    uv run python -m src.cli.train_gnn \
        --dataset {{GRAPH_DATASET}} \
        --jsonl {{TRAINING_JSONL}} \
        --splits-dir {{SPLITS_DIR}} \
        --checkpoint-dir {{CHECKPOINTS_DIR}} \
        --no-stance-edges --run-name no-stance \
        --epochs 50 --verbose
    @echo "=== Run A: no-stance edges, no epistemic (text-only floor) ==="
    uv run python -m src.cli.train_gnn \
        --dataset {{GRAPH_DATASET}} \
        --jsonl {{TRAINING_JSONL}} \
        --splits-dir {{SPLITS_DIR}} \
        --checkpoint-dir {{CHECKPOINTS_DIR}} \
        --no-stance-edges --no-epistemic --run-name no-stance-no-epistemic \
        --epochs 50 --verbose
    @echo "=== Run D: Pathway B — modality-learned Pramana ==="
    uv run python -m src.cli.train_gnn \
        --dataset {{GRAPH_DATASET}} \
        --jsonl {{TRAINING_JSONL}} \
        --splits-dir {{SPLITS_DIR}} \
        --checkpoint-dir {{CHECKPOINTS_DIR}} \
        --use-modality-learning --aux-loss-weight 0.1 --run-name pathway-b \
        --epochs 50 --verbose


[doc("Run evaluation on all 4 ablation runs (test set)")]
eval:
    mkdir -p {{RESULTS_DIR}}
    @echo "=== Evaluating Run C: full graph ==="
    uv run python -m src.cli.evaluate_gnn \
        --checkpoint {{CHECKPOINTS_DIR}}/best_model.pt \
        --dataset {{GRAPH_DATASET}} \
        --jsonl {{TRAINING_JSONL}} \
        --splits-dir {{SPLITS_DIR}} \
        --output {{RESULTS_DIR}}/full
    @echo "=== Evaluating Run B: no-stance, epistemic present ==="
    uv run python -m src.cli.evaluate_gnn \
        --checkpoint {{CHECKPOINTS_DIR}}/no-stance/best_model.pt \
        --dataset {{GRAPH_DATASET}} \
        --jsonl {{TRAINING_JSONL}} \
        --splits-dir {{SPLITS_DIR}} \
        --no-stance-edges \
        --output {{RESULTS_DIR}}/no-stance
    @echo "=== Evaluating Run A: no-stance, no epistemic ==="
    uv run python -m src.cli.evaluate_gnn \
        --checkpoint {{CHECKPOINTS_DIR}}/no-stance-no-epistemic/best_model.pt \
        --dataset {{GRAPH_DATASET}} \
        --jsonl {{TRAINING_JSONL}} \
        --splits-dir {{SPLITS_DIR}} \
        --no-stance-edges --no-epistemic \
        --output {{RESULTS_DIR}}/no-stance-no-epistemic
    @echo "=== Evaluating Run D: Pathway B ==="
    uv run python -m src.cli.evaluate_gnn \
        --checkpoint {{CHECKPOINTS_DIR}}/pathway-b/best_model.pt \
        --dataset {{GRAPH_DATASET}} \
        --jsonl {{TRAINING_JSONL}} \
        --splits-dir {{SPLITS_DIR}} \
        --use-modality-learning \
        --output {{RESULTS_DIR}}/pathway-b


# ── Test ─────────────────────────────────────────────────────────────────────
[doc("Lint (ruff) and run test suite")]
test:
    uv run ruff format . && uv run ruff check .
    uv run pytest tests/ -v --tb=short


# ── Full pipeline ─────────────────────────────────────────────────────────────
[doc("Full pipeline: build → validate → filter → check-train → report (logs saved to runs/<RUN_ID>/)")]
run RUN_ID="":
    #!/usr/bin/env bash
    set -euo pipefail
    RUN_ID="${RUN_ID:-$(date -u +"%Y-%m-%d_%H-%M-%S")}"
    echo "RUN_ID=$RUN_ID"
    mkdir -p "runs/$RUN_ID/logs"

    just build                       2>&1 | tee "runs/$RUN_ID/logs/build.log"
    just validate                    2>&1 | tee "runs/$RUN_ID/logs/validate.log"
    just filter                      2>&1 | tee "runs/$RUN_ID/logs/filter.log"
    just check-train                 2>&1 | tee "runs/$RUN_ID/logs/check-train.log"
    just report                      2>&1 | tee "runs/$RUN_ID/logs/report.log"

    echo "Done → runs/$RUN_ID/"


# ── Clean ────────────────────────────────────────────────────────────────────
[doc("Delete all generated outputs: out/, data/processed/, data/summary/, runs/")]
clean:
    rm -rf out/ data/processed/ data/summary/ runs/
    @echo "Cleaned generated outputs."
