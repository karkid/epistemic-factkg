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


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  DEV                                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

[group("Dev")]
[doc("Install dependencies and set up the environment")]
init:
    uv venv && uv sync && uv pip install -e ".[dev,notebook]"


[group("Dev")]
[doc("Lint and format check (ruff)")]
lint:
    uv run ruff format --check . && uv run ruff check .


[group("Dev")]
[doc("Auto-fix lint and format issues (ruff)")]
fix:
    uv run ruff format . && uv run ruff check --fix .


[group("Dev")]
[doc("Run test suite (pytest)")]
test:
    uv run pytest tests/ -v --tb=short


[group("Dev")]
[doc("Delete all generated outputs: out/, data/processed/, data/summary/, runs/")]
clean:
    rm -rf out/ data/processed/ data/summary/ runs/
    @echo "Cleaned generated outputs."


[group("Dev")]
[doc("Run a pipeline or one step: just run <data|model> [STEP]")]
run PIPELINE="" STEP="":
    #!/usr/bin/env bash
    set -euo pipefail
    PIPELINE="{{PIPELINE}}"
    STEP="{{STEP}}"
    case "${PIPELINE}" in
      data)
        case "${STEP}" in
          "")        just build && just validate && just report ;;
          rebuild)   just build rebuild=true && just validate && just report ;;
          build)     just build ;;
          validate)  just validate ;;
          report)    just report ;;
          *) echo "Unknown data step '${STEP}'. Available: rebuild  build  validate  report"; exit 1 ;;
        esac
        ;;
      model)
        case "${STEP}" in
          "")      just graph && just train && just eval ;;
          build)   just graph ;;
          train)   just train ;;
          eval)    just eval ;;
          *) echo "Unknown model step '${STEP}'. Available: build  train  eval"; exit 1 ;;
        esac
        ;;
      *)
        echo "Usage: just run <data|model> [STEP]"
        echo "  data  steps: rebuild  build  validate  report"
        echo "  model steps: build  train  eval"
        exit 1
        ;;
    esac


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  DATA PIPELINE                                                             ║
# ║  Order: build → validate → report  (pass rebuild=true to re-simulate AI2THOR first)
# ╚═════════════════════════════════════════════════════════════════════════════╝

[group("Data Pipeline")]
[doc("Build training data: generate synthetic, merge sources, filter, split (rebuild=true re-simulates AI2THOR first)")]
build rebuild="false":
    #!/usr/bin/env bash
    set -euo pipefail
    REBUILD_FLAG=""
    if [ "{{rebuild}}" = "true" ]; then
        REBUILD_FLAG="--rebuild --config {{CONFIG}} --out-rdf {{KG_TTL}} --ai2thor-dir {{AI2THOR_RAW_DIR}}"
    elif [ ! -f "{{AI2THOR_CLAIMS}}" ]; then
        echo "Error: {{AI2THOR_CLAIMS}} not found. Run 'just build rebuild=true' to generate it." >&2
        exit 1
    fi
    if [ ! -f "{{SYNTHETIC_JSONL}}" ]; then
        echo "--- generating synthetic claims ---"
        mkdir -p {{SYNTHETIC_RAW_DIR}}
        uv run python -m src.pipeline.data.generate synthetic \
            --config {{CONFIG}} \
            --registry {{REGISTRY}} \
            --seed-pool {{SEED_POOL}} \
            --ai2thor-claims {{AI2THOR_CLAIMS}} \
            --n-records 1000 \
            --output {{SYNTHETIC_JSONL}}
    fi
    mkdir -p out/unified out/training out/intermediate
    uv run python -m src.pipeline.data.build \
        $REBUILD_FLAG \
        --registry {{REGISTRY}} \
        --averitec {{RAW_AVERITEC_TRAIN}} {{RAW_AVERITEC_DEV}} \
        --ai2thor {{AI2THOR_CLAIMS}} \
        --synthetic {{SYNTHETIC_JSONL}} \
        --unified-out {{UNIFIED_JSONL}} \
        --training-out {{TRAINING_JSONL}} \
        --intermediate-dir out/intermediate
    mkdir -p {{SPLITS_DIR}}
    uv run python -m src.pipeline.data.split_dataset \
        --input {{TRAINING_JSONL}} \
        --output-dir {{SPLITS_DIR}} \
        --seed 42 \
        --verbose


[group("Data Pipeline")]
[doc("Validate unified JSONL schema + training Pramana distribution (ADR-012)")]
validate:
    mkdir -p {{REPORT_DIR}}
    uv run python -m src.pipeline.data.validate unified \
        --files {{UNIFIED_JSONL}} \
        --out {{VALIDATION_JSON}}
    uv run python -m src.pipeline.data.validate training \
        --input {{TRAINING_JSONL}} \
        --config {{CONFIG}} \
        --out {{TRAINING_VALIDATION}}


[group("Data Pipeline")]
[doc("Generate dataset quality report (markdown + charts) from validation output")]
report:
    uv run python -m src.pipeline.model.report \
        --summary {{VALIDATION_JSON}} \
        --out_dir {{REPORT_DIR}} \
        --training-summary {{TRAINING_VALIDATION}} \
        --title "Epistemic FactKG Dataset Report"


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  MODEL PIPELINE                                                            ║
# ║  Order: graph → train → eval                                               ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

[group("Model Pipeline")]
[doc("Build PyG HeteroData graph dataset from filtered training JSONL")]
graph:
    mkdir -p out/graphs
    uv run python -m src.pipeline.model.build_graphs \
        --input {{TRAINING_JSONL}} \
        --output {{GRAPH_DATASET}} \
        --embed-cache out/graphs/embed_cache.pkl \
        --verbose


[group("Model Pipeline")]
[doc("Train EpistemicHGNN (multi-head neuro-symbolic — stance + IS multi-task loss)")]
train:
    mkdir -p {{CHECKPOINTS_DIR}}
    uv run python -m src.pipeline.model.train \
        --dataset {{GRAPH_DATASET}} \
        --jsonl {{TRAINING_JSONL}} \
        --splits-dir {{SPLITS_DIR}} \
        --checkpoint-dir {{CHECKPOINTS_DIR}} \
        --epochs 50 \
        --lr 1e-3 \
        --batch-size 32 \
        --device cpu \
        --verbose


[group("Model Pipeline")]
[doc("Evaluate EpistemicHGNN on test set — outputs stance, IS, and verdict metrics")]
eval:
    mkdir -p {{RESULTS_DIR}}
    uv run python -m src.pipeline.model.evaluate \
        --checkpoint {{CHECKPOINTS_DIR}}/best_model.pt \
        --jsonl {{TRAINING_JSONL}} \
        --splits-dir {{SPLITS_DIR}} \
        --output {{RESULTS_DIR}}
