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
KG_TTL              := "out/model/knowledge_graph.ttl"
UNIFIED_JSONL       := "out/data/unified/epistemic_factkg.jsonl"
VALIDATION_JSON     := "out/reports/data/validation.json"
REPORT_DIR          := "out/reports/data"
TRAINING_JSONL      := "out/data/training/epistemic_factkg_training.jsonl"
TRAINING_VALIDATION := "out/reports/data/training_validation.json"
GRAPH_DATASET       := "out/model/graphs/graph_dataset.pt"
GRAPH_DATASET_NLI   := "out/model/graphs/graph_dataset_nli.pt"
SPLITS_DIR          := "out/data/splits"
MODEL_NAME          := "v1-hgnn"
CHECKPOINTS_DIR     := "out/model/" + MODEL_NAME + "/checkpoints"
MODEL_REPORT_DIR    := "out/reports/model/" + MODEL_NAME
RESULTS_DIR         := "out/reports/model/" + MODEL_NAME + "/eval"


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
[doc("Delete all generated outputs: out/ (includes runs/), data/processed/, data/summary/")]
clean:
    rm -rf out/ data/processed/ data/summary/
    @echo "Cleaned generated outputs."


[group("Dev")]
[doc("Run a pipeline or step: just run <data|model> [STEP] [MODELS='all']")]
run PIPELINE="" STEP="" MODELS="all":
    #!/usr/bin/env bash
    set -euo pipefail
    PIPELINE="{{PIPELINE}}"
    STEP="{{STEP}}"
    MODELS="{{MODELS}}"
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
          list)    uv run python -m src.pipeline.model.orchestrate list ;;
          "")      just graph && uv run python -m src.pipeline.model.orchestrate run \
                       --models "${MODELS}" \
                       --jsonl {{TRAINING_JSONL}} \
                       --splits-dir {{SPLITS_DIR}} \
                       --graph {{GRAPH_DATASET}} ;;
          build)   just graph ;;
          train)   uv run python -m src.pipeline.model.orchestrate train \
                       --models "${MODELS}" \
                       --jsonl {{TRAINING_JSONL}} \
                       --splits-dir {{SPLITS_DIR}} ;;
          eval)    uv run python -m src.pipeline.model.orchestrate eval \
                       --models "${MODELS}" \
                       --jsonl {{TRAINING_JSONL}} \
                       --splits-dir {{SPLITS_DIR}} ;;
          compare) uv run python -m src.pipeline.model.orchestrate compare \
                       --models "${MODELS}" ;;
          *) echo "Unknown model step '${STEP}'. Available: list  build  train  eval  compare"; exit 1 ;;
        esac
        ;;
      *)
        echo "Usage: just run <data|model> [STEP] [MODELS]"
        echo "  data  steps: rebuild  build  validate  report"
        echo "  model steps: list  build  train  eval  compare"
        echo "  MODELS: comma-separated names or 'all' (default)"
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
            --output {{SYNTHETIC_JSONL}}
    fi
    mkdir -p out/data/unified out/data/training out/data/intermediate
    uv run python -m src.pipeline.data.build \
        $REBUILD_FLAG \
        --registry {{REGISTRY}} \
        --averitec {{RAW_AVERITEC_TRAIN}} {{RAW_AVERITEC_DEV}} \
        --ai2thor {{AI2THOR_CLAIMS}} \
        --synthetic {{SYNTHETIC_JSONL}} \
        --unified-out {{UNIFIED_JSONL}} \
        --training-out {{TRAINING_JSONL}} \
        --intermediate-dir out/data/intermediate
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
    mkdir -p out/model/graphs
    uv run python -m src.pipeline.model.build_graphs \
        --input {{TRAINING_JSONL}} \
        --output {{GRAPH_DATASET}} \
        --embed-cache out/model/graphs/embed_cache.pkl \
        --verbose


[group("Model Pipeline")]
[doc("Build NLI-enhanced graph dataset for v3-nli (403d evidence features)")]
graph-nli:
    mkdir -p out/model/graphs
    uv run python -m src.pipeline.model.build_graphs \
        --input {{TRAINING_JSONL}} \
        --output {{GRAPH_DATASET_NLI}} \
        --embed-cache out/model/graphs/embed_cache.pkl \
        --use-nli \
        --verbose


[group("Model Pipeline")]
[doc("Train a model (default: MODEL_NAME). Override: just train baseline")]
train model=MODEL_NAME:
    mkdir -p out/model/{{model}}/checkpoints out/reports/model/{{model}}
    uv run python -m src.pipeline.model.train \
        --model {{model}} \
        --model-name {{model}} \
        --jsonl {{TRAINING_JSONL}} \
        --splits-dir {{SPLITS_DIR}} \
        --checkpoint-dir out/model/{{model}}/checkpoints \
        --report-dir out/reports/model/{{model}} \
        --epochs 100 \
        --lr 3e-4 \
        --batch-size 32 \
        --device cpu \
        --verbose


[group("Model Pipeline")]
[doc("Evaluate a model on test set (default: MODEL_NAME). Override: just eval baseline")]
eval model=MODEL_NAME:
    mkdir -p out/reports/model/{{model}}/eval
    uv run python -m src.pipeline.model.evaluate \
        --model {{model}} \
        --model-name {{model}} \
        --checkpoint out/model/{{model}}/checkpoints/best_model.pt \
        --jsonl {{TRAINING_JSONL}} \
        --splits-dir {{SPLITS_DIR}} \
        --output out/reports/model/{{model}}/eval


[group("Model Pipeline")]
[doc("Compare two evaluated models side by side: just compare v1-hgnn baseline")]
compare model1 model2:
    uv run python -m src.pipeline.model.compare \
        --model1 {{model1}} --dir1 out/reports/model/{{model1}}/eval \
        --model2 {{model2}} --dir2 out/reports/model/{{model2}}/eval \
        --out out/reports/model/comparison_{{model1}}_vs_{{model2}}.md


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  DEMO                                                                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

[group("Demo")]
[doc("Launch the Streamlit demo app (requires trained checkpoints)")]
app:
    uv run streamlit run app/app.py
