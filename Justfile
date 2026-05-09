# Epistemic FactKG
# Run `just` to list all commands.

[default]
[doc("List available commands")]
default:
    @just --list

# ── Variables ────────────────────────────────────────────────────────────────
CONFIG              := "configs/config.yaml"
AI2THOR_RAW_DIR     := "data/raw/ai2thor"
AI2THOR_CLAIMS      := AI2THOR_RAW_DIR + "/claims_all.jsonl"
RAW_AVERITEC_TRAIN  := "data/raw/averitec/train.json"
RAW_AVERITEC_DEV    := "data/raw/averitec/dev.json"
KG_TTL              := "out/knowledge_graph.ttl"
UNIFIED_JSONL       := "out/unified/epistemic_factkg.jsonl"
VALIDATION_JSON     := "out/report/validation.json"
REPORT_DIR          := "out/report"


# ── Setup ────────────────────────────────────────────────────────────────────
[doc("Install dependencies and set up the environment")]
init:
    uv venv && uv sync && uv pip install -e ".[dev,notebook]"


# ── Build ────────────────────────────────────────────────────────────────────
[doc("Build KG → generate AI2THOR claims → convert + merge all datasets to unified JSONL")]
build max_contexts="10" n_claims="2000":
    uv run python -m src.cli.build_rdf \
        --config {{CONFIG}} --out {{KG_TTL}} --verbose
    uv run python -m src.cli.build_claims {{KG_TTL}} \
        --output-dir {{AI2THOR_RAW_DIR}} \
        --max-contexts {{max_contexts}} \
        --n-claims {{n_claims}} \
        --verbose
    uv run python -m src.cli.convert_to_unified \
        --config {{CONFIG}} \
        --averitec {{RAW_AVERITEC_TRAIN}} {{RAW_AVERITEC_DEV}} \
        --ai2thor {{AI2THOR_CLAIMS}} \
        --output {{UNIFIED_JSONL}} \
        --intermediate_dir out/intermediate


# ── Validate ─────────────────────────────────────────────────────────────────
[doc("Validate the unified JSONL (schema + semantic + pramana checks for all datasets)")]
validate:
    mkdir -p {{REPORT_DIR}}
    uv run python -m src.cli.validate_unified_dataset \
        --files {{UNIFIED_JSONL}} \
        --out {{VALIDATION_JSON}}


# ── Report ───────────────────────────────────────────────────────────────────
[doc("Generate dataset report (markdown + charts) from validation output")]
report:
    uv run python -m src.cli.build_dataset_report \
        --summary {{VALIDATION_JSON}} \
        --out_dir {{REPORT_DIR}} \
        --title "Epistemic FactKG Dataset Report"


# ── Test ─────────────────────────────────────────────────────────────────────
[doc("Lint (ruff) and run test suite")]
test:
    uv run ruff format . && uv run ruff check .
    uv run pytest tests/ -v --tb=short


# ── Full pipeline ─────────────────────────────────────────────────────────────
[doc("Full pipeline: build → validate → report (logs saved to runs/<RUN_ID>/)")]
run RUN_ID="" max_contexts="10" n_claims="2000":
    #!/usr/bin/env bash
    set -euo pipefail
    RUN_ID="${RUN_ID:-$(date -u +"%Y-%m-%d_%H-%M-%S")}"
    echo "RUN_ID=$RUN_ID"
    mkdir -p "runs/$RUN_ID/logs"

    just build {{max_contexts}} {{n_claims}} 2>&1 | tee "runs/$RUN_ID/logs/build.log"
    just validate                              2>&1 | tee "runs/$RUN_ID/logs/validate.log"
    just report                               2>&1 | tee "runs/$RUN_ID/logs/report.log"

    echo "Done → runs/$RUN_ID/"


# ── Clean ────────────────────────────────────────────────────────────────────
[doc("Delete all generated outputs: out/, data/processed/, data/summary/, runs/")]
clean:
    rm -rf out/ data/processed/ data/summary/ runs/
    @echo "Cleaned generated outputs."
