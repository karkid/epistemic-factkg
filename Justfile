# Epistemic FactKG — Task Runner
# Tip: just --list   |   just <command> --help

[default]
[doc("List available commands")]
default:
    @just --list

# ----------------------------
# Global variables (edit here)
# ----------------------------
RAW_AVERITEC_TRAIN := "data/raw/averitec/train.json"
RAW_AVERITEC_DEV   := "data/raw/averitec/dev.json"
RAW_AI2THOR_CLAIMS := "data/raw/ai2thor/claims_all.jsonl"

PROCESSED_DIR      := "data/processed"
SUMMARY_DIR        := "data/summary"

KG_TTL             := "out/knowledge_graph.ttl"
VIZ_HTML           := "out/visualizer/knowledge_graph.html"


# ----------------------------
# Setup & Dev
# ----------------------------
[doc("Initialize environment (uv venv + sync + editable install)")]
init:
    uv venv
    uv sync
    uv pip install -e ".[dev,notebook]"

[doc("Format + lint code")]
dev:
    uv run ruff format .
    uv run ruff check .

[doc("Run test suite")]
test:
    uv run python -m pytest tests/ -v --tb=short --cov-fail-under=10

[doc("Show environment info")]
info:
    uv --version
    uv pip list


# ----------------------------
# Knowledge graph (AI2-THOR)
# ----------------------------
[doc("Build RDF knowledge graph from AI2-THOR config")]
build-kg:
    uv run python -m src.cli.build_rdf --config configs/ai2thor_default.yaml --out {{KG_TTL}} --verbose

[doc("Visualize knowledge graph as interactive HTML")]
viz-kg file=KG_TTL:
    rm -rf out/visualizer
    uv run python -m src.cli.build_viz {{file}} --output {{VIZ_HTML}}

[doc("Open KG visualization in browser")]
open-viz:
    open {{VIZ_HTML}}


# ----------------------------
# Claim generation (AI2-THOR)
# ----------------------------
[doc("Generate AI2-THOR claims from KG → data/raw/ai2thor/")]
build-claims max_contexts="10" n_claims="2000":
    uv run python -m src.cli.build_claims {{KG_TTL}} \
        --output-dir data/raw/ai2thor \
        --max-contexts {{max_contexts}} \
        --n-claims {{n_claims}} \
        --verbose

[doc("Validate raw AI2-THOR claims against KG")]
validate-claims claims_file=RAW_AI2THOR_CLAIMS:
    uv run python -m src.cli.validate_claims {{KG_TTL}} \
        --claims-file {{claims_file}} \
        --verbose


# ----------------------------
# Unified conversion
# ----------------------------
[doc("Convert AVeriTeC + AI2-THOR to unified v2.0 JSONL → data/processed/")]
convert-unified:
    uv run python -m src.cli.convert_to_unified \
        --averitec_inputs {{RAW_AVERITEC_TRAIN}} {{RAW_AVERITEC_DEV}} \
        --ai2thor_inputs {{RAW_AI2THOR_CLAIMS}} \
        --output_dir {{PROCESSED_DIR}}

[doc("Validate unified JSONL files against v2.0 schema")]
validate-unified out="{{SUMMARY_DIR}}/unified_validation_summary.json":
    mkdir -p {{SUMMARY_DIR}}
    uv run python -m src.cli.validate_unified_dataset \
        --files \
            {{PROCESSED_DIR}}/averitec_train.jsonl \
            {{PROCESSED_DIR}}/averitec_dev.jsonl \
            {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --out {{out}} \
        --max_examples 5


# ----------------------------
# AI2-THOR dataset splitting
# ----------------------------
[doc("Split AI2-THOR by floorplan — percentage mode (default 80/10/10)")]
split-ai2thor train_pct="80" dev_pct="10" seed="13":
    uv run python -m src.cli.split_ai2thor \
        --input {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --output_dir {{PROCESSED_DIR}} \
        --mode pct \
        --train_pct {{train_pct}} --dev_pct {{dev_pct}} \
        --seed {{seed}}

[doc("Split AI2-THOR by floorplan — exact floorplan counts")]
split-ai2thor-counts n_train="6" n_dev="1" n_test="1" seed="13":
    uv run python -m src.cli.split_ai2thor \
        --input {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --output_dir {{PROCESSED_DIR}} \
        --mode counts \
        --n_train_floorplans {{n_train}} \
        --n_dev_floorplans {{n_dev}} \
        --n_test_floorplans {{n_test}} \
        --seed {{seed}}

[doc("Split AI2-THOR by explicit floorplan lists")]
split-ai2thor-lists train="" dev="" test="":
    uv run python -m src.cli.split_ai2thor \
        --input {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --output_dir {{PROCESSED_DIR}} \
        --mode lists \
        --train_floorplans {{train}} \
        --dev_floorplans {{dev}} \
        --test_floorplans {{test}}

[doc("Validate AI2-THOR split files (train/dev/test)")]
validate-ai2thor-split out="{{SUMMARY_DIR}}/ai2thor_split_validation_summary.json":
    mkdir -p {{SUMMARY_DIR}}
    uv run python -m src.cli.validate_unified_dataset \
        --files \
            {{PROCESSED_DIR}}/ai2thor_train.jsonl \
            {{PROCESSED_DIR}}/ai2thor_dev.jsonl \
            {{PROCESSED_DIR}}/ai2thor_test.jsonl \
        --out {{out}} \
        --max_examples 5


# ----------------------------
# Pipelines (timestamped run logs)
# ----------------------------
[doc("PIPELINE: Convert + validate all unified datasets")]
pipeline-data RUN_ID="":
    #!/usr/bin/env bash
    set -euo pipefail
    RUN_ID="${RUN_ID:-$(date -u +"%Y-%m-%d_%H-%M-%S")}"
    echo "RUN_ID=$RUN_ID"
    mkdir -p "runs/$RUN_ID/summary" "runs/$RUN_ID/logs"

    uv run python -m src.cli.convert_to_unified \
        --averitec_inputs {{RAW_AVERITEC_TRAIN}} {{RAW_AVERITEC_DEV}} \
        --ai2thor_inputs {{RAW_AI2THOR_CLAIMS}} \
        --output_dir {{PROCESSED_DIR}} \
        | tee "runs/$RUN_ID/logs/convert.log"

    uv run python -m src.cli.validate_unified_dataset \
        --files \
            {{PROCESSED_DIR}}/averitec_train.jsonl \
            {{PROCESSED_DIR}}/averitec_dev.jsonl \
            {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --out "runs/$RUN_ID/summary/validation.json" \
        --max_examples 5 \
        | tee "runs/$RUN_ID/logs/validate.log"

    echo "Done → runs/$RUN_ID/"

[doc("PIPELINE: Split AI2-THOR + validate splits")]
pipeline-split RUN_ID="" mode="pct" train_pct="80" dev_pct="10" seed="13":
    #!/usr/bin/env bash
    set -euo pipefail
    RUN_ID="${RUN_ID:-$(date -u +"%Y-%m-%d_%H-%M-%S")}"
    echo "RUN_ID=$RUN_ID"
    mkdir -p "runs/$RUN_ID/summary" "runs/$RUN_ID/logs"

    uv run python -m src.cli.split_ai2thor \
        --input {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --output_dir {{PROCESSED_DIR}} \
        --mode {{mode}} \
        --train_pct {{train_pct}} --dev_pct {{dev_pct}} \
        --seed {{seed}} \
        | tee "runs/$RUN_ID/logs/split.log"

    uv run python -m src.cli.validate_unified_dataset \
        --files \
            {{PROCESSED_DIR}}/ai2thor_train.jsonl \
            {{PROCESSED_DIR}}/ai2thor_dev.jsonl \
            {{PROCESSED_DIR}}/ai2thor_test.jsonl \
        --out "runs/$RUN_ID/summary/split_validation.json" \
        --max_examples 5 \
        | tee "runs/$RUN_ID/logs/validate_split.log"

    echo "Done → runs/$RUN_ID/"

[doc("PIPELINE: Full end-to-end (KG → claims → convert → validate → split → validate)")]
pipeline-all RUN_ID="" max_contexts="10" n_claims="2000":
    just build-claims {{max_contexts}} {{n_claims}}
    just validate-claims
    just pipeline-data "{{RUN_ID}}"
    just pipeline-split "{{RUN_ID}}"


# ----------------------------
# Analysis & Reports
# ----------------------------
[doc("Build dataset report (md + plots) from a run's validation summary")]
report RUN_ID:
    mkdir -p runs/{{RUN_ID}}/report
    uv run python -m src.cli.build_dataset_report \
        --summary runs/{{RUN_ID}}/summary/validation.json \
        --out_dir runs/{{RUN_ID}}/report \
        --title "Epistemic FactKG — Run {{RUN_ID}}"

[doc("Analyze AVeriTeC raw data (claim types, evidence modalities, etc.)")]
analyze-averitec:
    uv run python -m src.cli.analyze_averitec_raw \
        --inputs {{RAW_AVERITEC_TRAIN}} {{RAW_AVERITEC_DEV}} \
        --out runs/averitec_profile.json
