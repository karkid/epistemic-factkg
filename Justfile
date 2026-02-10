# Epistemic FactKG - Essential Commands
# -----------------------------------
# Convention:
# - raw data:     data/raw/
# - processed:    data/processed/
# - schema:       data/schema/
# - summaries:    data/summary/ or runs/<RUN_ID>/summary/
# - run logs:     runs/<RUN_ID>/logs/
#
# Tip:
#   just --list
#   just <command> --help   (if underlying python CLI supports it)

[default]
[doc("List available commands")]
default:
    @just --list

[doc("Show recommended pipeline usage")]
docs:
    @echo "Recommended:"
    @echo "  just pipeline-all"
    @echo "  just pipeline-all n_claims=5000 max_contexts=50"
    @echo "  just pipeline-split-ai2thor mode=counts n_train_floorplans=6 n_dev_floorplans=1 n_test_floorplans=1"

# ----------------------------
# Global variables (edit here)
# ----------------------------
RAW_AVERITEC_TRAIN := "data/raw/averitec/train.json"
RAW_AVERITEC_DEV   := "data/raw/averitec/dev.json"
RAW_AI2THOR_CLAIMS := "data/raw/ai2thor/claims_all.jsonl"

PROCESSED_DIR      := "data/processed"
SCHEMA_PATH        := "data/schema/unified_schema.json"
SUMMARY_DIR        := "data/summary"

KG_TTL             := "out/knowledge_graph.ttl"
VIZ_HTML           := "out/visualizer/knowledge_graph.html"


# ----------------------------
# Setup & Dev
# ----------------------------
[doc("Initialize environment (uv venv + sync + editable install)")]
init:
    @echo "🏗️ Setting up environment..."
    uv venv
    uv sync
    uv pip install -e ".[dev,notebook]"

[doc("Format + lint code")]
dev:
    @echo "🔧 Formatting code..."
    uv run ruff format .
    @echo "🔍 Linting code..."
    uv run ruff check .

[doc("Run test suite")]
test:
    @echo "🧪 Running tests..."
    uv run python -m pytest tests/ -v --tb=short --cov-fail-under=10

[doc("Show environment info")]
info:
    @echo "📦 Environment:"
    uv --version
    uv pip list


# ----------------------------
# Knowledge graph (AI2-THOR)
# ----------------------------
[doc("Build knowledge graph (RDF TTL) from AI2-THOR config")]
build-kg:
    uv run python -m src.cli.build_rdf --config configs/ai2thor_default.yaml --out {{KG_TTL}} --verbose

[doc("Visualize knowledge graph (HTML)")]
viz-kg file=KG_TTL:
    @echo "🎨 Creating visualization..."
    rm -rf out/visualizer
    uv run python -m src.cli.build_viz {{file}} --output {{VIZ_HTML}}

[doc("Open KG visualization in browser")]
open-viz:
    open {{VIZ_HTML}}


# ----------------------------
# Claim generation (AI2-THOR)
# ----------------------------
[doc("Generate AI2-THOR claims from KG (writes into data/raw/ai2thor/)")]
build-claims max_contexts="10" n_claims="2000":
    @echo "🧾 Building AI2-THOR claims..."
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
[doc("Convert Averitec + AI2-THOR into unified schema (writes JSONL into data/processed/)")]
convert-unified:
    uv run python -m src.cli.convert_to_unified \
        --averitec_inputs {{RAW_AVERITEC_TRAIN}} {{RAW_AVERITEC_DEV}} \
        --ai2thor_inputs {{RAW_AI2THOR_CLAIMS}} \
        --output_dir {{PROCESSED_DIR}}

[doc("Validate unified JSONL datasets (writes summary JSON)")]
validate-unified out="{{SUMMARY_DIR}}/unified_validation_summary.json":
    mkdir -p {{SUMMARY_DIR}}
    uv run python -m src.cli.validate_unified_dataset \
        --schema {{SCHEMA_PATH}} \
        --files {{PROCESSED_DIR}}/averitec_train.jsonl {{PROCESSED_DIR}}/averitec_dev.jsonl {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --out {{out}} \
        --max_examples 5


# ----------------------------
# AI2-THOR split
# ----------------------------
[doc("Split AI2-THOR by floorplan (simple hash split; may produce empty test if floorplans are few)")]
split-ai2thor:
    uv run python -m src.cli.split_ai2thor_by_floorplan \
        --input {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --output_dir {{PROCESSED_DIR}} \
        --train_pct 80 --dev_pct 10

[doc("Split AI2-THOR by floorplan (configurable; recommended)")]
split-ai2thor-config mode="pct" train_pct="80" dev_pct="10" seed="13":
    uv run python -m src.cli.split_ai2thor_by_floorplan_configurable \
        --input {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --output_dir {{PROCESSED_DIR}} \
        --mode {{mode}} \
        --train_pct {{train_pct}} --dev_pct {{dev_pct}} \
        --seed {{seed}}

[doc("Validate AI2-THOR split files (train/dev/test)")]
validate-ai2thor-split out="{{SUMMARY_DIR}}/ai2thor_split_validation_summary.json":
    mkdir -p {{SUMMARY_DIR}}
    uv run python -m src.cli.validate_unified_dataset \
        --schema {{SCHEMA_PATH}} \
        --files {{PROCESSED_DIR}}/ai2thor_train.jsonl {{PROCESSED_DIR}}/ai2thor_dev.jsonl {{PROCESSED_DIR}}/ai2thor_test.jsonl \
        --out {{out}} \
        --max_examples 5

[doc("Split AI2-THOR by floorplan using percentage split (mode=pct)")]
split-ai2thor-pct train_pct="80" dev_pct="10" seed="13":
    uv run python -m src.cli.split_ai2thor_by_floorplan_configurable \
        --input {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --output_dir {{PROCESSED_DIR}} \
        --mode pct \
        --train_pct {{train_pct}} --dev_pct {{dev_pct}} \
        --seed {{seed}}

[doc("Split AI2-THOR by floorplan using exact floorplan counts (mode=counts)")]
split-ai2thor-counts n_train="6" n_dev="1" n_test="1" seed="13":
    uv run python -m src.cli.split_ai2thor_by_floorplan_configurable \
        --input {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --output_dir {{PROCESSED_DIR}} \
        --mode counts \
        --n_train_floorplans {{n_train}} \
        --n_dev_floorplans {{n_dev}} \
        --n_test_floorplans {{n_test}} \
        --seed {{seed}}

[doc("Split AI2-THOR by explicit floorplan lists (mode=lists). Put remaining into train automatically.")]
split-ai2thor-lists train="" dev="" test="":
    uv run python -m src.cli.split_ai2thor_by_floorplan_configurable \
        --input {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --output_dir {{PROCESSED_DIR}} \
        --mode lists \
        --train_floorplans {{train}} \
        --dev_floorplans {{dev}} \
        --test_floorplans {{test}}


# ----------------------------
# Pipelines (Run IDs + logging)
# ----------------------------
[doc("PIPELINE: Convert + validate unified datasets with run-id logs")]
pipeline-data RUN_ID="":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{RUN_ID}}" = "" ]; then
        RUN_ID=$(date -u +"%Y-%m-%d_%H-%M-%S")
    else
        RUN_ID="{{RUN_ID}}"
    fi
    echo "🚀 RUN_ID=$RUN_ID"
    mkdir -p "runs/$RUN_ID/summary"
    mkdir -p "runs/$RUN_ID/logs"

    uv run python -m src.cli.convert_to_unified \
        --averitec_inputs {{RAW_AVERITEC_TRAIN}} {{RAW_AVERITEC_DEV}} \
        --ai2thor_inputs {{RAW_AI2THOR_CLAIMS}} \
        --output_dir {{PROCESSED_DIR}} \
        | tee "runs/$RUN_ID/logs/convert_unified.log"

    uv run python -m src.cli.validate_unified_dataset \
        --schema {{SCHEMA_PATH}} \
        --files {{PROCESSED_DIR}}/averitec_train.jsonl {{PROCESSED_DIR}}/averitec_dev.jsonl {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --out "runs/$RUN_ID/summary/unified_validation_summary.json" \
        --max_examples 5 \
        | tee "runs/$RUN_ID/logs/validate_unified.log"

    echo "✅ pipeline-data done → runs/$RUN_ID/summary/unified_validation_summary.json"

[doc("PIPELINE: Split AI2-THOR + validate splits with run-id logs")]
pipeline-split-ai2thor RUN_ID="" mode="pct" train_pct="80" dev_pct="10" seed="13":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{RUN_ID}}" = "" ]; then
        RUN_ID=$(date -u +"%Y-%m-%d_%H-%M-%S")
    else
        RUN_ID="{{RUN_ID}}"
    fi
    echo "🚀 RUN_ID=$RUN_ID"
    mkdir -p "runs/$RUN_ID/summary"
    mkdir -p "runs/$RUN_ID/logs"

    uv run python -m src.cli.split_ai2thor_by_floorplan_configurable \
        --input {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --output_dir {{PROCESSED_DIR}} \
        --mode {{mode}} \
        --train_pct {{train_pct}} --dev_pct {{dev_pct}} \
        --seed {{seed}} \
        | tee "runs/$RUN_ID/logs/split_ai2thor.log"

    uv run python -m src.cli.validate_unified_dataset \
        --schema {{SCHEMA_PATH}} \
        --files {{PROCESSED_DIR}}/ai2thor_train.jsonl {{PROCESSED_DIR}}/ai2thor_dev.jsonl {{PROCESSED_DIR}}/ai2thor_test.jsonl \
        --out "runs/$RUN_ID/summary/ai2thor_split_validation_summary.json" \
        --max_examples 5 \
        | tee "runs/$RUN_ID/logs/validate_ai2thor_split.log"

    echo "✅ pipeline-split-ai2thor done → runs/$RUN_ID/summary/ai2thor_split_validation_summary.json"

[doc("PIPELINE: Split AI2-THOR + validate (counts mode)")]
pipeline-split-ai2thor-counts RUN_ID="" n_train="6" n_dev="1" n_test="1" seed="13":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{RUN_ID}}" = "" ]; then
        RUN_ID=$(date -u +"%Y-%m-%d_%H-%M-%S")
    else
        RUN_ID="{{RUN_ID}}"
    fi
    echo "🚀 RUN_ID=$RUN_ID"
    mkdir -p "runs/$RUN_ID/summary"
    mkdir -p "runs/$RUN_ID/logs"

    uv run python -m src.cli.split_ai2thor_by_floorplan_configurable \
        --input {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --output_dir {{PROCESSED_DIR}} \
        --mode counts \
        --n_train_floorplans {{n_train}} \
        --n_dev_floorplans {{n_dev}} \
        --n_test_floorplans {{n_test}} \
        --seed {{seed}} \
        | tee "runs/$RUN_ID/logs/split_ai2thor.log"

    uv run python -m src.cli.validate_unified_dataset \
        --schema {{SCHEMA_PATH}} \
        --files {{PROCESSED_DIR}}/ai2thor_train.jsonl {{PROCESSED_DIR}}/ai2thor_dev.jsonl {{PROCESSED_DIR}}/ai2thor_test.jsonl \
        --out "runs/$RUN_ID/summary/ai2thor_split_validation_summary.json" \
        --max_examples 5 \
        | tee "runs/$RUN_ID/logs/validate_ai2thor_split.log"


[doc("PIPELINE: Full end-to-end data pipeline (claims → convert → validate → split → validate)")]
pipeline-all RUN_ID="" max_contexts="10" n_claims="2000":
    just build-claims {{max_contexts}} {{n_claims}}
    just validate-claims
    just pipeline-data "{{RUN_ID}}"
    just pipeline-split-ai2thor "{{RUN_ID}}"


# ----------------------------
# Reports
# ----------------------------
[doc("Build dataset report (md + plots) from a validation summary JSON (requires RUN_ID)")]
report-dataset RUN_ID:
    @set -euo pipefail
    mkdir -p runs/{{RUN_ID}}/report
    uv run python -m src.cli.build_dataset_report \
        --summary runs/{{RUN_ID}}/summary/unified_validation_summary.json \
        --out_dir runs/{{RUN_ID}}/report \
        --title "Epistemic FactKG Dataset Report (RUN {{RUN_ID}})"
        
[doc("Analyze Averitec raw data (claim types, evidence types, etc.) and write profile JSON")]
analyze-averitec:
    uv run python -m src.cli.analyze_averitec_raw \
    --inputs data/raw/averitec/train.json data/raw/averitec/dev.json \
    --out runs/averitec_profile.json
