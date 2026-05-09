# Epistemic FactKG
# Run `just` to list all commands.

[default]
[doc("List available commands")]
default:
    @just --list

# ── Variables ────────────────────────────────────────────────────────────────
RAW_AVERITEC_TRAIN := "data/raw/averitec/train.json"
RAW_AVERITEC_DEV   := "data/raw/averitec/dev.json"
RAW_AI2THOR_CLAIMS := "data/raw/ai2thor/claims_all.jsonl"
PROCESSED_DIR      := "data/processed"
KG_TTL             := "out/knowledge_graph.ttl"
VIZ_HTML           := "out/visualizer/knowledge_graph.html"


# ── Setup ────────────────────────────────────────────────────────────────────
[doc("Install dependencies and set up the environment")]
init:
    uv venv && uv sync && uv pip install -e ".[dev,notebook]"

[doc("Format and lint source code")]
dev:
    uv run ruff format . && uv run ruff check .

[doc("Run the test suite")]
test:
    uv run pytest tests/ -v --tb=short


# ── Build ────────────────────────────────────────────────────────────────────
[doc("Build the AI2-THOR RDF knowledge graph → out/knowledge_graph.ttl")]
build-kg:
    uv run python -m src.cli.build_rdf \
        --config configs/ai2thor_default.yaml --out {{KG_TTL}} --verbose

[doc("Generate AI2-THOR claims from the KG → data/raw/ai2thor/claims_all.jsonl")]
build-claims max_contexts="10" n_claims="2000":
    uv run python -m src.cli.build_claims {{KG_TTL}} \
        --output-dir data/raw/ai2thor \
        --max-contexts {{max_contexts}} \
        --n-claims {{n_claims}} \
        --verbose


# ── Data pipeline ────────────────────────────────────────────────────────────
[doc("Convert all datasets to unified v2.0 JSONL → data/processed/")]
convert:
    uv run python -m src.cli.convert_to_unified \
        --averitec_inputs {{RAW_AVERITEC_TRAIN}} {{RAW_AVERITEC_DEV}} \
        --ai2thor_inputs {{RAW_AI2THOR_CLAIMS}} \
        --output_dir {{PROCESSED_DIR}}

[doc("Validate all processed JSONL files against the v2.0 schema")]
validate:
    mkdir -p data/summary
    uv run python -m src.cli.validate_unified_dataset \
        --files \
            {{PROCESSED_DIR}}/averitec_train.jsonl \
            {{PROCESSED_DIR}}/averitec_dev.jsonl \
            {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --out data/summary/validation.json

[doc("Split AI2-THOR by floorplan into train/dev/test. mode: pct | counts | lists")]
split mode="pct" train_pct="80" dev_pct="10" n_train="6" n_dev="1" n_test="1" seed="13":
    uv run python -m src.cli.split_ai2thor \
        --input {{PROCESSED_DIR}}/ai2thor_claims_all.jsonl \
        --output_dir {{PROCESSED_DIR}} \
        --mode {{mode}} \
        --train_pct {{train_pct}} --dev_pct {{dev_pct}} \
        --n_train_floorplans {{n_train}} \
        --n_dev_floorplans {{n_dev}} \
        --n_test_floorplans {{n_test}} \
        --seed {{seed}}

[doc("Full pipeline: build-claims → convert → validate → split (logs → runs/<RUN_ID>/)")]
run RUN_ID="" max_contexts="10" n_claims="2000":
    #!/usr/bin/env bash
    set -euo pipefail
    RUN_ID="${RUN_ID:-$(date -u +"%Y-%m-%d_%H-%M-%S")}"
    echo "RUN_ID=$RUN_ID"
    mkdir -p "runs/$RUN_ID/logs"

    just build-claims {{max_contexts}} {{n_claims}} | tee "runs/$RUN_ID/logs/build.log"
    just convert                                   | tee "runs/$RUN_ID/logs/convert.log"
    just validate                                  | tee "runs/$RUN_ID/logs/validate.log"
    just split                                     | tee "runs/$RUN_ID/logs/split.log"

    echo "Done → runs/$RUN_ID/"


# ── Extras ───────────────────────────────────────────────────────────────────
[doc("Build and open the interactive KG visualization in the browser")]
viz:
    rm -rf out/visualizer
    uv run python -m src.cli.build_viz {{KG_TTL}} --output {{VIZ_HTML}}
    open {{VIZ_HTML}}

[doc("Generate a dataset report (md + plots) from a run. Usage: just report <RUN_ID>")]
report RUN_ID:
    mkdir -p runs/{{RUN_ID}}/report
    uv run python -m src.cli.build_dataset_report \
        --summary runs/{{RUN_ID}}/summary/validation.json \
        --out_dir runs/{{RUN_ID}}/report \
        --title "Epistemic FactKG — Run {{RUN_ID}}"
