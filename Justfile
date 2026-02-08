# Epistemic FactKG - Essential Commands

[default]
[doc("List available commands")]
default:
    @just --list

# Setup
[doc("Initialize environment")]
init:
    @echo "🏗️ Setting up environment..."
    uv venv
    uv sync
    uv pip install -e ".[dev,notebook]"

# Core Commands
[doc("Build knowledge graph")]
build:
    uv run python -m src.cli.build_rdf --config configs/ai2thor_default.yaml --out out/knowledge_graph.ttl --verbose

[doc("Build semantic claims (WIP)")]
build-semantic config="":
    @echo "🧠 Building semantic claims..."
    uv run python scripts/semantic_test.py {{ if config != "" { "--config " + config } else { "" } }}

[doc("Visualize knowledge graph")]
viz file="out/knowledge_graph.ttl":
    @echo "🎨 Creating visualization..."
    rm -rf out/visualizer
    uv run python -m src.cli.build_viz {{file}} --output out/visualizer/knowledge_graph.html

[doc("Open visualization in browser")]
open:
    open out/visualizer/knowledge_graph.html

[doc("Run test suite")]
test:
    @echo "🧪 Running all tests in tests/ folder..."
    uv run python -m pytest tests/ -v --tb=short --cov-fail-under=10

[doc("Format and lint code")]
dev:
    @echo "🔧 Formatting code..."
    uv run ruff format .
    @echo "🔍 Linting code..."
    uv run ruff check .

[doc("Show environment info")]
info:
    @echo "📦 Environment:"
    uv --version
    uv pip list

[doc("build claims")]
build-claims:
    @echo "This is a temporary command for testing purposes."
    uv run python -m src.cli.build_claims out/knowledge_graph.ttl \
    --output-dir out \
    --max-contexts 5 \
    --n-claims 20 \
    --verbose

[doc("validate claims")]
validate-claims:
    uv run python -m src.cli.validate_claims out/knowledge_graph.ttl \
    --claims-file out/claims_all.jsonl \
    --verbose