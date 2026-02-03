[default]
[doc("List all available recipes")]
default:
	@just --list

[private]
[doc("Check if uv is installed")]
check-uv:
   @uv --version >/dev/null 2>&1 || (echo "Error: uv is not installed" && exit 1)

[doc("Create a virtual environment and synchronize dependencies")]
init: check-uv
    @echo "Creating virtual environment and syncing dependencies..."
    uv venv
    uv sync
    @echo "Installing dependencies..."
    uv pip install -e ".[dev,notebook]"

[confirm("This will delete your virtual environment and reinstall all dependencies. Continue?")]
[doc("Reinstall all dependencies from scratch")]
reinstall:
	@echo "Deleting old virtual environment..."
	@rm -rf .venv
	just init

[doc("Install all dependencies in the environment")]
install:
    uv sync

[doc("Add a new package to the environment")]
add pkg:
    uv add {{pkg}}

[doc("Remove a package from the environment")]
remove pkg:
    uv remove {{pkg}}

[doc("Upgrade all dependencies to their latest versions")]
upgrade-deps:
    uv pip install -U pip
    uv pip install --upgrade-deps

[doc("Show uv version and environment details")]
info:
    @echo "📦 Environment Info:"
    uv --version
    uv pip list

[doc("Format code using ruff")]
format:
    uv run ruff format .

[doc("Lint code using ruff")]
lint:
    uv run ruff check .

# Knowledge Graph Generation Commands

[doc("Generate knowledge graph (optional: scene, config, output)")]
kg-build generator="ai2thor" scene="" config="" output="output/knowledge_graph.ttl":
    uv run python -m src.generators.cli {{generator}} --output {{output}} {{ if scene != "" { "--scenes " + scene } else { "" } }} {{ if config != "" { "--config " + config } else { "" } }}

# Dataset Generation Commands

[doc("Convert RDF/TTL to dataset with default parameters")]
rdf-to-dataset ttl_path="output/knowledge_graph.ttl" output_path="data/processed/datasets/thor_pratyaksa.jsonl":
    uv run python -m src.generators.rdf.rdf_to_dataset --ttl "{{ttl_path}}" --output "{{output_path}}" --seed 42 --onehop_per_floorplan 200 --neg_pairs_per_floorplan 60 --conj_per_floorplan 80

[doc("Convert RDF/TTL to dataset with custom parameters")]
rdf-to-dataset-custom ttl_path output_path seed="42" onehop="200" neg_pairs="60" conj="80":
    uv run python -m src.generators.rdf.rdf_to_dataset --ttl "{{ttl_path}}" --output "{{output_path}}" --seed {{seed}} --onehop_per_floorplan {{onehop}} --neg_pairs_per_floorplan {{neg_pairs}} --conj_per_floorplan {{conj}}

[doc("Validate dataset")]
validate-dataset dataset_path strict="false":
    uv run python -m src.schema.validate_dataset --input "{{dataset_path}}" {{ if strict == "true" { "--strict" } else { "" } }}

[doc("Full pipeline: Generate RDF, convert to dataset, and validate")]
full-pipeline ttl_path="output/knowledge_graph.ttl" dataset_path="data/processed/datasets/thor_pratyaksa.jsonl":
    just rdf-to-dataset "{{ttl_path}}" "{{dataset_path}}"
    just validate-dataset "{{dataset_path}}"

[doc("Check RDF dataset quality and structure")]
rdf-checker dataset_path="data/processed/datasets/thor_pratyaksa.jsonl":
    @echo "🔍 Checking dataset quality: {{dataset_path}}"
    @echo ""
    @echo "📊 Checking pramana_type (should be 0):"
    @grep -c '"pramana_type"' "{{dataset_path}}" || echo "✅ No pramana_type found (expected)"
    @echo ""
    @echo "📊 Checking evidence_type (should be >0):"
    #!/bin/bash
    @evidence_count=$(grep -Roh '"evidence_type"' "{{dataset_path}}" | wc -l | tr -d ' '); echo "Found: ${evidence_count} evidence_type instances"; if [ "${evidence_count}" -gt 0 ]; then echo "✅ evidence_type present"; else echo "❌ No evidence types found"; fi
    @echo ""
    @echo "📊 Checking for 'relation type' (should be 0):"
    #!/bin/bash
    @relation_type_count=$(grep -Roh 'relation type' "{{dataset_path}}" | wc -l | tr -d ' '); echo "Found: ${relation_type_count} relation type instances"; if [ "${relation_type_count}" -eq 0 ]; then echo "✅ No 'relation type' found"; else echo "❌ Found 'relation type' instances"; grep -n 'relation type' "{{dataset_path}}" | head; fi
    @echo ""
    @echo "📊 Checking inScene/hasObject relations (optional filter):"
    #!/bin/bash
    @inscene_count=$(grep -Roh '"inScene"\|"hasObject"' "{{dataset_path}}" | wc -l | tr -d ' '); echo "Found: ${inscene_count} inScene/hasObject relations"

# Visualization Commands

[doc("Visualize knowledge graph file")]
kg-viz file="output/knowledge_graph.ttl" output="output/visualizer/knowledge_graph.html":
    rm -rf output/visualizer
    uv run python -m src.visualizer.cli --input {{file}} --output {{output}}

[doc("Open the knowledge graph visualization in browser")]
kg-open:
    open output/visualizer/knowledge_graph.html