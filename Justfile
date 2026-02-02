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
kg-build generator="ai2thor" scene="" config="" output="knowledge_graph.ttl":
    uv run python -m src.generators.cli {{generator}} --output {{output}} {{ if scene != "" { "--scenes " + scene } else { "" } }} {{ if config != "" { "--config " + config } else { "" } }}

# Visualization Commands

[doc("Visualize knowledge graph file")]
kg-viz file="knowledge_graph.ttl" title="Knowledge Graph":
    uv run python -m src.visualization.kg_visualizer --input {{file}} --title "{{title}}" --output out/kg_visualization.html

[doc("Open the knowledge graph visualization in browser")]
kg-open:
    open out/kg_visualization.html