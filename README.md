# Epistemic FactKG

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A knowledge graph generation and claim validation framework for epistemic fact checking. This project processes structured data from AI2-THOR simulations and AVERITEC datasets to build RDF knowledge graphs and generate verifiable claims with epistemic labels.

## Features

- 🏗️ **Knowledge Graph Construction**: Build RDF/TTL knowledge graphs from AI2-THOR simulation data
- 📝 **Claim Generation**: Automatically generate claims from knowledge graphs with configurable complexity
- ✅ **Claim Validation**: Validate claims against ontologies and schemas
- 🔄 **Dataset Conversion**: Unified schema for AI2-THOR and AVERITEC datasets
- 🎨 **Visualization**: Interactive HTML visualizations of knowledge graphs
- 🔍 **SPARQL Queries**: Query knowledge graphs using standard RDF tools

## Pre-installation

Before setting up the project, ensure you have the required tools installed:

### 1. Install Python 3.14+

**Windows:**
```powershell
# Using winget
winget install Python.Python.3.14

# Or download from python.org
# https://www.python.org/downloads/
```

**macOS:**
```bash
# Using Homebrew
brew install python@3.14
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.14 python3.14-venv

# Fedora
sudo dnf install python3.14
```

### 2. Install uv (Package Manager)

**Windows:**
```powershell
# Using PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

**macOS/Linux:**
```bash
# Using curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### 3. Install Just (Task Runner)

**Windows:**
```powershell
# Using Cargo (Rust)
cargo install just

# Or using Scoop
scoop install just

# Or download pre-built binary from:
# https://github.com/casey/just/releases
```

**macOS:**
```bash
# Using Homebrew
brew install just
```

**Linux:**
```bash
# Using Cargo (Rust)
cargo install just

# Or download pre-built binary
wget https://github.com/casey/just/releases/latest/download/just-x86_64-unknown-linux-musl.tar.gz
tar -xzf just-x86_64-unknown-linux-musl.tar.gz
sudo mv just /usr/local/bin/
```

### 4. Verify Installation

```bash
python --version    # Should show 3.14.x or higher
uv --version        # Should show uv version
just --version      # Should show just version
```

## Installation

### Prerequisites

- Python 3.14 or higher ✓
- [uv](https://github.com/astral-sh/uv) package manager ✓
- [Just](https://github.com/casey/just) task runner ✓

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd epistemic-factkg

# Initialize environment with uv
just init

# Or manually with uv
uv venv
uv sync
uv pip install -e ".[dev]"
```

## Data Acquisition

### AVERITEC Dataset

The AVERITEC (Automated VERIfication of Textual Claims) dataset needs to be placed in `data/raw/averitec/`:

1. **Download AVERITEC data:**
   - Visit: [AVERITEC Dataset](https://fever.ai/)
   - Or use the official repository/API if available

2. **Place files in the correct location:**
   ```
   data/raw/averitec/
   ├── train.json
   ├── dev.json
   └── test.json
   ```

### AI2-THOR Data

AI2-THOR claims are generated from simulations:

1. **Generate knowledge graph:**
   ```bash
   just build-kg
   ```
   This creates `out/knowledge_graph.ttl`

2. **Generate claims:**
   ```bash
   just build-claims n_claims=2000 max_contexts=10
   ```
   This creates `data/raw/ai2thor/claims_all.jsonl`

### Directory Structure

After data acquisition and processing:
```
data/
├── raw/
│   ├── averitec/
│   │   ├── train.json      # Original AVERITEC training data
│   │   ├── dev.json        # Original AVERITEC dev data
│   │   └── test.json       # Original AVERITEC test data
│   └── ai2thor/
│       └── claims_all.jsonl # Generated AI2-THOR claims
├── processed/
│   ├── averitec_train.jsonl  # Unified format
│   ├── averitec_dev.jsonl    # Unified format
│   ├── ai2thor_claims_all.jsonl # Unified format
│   ├── ai2thor_train.jsonl   # Split dataset
│   ├── ai2thor_dev.jsonl     # Split dataset
│   └── ai2thor_test.jsonl    # Split dataset
├── schema/
│   └── unified_schema.json   # Validation schema
└── summary/
    └── *.json                # Validation summaries

out/
├── knowledge_graph.ttl       # RDF knowledge graph
└── visualizer/
    └── knowledge_graph.html  # Interactive visualization

runs/
└── <RUN_ID>/                 # Pipeline runs with timestamps
    ├── logs/
    └── summary/
```

## Quick Start

The project uses [Just](https://github.com/casey/just) for task automation. View all available commands:

```bash
just --list
```

### Basic Workflow

1. **Build a Knowledge Graph** from AI2-THOR data:
   ```bash
   just build-kg
   ```

2. **Generate Claims** from the knowledge graph:
   ```bash
   just build-claims n_claims=2000 max_contexts=10
   ```

3. **Visualize the Knowledge Graph**:
   ```bash
   just viz-kg
   just open-viz
   ```

4. **Convert Datasets** to unified schema:
   ```bash
   just convert-ai2thor
   just convert-averitec
   ```

### Run Complete Pipeline

```bash
# Run full pipeline with default settings
just pipeline-all

# Or with custom parameters
just pipeline-all n_claims=5000 max_contexts=50
```

## Output Examples

### Generated Claims (JSONL)

Each claim follows the unified schema:

```json
{
  "id": "ai2thor-kitchenFloorPlan01-000123",
  "claim": "The apple is in the refrigerator.",
  "verdict": {
    "label": "SUPPORTED",
    "confidence": 1.0
  },
  "epistemic": {
    "proof_type": "deductive",
    "modality": "actual",
    "temporal": "present"
  },
  "evidence": [
    {
      "triple": ["Apple_01", "isInside", "Fridge_01"],
      "text": "Apple_01 is inside Fridge_01"
    }
  ],
  "metadata": {
    "source": "ai2thor",
    "scene": "FloorPlan1_physics",
    "context_id": "context_001"
  }
}
```

### Knowledge Graph (TTL)

```turtle
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Apple_01 a ex:Apple ;
    ex:isInside ex:Fridge_01 ;
    ex:hasTemperature "cold" ;
    ex:isEdible true .

ex:Fridge_01 a ex:Refrigerator ;
    ex:isReceptacle true ;
    ex:isOpen false .
```

### Visualization Output

The HTML visualization provides:
- **Interactive network graph** with drag-and-drop nodes
- **Color-coded entities** by type (objects, locations, properties)
- **Hover tooltips** with entity details
- **Zoom and pan** controls
- **Search/filter** capabilities

## Project Structure

```
epistemic-factkg/
├── configs/              # Configuration files
│   └── ai2thor_default.yaml
├── data/
│   ├── raw/             # Raw datasets (AI2-THOR, AVERITEC)
│   ├── processed/       # Processed unified datasets
│   ├── schema/          # JSON schemas
│   └── summary/         # Analysis summaries
├── src/
│   ├── adapters/        # Dataset-specific adapters (AI2-THOR, AVERITEC)
│   ├── cli/             # Command-line interface scripts
│   ├── core/            # Core functionality (claims, graphs, ontology)
│   ├── infra/           # RDF infrastructure
│   ├── pipelines/       # Data processing pipelines
│   ├── visualizer/      # Graph visualization tools
│   └── utils/           # Utility functions
├── tests/               # Test suite
├── Justfile             # Task automation
└── pyproject.toml       # Project dependencies
```

## Key Commands

### Development

```bash
just dev          # Format and lint code
just test         # Run test suite
just info         # Show environment info
```

### Knowledge Graph Operations

```bash
just build-kg                    # Build RDF knowledge graph
just viz-kg                      # Create HTML visualization
just build-claims                # Generate claims from KG
```

### Dataset Operations

```bash
just convert-ai2thor            # Convert AI2-THOR to unified schema
just convert-averitec           # Convert AVERITEC to unified schema
just validate-unified           # Validate unified datasets
```

### Advanced: AI2-THOR Dataset Splitting

Split AI2-THOR data by floor plans using different modes:

**Percentage-based split:**
```bash
just split-ai2thor-pct train_pct=80 dev_pct=10 seed=13
```

**Count-based split (by number of floor plans):**
```bash
just split-ai2thor-counts n_train=6 n_dev=1 n_test=1 seed=13
```

**Explicit floor plan lists:**
```bash
just split-ai2thor-lists \
  train="FloorPlan1,FloorPlan2,FloorPlan3" \
  dev="FloorPlan4" \
  test="FloorPlan5"
```

**Validate the split:**
```bash
just validate-ai2thor-split
```

### Analysis

```bash
just report                     # Generate dataset report
just analyze-averitec           # Analyze AVERITEC raw data
```

## Viewing Knowledge Graphs

### Option 1: Built-in Visualizer

```bash
just viz-kg
just open-viz
```

### Option 2: Protégé

Download and open TTL files with [Protégé](https://protege.stanford.edu/)

### Option 3: Jena Fuseki (SPARQL Interface)

For semantic web exploration with SPARQL queries:

```bash
# Install Fuseki
brew install fuseki

# Run server
fuseki-server

# Open browser
open http://localhost:3030
```

Upload your TTL/JSON-LD files and explore triples.

## Configuration

Edit `configs/ai2thor_default.yaml` to customize:
- Scene types and floor plans
- Object randomization settings
- Claim generation parameters
- Ontology mappings

## Development

### Code Quality

```bash
# Format code
uv run ruff format .

# Run linter
uv run ruff check .

# Run tests with coverage
uv run pytest tests/ -v --cov-fail-under=10
```

### Adding New Adapters

To add support for new datasets:

1. Create adapter in `src/adapters/<dataset_name>/`
2. Implement required interfaces from `src/core/ports/`
3. Add pipeline in `src/pipelines/`
4. Update unified schema if needed

## Dependencies

Key dependencies:
- **ai2thor**: 3D simulation environment
- **rdflib**: RDF graph manipulation
- **pyvis**: Interactive network visualization
- **jsonschema**: Schema validation
- **sentence-transformers**: Semantic analysis

See [pyproject.toml](pyproject.toml) for full dependency list.

## Testing

```bash
# Run all tests
just test

# Run specific test file
uv run pytest tests/test_graph_builder.py -v

# Run with coverage report
uv run pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### Common Issues

**1. Python version mismatch**
```bash
# Error: "requires-python = >=3.14"
Solution: Ensure Python 3.14+ is installed and active
python --version
```

**2. uv not found**
```powershell
# Windows: Reload PowerShell profile after installing uv
. $PROFILE
# Or restart your terminal
```

**3. Just command not found**
```bash
# Ensure Just is in your PATH
which just  # macOS/Linux
where.exe just  # Windows

# Alternative: Run commands directly with uv
uv run python -m src.cli.build_rdf --help
```

**4. Missing AVERITEC data**
```bash
# Error: FileNotFoundError for train.json/dev.json
Solution: Download AVERITEC dataset and place in data/raw/averitec/
```

**5. AI2-THOR simulation errors**
```bash
# Error: Display or graphics-related issues
Solution: AI2-THOR requires a display or Xvfb on Linux
# Linux headless:
sudo apt-get install xvfb
xvfb-run python -m src.cli.build_rdf ...
```

**6. RDF/TTL parsing errors**
```bash
# Error: "Invalid Turtle syntax"
Solution: Rebuild the knowledge graph
just build-kg
```

**7. Memory issues with large datasets**
```bash
# Reduce batch size or number of claims
just build-claims n_claims=1000 max_contexts=5
```

**8. Permission errors on Windows**
```powershell
# Run PowerShell as Administrator or adjust execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Getting Help

- Check the [Justfile](Justfile) for all available commands and parameters
- Review pipeline documentation in [src/pipelines/README.md](src/pipelines/README.md)
- Examine configuration files in [configs/](configs/)
- Run commands with `--help` flag for detailed usage:
  ```bash
  uv run python -m src.cli.build_claims --help
  ```

### Debug Mode

Enable verbose logging:
```bash
# Add --verbose flag to most commands
just build-kg --verbose
uv run python -m src.cli.build_claims out/knowledge_graph.ttl --verbose
```

## Performance Tips

- **Parallel Processing**: Use multiple cores for claim generation
- **Incremental Builds**: Work with smaller datasets during development
- **Caching**: Reuse generated knowledge graphs instead of rebuilding
- **Monitoring**: Check `runs/<RUN_ID>/logs/` for pipeline execution logs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for full details.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow code style**: Run `just dev` to format and lint
3. **Add tests**: Ensure `just test` passes with adequate coverage
4. **Update documentation**: Reflect changes in README and docstrings
5. **Submit a pull request** with a clear description of changes

### Development Workflow

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes and test
just dev
just test

# Commit with clear messages
git commit -m "Add: description of your changes"

# Push and create PR
git push origin feature/your-feature-name
```

## Citation

If you use this project in your research, please cite it as follows. See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

**BibTeX:**
```bibtex
@software{epistemic_factkg_2026,
  title = {Epistemic FactKG: A Knowledge Graph Framework for Epistemic Fact Checking},
  author = {{Epistemic FactKG Contributors}},
  year = {2026},
  url = {https://github.com/yourusername/epistemic-factkg},
  version = {0.1.0},
  note = {A framework for generating and validating claims with epistemic labels from AI2-THOR and AVERITEC datasets}
}
```

**APA:**
```
Epistemic FactKG Contributors. (2026). Epistemic FactKG: A Knowledge Graph Framework 
for Epistemic Fact Checking (Version 0.1.0) [Computer software]. 
https://github.com/yourusername/epistemic-factkg
```

**MLA:**
```
Epistemic FactKG Contributors. Epistemic FactKG: A Knowledge Graph Framework for 
Epistemic Fact Checking. Version 0.1.0, 2026, 
https://github.com/yourusername/epistemic-factkg.
```

### Related Publications

If this work is based on or extends published research, please also cite:

**AVERITEC Dataset:**
```bibtex
@inproceedings{schlich2024averitec,
  title={AVERITEC: A Dataset for Real-world Claim Verification with Evidence from the Web},
  author={Schlich, Michael and Jiang, Zhijiang and Akhtar, Sabrina and others},
  booktitle={Proceedings of NeurIPS Datasets and Benchmarks Track},
  year={2024}
}
```

**AI2-THOR Environment:**
```bibtex
@article{kolve2017ai2thor,
  title={AI2-THOR: An Interactive 3D Environment for Visual AI},
  author={Kolve, Eric and Mottaghi, Roozbeh and Han, Winson and others},
  journal={arXiv preprint arXiv:1712.05474},
  year={2017}
}
```

### Key Libraries:

**RDFLib** (for RDF processing):
```bibtex
@misc{rdflib,
  title={RDFLib: A Python library for working with RDF},
  author={{RDFLib Team}},
  howpublished={\url{https://github.com/RDFLib/rdflib}},
  year={2023}
}
```

**Sentence Transformers** (if using for semantic similarity):
```bibtex
@inproceedings{reimers-2019-sentence-bert,
  title={Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author={Reimers, Nils and Gurevych, Iryna},
  booktitle={Proceedings of EMNLP-IJCNLP},
  year={2019}
}
```

## Acknowledgments

- **AI2-THOR** team for the interactive 3D environment
- **AVERITEC** dataset contributors for fact verification data
- Open-source community for dependencies (RDFLib, PyVis, etc.)
