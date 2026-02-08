# Refactored AI2-THOR Claim Generation Pipeline

This directory contains a refactored and organized claim generation pipeline for AI2-THOR RDF data. The refactoring improves maintainability, testability, and extensibility of the original claim generation code.

## Architecture Overview

The refactored pipeline follows a modular architecture with clear separation of concerns:

```
src/pipelines/
├── build_claims.py      # Main entry point (CLI + legacy compatibility)
├── config.py           # Configuration classes
├── factory.py          # Component factory for dependency injection
├── data_loader.py      # RDF data loading and grouping
├── claim_builder.py    # Main orchestrator class
└── README.md          # This documentation
```

## Key Components

### 1. Configuration (`config.py`)
- **`ClaimGenerationConfig`**: Main configuration for claim generation parameters
- **`AI2ThorDataConfig`**: AI2-THOR specific data processing configuration  
- **`ClaimGenerationResult`**: Result object with success status and metadata

### 2. Factory (`factory.py`)
- **`AI2ThorClaimPipelineFactory`**: Creates claim generation pipeline components
- Handles dependency injection and component initialization
- Provides reusable boolean predicate checker

### 3. Data Loader (`data_loader.py`)
- **`RDFDataLoader`**: Handles TTL file loading and context grouping
- Isolates RDF processing logic
- Provides clean interface for data operations

### 4. Claim Builder (`claim_builder.py`)
- **`AI2ThorClaimBuilder`**: Main orchestrator class
- Coordinates data loading, pipeline creation, and claim generation
- Handles error management and logging

## Key Improvements

### 🏗️ **Modularity**
- Separated concerns into focused classes
- Clear interfaces between components
- Easy to test individual components

### 🔧 **Configuration Management**
- Centralized configuration in dataclasses
- Type-safe configuration objects
- Easy to extend with new parameters

### 🔍 **Logging & Debugging**
- Comprehensive logging throughout the pipeline
- Configurable log levels (--verbose flag)
- Better error reporting and debugging

### 🧪 **Testability**
- Dependency injection via factory pattern
- Mockable components
- Clear input/output interfaces

### 📈 **Extensibility**
- Easy to add new claim types
- Pluggable data sources
- Configurable predicates and patterns

### ⚡ **Performance**
- Same performance as original
- Better resource management
- Clear memory usage patterns

## Usage Examples

### Basic Usage
```bash
# Generate claims with default settings
uv run python src/pipelines/build_claims.py

# Use custom input/output paths
uv run python src/pipelines/build_claims.py \
    --ttl data/my_graph.ttl \
    --output results/my_claims.jsonl

# Generate fewer claims for testing
uv run python src/pipelines/build_claims.py \
    --max_contexts 2 \
    --onehop_per_context 50
```

### Advanced Usage
```bash
# Verbose logging for debugging
uv run python src/pipelines/build_claims.py --verbose

# Machine-readable JSON output
uv run python src/pipelines/build_claims.py --json

# Custom seed for reproducibility
uv run python src/pipelines/build_claims.py --seed 12345
```

### Programmatic Usage
```python
from src.pipelines.config import ClaimGenerationConfig
from src.pipelines.claim_builder import AI2ThorClaimBuilder

# Create configuration
config = ClaimGenerationConfig(
    ttl_path=Path("data/knowledge_graph.ttl"),
    output_path=Path("output/claims.jsonl"),
    seed=42,
    max_contexts=5,
)

# Build claims
builder = AI2ThorClaimBuilder()
result = builder.build_claims(config)

if result.success:
    print(f"Generated {result.total_records} claims from {result.contexts_processed} contexts")
else:
    print(f"Error: {result.error_message}")
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ttl_path` | Path | `out/knowledge_graph.ttl` | Input TTL file path |
| `output_path` | Path | `out/claims.jsonl` | Output JSONL file path |
| `seed` | int | `42` | Random seed for reproducibility |
| `max_contexts` | int | `0` | Max contexts to process (0 = all) |
| `onehop_per_context` | int | `200` | One-hop claims per context |
| `neg_pairs_per_context` | int | `60` | Negation pairs per context |
| `conj_per_context` | int | `80` | Conjunction claims per context |

## Backward Compatibility

The refactored code maintains 100% backward compatibility:

- **Legacy function**: `build_ai2thor_claims()` still works with the same interface
- **Same CLI arguments**: All original command-line arguments supported
- **Same output format**: Generates identical JSONL output
- **Same performance**: No performance regression

## Extending the Pipeline

### Adding New Claim Types

1. Extend the pipeline in `factory.py`
2. Add configuration in `config.py`
3. Update the claim builder in `claim_builder.py`

### Supporting New Data Sources

1. Create new data config class in `config.py`
2. Create new data loader in `data_loader.py`
3. Create new factory in `factory.py`
4. Create new builder class in `claim_builder.py`

### Custom Predicates

1. Update `AI2ThorDataConfig.context_predicates`
2. Extend the predicate lexicon
3. Update boolean predicate detection logic

## Testing

The modular architecture makes it easy to test components in isolation:

```python
# Test data loading
loader = RDFDataLoader(data_config)
triples, grouped = loader.load_and_group_triples(config)

# Test factory
factory = AI2ThorClaimPipelineFactory()
pipeline = factory.create_pipeline(config)

# Test claim builder
builder = AI2ThorClaimBuilder(data_config, factory, loader)
result = builder.build_claims(config)
```

## Migration Notes

If you're migrating from the old code:

1. **No changes needed** for basic usage - the CLI interface is identical
2. **For programmatic usage** - use the new `AI2ThorClaimBuilder` class instead of calling `build_ai2thor_claims()` directly
3. **For extensions** - use the new modular components instead of modifying the monolithic function
4. **For testing** - leverage the new modular architecture for better test isolation

## Dependencies

The refactored pipeline uses the same dependencies as the original:
- All existing RDF processing libraries
- All existing AI2-THOR adapters  
- All existing NLG components
- All existing claim generation logic

The refactoring only reorganizes the code structure without changing the underlying algorithms or data processing logic.