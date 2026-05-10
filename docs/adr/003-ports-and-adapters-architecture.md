# ADR-003: Ports and Adapters (Hexagonal) Architecture

## Status

Accepted

## Context

The project began with two data sources (AI2-THOR and AVeriTeC) and anticipates future datasets (FEVER, FactKG, or others). The initial implementation risk was that adding a third source would require modifying the conversion pipeline's core logic — creating fragile coupling between dataset-specific code and shared infrastructure.

Alternatives considered:

1. **Standalone conversion scripts per dataset** — simple; one `convert_averitec.py`, one `convert_ai2thor.py`; but adding a third dataset means a new script, a new CLI entry point, and potentially divergent output formats unless disciplined
2. **Single monolithic converter with dataset conditionals** — one file, `if dataset == "averitec": ...`; grows unmanageable as datasets accumulate
3. **Ports and Adapters (Hexagonal)** — abstract interfaces in `core/ports/`; dataset-specific implementations in `adapters/{name}/`; pipeline dispatches via a registry

## Decision

Use **Ports and Adapters**:

- `src/core/ports/dataset/converter.py` — `DatasetConverter` ABC defining `dataset_name`, `convert_one()`, `infer_pramana()`, and a default `convert_file()` implementation
- `src/core/ports/dataset/validator.py` — `DatasetValidator` ABC for per-dataset semantic checks
- `src/adapters/{name}/converter.py` — one subclass per dataset
- `src/adapters/{name}/validator.py` — one subclass per dataset
- `src/cli/convert_to_unified.py` — `CONVERTERS` dict dispatches by dataset name

## Consequences

**Positive:**
- Adding a new dataset = 2 files + 2 registration lines; zero core code changes
- Each adapter is independently testable without running the full pipeline
- `DatasetConverter.convert_file()` provides the loop infrastructure; adapters only need to implement per-record logic
- Clear separation between domain logic (`core/`) and source-specific details (`adapters/`)

**Negative:**
- Slightly more indirection than standalone scripts — a contributor must understand the ABC + dispatch pattern to add a dataset
- For one-off dataset conversions, the overhead of implementing the ABC may not be worth the structure

**Tradeoff:**
The indirection cost is paid once per dataset and amortised across all future pipeline runs. Given the project expects at least 3–4 datasets over its lifetime, the investment in structure is justified.
