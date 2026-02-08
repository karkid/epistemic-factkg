# Test Organization

## Test Files Overview

### Core Tests (Fast & Reliable)
- **`test_exceptions.py`** - Exception handling and error scenarios
- **`test_project_status.py`** - Project structure and API readiness validation  
- **`test_data_source.py`** - Data source API testing with dummy data
- **`test_graph_builder.py`** - Knowledge graph builder API with dummy data

### Test Configuration
- **`conftest.py`** - Lightweight fixtures with dummy data only

## Why We Specify Exact Test Files

### In `just test` command:
```bash
pytest tests/test_exceptions.py tests/test_project_status.py tests/test_data_source.py tests/test_graph_builder.py
```

**Reasons for explicit file specification:**

1. **Performance** - Only runs fast dummy data tests (0.24s vs 30+ seconds)
2. **Reliability** - Avoids broken tests that require heavy AI2THOR setup  
3. **Control** - Explicit about which tests run in CI/development
4. **Isolation** - Prevents accidental execution of resource-heavy tests
5. **Predictability** - Same tests run every time, in same order

### Alternative: `just test-all`
```bash 
pytest tests/  # Runs ALL tests in folder
```

**Use when:**
- You want to run any new test files added
- Testing comprehensive coverage
- Don't mind longer execution times

## Test Philosophy

✅ **Current Approach**: Lightweight dummy data tests
- Fast execution (< 1 second)
- No external dependencies  
- Pure data structure validation
- 100% reliable

❌ **Previous Approach**: Heavy integration tests
- Slow execution (30+ seconds)
- AI2THOR scene loading
- Resource management issues
- Unreliable in CI environments

## Adding New Tests

When adding new test files:
1. Add to `tests/` folder
2. Use dummy data approach
3. Update `just test` command to include new file
4. OR rely on `just test-all` for discovery