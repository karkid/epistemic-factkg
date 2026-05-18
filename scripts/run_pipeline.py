"""Cross-platform replacement for the bash routing logic in `just run`.

Usage: uv run python scripts/run_pipeline.py <data|model> [STEP] [MODELS]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def just(*args: str) -> None:
    subprocess.run(["just", *args], check=True)


def uv(*args: str) -> None:
    subprocess.run(["uv", "run", "python", "-m", *args], check=True)


TRAINING_JSONL = "out/data/training/epistemic_factkg_training.jsonl"
SPLITS_DIR     = "out/data/splits"
GRAPH_DATASET  = "out/model/graphs/graph_dataset.pt"


def main() -> None:
    argv = sys.argv[1:]
    pipeline = argv[0] if len(argv) > 0 else ""
    step     = argv[1] if len(argv) > 1 else ""
    models   = argv[2] if len(argv) > 2 else "all"

    if pipeline == "data":
        if step in ("", "all"):
            just("build"); just("validate"); just("report")
        elif step == "rebuild":
            just("build", "rebuild=true"); just("validate"); just("report")
        elif step == "build":
            just("build")
        elif step == "validate":
            just("validate")
        elif step == "report":
            just("report")
        else:
            print(f"Unknown data step '{step}'. Available: rebuild  build  validate  report", file=sys.stderr)
            sys.exit(1)

    elif pipeline == "model":
        if step == "list":
            uv("src.pipeline.model.orchestrate", "list")
        elif step in ("", "all"):
            if not Path(GRAPH_DATASET).exists():
                just("graph")
            uv("src.pipeline.model.orchestrate", "run",
               "--models", models,
               "--jsonl", TRAINING_JSONL,
               "--splits-dir", SPLITS_DIR,
               "--graph", GRAPH_DATASET)
        elif step == "rebuild":
            just("graph")
            uv("src.pipeline.model.orchestrate", "run",
               "--models", models,
               "--jsonl", TRAINING_JSONL,
               "--splits-dir", SPLITS_DIR,
               "--graph", GRAPH_DATASET)
        elif step == "build":
            just("graph")
        elif step == "train":
            uv("src.pipeline.model.orchestrate", "train",
               "--models", models,
               "--jsonl", TRAINING_JSONL,
               "--splits-dir", SPLITS_DIR)
        elif step == "eval":
            uv("src.pipeline.model.orchestrate", "eval",
               "--models", models,
               "--jsonl", TRAINING_JSONL,
               "--splits-dir", SPLITS_DIR)
        elif step == "compare":
            uv("src.pipeline.model.orchestrate", "compare", "--models", models)
        else:
            print(f"Unknown model step '{step}'. Available: list  build  train  eval  compare", file=sys.stderr)
            sys.exit(1)

    else:
        print("Usage: just run <data|model> [STEP] [MODELS]", file=sys.stderr)
        print("  data  steps: rebuild  build  validate  report", file=sys.stderr)
        print("  model steps: list  build  train  eval  compare", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
