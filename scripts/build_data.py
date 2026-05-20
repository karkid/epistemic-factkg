"""Cross-platform replacement for the bash logic in `just build`.

Usage: uv run python scripts/build_data.py [true|false]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CONFIG       = "configs/config.yaml"
REGISTRY     = "data/registry/source_trust_registry.jsonl"
SEED_POOL    = "data/registry/seed_pool.jsonl"
AI2THOR_DIR  = "data/raw/ai2thor"
AI2THOR_CLAIMS   = f"{AI2THOR_DIR}/claims_all.jsonl"
SYNTHETIC_DIR    = "data/raw/synthetic"
SYNTHETIC_JSONL  = f"{SYNTHETIC_DIR}/synthetic_current.jsonl"
UNIFIED_JSONL    = "out/data/unified/epistemic_factkg.jsonl"
TRAINING_JSONL   = "out/data/training/epistemic_factkg_training.jsonl"
SPLITS_DIR       = "out/data/splits"
KG_TTL           = "out/model/knowledge_graph.ttl"


def run(*cmd: str) -> None:
    subprocess.run(list(cmd), check=True)


def uv(*args: str) -> None:
    run("uv", "run", "python", "-m", *args)


def main() -> None:
    rebuild = len(sys.argv) > 1 and sys.argv[1].lower() == "true"

    if not rebuild and not Path(AI2THOR_CLAIMS).exists():
        print(
            f"{AI2THOR_CLAIMS} not found — triggering full rebuild automatically.",
            file=sys.stderr,
        )
        rebuild = True

    rebuild_flags: list[str] = []
    if rebuild:
        rebuild_flags = [
            "--rebuild",
            "--config", CONFIG,
            "--out-rdf", KG_TTL,
            "--ai2thor-dir", AI2THOR_DIR,
        ]

    if not Path(SYNTHETIC_JSONL).exists():
        print("--- generating synthetic claims ---")
        Path(SYNTHETIC_DIR).mkdir(parents=True, exist_ok=True)
        uv(
            "src.pipeline.data.generate", "synthetic",
            "--config", CONFIG,
            "--registry", REGISTRY,
            "--seed-pool", SEED_POOL,
            "--ai2thor-claims", AI2THOR_CLAIMS,
            "--output", SYNTHETIC_JSONL,
        )

    print("--- enriching source trust registry ---")
    run("uv", "run", "python", "scripts/enrich_registry.py")

    for d in ["out/data/unified", "out/data/training", "out/data/intermediate"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    uv(
        "src.pipeline.data.build",
        *rebuild_flags,
        "--registry", REGISTRY,
        "--averitec", "data/raw/averitec/train.json", "data/raw/averitec/dev.json",
        "--ai2thor", AI2THOR_CLAIMS,
        "--synthetic", SYNTHETIC_JSONL,
        "--unified-out", UNIFIED_JSONL,
        "--training-out", TRAINING_JSONL,
        "--intermediate-dir", "out/data/intermediate",
    )

    Path(SPLITS_DIR).mkdir(parents=True, exist_ok=True)
    uv(
        "src.pipeline.data.split_dataset",
        "--input", TRAINING_JSONL,
        "--output-dir", SPLITS_DIR,
        "--seed", "42",
        "--verbose",
    )


if __name__ == "__main__":
    main()
