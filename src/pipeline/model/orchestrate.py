"""Multi-model orchestrator — trains, evaluates, and compares registered models.

Sub-commands:
  list     Print all model names registered in MODELS.
  train    Train one or more models sequentially.
  eval     Evaluate one or more models on the test split.
  run      Full pipeline: train + eval for each model, then compare all.
  compare  Generate comparison report for a set of models.

Usage (via just):
  uv run python -m src.pipeline.model.orchestrate list
  uv run python -m src.pipeline.model.orchestrate train --models v1-hgnn,baseline
  uv run python -m src.pipeline.model.orchestrate run --models all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import combinations
from pathlib import Path

from src.model.models import MODELS


def _auto_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_device(device: str | None) -> str:
    return device if device else _auto_device()


def _resolve_models(models_arg: str) -> list[str]:
    """Return model name list from 'all' or 'm1,m2,...'."""
    if not models_arg or models_arg.strip().lower() == "all":
        return list(MODELS.keys())
    names = [m.strip() for m in models_arg.split(",") if m.strip()]
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        print(
            f"Unknown model(s): {unknown}. Registered: {list(MODELS)}", file=sys.stderr
        )
        sys.exit(1)
    return names


def _run(cmd: list[str]) -> None:
    """Run a subprocess and exit on failure."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _checkpoint(model: str) -> str:
    return f"out/model/{model}/checkpoints/best_model.pt"


def _report_dir(model: str) -> str:
    return f"out/reports/model/{model}"


def _results_dir(model: str) -> str:
    return f"out/reports/model/{model}/eval"


# ── Sub-command handlers ──────────────────────────────────────────────────────


def cmd_list(_args: argparse.Namespace) -> None:
    for name in MODELS:
        print(name)


def cmd_train(args: argparse.Namespace) -> None:
    models = _resolve_models(args.models)
    device = _resolve_device(args.device)
    for model in models:
        print(f"\n{'=' * 60}\nTraining: {model}\n{'=' * 60}")
        Path(f"out/model/{model}/checkpoints").mkdir(parents=True, exist_ok=True)
        Path(_report_dir(model)).mkdir(parents=True, exist_ok=True)
        _run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "src.pipeline.model.train",
                "--model",
                model,
                "--model-name",
                model,
                "--jsonl",
                args.jsonl,
                "--splits-dir",
                args.splits_dir,
                "--checkpoint-dir",
                f"out/model/{model}/checkpoints",
                "--report-dir",
                _report_dir(model),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--batch-size",
                str(args.batch_size),
                "--device",
                device,
                "--verbose",
            ]
        )


def cmd_eval(args: argparse.Namespace) -> None:
    models = _resolve_models(args.models)
    device = _resolve_device(getattr(args, "device", None))
    for model in models:
        ckpt = _checkpoint(model)
        if not Path(ckpt).exists():
            print(f"Checkpoint not found: {ckpt} — skipping {model}", file=sys.stderr)
            continue
        print(f"\n{'=' * 60}\nEvaluating: {model}\n{'=' * 60}")
        Path(_results_dir(model)).mkdir(parents=True, exist_ok=True)
        _run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "src.pipeline.model.evaluate",
                "--model",
                model,
                "--model-name",
                model,
                "--checkpoint",
                ckpt,
                "--jsonl",
                args.jsonl,
                "--splits-dir",
                args.splits_dir,
                "--output",
                _results_dir(model),
                "--device",
                device,
            ]
        )


def cmd_compare(args: argparse.Namespace) -> None:
    models = _resolve_models(args.models)
    if len(models) < 2:
        print("compare requires at least 2 models.", file=sys.stderr)
        sys.exit(1)
    for m1, m2 in combinations(models, 2):
        dir1 = _results_dir(m1)
        dir2 = _results_dir(m2)
        if not Path(dir1).exists() or not Path(dir2).exists():
            print(
                f"Eval results missing for {m1} or {m2} — skipping comparison.",
                file=sys.stderr,
            )
            continue
        out_path = f"out/reports/model/comparison_{m1}_vs_{m2}.md"
        print(f"\nComparing {m1} vs {m2} → {out_path}")
        _run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "src.pipeline.model.compare",
                "--model1",
                m1,
                "--dir1",
                dir1,
                "--model2",
                m2,
                "--dir2",
                dir2,
                "--out",
                out_path,
            ]
        )


def cmd_run(args: argparse.Namespace) -> None:
    models = _resolve_models(args.models)
    print(f"Running full pipeline for: {models}")

    # Train + eval per model
    train_args = argparse.Namespace(
        models=",".join(models),
        jsonl=args.jsonl,
        splits_dir=args.splits_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=_resolve_device(args.device),
    )
    cmd_train(train_args)

    eval_args = argparse.Namespace(
        models=",".join(models),
        jsonl=args.jsonl,
        splits_dir=args.splits_dir,
        device=_resolve_device(args.device),
    )
    cmd_eval(eval_args)

    # Compare if more than one model evaluated successfully
    if len(models) >= 2:
        compare_args = argparse.Namespace(models=",".join(models))
        cmd_compare(compare_args)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-model orchestrator for train / eval / compare.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Print registered model names")

    def _add_models(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--models", default="all", help="Comma-separated model names, or 'all'"
        )

    def _add_data(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--jsonl", default="out/data/training/epistemic_factkg_training.jsonl"
        )
        p.add_argument("--splits-dir", default="out/data/splits")

    def _add_train_hp(p: argparse.ArgumentParser) -> None:
        p.add_argument("--epochs", type=int, default=50)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument(
            "--device",
            default=None,
            help="cuda or cpu (default: auto-detect)",
        )

    p_train = sub.add_parser("train", help="Train specified models")
    _add_models(p_train)
    _add_data(p_train)
    _add_train_hp(p_train)

    p_eval = sub.add_parser("eval", help="Evaluate specified models")
    _add_models(p_eval)
    _add_data(p_eval)

    p_compare = sub.add_parser("compare", help="Generate comparison report")
    _add_models(p_compare)

    p_run = sub.add_parser("run", help="Full pipeline: train + eval + compare")
    _add_models(p_run)
    _add_data(p_run)
    _add_train_hp(p_run)

    args = ap.parse_args()
    {
        "list": cmd_list,
        "train": cmd_train,
        "eval": cmd_eval,
        "compare": cmd_compare,
        "run": cmd_run,
    }[args.command](args)


if __name__ == "__main__":
    main()
