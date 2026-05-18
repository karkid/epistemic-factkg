"""Optuna-based hyperparameter search for EpistemicHGNN.

Usage (from train.py — automatic):
    hparams = load_best_hparams()          # None if no saved file
    if hparams is None:
        hparams = run_search(...)          # runs Optuna, saves to configs/hparams/

Usage (standalone):
    uv run python -m src.pipeline.model.hparam_search \\
        --model v3-nli --jsonl out/data/training/epistemic_factkg_training.jsonl \\
        --n-trials 30
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from src.model.config import GraphConfig
from src.model.data.types import NUM_STANCE, NUM_VERDICT
from src.model.models import MODELS
from src.model.models.nlihybridhgnn import NLIHybridHGNN
from src.model.training.config import TrainConfig
from src.model.training.trainer import Trainer

# ── Per-model save location (committed to git) ───────────────────────────────
HPARAMS_DIR = Path("configs/hparams")

def hparams_path(model: str) -> Path:
    """Return the canonical hparams file path for a given model key."""
    return HPARAMS_DIR / f"{model}_best_hparams.json"


def load_best_hparams(model: str) -> dict | None:
    """Return saved best hparams for *model* if the file exists, else None.

    Keys match argparse dest names (lr, dropout, hidden_dim, heads,
    is_loss_weight, weight_decay) plus _meta fields prefixed with '_'.
    """
    path = hparams_path(model)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        # strip internal meta keys before returning to caller
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return None


def run_search(
    model_key: str,
    graph_cfg: GraphConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stance_weights: torch.Tensor | None = None,
    verdict_weights: torch.Tensor | None = None,
    n_trials: int = 30,
    save_path: Path | None = None,  # defaults to configs/hparams/{model_key}_best_hparams.json
) -> dict:
    """Run Optuna search, save best params, return them.

    Searched params:
        lr, weight_decay, dropout, hidden_dim, heads, is_loss_weight
    Fixed during search (not model hyperparams):
        epochs=20, patience=5, verdict_loss_weight=1.0
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("Optuna not installed. Run:  uv add optuna", file=sys.stderr)
        sys.exit(1)

    model_cls = MODELS[model_key]
    if save_path is None:
        save_path = hparams_path(model_key)

    def objective(trial: optuna.Trial) -> float:
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        heads = trial.suggest_categorical("heads", [2, 4])
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)

        model = model_cls(graph_cfg, hidden_dim, heads, dropout)
        cfg = TrainConfig(
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            dropout=dropout,
            hidden_dim=hidden_dim,
            heads=heads,
            is_loss_weight=trial.suggest_float("is_loss_weight", 0.5, 5.0),
            verdict_loss_weight=1.0,
            epochs=20,
            patience=5,
        )
        trainer = Trainer(model, cfg, stance_weights, verdict_weights)
        trainer.fit(train_loader, val_loader, verbose=False)
        return trainer._best_val_loss

    print(f"[{model_key}] Running hyperparameter search: {n_trials} trials …")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **best,
        "_best_val_loss": round(study.best_value, 4),
        "_n_trials": n_trials,
        "_model": model_key,
    }
    save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[{model_key}] Best val_loss : {study.best_value:.4f}")
    print(f"[{model_key}] Best params   : {best}")
    print(f"[{model_key}] Saved          → {save_path}")
    return best


# ── Standalone entry-point ────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run Optuna HPO for EpistemicHGNN.")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model", default="v3-nli")
    ap.add_argument("--splits-dir", default="out/data/splits")
    ap.add_argument("--registry", default="data/registry/source_trust_registry.jsonl")
    ap.add_argument("--embed-cache", default="out/model/graphs/embed_cache.pkl")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--save-path", default=None, help="Override save path (default: configs/hparams/{model}_best_hparams.json)")
    return ap


def main() -> None:
    from src.epistemic.registry import load_source_trust_registry
    from src.model.data.builder import ClaimGraphBuilder
    from src.model.data.featurizer import Featurizer

    args = _build_parser().parse_args()
    jsonl_path = Path(args.jsonl)
    splits_dir = Path(args.splits_dir)

    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    def _load_indices(split: str) -> list[int]:
        p = splits_dir / f"{split}_indices.json"
        return json.loads(p.read_text(encoding="utf-8"))["indices"]

    train_idx = _load_indices("train")
    val_idx = _load_indices("val")

    is_nli = MODELS.get(args.model) is NLIHybridHGNN
    graph_cfg = GraphConfig.v2() if is_nli else GraphConfig.v1()
    cache_path = Path("out/model/graphs") / f"split_cache_{args.model}.pkl"

    if cache_path.exists():
        print(f"Loading graph cache: {cache_path}")
        cached = pickle.loads(cache_path.read_bytes())
        train_graphs = cached["train"]
        val_graphs = cached["val"]
    else:
        featurizer = Featurizer(cache_path=args.embed_cache)
        registry = load_source_trust_registry(args.registry)
        builder = ClaimGraphBuilder(registry, featurizer, use_nli=is_nli)

        def _build(indices: list[int], split: str) -> list:
            graphs = []
            for idx in indices:
                try:
                    g = builder.build(records[idx])
                    if g is not None:
                        graphs.append(g)
                except Exception:
                    pass
            print(f"{split}: {len(graphs)} graphs")
            return graphs

        train_graphs = _build(train_idx, "train")
        val_graphs = _build(val_idx, "val")
        featurizer.save_cache()

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    all_stance_y = torch.cat([g["evidence"].stance_y for g in train_graphs])
    s_counts = torch.bincount(all_stance_y, minlength=NUM_STANCE).float().clamp(min=1.0)
    stance_weights = s_counts.sum() / (NUM_STANCE * s_counts)

    all_verdict_y = torch.cat([g["claim"].y for g in train_graphs])
    v_counts = torch.bincount(all_verdict_y, minlength=NUM_VERDICT).float().clamp(min=1.0)
    verdict_weights = v_counts.sum() / (NUM_VERDICT * v_counts)

    run_search(
        model_key=args.model,
        graph_cfg=graph_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        stance_weights=stance_weights,
        verdict_weights=verdict_weights,
        n_trials=args.n_trials,
        save_path=Path(args.save_path) if args.save_path else None,
    )


if __name__ == "__main__":
    main()
