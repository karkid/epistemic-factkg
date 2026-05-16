"""Model registry — maps model name strings to model classes.

To add a new model:
  1. Create src/model/models/<name>.py with your nn.Module class.
     Constructor must accept: (graph_config, hidden_dim, heads, dropout).
     forward() must return: {"stance_logits", "is_pred", "verdict_logits"}.
     predict() must return: {"stance_pred", "stance_logits", "is_pred", "verdict"}.
  2. Import it here and add to MODELS.
"""

from src.model.models.epistemichgnn import EpistemicHGNN

MODELS: dict[str, type] = {
    "v1-hgnn": EpistemicHGNN,
    # "baseline": BaselineGNN,  ← register new models here
}
