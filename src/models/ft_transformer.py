"""FT-Transformer wrapper for HMDA lending classification.
Architecture: each feature gets its own learned embedding (Feature Tokenizer),
then a standard Transformer encoder processes all feature tokens together,
and a classification head outputs a single logit for approve/deny.
Reference: Gorishniy et al. (2021), 'Revisiting Deep Learning Models for
Tabular Data', arXiv:2106.11959.
We use the rtdl_revisiting_models library (v0.0.2) from Yandex Research,
which provides the FTTransformer class with sensible defaults via
get_default_kwargs().
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from rtdl_revisiting_models import FTTransformer

class FTTransformerLending(nn.Module):
    """FT-Transformer for binary loan approval classification.
    n_num: number of numeric features (e.g. loan amount, income).
    cat_cardinalities: list of unique value counts per categorical feature.
    d_block: internal dimension of each Transformer block (controls capacity).
    n_blocks: number of stacked Transformer layers.
    """
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: list[int],
        d_block: int = 192,
        n_blocks: int = 3,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_num = n_num
        self.cat_cardinalities = cat_cardinalities
        self.d_block = d_block
        # Start from the library's recommended defaults, then override
        # with our tuned hyperparameters for this specific dataset.
        backbone_kwargs = FTTransformer.get_default_kwargs(n_blocks=n_blocks)
        backbone_kwargs.update({
            "d_block": d_block,
            "attention_dropout": attention_dropout,
            "ffn_dropout": ffn_dropout,
            "residual_dropout": residual_dropout,
            "d_out": 1,  # single logit output for binary classification
        })
        # Internal flag not meant for user code
        backbone_kwargs.pop("_is_default", None)
        self.backbone = FTTransformer(
            n_cont_features=n_num,
            cat_cardinalities=cat_cardinalities if cat_cardinalities else [],
            **backbone_kwargs,
        )

    def forward(
        self,
        x_num: torch.Tensor,           # float32 (batch, n_num)
        x_cat: torch.Tensor | None,    # int64   (batch, n_cat) or None
    ) -> torch.Tensor:
        """Return raw logit, shape (batch, 1)."""
        return self.backbone(x_num, x_cat)

    def predict_proba(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor | None,
    ) -> np.ndarray:
        """Return P(approved) as a 1-D numpy array. No gradient tracking."""
        self.eval()
        with torch.no_grad():
            logit = self.forward(x_num, x_cat)
            prob = torch.sigmoid(logit).squeeze(1)
        return prob.cpu().numpy()

    def n_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
