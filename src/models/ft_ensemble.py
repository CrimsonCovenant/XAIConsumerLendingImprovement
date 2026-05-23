"""Deep ensemble of FT-Transformer members with bagged subsampling.
K independently-seeded models, each trained on a unique stratified subsample
of the training data. This is Bootstrap Aggregating (bagging):
  - Forces ensemble diversity by giving each member different training data
  - 6x faster per member since each only sees 15% of the full dataset
  - Final prediction is the average of all K member predictions
Uncertainty = standard deviation across member predictions (free, no extra compute).
"""
from __future__ import annotations
import json
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from src.models.ft_transformer import FTTransformerLending
from src.models.ft_train import train_one_member, make_loader
from src.utils.reproducibility import seed_everything

def _stratified_subsample(
    x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray,
    fraction: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Take a random subsample that preserves the approve/deny class ratio.
    Floor at 1000 rows to avoid degenerate tiny samples.
    """
    if fraction >= 1.0:
        return x_num, x_cat, y
    n = len(y)
    n_sample = max(int(n * fraction), 1000)
    idx = np.arange(n)
    _, sample_idx = train_test_split(
        idx, test_size=n_sample / n, random_state=seed, stratify=y,
    )
    return x_num[sample_idx], x_cat[sample_idx], y[sample_idx]

class FTEnsemble:
    """K-member bagged deep ensemble for one HMDA year.
    K: number of ensemble members (default 5).
    subsample_fraction: fraction of training data each member sees (default 15%).
    model_kwargs: passed to FTTransformerLending (n_num, cat_cardinalities, etc.)
    train_kwargs: passed to train_one_member (lr, weight_decay, n_epochs, etc.)
    """
    def __init__(
        self,
        K: int = 5,
        base_seed: int = 42,
        subsample_fraction: float = 0.15,
        model_kwargs: dict | None = None,
        train_kwargs: dict | None = None,
    ):
        self.K = K
        self.base_seed = base_seed
        self.subsample_fraction = subsample_fraction
        self.model_kwargs = model_kwargs or {}
        self.train_kwargs = train_kwargs or {}
        self.members_: list[FTTransformerLending] = []
        self.member_results_: list[dict] = []
        self.is_fitted = False

    def fit(
        self,
        x_num_tr: np.ndarray, x_cat_tr: np.ndarray, y_tr: np.ndarray,
        x_num_val: np.ndarray, x_cat_val: np.ndarray, y_val: np.ndarray,
        device: torch.device,
    ) -> "FTEnsemble":
        """Train all K ensemble members sequentially.
        Each member gets its own random seed, its own 15% subsample of
        the training data, and independently trains with early stopping.
        Validation data is shared so early stopping is consistent.
        """
        self.members_ = []
        self.member_results_ = []
        n_full = len(y_tr)
        for k in range(self.K):
            seed = self.base_seed + k
            seed_everything(seed)
            # Each member gets a unique stratified subsample (bagging)
            xn_sub, xc_sub, y_sub = _stratified_subsample(
                x_num_tr, x_cat_tr, y_tr,
                fraction=self.subsample_fraction, seed=seed,
            )
            n_sub = len(y_sub)
            print(
                f"    Member {k+1}/{self.K} (seed={seed}) -- "
                f"{n_sub:,}/{n_full:,} rows ({n_sub/n_full:.0%}) ...",
                flush=True,
            )
            model = FTTransformerLending(**self.model_kwargs).to(device)
            result = train_one_member(
                model,
                xn_sub, xc_sub, y_sub,
                x_num_val, x_cat_val, y_val,
                device=device,
                **self.train_kwargs,
            )
            print(
                f"      best_val_auc={result['best_val_auc']:.4f}  "
                f"epochs={result['epochs_run']}  "
                f"time={result['wall_clock_s']:.0f}s"
            )
            self.members_.append(model)
            # Store result without the state_dict (already in the model)
            self.member_results_.append({
                k2: v for k2, v in result.items() if k2 != "best_state_dict"
            })
        self.is_fitted = True
        return self

    def predict_proba(
        self, x_num: np.ndarray, x_cat: np.ndarray,
        device: torch.device, batch_size: int = 4096,
    ) -> np.ndarray:
        """Average predicted probability across all K members. Shape (n,)."""
        assert self.is_fitted, "Call fit() first or load() a saved ensemble."
        all_probs = []
        for model in self.members_:
            model.eval()
            batches = make_loader(
                x_num, x_cat, np.zeros(len(x_num)),
                batch_size, shuffle=False, device=device,
            )
            probs = []
            with torch.no_grad():
                for xn, xc, _ in batches:
                    logit = model(xn, xc).squeeze(1)
                    probs.append(torch.sigmoid(logit).cpu().numpy())
            all_probs.append(np.concatenate(probs))
        return np.mean(all_probs, axis=0)

    def predict_proba_with_uncertainty(
        self, x_num: np.ndarray, x_cat: np.ndarray,
        device: torch.device, batch_size: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_prob, std_prob).
        std is the disagreement between ensemble members, which serves as
        a free measure of epistemic uncertainty (how unsure the model is).
        """
        all_probs = []
        for model in self.members_:
            model.eval()
            batches = make_loader(
                x_num, x_cat, np.zeros(len(x_num)),
                batch_size, shuffle=False, device=device,
            )
            probs = []
            with torch.no_grad():
                for xn, xc, _ in batches:
                    logit = model(xn, xc).squeeze(1)
                    probs.append(torch.sigmoid(logit).cpu().numpy())
            all_probs.append(np.concatenate(probs))
        stack = np.array(all_probs)
        return stack.mean(axis=0), stack.std(axis=0)

    def save(self, dir_path: str) -> None:
        """Save all member weights and metadata to a directory.
        Creates member_0.pt through member_{K-1}.pt plus ensemble_meta.json.
        """
        os.makedirs(dir_path, exist_ok=True)
        for k, model in enumerate(self.members_):
            torch.save(
                model.state_dict(),
                os.path.join(dir_path, f"member_{k}.pt"),
            )
        meta = {
            "K": self.K,
            "base_seed": self.base_seed,
            "subsample_fraction": self.subsample_fraction,
            "model_kwargs": self.model_kwargs,
            "member_results": self.member_results_,
        }
        with open(os.path.join(dir_path, "ensemble_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

    @classmethod
    def load(cls, dir_path: str, device: torch.device) -> "FTEnsemble":
        """Reconstruct a saved ensemble from disk.
        Reads ensemble_meta.json for architecture params, then loads
        each member's weights from member_k.pt files.
        """
        with open(os.path.join(dir_path, "ensemble_meta.json")) as f:
            meta = json.load(f)
        ens = cls(
            K=meta["K"],
            base_seed=meta["base_seed"],
            subsample_fraction=meta.get("subsample_fraction", 0.15),
            model_kwargs=meta["model_kwargs"],
        )
        ens.members_ = []
        for k in range(meta["K"]):
            model = FTTransformerLending(**meta["model_kwargs"]).to(device)
            state = torch.load(
                os.path.join(dir_path, f"member_{k}.pt"),
                map_location=device, weights_only=True,
            )
            model.load_state_dict(state)
            model.eval()
            ens.members_.append(model)
        ens.member_results_ = meta["member_results"]
        ens.is_fitted = True
        return ens
