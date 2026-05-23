"""Training loop for one FT-Transformer ensemble member.
Handles the full train/validate cycle with early stopping on validation
AUC. Returns the best model weights and training history.
Performance: all data is pre-loaded to the GPU/MPS device once via
DeviceData, so there are no per-batch CPU-to-GPU copies during training.
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

def _class_pos_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss so approved and denied
    applications contribute equally to the loss despite class imbalance.
    pos_weight = n_denied / n_approved.
    """
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)

class DeviceData:
    """Pre-loads numpy arrays as tensors on the target device once.
    iter_batches() uses torch.randperm for on-device shuffled batching,
    which is much faster than converting numpy slices each batch.
    """
    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray, device: torch.device):
        self.x_num = torch.tensor(x_num, dtype=torch.float32, device=device)
        self.x_cat = (
            torch.tensor(x_cat, dtype=torch.long, device=device)
            if x_cat.shape[1] > 0
            else None
        )
        self.y = torch.tensor(y, dtype=torch.float32, device=device)
        self.n = len(y)

    def iter_batches(self, batch_size: int, shuffle: bool = True):
        """Yield (x_num, x_cat, y) batches. Shuffling happens on-device."""
        if shuffle:
            perm = torch.randperm(self.n, device=self.x_num.device)
        else:
            perm = torch.arange(self.n, device=self.x_num.device)
        for start in range(0, self.n, batch_size):
            idx = perm[start : start + batch_size]
            xn = self.x_num[idx]
            xc = self.x_cat[idx] if self.x_cat is not None else None
            yb = self.y[idx]
            yield xn, xc, yb

def make_loader(
    x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray,
    batch_size: int, shuffle: bool, device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]]:
    """Wrap numpy arrays as a simple list-of-batches for inference only.
    For training, use DeviceData.iter_batches() instead since it avoids
    the per-batch numpy-to-tensor copy that this function performs.
    """
    n = len(y)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    batches = []
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        xn = torch.tensor(x_num[idx], dtype=torch.float32, device=device)
        xc = (
            torch.tensor(x_cat[idx], dtype=torch.long, device=device)
            if x_cat.shape[1] > 0
            else None
        )
        yb = torch.tensor(y[idx], dtype=torch.float32, device=device)
        batches.append((xn, xc, yb))
    return batches

def train_one_member(
    model: nn.Module,
    x_num_tr: np.ndarray, x_cat_tr: np.ndarray, y_tr: np.ndarray,
    x_num_val: np.ndarray, x_cat_val: np.ndarray, y_val: np.ndarray,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    n_epochs: int = 100,
    batch_size: int = 4096,
    patience: int = 10,
    device: torch.device,
) -> dict:
    """Train one ensemble member with early stopping on validation AUC.
    Returns dict with: best_val_auc, best_state_dict, epochs_run,
    wall_clock_s, history.
    """
    # Pre-load all data to device once to avoid per-batch copies
    train_data = DeviceData(x_num_tr, x_cat_tr, y_tr, device)
    val_data = DeviceData(x_num_val, x_cat_val, y_val, device)
    pos_weight = _class_pos_weight(y_tr, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine LR schedule with 5-epoch linear warmup.
    # Warmup prevents large early gradients from destabilizing the
    # randomly initialized attention weights.
    warmup_epochs = 5
    def lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return max(ep / warmup_epochs, 1e-3)
        progress = (ep - warmup_epochs) / max(n_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_auc = -1.0
    best_state: dict | None = None
    patience_count = 0
    history: list[dict] = []
    t0 = time.time()
    for epoch in range(n_epochs):
        # Training pass
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        for xn, xc, yb in train_data.iter_batches(batch_size, shuffle=True):
            optimizer.zero_grad()
            logit = model(xn, xc).squeeze(1)
            loss = criterion(logit, yb)
            loss.backward()
            # Clip gradients to prevent exploding gradient in Transformer
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
            n_seen += len(yb)
        scheduler.step()
        # Validation pass -- compute AUC to decide early stopping
        model.eval()
        val_probs = []
        with torch.no_grad():
            for xn, xc, _ in val_data.iter_batches(4096, shuffle=False):
                logit = model(xn, xc).squeeze(1)
                val_probs.append(torch.sigmoid(logit).cpu().numpy())
        val_probs = np.concatenate(val_probs)
        val_auc = (
            roc_auc_score(y_val, val_probs)
            if len(np.unique(y_val)) > 1
            else 0.5
        )
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / max(n_seen, 1),
            "val_auc": val_auc,
            "lr": scheduler.get_last_lr()[0],
        })
        # Early stopping: save best weights, stop if no improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break
    # Restore the best weights found during training
    if best_state is not None:
        model.load_state_dict(best_state)
    # Free device memory
    del train_data, val_data
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "best_val_auc": best_val_auc,
        "best_state_dict": best_state,
        "epochs_run": epoch + 1,
        "wall_clock_s": time.time() - t0,
        "history": history,
    }
