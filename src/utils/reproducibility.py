"""Reproducibility primitives for the XAI Consumer Lending pipeline.
Every script in this project must call seed_everything() before any
stochastic operation and assert_clean_git() before producing audit artifacts.
"""
import os
import random
import hashlib
import json
import subprocess
import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    """Lock down all sources of randomness so results are fully reproducible.
    We seed Python's built-in random, NumPy, and all PyTorch backends
    (CPU, CUDA, and Apple Silicon MPS). Deterministic algorithms are
    enabled with warn_only=True because some PyTorch ops do not have
    deterministic implementations and would otherwise crash.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # warn_only=True lets us keep running when an op has no deterministic
    # variant, while still getting warnings so we know which ops are affected.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def assert_clean_git() -> str:
    """Refuse to run on a dirty git tree and return the current SHA.
    This ensures every audit artifact can be traced back to an exact
    commit, which is critical for regulatory reproducibility.
    """
    dirty = subprocess.run(["git", "diff", "--quiet"]).returncode
    if dirty != 0:
        raise RuntimeError(
            "Refusing to run on a dirty git tree. Commit or stash first."
        )
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    return sha

def file_sha256(path: str) -> str:
    """Compute SHA-256 hash of a file for audit provenance tracking.
    Used to verify that input data has not changed between pipeline runs.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def select_device(force_cpu: bool = False) -> torch.device:
    """Pick the fastest available hardware accelerator.
    Priority: CPU (if forced) > MPS (Apple Silicon) > CUDA > CPU.
    We default to MPS on Apple Silicon for a 3-5x training speedup
    over CPU, but audit artifacts use force_cpu=True so results are
    hardware-independent.
    """
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def write_manifest(out_dir: str, **fields) -> None:
    """Save a _run_manifest.json capturing everything needed to reproduce
    this run: git SHA, random seed, device, hyperparameters, metrics, and
    wall-clock time. This is the audit trail for every pipeline step.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "_run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(fields, f, indent=2, sort_keys=True, default=str)

def get_library_versions() -> dict[str, str]:
    """Snapshot the versions of all key libraries so we can detect
    environment differences if results ever fail to reproduce.
    """
    import pandas as pd
    import sklearn
    versions = {
        "python": f"{__import__('sys').version}",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "torch": torch.__version__,
    }
    # Try importing optional libraries. Not every script needs all of these
    # so missing ones are silently skipped.
    for lib_name in ["captum", "shap", "dice_ml", "mlflow", "optuna"]:
        try:
            lib = __import__(lib_name)
            versions[lib_name] = getattr(lib, "__version__", "installed (version unknown)")
        except ImportError:
            pass
    return versions
