"""IO utilities: read yaml, make directories, save ckpt, append metrics."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, extra=None):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)

    # Also save a weight-only .pth (model state_dict only, for inference)
    pth_path = path.with_suffix(".pth")
    torch.save(model.state_dict(), pth_path)


def save_metrics_line(path, metrics):
    with open(path, "a") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def save_config_snapshot(config, path):
    with open(path, "w") as f:
        yaml.safe_dump(config, f, allow_unicode=True)


# Path keys under config["data"] (rewritten as original filename + new root dir)
DATA_PATH_KEYS = (
    "splits_json", "graph_dir", "fragment_embedding_cache",
    "chem_cache", "esm_cache", "fp_cache", "maccs_cache", "prop_cache",
)


def apply_versioned_paths(config, data_root, output_dir):
    """Rewrite every path under config["data"] to <data_root>/<original filename>
    and override output_dir.

    Versioning convention: preprocessing/training output dirs are suffixed with
    the execution date (data_preprocessed_YYYYMMDD, training/outputs_YYYYMMDD);
    stage scripts specify them via --data-root / --output-dir (defaulting to the
    day the script runs). This lets multiple dated data/model versions coexist
    under the same directory.
    """
    data = config.setdefault("data", {})
    for key in DATA_PATH_KEYS:
        val = data.get(key)
        if val:
            data[key] = str(Path(data_root) / Path(str(val)).name)
    config["output_dir"] = str(output_dir)
