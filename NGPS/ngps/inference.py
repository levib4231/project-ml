from __future__ import annotations

import time

import numpy as np
import torch

from datasets.volume_folder import validate_volume
from .config import NGPSConfig
from .nafnet import build_nafnet
from .seed import seed_everything


def validate_selection(metadata):
    """Reject non-selected checkpoints before any held-out data are opened."""
    selection = metadata.get("selection", {})
    epochs = metadata.get("training_config", {}).get("epochs")
    if (selection.get("rule") != "fixed_final_epoch"
            or not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1
            or selection.get("epoch") != epochs or metadata.get("epoch") != epochs
            or selection.get("clean_references_used") is not False):
        raise ValueError("Held-out evaluation requires the predeclared final-epoch checkpoint")


def load_model(checkpoint, device, *, require_selected=False):
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    metadata = state.get("metadata", {})
    if metadata.get("architecture_version") != 2:
        raise ValueError("Legacy checkpoint lacks global-residual architecture version 2; retrain, do not silently reinterpret")
    if require_selected:
        validate_selection(metadata)
    seed_everything(metadata["seed"])
    model = build_nafnet(NGPSConfig(**state["ngps_config"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state["metadata"]


@torch.no_grad()
def infer_volume(model, volume, device):
    validate_volume(volume)
    outputs = [model(torch.from_numpy(image)[None, None].to(device))[0, 0].cpu().numpy()
               for image in volume]
    return validate_volume(np.stack(outputs).astype(np.float32))


def timed_inference(function, volume, device="cpu"):
    """Wall time includes all slices, host/device transfers and output assembly.

    Excludes file I/O and model loading. No warmup; first-volume overhead remains.
    """
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = function(volume)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return validate_volume(result), elapsed
