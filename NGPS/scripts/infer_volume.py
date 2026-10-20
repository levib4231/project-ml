from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tqdm import tqdm

from datasets.volume_folder import load_volume
from ngps import build_nafnet
from ngps.config import NGPSConfig


def _to_bchw(slice_: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(slice_.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise one .npy volume with a trained NGPS/NAFNet checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip-output", action="store_true", help="Explicitly clip predictions to [0, 1].")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError("Checkpoint must be a mapping containing a 'model' state dictionary.")
    cfg_state = state.get("ngps_config", {})
    if not isinstance(cfg_state, dict):
        raise ValueError("Checkpoint 'ngps_config' must be a mapping.")
    cfg = NGPSConfig(**cfg_state) if isinstance(cfg_state, dict) else cfg_state
    model = build_nafnet(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    volume = load_volume(args.input)
    outputs = []
    with torch.no_grad():
        for z in tqdm(range(volume.shape[0]), desc="Infer"):
            pred = model(_to_bchw(volume[z], device)).squeeze().cpu().numpy()
            prediction = pred.astype(np.float32)
            if not np.isfinite(prediction).all():
                raise ValueError(f"Model produced non-finite values at slice {z}.")
            outputs.append(np.clip(prediction, 0.0, 1.0) if args.clip_output else prediction)
    output = np.stack(outputs, axis=0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)
    print(f"Saved denoised volume to {args.output}")


if __name__ == "__main__":
    main()
