from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from datasets.preparation import file_hash
from datasets.volume_folder import load_volume
from ngps.inference import infer_volume, load_model, timed_inference
from ngps.provenance import provenance, write_json


def main():
    parser = argparse.ArgumentParser(description="Unclipped complete-volume NAFNet inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model, metadata = load_model(args.checkpoint, args.device)
    result, seconds = timed_inference(lambda v: infer_volume(model, v, args.device),
                                      load_volume(args.input), args.device)
    output = Path(args.output)
    if output.suffix != ".npy":
        output = Path(str(output) + ".npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Hash inputs before writing; retain their identities even if paths later move.
    checkpoint_hash, input_hash = file_hash(args.checkpoint), file_hash(args.input)
    np.save(output, result)
    write_json(str(output) + ".json", {**provenance(args.device), "inference_seconds": seconds,
               "shape": list(result.shape), "checkpoint": str(Path(args.checkpoint).resolve()),
               "checkpoint_sha256": checkpoint_hash, "input": str(Path(args.input).resolve()),
               "input_sha256": input_hash, "prediction_sha256": file_hash(output),
               "training": metadata, "manifest_hash": metadata["manifest_hash"], "clipping": False})
    print(f"Saved {output}; complete-volume inference {seconds:.6f} s")


if __name__ == "__main__":
    main()
