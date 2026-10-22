from __future__ import annotations

import hashlib
import json
import zipfile
from importlib import import_module
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch


def source_record(snapshot=None):
    """Fingerprint runtime code/configs, including untracked files, never data.

    Training also archives these small text files so a dirty run is recoverable.
    Paths in the archive are relative to NGPS (or the installed package root).
    """
    root = Path(__file__).resolve().parents[1]
    paths = [path for folder in ("ngps", "datasets", "scripts")
             for path in (root / folder).rglob("*.py")]
    paths += [path for path in (root / "configs").glob("*")
              if path.suffix in {".json", ".yaml", ".yml"}]
    paths += [path for folder in ("docs", "tests")
              for path in (root / folder).rglob("*") if path.is_file()]
    paths += [root / name for name in ("requirements.txt", "pyproject.toml")
              if (root / name).is_file()]
    contents = {str(path.relative_to(root)): path.read_bytes() for path in sorted(paths)}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}
    result = {"files_sha256": hashes, "tree_sha256": hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    if snapshot is not None:
        snapshot = Path(snapshot)
        with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in contents.items():
                archive.writestr(zipfile.ZipInfo(name), content)
        result["snapshot"] = snapshot.name
        result["snapshot_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    return result


def provenance(device, *, source_snapshot=None):
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status == "unavailable" else bool(status),
            "source": source_record(source_snapshot),
            "device": str(device), "device_name": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else platform.processor(),
            "platform": platform.platform(), "python": platform.python_version(),
            "torch": str(torch.__version__), "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "dependencies": {name: str(import_module(name).__version__) for name in
                             ("numpy", "scipy", "skimage", "cv2", "yaml", "torch")}}


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
