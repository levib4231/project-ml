from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
import platform
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

from datasets.manifest import sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty() -> bool | None:
    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _software_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "cv2", "torch", "yaml"):
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError:
            versions[name] = None
        else:
            versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


class RunRecorder:
    def __init__(
        self,
        output_path: str | Path,
        *,
        command: Sequence[str],
        config: Mapping[str, Any],
        seed: int,
        device: str,
        dataset_manifest_path: str | Path | None,
    ) -> None:
        self.output_path = Path(output_path)
        self.started = time.monotonic()
        manifest_path = Path(dataset_manifest_path) if dataset_manifest_path else None
        self.payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "command": list(command),
            "config": dict(config),
            "seed": int(seed),
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "software_versions": _software_versions(),
            "device": device,
            "dataset_manifest": str(manifest_path) if manifest_path else None,
            "dataset_manifest_sha256": sha256_file(manifest_path) if manifest_path else None,
            "started_at_utc": _utc_now(),
            "ended_at_utc": None,
            "wall_time_seconds": None,
        }
        self._write()

    def _write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.output_path)

    def finish(self, status: str, *, extra: Mapping[str, Any] | None = None) -> None:
        self.payload["status"] = status
        self.payload["ended_at_utc"] = _utc_now()
        self.payload["wall_time_seconds"] = time.monotonic() - self.started
        if extra:
            self.payload.update(dict(extra))
        self._write()
