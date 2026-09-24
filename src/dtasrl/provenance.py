from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dtasrl.config import PROJECT_ROOT, config_hash, public_config


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root: Path | None = None) -> str:
    """Fingerprint executable project sources, including uncommitted Python changes."""
    source_root = (root or PROJECT_ROOT) / "src"
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNCOMMITTED"


def git_is_dirty() -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return True


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_directory(config: dict[str, Any], seed: int) -> Path:
    experiment = str(config.get("experiment", "unnamed"))
    if config.get("study") == "ppo_attribution":
        groups = {
            "synthetic": "01_synthetic",
            "single_asset": "02_single_asset",
            "multi_asset": "03_multi_asset",
        }
        group = str(config.get("output_group", groups[config["kind"]]))
        if not re.fullmatch(r"[A-Za-z0-9_-]+", group):
            raise ValueError("output_group must be a single safe directory name")
        return (
            PROJECT_ROOT
            / "outputs"
            / group
            / "runs"
            / experiment
            / config_hash(config)
            / str(seed)
        )
    return PROJECT_ROOT / "artifacts" / "runs" / experiment / config_hash(config) / str(seed)


def runtime_metadata(config: dict[str, Any], seed: int, data_hash: str | None) -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "experiment": config.get("experiment"),
        "stage": config.get("stage", "pilot"),
        "seed": seed,
        "config_hash": config_hash(config),
        "data_hash": data_hash,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "source_tree_sha256": source_tree_sha256(),
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
    }


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            suffix=".partial",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def initialise_run(
    config: dict[str, Any], seed: int, data_hash: str | None = None
) -> tuple[Path, dict[str, Any]]:
    path = run_directory(config, seed)
    if config.get("study") == "ppo_attribution" and path.exists():
        if not config.get("_resume"):
            raise FileExistsError(f"Run already exists; use explicit --resume: {path}")
        existing = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if (
            existing["config_hash"] != config_hash(config)
            or existing["data_hash"] != data_hash
            or existing.get("source_tree_sha256") != source_tree_sha256()
        ):
            raise ValueError("Run resume identity mismatch")
        return path, existing
    path.mkdir(parents=True, exist_ok=True)
    metadata = runtime_metadata(config, seed, data_hash)
    write_json(public_config(config), path / "config.json")
    write_json(metadata, path / "metadata.json")
    return path, metadata
