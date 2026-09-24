"""Durable local artifacts; consumers trust completed manifests, never partial files."""
from __future__ import annotations

import json
import os
import zipfile
from importlib.metadata import distributions
from pathlib import Path

import pandas as pd

from dtasrl.config import PROJECT_ROOT
from dtasrl.provenance import sha256_file, utc_now, write_json


def save_table(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".csv", ".parquet"):
        target = stem.with_suffix(suffix)
        temp = target.with_name(target.name + ".partial")
        if suffix == ".csv":
            frame.to_csv(temp, index=False, encoding="utf-8", date_format="%Y-%m-%d",
                         float_format="%.17g")
        else:
            frame.to_parquet(temp, index=False)
        with temp.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, target)


def snapshot_code(destination: Path) -> None:
    target = destination / "source.zip"
    if target.exists():
        return
    temp = target.with_suffix(".partial")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("src", "scripts", "configs"):
            for path in sorted((PROJECT_ROOT / folder).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(PROJECT_ROOT).as_posix())
        for name in ("pyproject.toml", "uv.lock", "environment.yml", "README.md"):
            path = PROJECT_ROOT / name
            if path.exists():
                archive.write(path, name)
    os.replace(temp, target)
    write_json({d.metadata["Name"]: d.version for d in distributions()},
               destination / "dependencies.json")


def inventory(directory: Path) -> dict:
    rows = []
    for path in sorted(directory.rglob("*")):
        if (not path.is_file() or path.name in ("manifest.json", "inventory.csv",
                "inventory.parquet", ".training.lock") or ".partial" in path.name):
            continue
        rows.append({"path": path.relative_to(directory).as_posix(),
                     "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    save_table(pd.DataFrame(rows), directory / "inventory")
    result = {"status": "complete", "created_at_utc": utc_now(), "files": rows}
    write_json(result, directory / "manifest.json")
    return result


def verify_inventory(directory: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("incomplete artifact manifest")
    for item in manifest["files"]:
        path = directory / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"artifact missing or corrupt: {path}")
