"""Freeze study inputs in notebook 00; experiments only load these artifacts."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict

import numpy as np
import pandas as pd

from dtasrl.config import PROJECT_ROOT, resolve_project_path
from dtasrl.data.panel import fit_experiment_inputs
from dtasrl.envs.calibration import (
    calibration_settings,
    generate_calibration_market,
    prepare_calibration,
)
from dtasrl.provenance import write_json
from dtasrl.storage import inventory, save_table, verify_inventory


def _write_market(data, stem):
    frame = pd.DataFrame({"date": data["dates"]})
    schema = {}
    for key in (
        "prices",
        "opens",
        "features",
        "tradable",
        "forced_liquidation",
        "conditional_up_probability",
        "reference_risky_weight",
        "zero_cost_kelly_risky_weight",
    ):
        if key not in data:
            continue
        arr = np.asarray(data[key])
        schema[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        for i, column in enumerate(arr.reshape(len(arr), -1).T):
            frame[f"{key}_{i}"] = column
    save_table(frame, stem)
    meta = {k: v for k, v in data.items() if k not in schema and k != "dates"}
    write_json({"arrays": schema, "metadata": meta}, stem.with_suffix(".json"))


def _read_market(stem):
    frame = pd.read_parquet(stem.with_suffix(".parquet"))
    schema = json.loads(stem.with_suffix(".json").read_text())
    data = schema["metadata"]
    data["dates"] = frame.date.to_numpy()
    for key, spec in schema["arrays"].items():
        columns = [f"{key}_{i}" for i in range(int(np.prod(spec["shape"][1:])))]
        data[key] = frame[columns].to_numpy(dtype=spec["dtype"]).reshape(spec["shape"])
    return data


def freeze_real_inputs(config):
    prepared, digest = fit_experiment_inputs(config)
    identity = {
        "data_hash": digest,
        "split": prepared["provenance"]["split"],
        "observed": config["observation_tickers"],
        "schema": 1,
    }
    folder = PROJECT_ROOT / "data/processed/inputs/real"
    if (folder / "manifest.json").exists():
        try:
            verify_inventory(folder)
            if json.loads((folder / "metadata.json").read_text(encoding="utf-8")) == {
                **identity,
                "provenance": prepared["provenance"],
                "tickers": config["tickers"],
            }:
                return folder
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            pass
    staging = folder.with_name(f".{folder.name}.partial-{uuid.uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    for split in ("train", "validation", "test"):
        _write_market(prepared[split], staging / split)
    write_json(prepared["scaler"], staging / "scaler.json")
    write_json(
        {**identity, "provenance": prepared["provenance"], "tickers": config["tickers"]},
        staging / "metadata.json",
    )
    inventory(staging)
    verify_inventory(staging)
    if folder.exists():
        shutil.rmtree(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, folder)
    return folder


def load_real_inputs(config):
    folder = resolve_project_path(config["data"]["frozen_inputs"])
    verify_inventory(folder)
    metadata = json.loads((folder / "metadata.json").read_text())
    if (
        metadata["split"] != config["split"]
        or metadata["observed"] != config["observation_tickers"]
        or metadata["provenance"]["processed_snapshot"] != config["data"]["snapshot"]
    ):
        raise ValueError("Frozen input configuration mismatch")
    positions = [metadata["tickers"].index(t) for t in config["tickers"]]
    result = {}
    for split in ("train", "validation", "test"):
        item = _read_market(folder / split)
        for key in ("prices", "opens", "tradable", "forced_liquidation"):
            item[key] = item[key][:, positions]
        item["tickers"] = config["tickers"]
        result[split] = item
    result["scaler"] = json.loads((folder / "scaler.json").read_text())
    result["provenance"] = metadata["provenance"]
    return result, metadata["data_hash"]


def freeze_synthetic_inputs(config):
    prepared, digest = prepare_calibration(config)
    settings = calibration_settings(config)
    label = "synthetic_high" if settings.signal_strength else "synthetic_zero"
    folder = PROJECT_ROOT / "data/processed/inputs" / label
    if (folder / "manifest.json").exists():
        try:
            verify_inventory(folder)
            existing = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            if existing.get("data_sha256") == digest and existing.get("test_paths") == config.get(
                "evaluation", {}
            ).get("test_paths", 100):
                return folder
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            pass
    staging = folder.with_name(f".{folder.name}.partial-{uuid.uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    for split in ("train", "validation", "test"):
        _write_market(prepared[split], staging / split)
    count = config.get("evaluation", {}).get("test_paths", 100)
    for i in range(count):
        data = generate_calibration_market(
            settings, settings.test_steps, settings.test_market_seed + i
        )
        _write_market(data, staging / f"market_{settings.test_market_seed + i}")
    write_json({**prepared["metadata"], "test_paths": count}, staging / "metadata.json")
    inventory(staging)
    verify_inventory(staging)
    if folder.exists():
        shutil.rmtree(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, folder)
    return folder


def load_synthetic_inputs(config):
    folder = resolve_project_path(config["data"]["frozen_inputs"])
    verify_inventory(folder)
    meta = json.loads((folder / "metadata.json").read_text())
    if meta["settings"] != asdict(calibration_settings(config)):
        raise ValueError("Frozen synthetic settings mismatch")
    if meta["test_paths"] != config.get("evaluation", {}).get("test_paths", 100):
        raise ValueError("Frozen synthetic path count mismatch")
    return {
        **{s: _read_market(folder / s) for s in ("train", "validation", "test")},
        "metadata": meta,
    }, meta["data_sha256"]


def load_synthetic_market(config, market_seed):
    return _read_market(
        resolve_project_path(config["data"]["frozen_inputs"]) / f"market_{market_seed}"
    )
