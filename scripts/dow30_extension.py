"""Preparation and reporting for 30-stock joint-portfolio attribution.

The 30-stock experiment shares feature, execution, PPO, and attribution
implementations with the 3- and 100-stock experiments. Its frozen inputs and
outputs are separate; it is not part of the base 18-task planner. Existing
model artifacts and planned source hashes are never rebound by this module.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dtasrl.config import PROJECT_ROOT, config_hash, load_config, public_config
from dtasrl.data.inputs import _write_market
from dtasrl.data.panel import fit_experiment_inputs
from dtasrl.provenance import run_directory, sha256_file, write_json
from dtasrl.storage import inventory, save_table, verify_inventory
from dtasrl.study_reports import EXPORT_DPI, figure, status_figure

CONFIG = PROJECT_ROOT / "configs/experiments/multi_dow30.yaml"
OUTPUT = PROJECT_ROOT / "outputs/04_dow30"
SEEDS = (101, 202, 303)


def _load_config() -> dict:
    config = load_config(CONFIG)
    snapshot = json.loads((PROJECT_ROOT / config["data"]["snapshot"]).read_text(encoding="utf-8"))
    tickers = snapshot["tickers_returned"]
    if (
        len(tickers) != 30
        or config["tickers"] != tickers
        or config["observation_tickers"] != tickers
        or config["experiment"] != "ppo_multi_dow30"
        or config["kind"] != "multi_asset"
        or config["stage"] != "extension"
        or config.get("output_group") != "04_dow30"
    ):
        raise ValueError("The extension configuration must match all 30 frozen constituents")
    return config


def prepare_inputs() -> Path:
    """Freeze all-stock features into a new directory; never replace the three-stock input."""
    config = _load_config()
    folder = PROJECT_ROOT / config["data"]["frozen_inputs"]
    if folder.exists():
        verify_inventory(folder)
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if (
            metadata["tickers"] != config["tickers"]
            or metadata["observed"] != config["observation_tickers"]
        ):
            raise ValueError("Existing 30-stock input has a different ticker definition")
        if metadata["split"] != config["split"]:
            raise ValueError("Existing 30-stock input has a different time split")
        snapshot_hash = json.loads(
            (PROJECT_ROOT / config["data"]["snapshot"]).read_text(encoding="utf-8")
        )["features_sha256"]
        if metadata["data_hash"] != snapshot_hash:
            raise ValueError("Existing 30-stock input does not match the current processed data")
        return folder

    prepared, digest = fit_experiment_inputs(config)
    prepared["provenance"]["case_selection"] = (
        "All 30 constituents fixed at 2026-08-31; no stock selected by test returns"
    )
    metadata = {
        "data_hash": digest,
        "split": config["split"],
        "observed": config["observation_tickers"],
        "schema": 1,
        "provenance": prepared["provenance"],
        "tickers": config["tickers"],
    }
    folder.parent.mkdir(parents=True, exist_ok=True)
    staging = folder.with_name(f".{folder.name}.partial-{uuid.uuid4().hex}")
    staging.mkdir(parents=False, exist_ok=False)
    for split in ("train", "validation", "test"):
        _write_market(prepared[split], staging / split)
    write_json(prepared["scaler"], staging / "scaler.json")
    write_json(metadata, staging / "metadata.json")
    inventory(staging)
    verify_inventory(staging)
    os.replace(staging, folder)
    return folder


def _complete_run(run: Path, config: dict, data_hash: str) -> bool:
    if not (run / "manifest.json").is_file():
        return False
    verify_inventory(run)
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    training = json.loads((run / "training_status.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run / "evaluation_status.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if (
        metadata["config_hash"] != config_hash(config)
        or metadata["data_hash"] != data_hash
        or training["status"] != "complete"
        or evaluation["status"] != "complete"
        or metrics["actual_transitions"] != config["total_timesteps"]
    ):
        raise ValueError(f"Completed run has a mismatched identity or step budget: {run}")
    return True


def _initial_weights_figure(run: Path, destination: Path, tickers: list[str]) -> Path:
    first = pd.read_parquet(run / "target_weights.parquet").iloc[0]
    frame = pd.DataFrame({"ticker": tickers, "weight": [float(first[t]) for t in tickers]})
    frame = frame.sort_values("weight", ascending=True).reset_index(drop=True)
    save_table(frame, destination / "initial_weights")
    fig, ax = plt.subplots(figsize=(8.2, 8.6), layout="constrained")
    ax.barh(frame.ticker, frame.weight, color="#2364AA")
    ax.set(xlabel="Initial target weight", ylabel="", title="First PPO allocation across 30 stocks")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    fig.savefig(destination / "initial_weights.png", dpi=EXPORT_DPI)
    plt.close(fig)
    write_json(
        {
            "source": "target_weights.parquet, first decision",
            "unit": "portfolio fraction",
            "dpi": EXPORT_DPI,
            "generator_sha256": sha256_file(Path(__file__)),
        },
        destination / "initial_weights.json",
    )
    return destination / "initial_weights.png"


def _monthly_weights_figure(run: Path, destination: Path, tickers: list[str]) -> Path:
    targets = pd.read_parquet(run / "target_weights.parquet")
    targets["decision_date"] = pd.to_datetime(targets["decision_date"])
    monthly = targets.set_index("decision_date")[tickers].resample("MS").mean()
    monthly = monthly.loc[monthly.index <= "2025-12-31"]
    save_table(monthly.reset_index(), destination / "monthly_target_weights")
    fig, ax = plt.subplots(figsize=(12.4, 8.4), layout="constrained")
    image = ax.imshow(monthly.to_numpy().T, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set_yticks(range(len(tickers)), tickers, fontsize=8)
    positions = np.arange(0, len(monthly), 12)
    ax.set_xticks(positions, monthly.index[positions].strftime("%Y"))
    ax.set(xlabel="Decision month", ylabel="", title="Mean monthly PPO target weights")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Portfolio weight", shrink=0.8)
    fig.savefig(destination / "monthly_target_weights.png", dpi=EXPORT_DPI)
    plt.close(fig)
    write_json(
        {
            "source": "target_weights.parquet",
            "aggregation": "calendar-month mean of daily target weights",
            "period_end": "2025-12-31",
            "unit": "portfolio fraction",
            "dpi": EXPORT_DPI,
            "generator_sha256": sha256_file(Path(__file__)),
        },
        destination / "monthly_target_weights.json",
    )
    return destination / "monthly_target_weights.png"


def report() -> dict:
    """Summarize only complete 30-stock runs, never the earlier three-stock study."""
    config = _load_config()
    frozen = PROJECT_ROOT / config["data"]["frozen_inputs"]
    verify_inventory(frozen)
    frozen_metadata = json.loads((frozen / "metadata.json").read_text(encoding="utf-8"))
    if (
        frozen_metadata["tickers"] != config["tickers"]
        or frozen_metadata["observed"] != config["observation_tickers"]
    ):
        raise ValueError("Frozen input is not the 30-stock configuration")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries, effects, figures = [], [], []
    status = _status(config, frozen_metadata["data_hash"])
    complete_seeds = set(status.loc[status.status.eq("complete"), "seed"])
    figure_status = status.assign(
        seed=status.seed.map({seed: i + 1 for i, seed in enumerate(SEEDS)})
    )
    figures.append(status_figure(figure_status, OUTPUT / "figures"))
    for seed in SEEDS:
        run = run_directory(config, seed)
        if seed not in complete_seeds:
            continue
        result = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        for strategy, values in result["controls"]["metrics"].items():
            summaries.append({"seed": seed, "strategy": strategy, **values})
        for name, value in result["controls"]["effects"].items():
            effects.append({"seed": seed, "effect": name, "net_log_wealth_difference": value})
        destination = OUTPUT / "figures" / str(seed)
        destination.mkdir(parents=True, exist_ok=True)
        controls = []
        for name in ("full_ppo", "equal_weight_hold", "freeze_1"):
            path = pd.read_parquet(run / "controls/main" / f"{name}.parquet")
            controls.append(
                path.set_index("execution_date").wealth_after.rename(name)
                / path.wealth_before.iloc[0]
            )
        wealth = pd.concat(controls, axis=1).reset_index()
        wealth["execution_date"] = pd.to_datetime(wealth["execution_date"])
        figures.append(
            figure(
                wealth,
                destination,
                "wealth",
                title=f"30-stock portfolio | Seed {SEEDS.index(seed) + 1}",
                ylabel="Wealth / initial wealth",
                x="execution_date",
                columns=list(wealth.columns[1:]),
            )
        )
        figures.append(_initial_weights_figure(run, destination, config["tickers"]))
        figures.append(_monthly_weights_figure(run, destination, config["tickers"]))
    summary = (
        pd.DataFrame(summaries)
        if summaries
        else pd.DataFrame(
            {
                "seed": pd.Series(dtype="int64"),
                "strategy": pd.Series(dtype="str"),
                "net_log_wealth": pd.Series(dtype="float64"),
            }
        )
    )
    effect_frame = (
        pd.DataFrame(effects)
        if effects
        else pd.DataFrame(
            {
                "seed": pd.Series(dtype="int64"),
                "effect": pd.Series(dtype="str"),
                "net_log_wealth_difference": pd.Series(dtype="float64"),
            }
        )
    )
    save_table(status, OUTPUT / "status")
    save_table(summary, OUTPUT / "summary")
    save_table(effect_frame, OUTPUT / "effects")
    write_json(public_config(config), OUTPUT / "configuration.json")
    write_json(frozen_metadata, OUTPUT / "input_metadata.json")
    write_json(
        {
            "experiment": config["experiment"],
            "expected_runs": len(SEEDS),
            "complete_runs": int(status.status.eq("complete").sum()),
            "formal_inference": False,
            "comparison_scope": "within the fixed 30-stock universe",
        },
        OUTPUT / "report_manifest.json",
    )
    inventory(OUTPUT)
    return {
        "status": status,
        "summary": summary,
        "effects": effect_frame,
        "figures": figures,
        "out": OUTPUT,
    }


def _status(config: dict, data_hash: str) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        run = run_directory(config, seed)
        if _complete_run(run, config, data_hash):
            state = "complete"
        elif (run / "evaluation_status.json").is_file():
            evaluation = json.loads((run / "evaluation_status.json").read_text(encoding="utf-8"))
            state = "evaluation_" + str(evaluation["status"])
        elif (run / "training_status.json").is_file():
            state = json.loads((run / "training_status.json").read_text(encoding="utf-8"))["status"]
        else:
            state = "not_started"
        rows.append(
            {
                "condition": "multi_dow30",
                "seed": seed,
                "status": state,
                "run": run.relative_to(PROJECT_ROOT).as_posix(),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "report"))
    args = parser.parse_args()
    print(
        prepare_inputs() if args.command == "prepare" else report()["status"].to_string(index=False)
    )
