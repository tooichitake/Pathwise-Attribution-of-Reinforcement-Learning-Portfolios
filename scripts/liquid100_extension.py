"""Preparation and reporting for 100-stock joint-portfolio attribution.

Data preparation is initiated by Notebook 00. Notebook 05 reads the frozen
inputs and starts training only when its explicit switch is enabled. The
candidate list is retrospective; coverage and liquidity use training dates
only. The universe is not the S&P 100 or a historical S&P 500 reconstruction.
"""

from __future__ import annotations

import argparse
import html
import importlib.metadata
import io
import json
import os
import re
import subprocess
import sys
import uuid
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import requests

from dtasrl.config import PROJECT_ROOT, config_hash, dump_yaml, load_config, public_config
from dtasrl.data.features import FEATURE_COLUMNS, engineer_features
from dtasrl.data.panel import PANEL_COLUMNS, _write_layer, fit_experiment_inputs
from dtasrl.data.yahoo import download_yahoo
from dtasrl.provenance import git_commit, run_directory, sha256_file, utc_now, write_json
from dtasrl.storage import inventory, save_table, verify_inventory
from dtasrl.study_reports import EXPORT_DPI, PALETTE, status_figure
from scripts.dow30_extension import _complete_run

DATA_CONFIG = PROJECT_ROOT / "configs/data/liquid100.yaml"
CONFIG = PROJECT_ROOT / "configs/experiments/multi_liquid100.yaml"
OUTPUT = PROJECT_ROOT / "outputs/05_liquid100"
REPORT = OUTPUT / "report"
DATA_REPORT = PROJECT_ROOT / "outputs/00_data/liquid100"
SEEDS = (101, 202, 303)
LABELS = {101: 1, 202: 2, 303: 3}


def _publish_bytes(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen source: {path}")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def parse_holdings(content: str) -> tuple[pd.DataFrame, str]:
    lines = content.lstrip("\ufeff").splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith("Ticker,")), None)
    date_line = next((line for line in lines if line.startswith("Fund Holdings as of,")), None)
    if header is None or date_line is None:
        raise ValueError("Holdings response is not the expected dated CSV")
    as_of = pd.Timestamp(next(iter(pd.read_csv(io.StringIO(date_line), header=None)[1])))
    frame = pd.read_csv(io.StringIO("\n".join(lines[header:])), on_bad_lines="skip")
    frame = frame.loc[frame["Asset Class"].eq("Equity")].copy()
    frame["ticker"] = frame["Ticker"].str.strip().str.replace(" ", "-", regex=False)
    frame["ticker"] = frame["ticker"].str.replace(".", "-", regex=False)
    if frame.ticker.duplicated().any() or len(frame) < 400:
        raise ValueError("Holdings response does not contain a credible large-cap equity list")
    return frame, as_of.date().isoformat()


def candidate_table(public_csv: str, holdings_csv: str) -> tuple[pd.DataFrame, dict]:
    public = pd.read_csv(io.StringIO(public_csv), dtype=str)
    required = {"Symbol", "Security", "GICS Sector", "CIK"}
    if not required <= set(public):
        raise ValueError("Public constituent list has changed schema")
    public["ticker"] = public.Symbol.str.replace(".", "-", regex=False)
    holdings, as_of = parse_holdings(holdings_csv)
    public["in_holdings_crosscheck"] = public.ticker.isin(holdings.ticker)
    if public.ticker.duplicated().any() or public.CIK.isna().any():
        raise ValueError("Constituent symbols must be unique and issuer CIKs must be present")
    # Only a mutually corroborated candidate is eligible. Do not import ETF
    # weights, prices, cash, futures, or fund returns into the learning task.
    public = public.rename(columns={"Security": "company", "GICS Sector": "sector"})
    public["issuer_id"] = public.CIK.str.lstrip("0").replace("", "0")
    return public.sort_values("ticker").reset_index(drop=True), {
        "public_list_symbols": len(public),
        "equity_holdings_symbols": len(holdings),
        "crosschecked_symbols": int(public.in_holdings_crosscheck.sum()),
        "public_only": sorted(set(public.ticker) - set(holdings.ticker)),
        "holdings_only": sorted(set(holdings.ticker) - set(public.ticker)),
        "holdings_as_of": as_of,
        "membership_semantics": (
            "current public constituent list crosschecked with IVV equity holdings"
        ),
        "not_index_reconstruction": True,
    }


def select_universe(
    raw: pd.DataFrame, candidates: pd.DataFrame, calendar: pd.DatetimeIndex, rules: dict
) -> pd.DataFrame:
    """Return all selection decisions; future rows never enter the decision."""
    dates = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
    start, end = (pd.Timestamp(rules["split"][key]) for key in ("train_start", "train_end"))
    training = dates[(dates >= start) & (dates <= end)]
    warmup = dates[dates < start][-int(rules["warmup_sessions"]) :]
    liquidity_dates = training[-int(rules["liquidity_sessions"]) :]
    if len(warmup) < int(rules["warmup_sessions"]) or len(liquidity_dates) < int(
        rules["liquidity_sessions"]
    ):
        raise ValueError("Calendar does not contain the prespecified training and warmup history")
    historical = raw.loc[pd.to_datetime(raw.date) <= end].copy()
    historical["date"] = pd.to_datetime(historical.date).dt.normalize()
    if historical.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker in selection input")
    groups = {ticker: group.set_index("date") for ticker, group in historical.groupby("ticker")}
    rows = []
    for candidate in candidates.itertuples(index=False):
        group = groups.get(candidate.ticker, pd.DataFrame(columns=["Adj Close", "Close", "Volume"]))
        price = pd.to_numeric(group["Adj Close"], errors="coerce")
        valid = price.notna() & np.isfinite(price) & price.gt(0)
        coverage = float(valid.reindex(training, fill_value=False).mean())
        warmup_complete = bool(valid.reindex(warmup, fill_value=False).all())
        quotes = group.reindex(liquidity_dates)
        amount = pd.to_numeric(quotes.Close, errors="coerce") * pd.to_numeric(
            quotes.Volume, errors="coerce"
        )
        amount = amount.where(np.isfinite(amount) & amount.gt(0))
        liquidity = float(amount.median()) if amount.notna().any() else float("nan")
        reason = "eligible"
        if not candidate.in_holdings_crosscheck:
            reason = "not_crosschecked_in_holdings"
        elif not warmup_complete:
            reason = "insufficient_pretraining_history"
        elif coverage < float(rules["minimum_training_coverage"]):
            reason = "insufficient_training_coverage"
        elif amount.notna().mean() < float(rules["minimum_training_coverage"]):
            reason = "insufficient_training_liquidity_data"
        elif not np.isfinite(liquidity):
            reason = "invalid_training_liquidity"
        # Class C GOOG began in 2014. Its vendor backfilled price series is
        # not evidence that Class C was independently tradable in 2008.
        elif candidate.ticker in rules.get("duplicate_class_preference", {}):
            reason = (
                "alternate_share_class_use_" + rules["duplicate_class_preference"][candidate.ticker]
            )
        rows.append(
            {
                "ticker": candidate.ticker,
                "company": candidate.company,
                "sector": candidate.sector,
                "issuer_id": candidate.issuer_id,
                "training_coverage": coverage,
                "warmup_complete": warmup_complete,
                "median_training_dollar_volume": liquidity,
                "liquidity_start": liquidity_dates[0].date().isoformat(),
                "liquidity_end": liquidity_dates[-1].date().isoformat(),
                "reason": reason,
                "eligible": reason == "eligible",
                "selected": False,
                "rank": np.nan,
            }
        )
    audit = pd.DataFrame(rows)
    eligible = audit.loc[audit.eligible].sort_values(
        ["median_training_dollar_volume", "ticker"], ascending=[False, True], kind="stable"
    )
    unique = eligible.drop_duplicates("issuer_id", keep="first")
    duplicate_indices = eligible.index.difference(unique.index)
    audit.loc[duplicate_indices, "eligible"] = False
    audit.loc[duplicate_indices, "reason"] = "duplicate_issuer_share_class"
    audit.loc[unique.index, "rank"] = np.arange(1, len(unique) + 1)
    selected_indices = unique.head(int(rules["target_stocks"])).index
    audit.loc[selected_indices, "selected"] = True
    audit.loc[selected_indices, "reason"] = "selected"
    audit.loc[audit.eligible & ~audit.selected, "reason"] = "below_liquidity_cutoff"
    return audit.sort_values("ticker").reset_index(drop=True)


def _sources(rules: dict) -> tuple[pd.DataFrame, dict]:
    raw_folder = PROJECT_ROOT / "data/raw"
    urls = {
        "constituents": rules["candidate_list_url"],
        "holdings": rules["holdings_crosscheck_url"],
    }
    contents, receipts = {}, {}
    for name, url in urls.items():
        response = requests.get(
            url, timeout=60, headers={"User-Agent": "DTASRL/0.1 research data pipeline"}
        )
        response.raise_for_status()
        if "<html" in response.text[:500].lower():
            raise ValueError(f"Unexpected HTML rather than CSV: {url}")
        target = raw_folder / f"liquid100_{name}_source.csv"
        _publish_bytes(response.content, target)
        contents[name] = response.content.decode("utf-8-sig")
        receipts[name] = {
            "url": url,
            "retrieved_at_utc": utc_now(),
            "path": _relative(target),
            "sha256": sha256_file(target),
        }
    candidates, crosscheck = candidate_table(contents["constituents"], contents["holdings"])
    save_table(candidates, raw_folder / "liquid100_constituents")
    return candidates, {"source_files": receipts, "candidate_crosscheck": crosscheck}


def _reference_calendar(rules: dict) -> tuple[pd.DatetimeIndex, dict]:
    snapshot = PROJECT_ROOT / rules["reference_snapshot"]
    metadata = json.loads(snapshot.read_text(encoding="utf-8"))
    path = PROJECT_ROOT / metadata["layers"]["processed"]["parquet"]
    if sha256_file(path) != metadata["features_sha256"]:
        raise ValueError("Reference Dow panel hash is invalid; do not silently change calendars")
    dates = pd.DatetimeIndex(pd.read_parquet(path, columns=["date"]).date.unique()).sort_values()
    if str(dates[-1].date()) != rules["expected_end"]:
        raise ValueError("Reference calendar does not end at the agreed market-data cutoff")
    return dates, {
        "path": _relative(snapshot),
        "sha256": sha256_file(snapshot),
        "panel_path": _relative(path),
        "panel_sha256": sha256_file(path),
    }


def _download_candidates(candidates: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, dict]:
    pieces, receipts, failures = [], [], []
    tickers = candidates.loc[candidates.in_holdings_crosscheck, "ticker"].tolist()
    # Serial small batches use the existing locked Yahoo download implementation.
    for offset in range(0, len(tickers), 20):
        batch = tickers[offset : offset + 20]
        frame, metadata = download_yahoo(batch, rules["start"], rules["end"])
        pieces.append(frame)
        receipts.append(metadata)
        failures.extend(metadata["failed_tickers"])
        print(
            f"Candidate price download: {min(offset + 20, len(tickers))}/{len(tickers)}", flush=True
        )
    raw = (
        pd.concat(pieces, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    )
    raw["date"] = pd.to_datetime(raw.date).dt.normalize()
    return raw, {
        "source": "yfinance",
        "retrieved_at_utc": utc_now(),
        "batches": receipts,
        "failed_tickers": sorted(set(failures)),
        "tickers_requested": tickers,
        "request": {
            "start": rules["start"],
            "end": rules["end"],
            "auto_adjust": False,
            "actions": True,
            "threads": False,
        },
    }


def _write_market_wide(data: dict, stem: Path) -> None:
    """Same frozen schema as existing loaders, without thousands of insertions."""
    arrays, columns = {}, {"date": data["dates"]}
    for key in ("prices", "opens", "features", "tradable", "forced_liquidation"):
        arr = np.asarray(data[key])
        arrays[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        columns.update({f"{key}_{i}": col for i, col in enumerate(arr.reshape(len(arr), -1).T)})
    save_table(pd.DataFrame(columns), stem)
    metadata = {k: v for k, v in data.items() if k not in arrays and k != "dates"}
    write_json({"arrays": arrays, "metadata": metadata}, stem.with_suffix(".json"))


def load_extension_config() -> dict:
    config = load_config(CONFIG)
    if len(config["tickers"]) != 100 or config["tickers"] != config["observation_tickers"]:
        raise ValueError("Prepare and freeze the 100-stock inputs in Notebook 00 first")
    if config["output_group"] != "05_liquid100" or config["experiment"] != "ppo_multi_liquid100":
        raise ValueError("Extension must remain isolated from the existing experiment directories")
    snapshot = json.loads((PROJECT_ROOT / config["data"]["snapshot"]).read_text(encoding="utf-8"))
    if snapshot["tickers_returned"] != config["tickers"]:
        raise ValueError("Selected universe differs from the frozen processed snapshot")
    return config


def freeze_inputs() -> Path:
    config = load_extension_config()
    folder = PROJECT_ROOT / config["data"]["frozen_inputs"]
    if folder.exists():
        verify_inventory(folder)
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        snapshot = json.loads(
            (PROJECT_ROOT / config["data"]["snapshot"]).read_text(encoding="utf-8")
        )
        if (
            metadata["tickers"] != config["tickers"]
            or metadata["split"] != config["split"]
            or metadata["data_hash"] != snapshot["features_sha256"]
        ):
            raise ValueError(
                "Existing frozen input has a different identity; it will not be replaced"
            )
        return folder
    prepared, digest = fit_experiment_inputs(config)
    prepared["provenance"]["case_selection"] = (
        "100 unique issuers from a current S&P 500 candidate list; training-only "
        "history eligibility and last-252-session median Close*Volume ranking"
    )
    prepared["provenance"]["selection_audit"] = config["data"]["selection_audit"]
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
    staging.mkdir(exist_ok=False)
    for split in ("train", "validation", "test"):
        _write_market_wide(prepared[split], staging / split)
    write_json(prepared["scaler"], staging / "scaler.json")
    write_json(metadata, staging / "metadata.json")
    inventory(staging)
    verify_inventory(staging)
    os.replace(staging, folder)
    return folder


def prepare_data() -> Path:
    """Create only liquid100-prefixed data; never invoke the overwriting Dow builder."""
    rules = load_config(DATA_CONFIG)
    protocol_hash = config_hash(rules, length=64)
    metadata_path = PROJECT_ROOT / "data/processed/liquid100_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata["status"] != "complete"
            or metadata["preparation_protocol_sha256"] != protocol_hash
        ):
            raise ValueError("Existing extension snapshot cannot be replaced implicitly")
        for layer in metadata["layers"].values():
            for kind in ("csv", "parquet"):
                if sha256_file(PROJECT_ROOT / layer[kind]) != layer[f"{kind}_sha256"]:
                    raise ValueError(f"Extension data hash mismatch: {layer[kind]}")
        return freeze_inputs()
    raw_receipt = PROJECT_ROOT / "data/raw/liquid100_metadata.json"
    calendar, reference = _reference_calendar(rules)
    if raw_receipt.exists():
        receipt = json.loads(raw_receipt.read_text(encoding="utf-8"))
        if (
            receipt["preparation_protocol_sha256"] != protocol_hash
            or receipt["reference_calendar"] != reference
        ):
            raise ValueError("Saved candidate download has different rules or calendar")
        raw_layer = receipt["layer"]
        if sha256_file(PROJECT_ROOT / raw_layer["parquet"]) != raw_layer["parquet_sha256"]:
            raise ValueError("Saved candidate market data failed its hash check")
        candidates = pd.read_parquet(PROJECT_ROOT / "data/raw/liquid100_constituents.parquet")
        raw = pd.read_parquet(PROJECT_ROOT / raw_layer["parquet"])
    else:
        candidates, sources = _sources(rules)
        raw, downloads = _download_candidates(candidates, rules)
        raw_layer = _write_layer(raw, PROJECT_ROOT / "data/raw", "liquid100_candidate_market_data")
        receipt = {
            **sources,
            **downloads,
            "status": "complete",
            "layer": raw_layer,
            "preparation_protocol_sha256": protocol_hash,
            "reference_calendar": reference,
            "packages": {
                p: importlib.metadata.version(p) for p in ("yfinance", "pandas", "numpy", "pyarrow")
            },
        }
        write_json(receipt, raw_receipt)
    audit = select_universe(raw, candidates, calendar, rules)
    save_table(audit, PROJECT_ROOT / "data/interim/liquid100_selection_audit")
    selected = sorted(audit.loc[audit.selected, "ticker"])
    if len(selected) != int(rules["target_stocks"]):
        raise ValueError(
            f"Only {len(selected)} eligible issuers; no substitution or fake history is allowed"
        )
    selected_raw = raw.loc[raw.ticker.isin(selected)].copy()
    interim = engineer_features(selected_raw)
    ratio = interim["Adj Close"] / interim["Close"]
    for name in ("Open", "High", "Low", "Close"):
        interim[f"adjusted_{name.lower()}"] = interim[name] * ratio
    quotes = selected_raw.pivot(index="date", columns="ticker", values="Adj Close").reindex(
        columns=selected
    )
    valid = np.isfinite(quotes) & quotes.gt(0)
    common = pd.DatetimeIndex(quotes.index[valid.all(axis=1)])
    if common.empty or str(common[-1].date()) != rules["expected_end"]:
        raise ValueError("Selected universe does not cover the terminal evaluation date")
    expected = calendar[(calendar >= common[0]) & (calendar <= common[-1])]
    panel = (
        interim.loc[interim.date.isin(expected)]
        .rename(
            columns={
                **{f"adjusted_{name}": name for name in ("open", "high", "low", "close")},
                "Volume": "volume",
            }
        )
        .copy()
    )
    panel["tradable"] = panel.volume.gt(0)
    panel = panel[list(PANEL_COLUMNS)].sort_values(["date", "ticker"]).reset_index(drop=True)
    values = panel[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if (
        len(panel) != len(expected) * len(selected)
        or not panel.groupby("date").ticker.nunique().eq(100).all()
    ):
        raise ValueError(
            "Selected panel has missing calendar rows; no test-based replacement is allowed"
        )
    if (
        not np.isfinite(values).all()
        or (values <= 0).any()
        or panel.volume.isna().any()
        or panel.volume.lt(0).any()
    ):
        raise ValueError("Selected panel has invalid quotes; repair the source, not the selection")
    post_warmup = panel.loc[
        panel.date >= expected[expected < pd.Timestamp(rules["split"]["train_start"])][-1]
    ]
    if not np.isfinite(post_warmup[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("Selected panel has incomplete features after training warmup")
    layers = {
        "raw": raw_layer,
        "interim": _write_layer(
            interim, PROJECT_ROOT / "data/interim", "liquid100_features_unscaled"
        ),
        "processed": _write_layer(panel, PROJECT_ROOT / "data/processed", "liquid100_panel"),
    }
    metadata = {
        "status": "complete",
        "name": rules["name"],
        "snapshot_id": "liquid100_" + utc_now(),
        "preparation_protocol_sha256": protocol_hash,
        "universe_condition": "current_sp500_training_eligible_liquid100_backfilled",
        "survivorship_bias": (
            "Present: current membership is retrospective; "
            "this is not a historical investable index"
        ),
        "membership_sources": receipt["source_files"],
        "candidate_crosscheck": receipt["candidate_crosscheck"],
        "reference_calendar": reference,
        "query": receipt["request"],
        "packages": receipt["packages"],
        "git_commit": git_commit(),
        "pipeline_sha256": sha256_file(Path(__file__)),
        "tickers_returned": selected,
        "candidate_count": len(candidates),
        "selection_rules": public_config(rules),
        "selection_audit": "data/interim/liquid100_selection_audit.parquet",
        "selection_sha256": sha256_file(
            PROJECT_ROOT / "data/interim/liquid100_selection_audit.parquet"
        ),
        "date_min": str(expected[0].date()),
        "date_max": str(expected[-1].date()),
        "common_trading_days": len(expected),
        "field_order": list(PANEL_COLUMNS),
        "feature_order": list(FEATURE_COLUMNS),
        "raw_sha256": raw_layer["parquet_sha256"],
        "features_sha256": layers["processed"]["parquet_sha256"],
        "layers": layers,
        "price_semantics": "total_return_adjusted_units_dividends_not_credited_again",
        "adjustment_vintage_policy": "single_download_session_no_splicing",
        "no_backward_fill": True,
        "no_test_based_substitution": True,
    }
    write_json(metadata, metadata_path)
    write_json(metadata, PROJECT_ROOT / "data/interim/liquid100_metadata.json")
    config = public_config(load_config(CONFIG))
    if config["tickers"] and config["tickers"] != selected:
        raise ValueError(
            "Existing experiment definition names another universe; it will not be replaced"
        )
    config["tickers"] = config["observation_tickers"] = selected
    config["universe_selection_sha256"] = metadata["selection_sha256"]
    dump_yaml(config, CONFIG)
    folder = freeze_inputs()
    data_report()
    return folder


def data_report() -> dict:
    config = load_extension_config()
    metadata = json.loads((PROJECT_ROOT / config["data"]["snapshot"]).read_text(encoding="utf-8"))
    panel = pd.read_parquet(PROJECT_ROOT / metadata["layers"]["processed"]["parquet"])
    audit = pd.read_parquet(PROJECT_ROOT / config["data"]["selection_audit"])
    selected = audit.loc[audit.selected].sort_values("rank").reset_index(drop=True)
    DATA_REPORT.mkdir(parents=True, exist_ok=True)
    coverage = []
    for name in ("train", "validation", "test"):
        block = panel.loc[
            panel.date.between(config["split"][f"{name}_start"], config["split"][f"{name}_end"])
        ]
        for ticker, group in block.groupby("ticker"):
            coverage.append(
                {
                    "split": name,
                    "ticker": ticker,
                    "rows": len(group),
                    "first_date": group.date.min(),
                    "last_date": group.date.max(),
                    "missing_features": int(group[list(FEATURE_COLUMNS)].isna().sum().sum()),
                    "tradable_days": int(group.tradable.sum()),
                }
            )
    coverage = pd.DataFrame(coverage)
    save_table(coverage, DATA_REPORT / "coverage")
    save_table(audit, DATA_REPORT / "selection_audit")
    save_table(selected, DATA_REPORT / "selected_stocks")
    write_json(metadata, DATA_REPORT / "source_metadata.json")
    figures = []
    sectors = selected.groupby("sector").size().sort_values()
    save_table(sectors.rename("stocks").reset_index(), DATA_REPORT / "sector_counts")
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    ax.barh(sectors.index, sectors.values, color=PALETTE[0], height=0.65)
    ax.set(xlabel="Selected stocks", title="Training-selected 100-stock universe")
    ax.grid(axis="y", visible=False)
    path = DATA_REPORT / "sector_counts.png"
    fig.savefig(path, dpi=EXPORT_DPI)
    plt.close(fig)
    figures.append(path)
    write_json(
        {
            "dpi": EXPORT_DPI,
            "generator_sha256": sha256_file(Path(__file__)),
            "selection_uses_test_returns": False,
        },
        DATA_REPORT / "figure_parameters.json",
    )
    inventory(DATA_REPORT)
    return {
        "selected": selected,
        "coverage": coverage,
        "audit": audit,
        "figures": figures,
        "metadata": metadata,
        "out": DATA_REPORT,
    }


def _line_figure(frame: pd.DataFrame, destination: Path, stem: str, title: str) -> Path:
    save_table(frame, destination / stem)
    fig, ax = plt.subplots(figsize=(10.5, 4.4), layout="constrained")
    styles = {
        "full_ppo": (PALETTE[0], "Learned policy"),
        "equal_weight_hold": (PALETTE[1], "Equal-weight hold"),
        "freeze_1": (PALETTE[2], "Initial-allocation hold"),
    }
    for name in frame.columns[1:]:
        color, label = styles.get(name, (PALETTE[3], name))
        ax.plot(
            pd.to_datetime(frame.iloc[:, 0]), frame[name], color=color, label=label, linewidth=1.05
        )
    ax.set(xlabel="Execution date", ylabel="Wealth / initial wealth", title=title)
    ax.legend(loc="upper left", fontsize=9)
    path = destination / f"{stem}.png"
    fig.savefig(path, dpi=EXPORT_DPI)
    plt.close(fig)
    write_json(
        {
            "dpi": EXPORT_DPI,
            "linewidth": 1.05,
            "markers": False,
            "generator_sha256": sha256_file(Path(__file__)),
        },
        destination / f"{stem}.json",
    )
    return path


def report() -> dict:
    config = load_extension_config()
    frozen = PROJECT_ROOT / config["data"]["frozen_inputs"]
    verify_inventory(frozen)
    data_hash = json.loads((frozen / "metadata.json").read_text(encoding="utf-8"))["data_hash"]
    REPORT.mkdir(parents=True, exist_ok=True)
    rows, summaries, effects, figures = [], [], [], []
    for seed in SEEDS:
        run = run_directory(config, seed)
        complete = _complete_run(run, config, data_hash)
        state = "complete" if complete else ("incomplete" if run.exists() else "not_started")
        for name in ("training", "evaluation"):
            path = run / f"{name}_status.json"
            if not complete and path.exists():
                state = name + "_" + json.loads(path.read_text(encoding="utf-8"))["status"]
        rows.append(
            {
                "condition": "multi_liquid100",
                "seed": seed,
                "seed_id": LABELS[seed],
                "status": state,
                "run": _relative(run),
            }
        )
        if not complete:
            continue
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["controls"]
        for name, value in metrics["metrics"].items():
            summaries.append({"seed": seed, "seed_id": LABELS[seed], "strategy": name, **value})
        for name, value in metrics["effects"].items():
            effects.append(
                {
                    "seed": seed,
                    "seed_id": LABELS[seed],
                    "effect": name,
                    "net_log_wealth_difference": value,
                }
            )
        j = metrics["metrics"]
        allocation = j["freeze_1"]["net_log_wealth"] - j["equal_weight_hold"]["net_log_wealth"]
        dynamic = j["full_ppo"]["net_log_wealth"] - j["freeze_1"]["net_log_wealth"]
        if not np.isclose(
            allocation + dynamic,
            j["full_ppo"]["net_log_wealth"] - j["equal_weight_hold"]["net_log_wealth"],
        ):
            raise AssertionError("Initial-allocation plus subsequent-trading identity failed")
        for name, value in (
            ("initial_allocation_A", allocation),
            ("subsequent_trading_D", dynamic),
            ("D_minus_A", dynamic - allocation),
        ):
            effects.append(
                {
                    "seed": seed,
                    "seed_id": LABELS[seed],
                    "effect": name,
                    "net_log_wealth_difference": value,
                }
            )
        destination = REPORT / "figures" / f"seed_{LABELS[seed]}"
        destination.mkdir(parents=True, exist_ok=True)
        paths = []
        for name in ("full_ppo", "equal_weight_hold", "freeze_1"):
            path = pd.read_parquet(run / "controls/main" / f"{name}.parquet")
            paths.append(
                path.set_index("execution_date").wealth_after.rename(name)
                / path.wealth_before.iloc[0]
            )
        wealth = pd.concat(paths, axis=1).reset_index()
        figures.append(
            _line_figure(
                wealth, destination, "wealth", f"100-stock portfolio | Seed {LABELS[seed]}"
            )
        )
        targets = pd.read_parquet(run / "target_weights.parquet")
        initial = (
            targets.iloc[0][["CASH", *config["tickers"]]].astype(float).sort_values(ascending=False)
        )
        save_table(
            initial.rename("weight").rename_axis("ticker").reset_index(),
            destination / "initial_weights_all",
        )
        top = initial.head(15)
        top = pd.concat([top, pd.Series({"Other holdings": float(initial.iloc[15:].sum())})])
        save_table(
            top.rename("weight").rename_axis("ticker").reset_index(),
            destination / "initial_weights_display",
        )
        fig, ax = plt.subplots(figsize=(8, 5.8), layout="constrained")
        ax.barh(top.index[::-1], top.values[::-1], color=PALETTE[0], height=0.65)
        ax.set(
            xlabel="Initial portfolio weight",
            title="Largest initial holdings; remainder aggregated",
        )
        ax.grid(axis="y", visible=False)
        from matplotlib.ticker import PercentFormatter

        ax.xaxis.set_major_formatter(PercentFormatter(1))
        path = destination / "initial_weights.png"
        fig.savefig(path, dpi=EXPORT_DPI)
        plt.close(fig)
        figures.append(path)
        monthly = (
            targets.assign(decision_date=pd.to_datetime(targets.decision_date))
            .set_index("decision_date")[config["tickers"]]
            .resample("MS")
            .mean()
            .loc[:"2025-12-31"]
        )
        save_table(monthly.reset_index(), destination / "monthly_weights_all")
        top_names = monthly.mean().nlargest(20).index
        save_table(monthly[top_names].reset_index(), destination / "monthly_weights_display")
        fig, ax = plt.subplots(figsize=(11, 6.4), layout="constrained")
        heat = ax.imshow(
            monthly[top_names].to_numpy().T,
            aspect="auto",
            cmap="Blues",
            vmin=0,
            vmax=max(float(monthly[top_names].max().max()), 0.01),
            interpolation="nearest",
        )
        ax.set_yticks(range(len(top_names)), top_names, fontsize=8)
        positions = np.arange(0, len(monthly), 12)
        ax.set_xticks(positions, monthly.index[positions].strftime("%Y"))
        ax.set(title="Monthly weights: 20 largest mean holdings", xlabel="Decision month")
        ax.grid(False)
        fig.colorbar(heat, ax=ax, label="Portfolio weight", shrink=0.8)
        path = destination / "monthly_weights.png"
        fig.savefig(path, dpi=EXPORT_DPI)
        plt.close(fig)
        figures.append(path)
        write_json(
            {
                "dpi": EXPORT_DPI,
                "monthly_display_selection": (
                    "top 20 by test mean weight for descriptive display only"
                ),
                "full_100_weights_saved": True,
                "generator_sha256": sha256_file(Path(__file__)),
            },
            destination / "allocation_figures.json",
        )
    status = pd.DataFrame(rows)
    summary = (
        pd.DataFrame(summaries)
        if summaries
        else pd.DataFrame(columns=["seed", "seed_id", "strategy", "net_log_wealth"])
    )
    effect_frame = (
        pd.DataFrame(effects)
        if effects
        else pd.DataFrame(columns=["seed", "seed_id", "effect", "net_log_wealth_difference"])
    )
    figures.insert(0, status_figure(status.assign(seed=status.seed_id), REPORT / "figures"))
    for name, frame in (("status", status), ("summary", summary), ("effects", effect_frame)):
        save_table(frame, REPORT / name)
    write_json(public_config(config), REPORT / "configuration.json")
    write_json(
        {
            "experiment": config["experiment"],
            "expected_runs": 3,
            "complete_runs": int(status.status.eq("complete").sum()),
            "formal_inference": False,
            "comparison_scope": "within the fixed training-selected 100-stock universe",
        },
        REPORT / "report_manifest.json",
    )
    inventory(REPORT)
    return {
        "status": status,
        "summary": summary,
        "effects": effect_frame,
        "figures": figures,
        "out": REPORT,
    }


def _check_training_capacity(config: dict) -> None:
    """Check live workers without starting, stopping or modifying any of them."""
    workers = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info["cmdline"] or []
            if all(part in command for part in ("dtasrl.cli", "experiment", "run")):
                workers.append(process.info["pid"])
                # Re-entering the same notebook must not damage an active run.
                if "--config" in command:
                    value = command[command.index("--config") + 1]
                    candidate = Path(value)
                    if not candidate.is_absolute():
                        candidate = Path(process.cwd()) / candidate
                    if candidate.resolve() == CONFIG.resolve():
                        raise RuntimeError(
                            f"Notebook 05 already has an active worker (PID {process.pid}). "
                            "Do not start a second copy."
                        )
        except (psutil.AccessDenied, psutil.NoSuchProcess, IndexError):
            continue
    if len(workers) >= 2:
        raise RuntimeError(
            f"Two or more formal workers are active (PIDs {workers}). "
            "Wait for capacity before manually starting Notebook 05."
        )
    reserve = float(config.get("training", {}).get("min_available_memory_gb", 8))
    available = psutil.virtual_memory().available / 2**30
    if available < reserve:
        raise RuntimeError(
            f"Available memory is {available:.2f} GiB; at least {reserve:.2f} GiB is required. "
            "No new worker was launched. Close unneeded applications and retry."
        )


def run_seeds() -> pd.DataFrame:
    """Notebook-only explicit invocation; sequential workers and visible progress."""
    from IPython.display import HTML, display

    config = load_extension_config()
    frozen = PROJECT_ROOT / config["data"]["frozen_inputs"]
    verify_inventory(frozen)
    data_hash = json.loads((frozen / "metadata.json").read_text(encoding="utf-8"))["data_hash"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in SEEDS:
        run = run_directory(config, seed)
        if _complete_run(run, config, data_hash):
            print(f"SKIP complete | 100 stocks | seed {LABELS[seed]}")
            records.append(
                {"seed_id": LABELS[seed], "action": "skipped_complete", "run": _relative(run)}
            )
            continue
        _check_training_capacity(config)
        command = [
            sys.executable,
            "-u",
            "-m",
            "dtasrl.cli",
            "experiment",
            "run",
            "--config",
            str(CONFIG),
            "--seed",
            str(seed),
        ]
        action = "started"
        if run.exists():
            pointer = run / "latest_checkpoint.json"
            if not pointer.is_file() or not (run / "training_identity.json").is_file():
                raise RuntimeError(
                    f"Incomplete run without safe checkpoint; inspect before removal: {run}"
                )
            checkpoint = run / json.loads(pointer.read_text(encoding="utf-8"))["path"]
            if not all(
                (checkpoint / name).is_file()
                for name in ("model.zip", "state.pkl", "checkpoint.json")
            ):
                raise RuntimeError(f"Incomplete checkpoint: {checkpoint}")
            command.append("--resume")
            action = "resumed"
        print(f"{action.upper()} | 100 stocks | seed {LABELS[seed]}", flush=True)
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        handle = display(HTML("<pre>Starting training...</pre>"), display_id=True)
        recent, frame = deque(maxlen=80), []
        ansi = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        try:
            with (OUTPUT / f"notebook_seed_{LABELS[seed]}.log").open(
                "a", encoding="utf-8"
            ) as transcript:
                transcript.write(f"\n{utc_now()} {action.upper()} PID={process.pid}\n")
                while character := process.stdout.read(1):
                    transcript.write(character)
                    if character in ("\r", "\n"):
                        transcript.flush()
                        rendered = ansi.sub("", "".join(frame)).strip()
                        frame.clear()
                        if rendered:
                            recent.append(rendered)
                            handle.update(
                                HTML(
                                    "<pre style='white-space:pre-wrap'>"
                                    f"{html.escape(rendered)}</pre>"
                                )
                            )
                    else:
                        frame.append(character)
                if frame:
                    recent.append(ansi.sub("", "".join(frame)).strip())
            return_code = process.wait()
        except KeyboardInterrupt:
            # The user interrupted this notebook. Do not leave an invisible
            # child training behind or silently start the next seed.
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
        if return_code:
            print("\n".join(recent))
            raise subprocess.CalledProcessError(return_code, command)
        if not _complete_run(run, config, data_hash):
            raise RuntimeError(f"Worker exited without a complete validated run: {run}")
        handle.update(HTML(f"<pre>COMPLETED | 100 stocks | seed {LABELS[seed]}</pre>"))
        records.append({"seed_id": LABELS[seed], "action": action, "run": _relative(run)})
    return pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "report"))
    args = parser.parse_args()
    print(
        prepare_data() if args.command == "prepare" else report()["status"].to_string(index=False)
    )
