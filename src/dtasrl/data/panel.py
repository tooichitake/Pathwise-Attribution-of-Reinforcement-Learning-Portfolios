"""Fixed-universe market snapshots and training-only portfolio input preparation.

Prices are total-return-adjusted units: dividends are already embedded and must
not be credited a second time. Fractional adjusted units are not broker shares.
The feature definitions and execution convention are shared across the study;
the single-stock diagnostics and 3-stock portfolios also share market inputs.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dtasrl.config import PROJECT_ROOT, public_config
from dtasrl.data.features import (
    FEATURE_COLUMNS,
    apply_scaler,
    engineer_features,
    fit_training_scaler,
)
from dtasrl.data.yahoo import download_yahoo
from dtasrl.provenance import git_commit, git_is_dirty, sha256_file, write_json

DATASET_NAME = "current_djia_2026_08_31_backfilled"
OBSERVATION_TICKERS = ("MSFT", "JPM", "JNJ")
DEFAULT_SPLIT = {
    "train_start": "2009-01-01",
    "train_end": "2018-12-31",
    "validation_start": "2019-01-01",
    "validation_end": "2019-12-31",
    "test_start": "2020-01-01",
    "test_end": "2026-08-31",
}
PANEL_COLUMNS = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    *FEATURE_COLUMNS,
    "tradable",
)


def _path(value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _write_layer(frame: pd.DataFrame, directory: Path, stem: str) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    csv_path, parquet_path = directory / f"{stem}.csv", directory / f"{stem}.parquet"
    csv_partial = csv_path.with_name(csv_path.name + ".partial")
    parquet_partial = parquet_path.with_name(parquet_path.name + ".partial")
    frame = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
    # CSV has no representation for pandas' optional column-axis name (Yahoo
    # attaches "Price"). Column labels and every data value remain unchanged.
    frame.columns.name = None
    frame.to_csv(csv_partial, index=False, encoding="utf-8", date_format="%Y-%m-%d")
    frame.to_parquet(parquet_partial, index=False)
    # A parsed CSV is required to preserve numerical rows, not binary encodings.
    recovered = pd.read_csv(csv_partial, parse_dates=["date"], float_precision="round_trip")
    pd.testing.assert_frame_equal(frame, recovered, check_dtype=False, check_exact=True)
    os.replace(csv_partial, csv_path)
    os.replace(parquet_partial, parquet_path)
    return {
        "csv": _relative(csv_path),
        "parquet": _relative(parquet_path),
        "csv_sha256": sha256_file(csv_path),
        "parquet_sha256": sha256_file(parquet_path),
        "rows": len(frame),
        "field_order": list(frame.columns),
        "date_min": frame["date"].min().date().isoformat(),
        "date_max": frame["date"].max().date().isoformat(),
        "csv_parquet_row_equality": True,
    }


def build_market_snapshot(config: dict[str, Any]) -> Path:
    """Replace the one current raw/interim/processed copy after validation."""
    name = str(config.get("name", DATASET_NAME))
    tickers = sorted(config["tickers"])
    if not tickers or len(tickers) != len(set(tickers)):
        raise ValueError("tickers must be a nonempty unique list")
    raw, source = download_yahoo(tickers, str(config["start"]), str(config["end"]))
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw = raw.sort_values(["date", "ticker"]).reset_index(drop=True)
    snapshot_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    roots = {layer: PROJECT_ROOT / "data" / layer for layer in ("raw", "interim", "processed")}
    raw_layer = _write_layer(raw, roots["raw"], "market_data")
    metadata: dict[str, Any] = {
        **source,
        "name": name,
        "snapshot_id": snapshot_id,
        "config": public_config(config),
        "query": source["request"],
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "packages": {
            key: importlib.metadata.version(key)
            for key in ("yfinance", "numpy", "pandas", "pyarrow")
        },
        "universe_condition": config.get("universe_condition", name),
        "survivorship_bias": (
            "Present: constituents fixed at 2026-08-31 and backfilled; "
            "not a historical point-in-time index universe."
        ),
        "tickers_requested": tickers,
        "immutable": False,
        "storage_policy": "single_current_copy_no_internal_backups",
        "adjustment_vintage_policy": "full_history_single_download_vintage_no_splicing",
        "price_semantics": "total_return_adjusted_units_dividends_not_credited_again",
        "layers": {"raw": raw_layer},
        "status": "building",
    }
    write_json(metadata, roots["raw"] / "metadata.json")
    try:
        if source.get("failed_tickers"):
            raise ValueError(f"failed tickers: {source['failed_tickers']}")
        if raw.duplicated(["date", "ticker"]).any():
            raise ValueError("duplicate date/ticker rows")
        if set(raw["ticker"]) != set(tickers):
            raise ValueError("returned ticker set does not match requested ticker set")
        prices = raw.pivot(index="date", columns="ticker", values="Adj Close")
        valid = np.isfinite(prices) & prices.gt(0)
        common_dates = prices.index[valid.all(axis=1)]
        if common_dates.empty:
            raise ValueError("no date has valid adjusted close prices for every ticker")
        for key, value in (
            ("expected_common_start", common_dates[0]),
            ("expected_common_end", common_dates[-1]),
        ):
            if config.get(key) and pd.Timestamp(config[key]) != value:
                raise ValueError(f"{key}: expected {config[key]}, obtained {value.date()}")
        # Features use each stock's pre-common history; incomplete IPO warmup
        # remains missing. No backwards fill or information from later dates.
        interim = engineer_features(raw)
        ratio = interim["Adj Close"] / interim["Close"]
        for field in ("Open", "High", "Low", "Close"):
            interim[f"adjusted_{field.lower()}"] = interim[field] * ratio
        panel = interim.loc[interim["date"].isin(common_dates)].copy()
        rename = {f"adjusted_{field}": field for field in ("open", "high", "low", "close")}
        panel = panel.rename(columns={**rename, "Volume": "volume"})
        quoted = panel[["open", "high", "low", "close"]].to_numpy(dtype=float)
        if not np.isfinite(quoted).all() or (quoted <= 0).any():
            raise ValueError("common-date adjusted OHLC has missing or nonpositive values")
        if panel["volume"].isna().any() or (panel["volume"] < 0).any():
            raise ValueError("common-date volume is missing or negative")
        panel["tradable"] = panel["volume"].gt(0)
        panel = panel[list(PANEL_COLUMNS)].reset_index(drop=True)
        if not panel.groupby("date")["ticker"].nunique().eq(len(tickers)).all():
            raise ValueError("common-date panel is incomplete")
        layers = {
            "raw": raw_layer,
            "interim": _write_layer(interim, roots["interim"], "features_unscaled"),
            "processed": _write_layer(panel, roots["processed"], "panel"),
        }
        union_after_start = prices.loc[common_dates[0] : common_dates[-1]].index
        excluded = union_after_start.difference(common_dates)
        metadata.update(
            {
                "status": "complete",
                "layers": layers,
                "date_min": common_dates[0].date().isoformat(),
                "date_max": common_dates[-1].date().isoformat(),
                "common_trading_days": len(common_dates),
                "excluded_incomplete_dates": [d.date().isoformat() for d in excluded],
                "tickers_returned": tickers,
                "field_order": list(PANEL_COLUMNS),
                "feature_order": list(FEATURE_COLUMNS),
                "feature_definition": {
                    "rsi_14d": (
                        "14-return simple moving-average RSI; "
                        "up=100, down=0, flat=50; incomplete=NaN"
                    )
                },
                "feature_missing_rows": {f: int(panel[f].isna().sum()) for f in FEATURE_COLUMNS},
                "corporate_action_counts": {
                    t: {
                        "dividend_rows": int((g["Dividends"].fillna(0) != 0).sum()),
                        "split_rows": int((g["Stock Splits"].fillna(0) != 0).sum()),
                    }
                    for t, g in raw.groupby("ticker")
                },
                "raw_sha256": layers["raw"]["parquet_sha256"],
                "features_sha256": layers["processed"]["parquet_sha256"],
            }
        )
        write_json(metadata, roots["processed"] / "metadata.json")
        write_json(metadata, roots["interim"] / "metadata.json")
        write_json(metadata, roots["raw"] / "metadata.json")
        (roots["raw"] / "last_failure.json").unlink(missing_ok=True)
    except Exception as exc:
        write_json(
            {"status": "failed", "error": str(exc), "raw": raw_layer},
            roots["raw"] / "last_failure.json",
        )
        raise
    return roots["processed"]


def _read_processed(snapshot: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _path(snapshot)
    manifest_path = path / "metadata.json" if path.is_dir() else path
    with manifest_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if "metadata" in metadata:
        with _path(metadata["metadata"]).open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    if metadata.get("status") != "complete":
        raise ValueError("processed snapshot is not complete")
    layer = metadata["layers"]["processed"]
    data_path = _path(layer["parquet"])
    if metadata["features_sha256"] != layer["parquet_sha256"]:
        raise RuntimeError("processed manifest contains inconsistent data hashes")
    if sha256_file(data_path) != layer["parquet_sha256"]:
        raise RuntimeError(f"processed data hash mismatch: {data_path}")
    frame = pd.read_parquet(data_path)
    if list(frame.columns) != list(PANEL_COLUMNS):
        raise ValueError("processed field order does not match the market panel schema")
    return frame, metadata


def fit_experiment_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Read processed data, fit per-ticker training scalers, and align all arrays.

    The leading row of every split is the previous close. Its observation is
    used for the first in-split next-open execution, but is not a test return.
    Features are feature-major over observation_tickers, independently of the
    subset that the policy may hold. No network access or raw processing occurs.
    """
    snapshot = config.get("data", {}).get("snapshot", "data/processed/metadata.json")
    frame, metadata = _read_processed(snapshot)
    observed = list(config.get("observation_tickers", OBSERVATION_TICKERS))
    tickers = list(config.get("tickers", observed))
    if not observed or len(set(observed)) != len(observed):
        raise ValueError("observation_tickers must be nonempty and unique")
    if not tickers or len(set(tickers)) != len(tickers) or not set(tickers) <= set(observed):
        raise ValueError("trade tickers must be a unique nonempty subset of observation_tickers")
    if not set(observed) <= set(metadata["tickers_returned"]):
        raise ValueError("some observation tickers are absent from the fixed universe")
    split = {**DEFAULT_SPLIT, **config.get("split", {})}
    boundaries = [pd.Timestamp(split[k]) for k in DEFAULT_SPLIT]
    if not all(a < b for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)):
        raise ValueError("training, validation and test boundaries must be strictly ordered")
    frame = frame.loc[frame["ticker"].isin(observed)].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    scaled, scalers = [], {}
    for ticker in observed:
        part = frame.loc[frame["ticker"] == ticker].copy()
        scaler = fit_training_scaler(part, split["train_end"], split["train_start"])
        scalers[ticker] = {
            "mean": scaler.mean,
            "scale": scaler.scale,
            "train_start": split["train_start"],
            "train_end": split["train_end"],
        }
        scaled.append(apply_scaler(part, scaler))
    scaled_frame = pd.concat(scaled, ignore_index=True)
    columns = pd.MultiIndex.from_product([FEATURE_COLUMNS, observed])
    features = scaled_frame.pivot(index="date", columns="ticker", values=list(FEATURE_COLUMNS))
    features = features.reindex(columns=columns).sort_index()
    closes = frame.pivot(index="date", columns="ticker", values="close").reindex(columns=tickers)
    opens = frame.pivot(index="date", columns="ticker", values="open").reindex(columns=tickers)
    tradable = frame.pivot(index="date", columns="ticker", values="tradable").reindex(
        columns=tickers
    )
    usable = np.isfinite(features).all(axis=1) & np.isfinite(closes).all(axis=1)
    usable &= np.isfinite(opens).all(axis=1)
    dates = pd.DatetimeIndex(features.index[usable])
    output: dict[str, Any] = {}
    for name in ("train", "validation", "test"):
        start, end = pd.Timestamp(split[f"{name}_start"]), pd.Timestamp(split[f"{name}_end"])
        positions = np.flatnonzero((dates >= start) & (dates <= end))
        expected = features.index[(features.index >= start) & (features.index <= end)]
        if len(positions) < 2 or len(positions) != len(expected) or positions[0] == 0:
            raise ValueError(f"{name} has incomplete features or no previous-close boundary")
        selected = dates[positions[0] - 1 : positions[-1] + 1]
        output[name] = {
            "prices": closes.loc[selected].to_numpy(dtype=float),
            "opens": opens.loc[selected].to_numpy(dtype=float),
            "features": features.loc[selected].to_numpy(dtype=float),
            "tradable": tradable.loc[selected].to_numpy(dtype=bool),
            "forced_liquidation": np.zeros((len(selected), len(tickers)), dtype=bool),
            "dates": selected.to_numpy(),
            "tickers": tickers,
            "observation_tickers": observed,
            "feature_names": list(FEATURE_COLUMNS),
            "feature_columns": [f"{ticker}:{feature}" for feature, ticker in columns],
            "period_start": str(start.date()),
            "period_end": str(end.date()),
        }
    output["scaler"] = scalers
    output["provenance"] = {
        "snapshot_id": metadata["snapshot_id"],
        "dataset_name": metadata["name"],
        "processed_snapshot": str(snapshot),
        "split": split,
        "universe_condition": metadata["universe_condition"],
        "case_selection": (
            "MSFT/JPM/JNJ predeclared static cases, not historical universe inference"
        ),
    }
    return output, metadata["features_sha256"]


def prepare_experiment_data(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from dtasrl.data.inputs import load_real_inputs

    return load_real_inputs(config)
