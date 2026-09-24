from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "return_1d",
    "momentum_21d",
    "momentum_63d",
    "volatility_21d",
    "rsi_14d",
    "macd",
    "macd_signal",
    "volume_z_20d",
)


@dataclass(frozen=True)
class TrainingScaler:
    mean: dict[str, float]
    scale: dict[str, float]


def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    result = 100 - 100 / (1 + relative_strength)
    # A completed all-up window is overbought, not neutral. Keep the initial
    # incomplete window missing so callers cannot mistake it for an observation.
    result = result.mask((gain > 0) & (loss == 0), 100.0)
    result = result.mask((gain == 0) & (loss == 0), 50.0)
    return result


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "Adj Close", "Volume"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    output: list[pd.DataFrame] = []
    for ticker, group in raw.sort_values("date").groupby("ticker", sort=True):
        item = group.copy()
        price = item["Adj Close"].astype(float)
        returns = price.pct_change(fill_method=None)
        item["return_1d"] = returns
        item["momentum_21d"] = price.pct_change(21, fill_method=None)
        item["momentum_63d"] = price.pct_change(63, fill_method=None)
        item["volatility_21d"] = returns.rolling(21, min_periods=21).std()
        item["rsi_14d"] = _rsi(price)
        ema12 = price.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = price.ewm(span=26, adjust=False, min_periods=26).mean()
        item["macd"] = ema12 - ema26
        item["macd_signal"] = item["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
        log_volume = np.log1p(item["Volume"].astype(float))
        rolling_mean = log_volume.rolling(20, min_periods=20).mean()
        rolling_std = log_volume.rolling(20, min_periods=20).std()
        item["volume_z_20d"] = (log_volume - rolling_mean) / rolling_std.replace(0, np.nan)
        item["ticker"] = ticker
        output.append(item)
    return (
        pd.concat(output, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    )


def fit_training_scaler(
    features: pd.DataFrame, train_end: str, train_start: str | None = None
) -> TrainingScaler:
    dates = pd.to_datetime(features["date"])
    mask = dates <= pd.Timestamp(train_end)
    if train_start is not None:
        if pd.Timestamp(train_start) > pd.Timestamp(train_end):
            raise ValueError("train_start must not exceed train_end")
        mask &= dates >= pd.Timestamp(train_start)
    training = features.loc[mask]
    if training.empty:
        raise ValueError("training slice is empty")
    mean = training[list(FEATURE_COLUMNS)].mean(skipna=True)
    if not np.isfinite(mean.to_numpy(dtype=float)).all():
        raise ValueError("training features contain no finite observations in some columns")
    scale = training[list(FEATURE_COLUMNS)].std(skipna=True).replace(0, 1.0).fillna(1.0)
    return TrainingScaler(mean.to_dict(), scale.to_dict())


def apply_scaler(features: pd.DataFrame, scaler: TrainingScaler) -> pd.DataFrame:
    output = features.copy()
    for column in FEATURE_COLUMNS:
        output[column] = (output[column] - scaler.mean[column]) / scaler.scale[column]
    return output
