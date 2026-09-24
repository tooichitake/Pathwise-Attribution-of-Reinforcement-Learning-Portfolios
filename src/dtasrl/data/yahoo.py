from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import yfinance as yf

FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits")


def _normalise_download(frame: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if frame.empty:
        raise RuntimeError("yfinance returned no rows")
    pieces: list[pd.DataFrame] = []
    if isinstance(frame.columns, pd.MultiIndex):
        first_level = set(map(str, frame.columns.get_level_values(0)))
        ticker_first = bool(first_level.intersection(tickers))
        for ticker in tickers:
            try:
                part = frame[ticker] if ticker_first else frame.xs(ticker, axis=1, level=1)
            except KeyError:
                continue
            part = part.copy()
            part["ticker"] = ticker
            pieces.append(part.reset_index())
    else:
        part = frame.copy()
        part["ticker"] = tickers[0]
        pieces.append(part.reset_index())
    if not pieces:
        raise RuntimeError("none of the requested tickers were returned")
    result = pd.concat(pieces, ignore_index=True)
    date_column = "Date" if "Date" in result.columns else result.columns[0]
    result = result.rename(columns={date_column: "date"})
    result["date"] = pd.to_datetime(result["date"], utc=True).dt.tz_convert(None)
    for field in FIELDS:
        if field not in result:
            result[field] = 0.0 if field in {"Dividends", "Stock Splits"} else float("nan")
    return (
        result[["date", "ticker", *FIELDS]].sort_values(["date", "ticker"]).reset_index(drop=True)
    )


def download_yahoo(
    tickers: list[str], start: str, end: str, batch_size: int = 5
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not tickers:
        raise ValueError("at least one ticker is required")
    request = {
        "tickers": sorted(set(map(str.upper, tickers))),
        "start": start,
        "end": end,
        "auto_adjust": False,
        "actions": True,
        "group_by": "ticker",
        "repair": False,
        "keepna": True,
    }
    pieces: list[pd.DataFrame] = []
    failures: list[str] = []
    retry_log: list[dict[str, Any]] = []
    delays = (0, 5, 15, 30)
    for offset in range(0, len(request["tickers"]), batch_size):
        pending = request["tickers"][offset : offset + batch_size]
        recovered: dict[str, pd.DataFrame] = {}
        for attempt, delay in enumerate(delays, start=1):
            if not pending:
                break
            if delay:
                time.sleep(delay)
            frame = yf.download(
                pending,
                start=start,
                end=end,
                auto_adjust=False,
                actions=True,
                group_by="ticker",
                repair=False,
                keepna=True,
                threads=False,
                progress=False,
            )
            if not frame.empty:
                normalised = _normalise_download(frame, pending)
                for ticker, ticker_frame in normalised.groupby("ticker"):
                    if ticker_frame["Adj Close"].notna().any():
                        recovered[str(ticker)] = ticker_frame
            missing = sorted(set(pending).difference(recovered))
            retry_log.append(
                {"attempt": attempt, "requested": pending, "missing_after_attempt": missing}
            )
            pending = missing
        pieces.extend(recovered.values())
        failures.extend(pending)
    if not pieces:
        raise RuntimeError("yfinance returned no usable rows after batched retries")
    raw = pd.concat(pieces, ignore_index=True)
    metadata = {
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "source": "yfinance",
        "yfinance_version": yf.__version__,
        "request": request,
        "failed_tickers": sorted(set(failures)),
        "retry_log": retry_log,
    }
    return raw.sort_values(["date", "ticker"]).reset_index(drop=True), metadata
