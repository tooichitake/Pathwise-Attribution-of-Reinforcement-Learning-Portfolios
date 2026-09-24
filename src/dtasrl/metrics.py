from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def performance_metrics(
    wealth: Iterable[float],
    target_weights: np.ndarray | None = None,
    executed_turnover: Iterable[float] | None = None,
) -> dict[str, float]:
    values = np.asarray(list(wealth), dtype=float)
    if (
        values.ndim != 1
        or values.size < 2
        or np.any(~np.isfinite(values))
        or np.any(values <= 0)
    ):
        raise ValueError("wealth must contain at least two finite positive observations")
    log_returns = np.diff(np.log(values))
    simple_returns = np.diff(values) / values[:-1]
    running_max = np.maximum.accumulate(values)
    drawdowns = values / running_max - 1.0
    std = float(np.std(log_returns, ddof=1)) if log_returns.size > 1 else 0.0
    metrics = {
        "net_log_wealth": float(np.log(values[-1] / values[0])),
        "cumulative_return": float(values[-1] / values[0] - 1.0),
        "annualised_sharpe": float(math.sqrt(252) * np.mean(log_returns) / std) if std > 0 else 0.0,
        "max_drawdown": float(np.min(drawdowns)),
        "mean_daily_return": float(np.mean(simple_returns)),
    }
    turnover: np.ndarray | None = None
    if executed_turnover is not None:
        turnover = np.asarray(list(executed_turnover), dtype=float)
        metrics["mean_turnover"] = float(np.mean(turnover)) if turnover.size else 0.0
        metrics["active_trading_days"] = float(np.mean(turnover > 1e-10)) if turnover.size else 0.0
    if target_weights is not None:
        weights = np.asarray(target_weights, dtype=float)
        risky = weights[:, 1:]
        exposure = risky.sum(axis=1)
        metrics["exposure_variance"] = float(np.var(exposure))
        metrics["mean_risky_exposure"] = float(np.mean(exposure))
        metrics["mean_concentration_hhi"] = float(np.mean(np.sum(risky**2, axis=1)))
        # Softmax policies assign every asset a strictly positive weight, so counting
        # consecutive non-zero weights would mechanically report the full sample as
        # the holding duration.  Instead, measure portfolio holding spells between
        # executed rebalances.  The initial allocation begins the first spell.
        if turnover is not None and turnover.size:
            if turnover.size != weights.shape[0]:
                raise ValueError("turnover and target_weights must have the same length")
            if np.any(exposure > 1e-6):
                rebalance_days = np.flatnonzero(turnover > 1e-10)
                starts = np.unique(np.r_[0, rebalance_days])
                durations = np.diff(np.r_[starts, turnover.size])
                metrics["mean_holding_duration"] = float(np.mean(durations))
            else:
                metrics["mean_holding_duration"] = 0.0
    return metrics


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted
