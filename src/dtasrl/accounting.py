"""Self-financing execution shared by environments and counterfactual paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExecutionResult:
    net_wealth_fraction: float
    target_weights: np.ndarray
    trade_value_fractions: np.ndarray
    turnover: float
    cost_fraction: float


def rebalance_portfolio(
    pre_trade_weights: np.ndarray,
    desired_weights: np.ndarray,
    cost_rate: float,
    tradable: np.ndarray | None = None,
    forced_liquidation: np.ndarray | None = None,
) -> ExecutionResult:
    """Execute post-cost target weights, with cash at index zero.

    Signed trade values and fees are fractions of wealth immediately before
    execution. With no constraints the retained wealth ``z`` solves
    ``z = 1 - cost_rate * sum(abs(z * desired[1:] - pre[1:]))``.
    Halted assets retain their pre-trade monetary value (hence their units at
    the unchanged execution price). The remaining post-cost wealth follows
    the desired relative weights of cash and tradable assets. A forced
    liquidation is an explicit executable settlement, overriding a halt.
    """
    pre = np.asarray(pre_trade_weights, dtype=np.float64)
    desired = np.asarray(desired_weights, dtype=np.float64)
    if pre.ndim != 1 or pre.size < 2 or desired.shape != pre.shape:
        raise ValueError("weights must have equal one-dimensional cash-plus-assets shapes")
    if (
        not np.all(np.isfinite(pre))
        or not np.all(np.isfinite(desired))
        or np.any(pre < -1e-12)
        or np.any(desired < -1e-12)
        or abs(float(pre.sum()) - 1.0) > 1e-9
        or abs(float(desired.sum()) - 1.0) > 1e-9
    ):
        raise ValueError("weights must be finite, nonnegative, and sum to one")
    rate = float(cost_rate)
    if not np.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError("cost_rate must be finite and in [0, 1)")
    n_assets = pre.size - 1
    tradable_array = (
        np.ones(n_assets, dtype=bool) if tradable is None else np.asarray(tradable, dtype=bool)
    )
    forced = (
        np.zeros(n_assets, dtype=bool)
        if forced_liquidation is None
        else np.asarray(forced_liquidation, dtype=bool)
    )
    if tradable_array.shape != (n_assets,) or forced.shape != (n_assets,):
        raise ValueError("execution flags must have one entry per risky asset")
    desired = np.maximum(desired, 0.0).copy()
    desired[0] += float(desired[1:][forced].sum())
    desired[1:][forced] = 0.0
    halted = ~tradable_array & ~forced
    frozen = np.r_[False, halted]
    free = np.r_[True, ~halted & ~forced]
    frozen_value = float(pre[frozen].sum())
    allocation = np.zeros_like(pre)
    free_desired = float(desired[free].sum())
    if free_desired > 0:
        allocation[free] = desired[free] / free_desired
    else:
        allocation[0] = 1.0
    # Post-execution holdings, before dividing by net wealth: a*z + b.
    offset = -frozen_value * allocation
    offset[frozen] = pre[frozen]
    risky_a = allocation[1:]
    risky_b_minus_pre = offset[1:] - pre[1:]

    z = 1.0
    if rate:
        # The equation is piecewise linear. Updating the active trade signs
        # normally solves it in two iterations at realistic commission rates.
        for _ in range(12):
            signs = np.sign(risky_a * z + risky_b_minus_pre)
            candidate = (1.0 - rate * float(signs @ risky_b_minus_pre)) / (
                1.0 + rate * float(signs @ risky_a)
            )
            residual = candidate + rate * float(
                np.abs(risky_a * candidate + risky_b_minus_pre).sum()
            ) - 1.0
            z = candidate
            if abs(residual) <= 5e-15:
                break
        else:
            lower, upper = frozen_value, 1.0
            for _ in range(64):
                z = (lower + upper) / 2.0
                residual = z + rate * float(
                    np.abs(risky_a * z + risky_b_minus_pre).sum()
                ) - 1.0
                if residual > 0:
                    upper = z
                else:
                    lower = z
    values = allocation * z + offset
    trades = values[1:] - pre[1:]
    turnover = float(np.abs(trades).sum())
    fee = rate * turnover
    if z <= 0 or z < frozen_value - 1e-12 or np.any(values < -1e-12):
        raise ArithmeticError("execution could not satisfy the self-financing constraints")
    return ExecutionResult(z, np.maximum(values, 0.0) / z, trades, turnover, fee)
