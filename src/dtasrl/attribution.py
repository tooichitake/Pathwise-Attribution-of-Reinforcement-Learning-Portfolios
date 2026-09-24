from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dtasrl.accounting import rebalance_portfolio


@dataclass(frozen=True)
class PathResult:
    wealth: np.ndarray
    target_weights: np.ndarray
    closing_weights: np.ndarray
    turnover: np.ndarray
    costs: np.ndarray
    pre_trade_wealth: np.ndarray
    pre_trade_weights: np.ndarray
    trade_values: np.ndarray

    @property
    def net_log_wealth(self) -> float:
        return float(np.log(self.wealth[-1] / self.wealth[0]))


def evaluate_target_path(
    target_weights: np.ndarray,
    gross_relatives: np.ndarray,
    transaction_cost_bps: float,
    initial_wealth: float = 1.0,
    overnight_gross_relatives: np.ndarray | None = None,
) -> PathResult:
    """Replay execution targets: overnight on old holdings, then intraday.

    ``gross_relatives`` describes execution-price to next-observation returns.
    With next-open execution it is the intraday open-to-close relative, and
    ``overnight_gross_relatives`` is the preceding close-to-open relative.
    Omitting overnight relatives retains close-to-close replay.
    """
    targets = np.asarray(target_weights, dtype=float)
    relatives = np.asarray(gross_relatives, dtype=float)
    if targets.shape != relatives.shape:
        raise ValueError("target_weights and gross_relatives must have equal shape")
    if not np.allclose(targets.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("every target-weight row must sum to one")
    if (
        not np.all(np.isfinite(targets))
        or not np.all(np.isfinite(relatives))
        or np.any(targets < -1e-10)
        or np.any(relatives <= 0)
    ):
        raise ValueError("weights must be nonnegative and relatives positive")
    overnight = (
        np.ones_like(relatives)
        if overnight_gross_relatives is None
        else np.asarray(overnight_gross_relatives, dtype=float)
    )
    if (
        overnight.shape != relatives.shape
        or not np.all(np.isfinite(overnight))
        or np.any(overnight <= 0)
    ):
        raise ValueError("overnight relatives must match the path and be finite and positive")
    if not np.isfinite(initial_wealth) or initial_wealth <= 0:
        raise ValueError("initial_wealth must be finite and positive")
    n_steps, n_assets = targets.shape
    wealth = np.empty(n_steps + 1)
    closing = np.empty_like(targets)
    turnover = np.empty(n_steps)
    costs = np.empty(n_steps)
    actual_targets = np.empty_like(targets)
    pre_trade_wealth = np.empty(n_steps)
    pre_trade_weights = np.empty_like(targets)
    trade_values = np.empty((n_steps, n_assets - 1))
    wealth[0] = initial_wealth
    pretrade = np.r_[1.0, np.zeros(n_assets - 1)]
    rate = transaction_cost_bps / 10_000.0
    for t in range(n_steps):
        overnight_relative = float(pretrade @ overnight[t])
        pre_trade_wealth[t] = wealth[t] * overnight_relative
        pre_trade_weights[t] = pretrade * overnight[t] / overnight_relative
        execution = rebalance_portfolio(pre_trade_weights[t], targets[t], rate)
        actual_targets[t] = execution.target_weights
        turnover[t] = execution.turnover
        costs[t] = pre_trade_wealth[t] * execution.cost_fraction
        trade_values[t] = pre_trade_wealth[t] * execution.trade_value_fractions
        relative = float(actual_targets[t] @ relatives[t])
        wealth[t + 1] = pre_trade_wealth[t] * execution.net_wealth_fraction * relative
        closing[t] = actual_targets[t] * relatives[t] / relative
        pretrade = closing[t]
    return PathResult(
        wealth, actual_targets, closing, turnover, costs,
        pre_trade_wealth, pre_trade_weights, trade_values,
    )


def freeze_k_path(
    full_target_weights: np.ndarray,
    gross_relatives: np.ndarray,
    k: int,
    transaction_cost_bps: float,
    overnight_gross_relatives: np.ndarray | None = None,
    initial_wealth: float = 1.0,
) -> PathResult:
    """Follow K targets, then hold the resulting asset units without rebalancing."""
    targets = np.asarray(full_target_weights, dtype=float)
    relatives = np.asarray(gross_relatives, dtype=float)
    overnight = (
        np.ones_like(relatives)
        if overnight_gross_relatives is None
        else np.asarray(overnight_gross_relatives, dtype=float)
    )
    if overnight.shape != relatives.shape:
        raise ValueError("overnight relatives must match the path")
    if not 1 <= k <= len(targets):
        raise ValueError("k must be between one and the number of decisions")
    rebuilt = targets.copy()
    prefix = evaluate_target_path(
        targets[:k], relatives[:k], transaction_cost_bps,
        initial_wealth=initial_wealth,
        overnight_gross_relatives=overnight[:k],
    )
    current = prefix.closing_weights[-1]
    for t in range(k, len(targets)):
        at_open = current * overnight[t] / float(current @ overnight[t])
        rebuilt[t] = at_open
        relative = float(at_open @ relatives[t])
        current = at_open * relatives[t] / relative
    return evaluate_target_path(
        rebuilt, relatives, transaction_cost_bps, initial_wealth=initial_wealth,
        overnight_gross_relatives=overnight,
    )


def exposure_composition(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(weights, dtype=float)
    risky = values[:, 1:]
    exposure = risky.sum(axis=1)
    n_risky = risky.shape[1]
    composition = np.empty_like(risky)
    previous = np.repeat(1.0 / n_risky, n_risky)
    for t, level in enumerate(exposure):
        if level > 1e-12:
            previous = risky[t] / level
        composition[t] = previous
    return exposure, composition


def compose_weights(exposure: np.ndarray, composition: np.ndarray) -> np.ndarray:
    e = np.asarray(exposure, dtype=float)
    q = np.asarray(composition, dtype=float)
    risky = e[:, None] * q
    return np.column_stack([1.0 - e, risky])


def two_by_two_attribution(
    full_target_weights: np.ndarray,
    gross_relatives: np.ndarray,
    transaction_cost_bps: float,
    overnight_gross_relatives: np.ndarray | None = None,
    initial_wealth: float = 1.0,
) -> dict[str, float | PathResult]:
    exposure, composition = exposure_composition(full_target_weights)
    mean_exposure = np.repeat(exposure.mean(), len(exposure))
    mean_composition = np.repeat(composition.mean(axis=0, keepdims=True), len(exposure), axis=0)
    mean_composition /= mean_composition.sum(axis=1, keepdims=True)
    paths = {
        "dynamic_exposure_dynamic_composition": compose_weights(exposure, composition),
        "dynamic_exposure_mean_composition": compose_weights(exposure, mean_composition),
        "mean_exposure_dynamic_composition": compose_weights(mean_exposure, composition),
        "mean_exposure_mean_composition": compose_weights(mean_exposure, mean_composition),
    }
    evaluated = {
        name: evaluate_target_path(
            path, gross_relatives, transaction_cost_bps,
            initial_wealth=initial_wealth,
            overnight_gross_relatives=overnight_gross_relatives,
        )
        for name, path in paths.items()
    }
    v11 = evaluated["dynamic_exposure_dynamic_composition"].net_log_wealth
    v10 = evaluated["dynamic_exposure_mean_composition"].net_log_wealth
    v01 = evaluated["mean_exposure_dynamic_composition"].net_log_wealth
    v00 = evaluated["mean_exposure_mean_composition"].net_log_wealth
    equal_composition = np.full_like(composition, 1.0 / composition.shape[1])
    equal_path = evaluate_target_path(
        compose_weights(mean_exposure, equal_composition), gross_relatives, transaction_cost_bps,
        initial_wealth=initial_wealth,
        overnight_gross_relatives=overnight_gross_relatives,
    )
    return {
        **evaluated,
        "timing_shapley": 0.5 * ((v11 - v01) + (v10 - v00)),
        "dynamic_selection_shapley": 0.5 * ((v11 - v10) + (v01 - v00)),
        "persistent_selection": v00 - equal_path.net_log_wealth,
        "equal_weight_same_exposure": equal_path,
    }
