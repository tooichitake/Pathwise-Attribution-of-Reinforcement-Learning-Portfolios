from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dtasrl.accounting import rebalance_portfolio


def logits_to_weights(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("logits must be a finite one-dimensional array")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    shifted = values / temperature - np.max(values / temperature)
    exp = np.exp(shifted)
    return exp / exp.sum()


@dataclass(frozen=True)
class StepRecord:
    time_index: int
    raw_action: np.ndarray
    pre_trade_weights: np.ndarray
    wealth_before: float
    wealth_after: float
    reward: float
    transaction_cost: float
    turnover: float
    target_weights: np.ndarray
    closing_weights: np.ndarray
    gross_relatives: np.ndarray
    overnight_gross_relatives: np.ndarray
    pre_trade_wealth: float
    trade_values: np.ndarray
    decision_weights: np.ndarray
    scaled_reward: float


class TargetWeightPortfolioEnv(gym.Env[np.ndarray, np.ndarray]):
    """Daily long-only target-weight environment with an explicit cash asset.

    `prices[t]` and `features[t]` are known when action t is selected. Supplying
    `execution_prices` executes at the next open, after the old holdings earn the
    overnight return. Without them, execution retains the legacy current-close
    convention. Costs are charged on actual risky-asset trade values.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray | None = None,
        transaction_cost_bps: float = 10.0,
        temperature: float = 1.0,
        tradable: np.ndarray | None = None,
        forced_liquidation: np.ndarray | None = None,
        initial_wealth: float = 1.0,
        execution_prices: np.ndarray | None = None,
        reward_scale: float = 1.0,
        record_history: bool = True,
    ) -> None:
        super().__init__()
        self.prices = np.asarray(prices, dtype=np.float64)
        if self.prices.ndim == 1:
            self.prices = self.prices[:, None]
        if self.prices.ndim != 2 or self.prices.shape[0] < 2:
            raise ValueError("prices must have shape (time >= 2, assets)")
        if np.any(~np.isfinite(self.prices)) or np.any(self.prices <= 0):
            raise ValueError("prices must be finite and strictly positive")
        self.n_assets = self.prices.shape[1]
        self.execution_prices = None
        if execution_prices is not None:
            execution = np.asarray(execution_prices, dtype=np.float64)
            if execution.ndim == 1:
                execution = execution[:, None]
            if execution.shape != self.prices.shape:
                raise ValueError("execution_prices must match prices")
            if np.any(~np.isfinite(execution)) or np.any(execution <= 0):
                raise ValueError("execution_prices must be finite and strictly positive")
            self.execution_prices = execution
        self.reward_scale = float(reward_scale)
        if not np.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive")
        self.record_history = bool(record_history)
        if features is None:
            features = np.zeros((self.prices.shape[0], self.n_assets), dtype=np.float64)
        self.features = np.asarray(features, dtype=np.float64)
        if self.features.shape[0] != self.prices.shape[0]:
            raise ValueError("features and prices must have the same time dimension")
        self.features = self.features.reshape(self.prices.shape[0], -1)
        if np.any(~np.isfinite(self.features)):
            raise ValueError("features must be finite")
        self.tradable = (
            np.ones_like(self.prices, dtype=bool)
            if tradable is None
            else np.asarray(tradable, dtype=bool)
        )
        if self.tradable.shape != self.prices.shape:
            raise ValueError("tradable must match prices")
        self.forced_liquidation = (
            np.zeros_like(self.prices, dtype=bool)
            if forced_liquidation is None
            else np.asarray(forced_liquidation, dtype=bool)
        )
        if self.forced_liquidation.shape != self.prices.shape:
            raise ValueError("forced_liquidation must match prices")
        self.cost_rate = float(transaction_cost_bps) / 10_000.0
        if not np.isfinite(self.cost_rate) or not 0 <= self.cost_rate < 1:
            raise ValueError("transaction costs must be finite and below 10000 bps")
        self.temperature = float(temperature)
        self.initial_wealth = float(initial_wealth)
        if not np.isfinite(self.initial_wealth) or self.initial_wealth <= 0:
            raise ValueError("initial wealth must be finite and positive")
        observation_size = (self.n_assets + 1) + self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.n_assets + 1,), dtype=np.float32
        )
        self.records: list[StepRecord] = []
        self.t = 0
        self.wealth = self.initial_wealth
        self.current_weights = np.r_[1.0, np.zeros(self.n_assets)]

    def _observation(self) -> np.ndarray:
        return np.concatenate([self.current_weights, self.features[self.t]]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        self.wealth = self.initial_wealth
        self.current_weights = np.r_[1.0, np.zeros(self.n_assets)]
        self.records = []
        return self._observation(), {"wealth": self.wealth}

    def step(self, action: np.ndarray):
        if self.t >= self.prices.shape[0] - 1:
            raise RuntimeError("step called after episode termination")
        action = np.asarray(action, dtype=float)
        if action.shape != (self.n_assets + 1,):
            raise ValueError(f"action must have shape {(self.n_assets + 1,)}")
        decision_weights = self.current_weights.copy()
        wealth_before = self.wealth
        if self.execution_prices is None:
            overnight = np.ones(self.n_assets + 1)
            execution_price = self.prices[self.t]
            execution_index = self.t
        else:
            execution_price = self.execution_prices[self.t + 1]
            overnight = np.r_[1.0, execution_price / self.prices[self.t]]
            execution_index = self.t + 1
        overnight_relative = float(decision_weights @ overnight)
        pre_trade_wealth = wealth_before * overnight_relative
        pre_trade_weights = decision_weights * overnight / overnight_relative
        execution = rebalance_portfolio(
            pre_trade_weights,
            logits_to_weights(action, self.temperature),
            self.cost_rate,
            self.tradable[execution_index],
            self.forced_liquidation[execution_index],
        )
        desired = execution.target_weights
        turnover = execution.turnover
        transaction_cost = pre_trade_wealth * execution.cost_fraction
        gross_relatives = np.r_[1.0, self.prices[self.t + 1] / execution_price]
        portfolio_relative = float(desired @ gross_relatives)
        self.wealth = pre_trade_wealth * execution.net_wealth_fraction * portfolio_relative
        closing_weights = desired * gross_relatives / portfolio_relative
        reward = float(np.log(self.wealth / wealth_before))
        scaled_reward = reward * self.reward_scale
        trade_values = pre_trade_wealth * execution.trade_value_fractions
        if self.record_history:
            self.records.append(StepRecord(
                time_index=self.t,
                raw_action=action.copy(),
                pre_trade_weights=pre_trade_weights,
                wealth_before=wealth_before,
                wealth_after=self.wealth,
                reward=reward,
                transaction_cost=transaction_cost,
                turnover=turnover,
                target_weights=desired.copy(),
                closing_weights=closing_weights.copy(),
                gross_relatives=gross_relatives.copy(),
                overnight_gross_relatives=overnight.copy(),
                pre_trade_wealth=pre_trade_wealth,
                trade_values=trade_values,
                decision_weights=decision_weights,
                scaled_reward=scaled_reward,
            ))
        self.current_weights = closing_weights
        self.t += 1
        terminated = self.t >= self.prices.shape[0] - 1
        return (
            self._observation(),
            scaled_reward,
            terminated,
            False,
            {
                "wealth": self.wealth,
                "transaction_cost": transaction_cost,
                "turnover": turnover,
                "target_weights": desired.copy(),
                "gross_relatives": gross_relatives.copy(),
                "overnight_gross_relatives": overnight.copy(),
                "pre_trade_wealth": pre_trade_wealth,
                "trade_values": trade_values,
                "net_log_reward": reward,
            },
        )

    def render(self):
        return {"time_index": self.t, "wealth": self.wealth, "weights": self.current_weights.copy()}
