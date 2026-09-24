"""Bounded, known-signal calibration with independent market and learner RNGs.

This module deliberately does not claim an optimal policy with transaction
costs. Its reference uses the known sign of conditional returns; the analytic
Kelly weight is optimal only for the zero-cost, one-period log objective.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dtasrl.envs.portfolio import TargetWeightPortfolioEnv


@dataclass(frozen=True)
class CalibrationSettings:
    signal_strength: float = 1.0
    persistence: float = 0.9
    return_size: float = 0.02
    initial_price: float = 100.0
    episode_steps: int = 512
    train_steps: int = 4096
    validation_steps: int = 2048
    test_steps: int = 2048
    train_market_seed: int = 7001
    validation_market_seed: int = 8001
    test_market_seed: int = 9001

    def __post_init__(self) -> None:
        if not 0 <= self.signal_strength <= 1:
            raise ValueError("signal_strength must lie in [0, 1]")
        if not 0 <= self.persistence < 1:
            raise ValueError("persistence must lie in [0, 1)")
        if not 0 < self.return_size < 1:
            raise ValueError("return_size must lie in (0, 1)")
        if not np.isfinite(self.initial_price) or self.initial_price <= 0:
            raise ValueError("initial_price must be finite and positive")
        for name in ("episode_steps", "train_steps", "validation_steps", "test_steps"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        seeds = (
            self.train_market_seed, self.validation_market_seed, self.test_market_seed,
        )
        if any(not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ValueError("market seeds must be nonnegative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("training, validation, and test market seeds must differ")


def calibration_settings(config: dict[str, Any]) -> CalibrationSettings:
    supplied = config.get("synthetic", {})
    if supplied.get("dgp", "bounded_binary_markov") != "bounded_binary_markov":
        raise ValueError("calibration requires synthetic.dgp=bounded_binary_markov")
    fields = CalibrationSettings.__dataclass_fields__
    return CalibrationSettings(**{name: supplied[name] for name in fields if name in supplied})


def kelly_risky_weight(up_probability: np.ndarray | float, return_size: float = 0.02):
    """Exact zero-cost long-only optimum for binary simple returns +/-r."""
    p = np.asarray(up_probability, dtype=float)
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("up_probability must be finite and lie in [0, 1]")
    if not np.isfinite(return_size) or not 0 < return_size < 1:
        raise ValueError("return_size must lie in (0, 1)")
    return np.clip((2.0 * p - 1.0) / return_size, 0.0, 1.0)


def _sample_market(
    settings: CalibrationSettings, steps: int, rng: np.random.Generator,
) -> dict[str, Any]:
    signal = np.empty(steps + 1, dtype=np.float64)
    signal[0] = rng.choice(np.array([-1.0, 1.0]))
    # Interleave return and transition innovations so extending a snapshot does
    # not restate its prefix. Each next return is conditioned on signal[t].
    innovations = rng.random((steps, 2))
    returns = np.empty(steps, dtype=np.float64)
    for t in range(steps):
        probability = 0.5 + settings.signal_strength * 0.15 * signal[t]
        returns[t] = (
            settings.return_size if innovations[t, 0] < probability else -settings.return_size
        )
        signal[t + 1] = (
            signal[t] if innovations[t, 1] < settings.persistence else -signal[t]
        )
    prices = np.r_[settings.initial_price, settings.initial_price * np.cumprod(1 + returns)]
    opens = np.r_[prices[0], prices[:-1]]
    probabilities = 0.5 + settings.signal_strength * 0.15 * signal
    return {
        "prices": prices[:, None],
        "opens": opens[:, None],
        "features": signal[:, None],
        "dates": np.arange(steps + 1),
        "tickers": ["SYNTH"],
        "observation_tickers": ["SYNTH"],
        "feature_names": ["signal"],
        "conditional_up_probability": probabilities,
        "reference_risky_weight": (probabilities > 0.5).astype(float),
        "zero_cost_kelly_risky_weight": kelly_risky_weight(probabilities, settings.return_size),
    }


def generate_calibration_market(
    settings: CalibrationSettings, steps: int, market_seed: int,
) -> dict[str, Any]:
    if not isinstance(steps, int) or steps < 2:
        raise ValueError("steps must be an integer of at least two")
    rng = np.random.default_rng(market_seed)
    initial_rng_state = rng.bit_generator.state
    result = _sample_market(settings, steps, rng)
    result["market_seed"] = int(market_seed)
    result["market_rng_initial_state"] = initial_rng_state
    result["market_rng_final_state"] = rng.bit_generator.state
    return result


def _update_array_hash(digest, data: dict[str, Any]) -> None:
    for name in ("prices", "opens", "features", "dates", "conditional_up_probability"):
        array = np.ascontiguousarray(data[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())


def prepare_calibration(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Return reproducible source snapshots and their deterministic identity.

    The 4096-step training snapshot documents the DGP and common market seed.
    Training itself uses CalibrationEnv's independently resampled episodes.
    All learner seeds share these fixed validation and test paths.
    """
    settings = calibration_settings(config)
    splits: dict[str, Any] = {}
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(settings), sort_keys=True).encode())
    digest.update(b"bounded_binary_markov_v1")
    for name in ("train", "validation", "test"):
        steps = getattr(settings, f"{name}_steps")
        market_seed = getattr(settings, f"{name}_market_seed")
        splits[name] = generate_calibration_market(settings, steps, market_seed)
        digest.update(name.encode())
        _update_array_hash(digest, splits[name])
    splits["metadata"] = {
        "dgp": "bounded_binary_markov_v1",
        "settings": asdict(settings),
        "time_axis": "synthetic integer trading step, not observed calendar dates",
        "training_sampling": "new independent innovations at every episode reset",
        "market_rng": "numpy.random.Generator(PCG64), independent of learner seed",
        "reference": "known-signal sign rule; not a transaction-cost optimum",
        "zero_signal_reference": "cash; exact expected-log optimum at zero costs",
        "training_snapshot_role": "provenance only; training uses resampled episodes",
        "data_sha256": digest.hexdigest(),
    }
    return splits, digest.hexdigest()


class CalibrationEnv(TargetWeightPortfolioEnv):
    """Resample training episodes without tying market randomness to PPO's seed.

    Pickling this environment saves the Generator and its current bit-generator
    state. A resume must restore the whole environment, rather than instantiate
    another environment from the initial market seed.
    """

    def __init__(self, settings: CalibrationSettings | None = None, **kwargs):
        self.settings = settings if settings is not None else CalibrationSettings()
        self.market_seed = self.settings.train_market_seed
        self.market_rng = np.random.default_rng(self.market_seed)
        first = _sample_market(self.settings, self.settings.episode_steps, self.market_rng)
        self.episode_index = 0
        self._episode_started = False
        digest = hashlib.sha256()
        _update_array_hash(digest, first)
        self.runtime_identity = {
            "dgp": "bounded_binary_markov_v1",
            "settings": asdict(self.settings),
            "initial_episode_sha256": digest.hexdigest(),
            "train_market_seed": self.market_seed,
        }
        super().__init__(
            prices=first["prices"], execution_prices=first["opens"],
            features=first["features"], **kwargs,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self._episode_started:
            data = _sample_market(self.settings, self.settings.episode_steps, self.market_rng)
            self.prices = data["prices"]
            self.execution_prices = data["opens"]
            self.features = data["features"]
            self.episode_index += 1
        self._episode_started = True
        observation, info = super().reset(seed=seed, options=options)
        info.update({
            "market_seed": self.market_seed,
            "episode_index": self.episode_index,
            "market_rng_state": self.market_rng.bit_generator.state,
        })
        return observation, info


def make_calibration_train_env(config: dict[str, Any]) -> CalibrationEnv:
    return CalibrationEnv(
        calibration_settings(config),
        transaction_cost_bps=float(config.get("transaction_cost_bps", 10)),
        temperature=float(config.get("temperature", 1)),
        initial_wealth=float(config.get("initial_wealth", 1)),
        reward_scale=float(config.get("reward_scale", 100)),
        record_history=bool(config.get("record_history", False)),
    )
