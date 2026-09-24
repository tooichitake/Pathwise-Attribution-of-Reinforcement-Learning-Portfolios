"""Resource-bounded SBX training with auditable, explicit checkpoint continuation.

Checkpoints are published only after optimizer updates, at rollout boundaries.
The environment and RNG states are retained, but continuation is deliberately not
advertised as bitwise identical across processes, libraries, or operating systems.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cloudpickle
import jax
import numpy as np
import pandas as pd
import psutil
from sbx import PPO as JaxPPO
from sbx import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from dtasrl.config import canonical_json, public_config
from dtasrl.provenance import git_commit, source_tree_sha256, utc_now, write_json

PACKAGES = (
    "sbx-rl",
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "tfp-nightly",
    "stable-baselines3",
    "numpy",
    "gymnasium",
)


class PPO(JaxPPO):
    """Flush statistics after each completed optimization, including the final rollout."""

    def train(self):
        start = time.perf_counter()
        super().train()
        self.logger.record(
            "train/optimizer_wall_seconds_including_jit", time.perf_counter() - start
        )
        self.logger.record("time/total_timesteps", self.num_timesteps)
        self.logger.record("rollout/mean_step_reward", float(self.rollout_buffer.rewards.mean()))
        self.logger.record("rollout/std_step_reward", float(self.rollout_buffer.rewards.std()))
        self.logger.dump(step=self.num_timesteps)


class TrainingInterrupted(RuntimeError):
    """A recoverable stop; ``checkpoint`` points to the last safe checkpoint."""

    def __init__(self, reason: str, checkpoint: Path | None):
        super().__init__(reason)
        self.reason = reason
        self.checkpoint = checkpoint


class _MemoryBoundaryStop(RuntimeError):
    pass


def _atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    write_json(value, temporary)
    _replace_with_retry(temporary, path)


def _replace_with_retry(source: Path, target: Path, attempts: int = 8) -> None:
    """Atomically publish one file, tolerating transient Windows file locks."""

    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == attempts - 1:
                raise
            time.sleep(min(0.05 * (2**attempt), 0.5))


def _environment_identity(env: Any) -> dict[str, Any]:
    base = env.unwrapped
    arrays = {}
    for name in ("prices", "execution_prices", "features", "tradable", "forced_liquidation"):
        value = getattr(base, name, None)
        if value is not None:
            array = np.ascontiguousarray(value)
            arrays[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
    parameters = {
        name: getattr(base, name, None)
        for name in ("cost_rate", "temperature", "initial_wealth", "reward_scale", "n_assets")
    }
    return {
        "class": f"{type(base).__module__}.{type(base).__qualname__}",
        "arrays": arrays,
        "parameters": parameters,
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
    }


def _identity(config: dict[str, Any], seed: int, env: Any) -> dict[str, Any]:
    immutable = public_config(config).copy()
    immutable["training"] = dict(immutable.get("training", {}))
    immutable["training"].pop("resume", None)
    return {
        "config_sha256": hashlib.sha256(canonical_json(immutable).encode()).hexdigest(),
        "source_sha256": source_tree_sha256(),
        "seed": int(seed),
        "environment": _environment_identity(env),
        "runtime": {
            "python": platform.python_version(),
            **{name: version(name) for name in PACKAGES},
        },
    }


def _algorithm_parameters(config: dict[str, Any], seed: int, env: Any) -> dict[str, Any]:
    specified = config.get("model", {})
    parameters = {
        "policy": "MlpPolicy",
        "env": env,
        "seed": seed,
        "device": "cpu",
        "verbose": int(specified.get("verbose", 0)),
        "learning_rate": float(specified.get("learning_rate", 3e-4)),
        "gamma": float(specified.get("gamma", 0.99)),
        "policy_kwargs": {"net_arch": specified.get("net_arch", [64, 64])},
    }
    if str(config.get("algorithm", "ppo")).lower() == "ppo":
        parameters["policy_kwargs"].update(ortho_init=False, optimizer_kwargs={"eps": 1e-5})
        parameters.update(
            {
                "n_steps": int(specified.get("n_steps", 2048)),
                "batch_size": int(specified.get("batch_size", 256)),
                "n_epochs": int(specified.get("n_epochs", 10)),
                "gae_lambda": float(specified.get("gae_lambda", 0.95)),
                "clip_range": float(specified.get("clip_range", 0.2)),
                "ent_coef": float(specified.get("ent_coef", 0.0)),
                "vf_coef": float(specified.get("vf_coef", 0.5)),
                "max_grad_norm": float(specified.get("max_grad_norm", 0.5)),
                "normalize_advantage": bool(specified.get("normalize_advantage", True)),
            }
        )
    else:
        parameters.update(
            {
                "buffer_size": int(specified.get("buffer_size", 100_000)),
                "batch_size": int(specified.get("batch_size", 256)),
                "learning_starts": int(specified.get("learning_starts", 100)),
                "tau": float(specified.get("tau", 0.005)),
                "train_freq": int(specified.get("train_freq", 1)),
                "gradient_steps": int(specified.get("gradient_steps", 1)),
                "ent_coef": specified.get("ent_coef", "auto"),
            }
        )
    return parameters


class _ResourceMonitor(BaseCallback):
    def __init__(
        self,
        reserve_bytes: int,
        interval: int,
        breach_patience: int = 3,
        hard_floor_bytes: int | None = None,
    ):
        super().__init__()
        self.reserve_bytes = reserve_bytes
        self.interval = max(1, interval)
        self.breach_patience = max(1, breach_patience)
        self.hard_floor_bytes = (
            max(0, reserve_bytes - 2**30)
            if hard_floor_bytes is None
            else max(0, hard_floor_bytes)
        )
        self.consecutive_breaches = 0
        self.last_sample = -self.interval
        self.peak_rss_bytes = 0
        self.minimum_available_bytes: int | None = None
        self.process = psutil.Process()

    def sample(self, enforce: bool = True) -> None:
        memory = self.process.memory_info()
        self.peak_rss_bytes = max(self.peak_rss_bytes, memory.rss, getattr(memory, "peak_wset", 0))
        available = psutil.virtual_memory().available
        self.minimum_available_bytes = min(self.minimum_available_bytes or available, available)
        if available < self.reserve_bytes:
            self.consecutive_breaches += 1
        else:
            self.consecutive_breaches = 0
        hard_breach = available < self.hard_floor_bytes
        sustained_breach = self.consecutive_breaches >= self.breach_patience
        if enforce and (hard_breach or sustained_breach):
            raise _MemoryBoundaryStop(
                f"available memory {available / 2**30:.2f} GiB is below "
                f"the reserved {self.reserve_bytes / 2**30:.2f} GiB "
                f"({self.consecutive_breaches} consecutive samples; "
                f"hard floor {self.hard_floor_bytes / 2**30:.2f} GiB)"
            )

    def _on_rollout_start(self) -> None:
        # This is after the preceding gradient updates and before new transitions.
        if self.num_timesteps - self.last_sample >= self.interval:
            self.sample()
            self.last_sample = self.num_timesteps

    def _on_step(self) -> bool:
        return True


class _SafeCheckpointCallback(BaseCallback):
    """Publish checkpoints only between completed PPO rollout/update cycles."""

    def __init__(
        self,
        run_path: Path,
        interval: int,
        initial_step: int,
        summary: Callable[[str, str | None], dict[str, Any]],
        stop_after_checkpoints: int | None = None,
    ):
        super().__init__()
        self.run_path = run_path
        self.interval = interval
        self.next_step = (initial_step // interval + 1) * interval
        self.summary = summary
        self.stop_after_checkpoints = stop_after_checkpoints
        self.checkpoint_count = 0
        self.latest: Path | None = None

    def _on_rollout_start(self) -> None:
        current = int(self.model.num_timesteps)
        if current < self.next_step:
            return
        self.latest = _save_checkpoint(self.model, self.run_path, self.summary("checkpointed"))
        _combine_learning_logs(self.run_path)
        _atomic_json(self.summary("running"), self.run_path / "training_status.json")
        self.checkpoint_count += 1
        while self.next_step <= current:
            self.next_step += self.interval
        if (
            self.stop_after_checkpoints == self.checkpoint_count
            and current < int(self.locals["total_timesteps"])
        ):
            raise TrainingInterrupted(
                "requested test interruption at safe boundary", self.latest
            )

    def _on_step(self) -> bool:
        return True


def _save_checkpoint(model: Any, run_path: Path, metadata: dict[str, Any]) -> Path:
    path = run_path / "checkpoints" / f"step_{model.num_timesteps:010d}"
    revision = 1
    while path.exists():
        revision += 1
        path = run_path / "checkpoints" / f"step_{model.num_timesteps:010d}_v{revision:03d}"
    # Renaming a populated directory is not reliably atomic on Windows and can
    # fail with WinError 5 when an indexer briefly opens a file. Create the
    # final directory once, atomically publish each component, and publish
    # checkpoint.json last as the completion marker. latest_checkpoint.json is
    # updated only after that marker exists, so incomplete directories are never
    # resume targets.
    path.mkdir(parents=True)
    model_partial = path / "model.partial.zip"
    model.save(model_partial)
    _replace_with_retry(model_partial, path / "model.zip")
    state = {
        "env": model.get_env(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "backend": "sbx_jax",
    }
    state_partial = path / "state.pkl.partial"
    with state_partial.open("wb") as handle:
        cloudpickle.dump(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    _replace_with_retry(state_partial, path / "state.pkl")
    if isinstance(model, SAC):
        replay_partial = path / "replay_buffer.pkl.partial"
        model.save_replay_buffer(replay_partial)
        _replace_with_retry(replay_partial, path / "replay_buffer.pkl")
    # This marker is deliberately last. A directory without it is incomplete.
    _atomic_json(metadata, path / "checkpoint.json")
    # Publish only once every checkpoint component has been written successfully.
    _atomic_json({"path": str(path.relative_to(run_path))}, run_path / "latest_checkpoint.json")
    return path


def _load_checkpoint(algorithm: type, path: Path) -> Any:
    # These files are project-generated trusted artifacts, never arbitrary downloads.
    with (path / "state.pkl").open("rb") as handle:
        state = cloudpickle.load(handle)
    model = algorithm.load(path / "model.zip", env=state["env"], device="cpu", force_reset=False)
    if algorithm is SAC:
        model.load_replay_buffer(path / "replay_buffer.pkl", truncate_last_traj=False)
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    if state.get("backend") != "sbx_jax":
        raise ValueError("Cannot resume a non-SBX checkpoint")
    return model


def _combine_learning_logs(run_path: Path) -> None:
    frames = []
    for path in sorted((run_path / "learning").glob("session_*/progress.csv")):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        frame["session"] = path.parent.name
        frames.append(frame)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(run_path / "progress.csv", index=False)


def train_model(config: dict[str, Any], seed: int, env: Any, run_path: str | Path) -> Any:
    """Train a final PPO/SAC checkpoint, or continue a matching incomplete run.

    ``_resume=True`` explicitly enables continuation. A completed run is never
    silently rerun. ``training`` controls memory reserve, checkpoints and threads;
    ``model`` controls algorithm parameters. The target counts environment steps.
    """
    run_path = Path(run_path)
    run_path.mkdir(parents=True, exist_ok=True)
    lock_path = run_path / ".training.lock"
    resume = bool(config.get("_resume", config.get("training", {}).get("resume", False)))
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        try:
            active = abs(psutil.Process(existing["pid"]).create_time() - existing["created"]) < 1
        except psutil.NoSuchProcess:
            active = False
        if active or not resume:
            raise FileExistsError("training directory is locked by an active or unresumed run")
        lock_path.unlink()
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "created": psutil.Process().create_time()}, handle)
    try:
        return _train_model(config, seed, env, run_path)
    finally:
        lock_path.unlink(missing_ok=True)


def _train_model(config: dict[str, Any], seed: int, env: Any, run_path: Path) -> Any:
    algorithm_name = str(config.get("algorithm", "ppo")).lower()
    if algorithm_name not in ("ppo", "sac"):
        raise ValueError("checkpointed training supports only ppo or sac")
    algorithm = PPO if algorithm_name == "ppo" else SAC
    target = int(config["total_timesteps"])
    if target <= 0:
        raise ValueError("total_timesteps must be positive")
    controls = config.get("training", {})
    requested_threads = controls.get("threads")
    if requested_threads is not None and int(requested_threads) < 1:
        raise ValueError("training.threads must be positive when supplied")
    if config.get("backend", "sbx_jax") != "sbx_jax":
        raise ValueError("Active training requires backend=sbx_jax")
    resume = bool(config.get("_resume", controls.get("resume", False)))
    expected = _identity(config, seed, env)
    identity_file = run_path / "training_identity.json"
    previous_summary: dict[str, Any] = {}
    checkpoint = None
    if identity_file.exists():
        if not resume:
            raise FileExistsError(f"training already exists at {run_path}; use explicit resume")
        if json.loads(identity_file.read_text(encoding="utf-8")) != expected:
            raise ValueError(
                "resume identity mismatch: configuration, data, seed or source changed"
            )
        status = json.loads((run_path / "training_status.json").read_text(encoding="utf-8"))
        if status["status"] == "complete":
            raise FileExistsError("completed training cannot be resumed or overwritten")
        pointer = json.loads((run_path / "latest_checkpoint.json").read_text(encoding="utf-8"))
        checkpoint = run_path / pointer["path"]
        previous_summary = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
        model = _load_checkpoint(algorithm, checkpoint)
    else:
        if resume:
            raise FileNotFoundError("no matching training identity/checkpoint exists to resume")
        if (run_path / "model.zip").exists() or (run_path / "final_checkpoint.zip").exists():
            raise FileExistsError("existing model without training identity; refusing overwrite")
        model = algorithm(**_algorithm_parameters(config, seed, env))
        _atomic_json(expected, identity_file)

    quantum = model.n_steps * model.n_envs if isinstance(model, PPO) else model.n_envs
    requested_interval = int(controls.get("checkpoint_interval_steps", 250_000))
    if requested_interval <= 0:
        raise ValueError("checkpoint_interval_steps must be positive")
    interval = max(quantum, math.ceil(requested_interval / quantum) * quantum)
    actual_target = math.ceil(target / quantum) * quantum
    reserve = float(controls.get("min_available_memory_gb", 8))
    if reserve < 0:
        raise ValueError("min_available_memory_gb cannot be negative")
    breach_patience = int(controls.get("memory_breach_patience", 3))
    if breach_patience < 1:
        raise ValueError("memory_breach_patience must be positive")
    hard_floor = float(controls.get("memory_hard_stop_gb", max(0.0, reserve - 1.0)))
    if hard_floor < 0 or hard_floor > reserve:
        raise ValueError("memory_hard_stop_gb must be between zero and the reserve")
    monitor = _ResourceMonitor(
        int(reserve * 2**30),
        int(controls.get("memory_check_interval_steps", 2048)),
        breach_patience=breach_patience,
        hard_floor_bytes=int(hard_floor * 2**30),
    )
    monitor.peak_rss_bytes = int(previous_summary.get("peak_rss_bytes", 0))
    session_start = time.perf_counter()
    training_seconds = float(previous_summary.get("training_seconds", 0.0))
    learn_started_at: float | None = None
    preceding_wall = float(previous_summary.get("active_wall_seconds", 0.0))
    sessions = run_path / "learning"
    sessions.mkdir(exist_ok=True)
    session_id = len(list(sessions.glob("session_*"))) + 1
    log_path = sessions / f"session_{session_id:03d}"
    console_metrics = bool(controls.get("console_metrics", False))
    log_formats = ["log", "csv", "json", "tensorboard"]
    if console_metrics:
        log_formats.insert(0, "stdout")
    model.set_logger(configure(str(log_path), log_formats))
    saved_parameters = {
        key: value
        for key, value in _algorithm_parameters(config, seed, env).items()
        if key != "env"
    }
    write_json(saved_parameters, run_path / "resolved_model_parameters.json")

    def summary(status: str, reason: str | None = None) -> dict[str, Any]:
        elapsed_training = training_seconds + (
            time.perf_counter() - learn_started_at if learn_started_at is not None else 0.0
        )
        return {
            "status": status,
            "reason": reason,
            "updated_at_utc": utc_now(),
            "stage": config.get("stage", "pilot"),
            "git_commit": git_commit(),
            "source_sha256": expected["source_sha256"],
            "algorithm": algorithm_name,
            "packages": {name: version(name) for name in PACKAGES},
            "seed": seed,
            "requested_timesteps": target,
            "rounded_target_timesteps": actual_target,
            "actual_timesteps": model.num_timesteps,
            "training_seconds": elapsed_training,
            "jit_compilation_included": True,
            "active_wall_seconds": preceding_wall + time.perf_counter() - session_start,
            "steps_per_training_second": (
                model.num_timesteps / elapsed_training if elapsed_training else None
            ),
            "peak_rss_bytes": monitor.peak_rss_bytes,
            "minimum_available_memory_bytes": monitor.minimum_available_bytes,
            "reserved_memory_gib": reserve,
            "memory_breach_patience": breach_patience,
            "memory_hard_stop_gib": hard_floor,
            "consecutive_memory_breaches": monitor.consecutive_breaches,
            "requested_threads": int(requested_threads) if requested_threads is not None else None,
            "thread_limit_enforced": False,
            "thread_policy": "jax_runtime_default",
            "progress_bar_enabled": bool(controls.get("progress_bar", False)),
            "console_metrics_enabled": console_metrics,
            "backend": "sbx_jax",
            "device": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
            "checkpoint_interval_steps": interval,
            "continuation_guarantee": "state_and_rng_restored; not_bitwise_guaranteed",
            "session": session_id,
        }

    _atomic_json(summary("running"), run_path / "training_status.json")
    checkpoint_callback: _SafeCheckpointCallback | None = None
    try:
        if checkpoint is None:
            checkpoint = _save_checkpoint(model, run_path, summary("initialized"))
        checkpoint_callback = _SafeCheckpointCallback(
            run_path,
            interval,
            int(model.num_timesteps),
            summary,
            config.get("_stop_after_chunks"),
        )
        monitor.sample()
        remaining = actual_target - int(model.num_timesteps)
        learn_started_at = time.perf_counter()
        model.learn(
            total_timesteps=remaining,
            reset_num_timesteps=False,
            callback=[monitor, checkpoint_callback],
            progress_bar=bool(controls.get("progress_bar", False)),
            log_interval=1,
        )
        training_seconds += time.perf_counter() - learn_started_at
        learn_started_at = None
        monitor.sample(enforce=False)
        checkpoint = _save_checkpoint(model, run_path, summary("checkpointed"))
        _combine_learning_logs(run_path)
        _atomic_json(summary("running"), run_path / "training_status.json")
    except _MemoryBoundaryStop as error:
        checkpoint = _save_checkpoint(model, run_path, summary("interrupted", str(error)))
        _atomic_json(summary("interrupted", str(error)), run_path / "training_status.json")
        _combine_learning_logs(run_path)
        raise TrainingInterrupted(str(error), checkpoint) from error
    except (KeyboardInterrupt, TrainingInterrupted) as error:
        # An arbitrary KeyboardInterrupt can interrupt an optimizer update. Never
        # publish that half-update: resume the previously completed checkpoint.
        checkpoint = (checkpoint_callback.latest if checkpoint_callback else None) or checkpoint
        result = summary("interrupted", str(error) or "keyboard interrupt")
        if checkpoint is not None:
            result["resume_checkpoint"] = str(checkpoint)
            checkpoint_metadata = json.loads((checkpoint / "checkpoint.json").read_text())
            result["resume_timesteps"] = checkpoint_metadata["actual_timesteps"]
        _atomic_json(result, run_path / "training_status.json")
        _combine_learning_logs(run_path)
        raise TrainingInterrupted(result["reason"], checkpoint) from error
    except Exception as error:
        _atomic_json(
            summary("failed", f"{type(error).__name__}: {error}"),
            run_path / "training_status.json",
        )
        raise
    finally:
        if learn_started_at is not None:
            training_seconds += time.perf_counter() - learn_started_at
            learn_started_at = None
        model.logger.close()

    model.save(run_path / "final_checkpoint.partial.zip")
    _replace_with_retry(
        run_path / "final_checkpoint.partial.zip", run_path / "final_checkpoint.zip"
    )
    _combine_learning_logs(run_path)
    result = summary("complete")
    _atomic_json(result, run_path / "training_summary.json")
    _atomic_json(result, run_path / "training_status.json")
    return model
