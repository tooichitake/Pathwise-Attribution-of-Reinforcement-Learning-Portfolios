"""Matched study orchestration. Experiments only read frozen processed inputs."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dtasrl.attribution import evaluate_target_path, freeze_k_path, two_by_two_attribution
from dtasrl.data.panel import prepare_experiment_data
from dtasrl.envs.portfolio import TargetWeightPortfolioEnv
from dtasrl.metrics import performance_metrics
from dtasrl.provenance import initialise_run, write_json
from dtasrl.storage import inventory, save_table, snapshot_code
from dtasrl.target_weight_exports import write_target_weight_run_exports
from dtasrl.training import train_model


def make_env(data: dict, config: dict, *, training: bool = False):
    return TargetWeightPortfolioEnv(
        prices=data["prices"],
        features=data["features"],
        execution_prices=data.get("opens"),
        tradable=data.get("tradable"),
        forced_liquidation=data.get("forced_liquidation"),
        transaction_cost_bps=float(config.get("transaction_cost_bps", 10)),
        temperature=float(config.get("temperature", 1)),
        initial_wealth=float(config.get("initial_wealth", 1_000_000)),
        reward_scale=float(config.get("reward_scale", 100)),
        record_history=not training,
    )


def evaluate_model(model, data: dict, config: dict) -> pd.DataFrame:
    env = make_env(data, config)
    observation, _ = env.reset()
    done = False
    rows = []
    while not done:
        actor_mean = np.asarray(model.policy._predict(observation[None, :], deterministic=True))[0]
        action, _ = model.predict(observation, deterministic=True)
        observation, _, done, truncated, _ = env.step(action)
        if truncated:
            raise RuntimeError("Unexpected evaluation truncation")
        rec = env.records[-1]
        t = rec.time_index
        row = {
            "step": t,
            "decision_date": data["dates"][t],
            "execution_date": data["dates"][t + 1],
            "next_observation_date": data["dates"][t + 1],
            "wealth_before": rec.wealth_before,
            "pre_trade_wealth": rec.pre_trade_wealth,
            "wealth_after": rec.wealth_after,
            "reward": rec.reward,
            "training_reward": rec.scaled_reward,
            "transaction_cost": rec.transaction_cost,
            "turnover": rec.turnover,
        }
        # Full float32 observation actually supplied to the policy, including holdings.
        for i, value in enumerate(
            np.r_[rec.decision_weights, data["features"][t]].astype(np.float32)
        ):
            row[f"observation_{i}"] = float(value)
        for i in range(len(rec.target_weights)):
            label = "cash" if i == 0 else f"asset_{i}"
            for prefix, values in (
                ("raw_action", rec.raw_action),
                ("actor_mean", actor_mean),
                ("decision_weight", rec.decision_weights),
                ("pre_trade", rec.pre_trade_weights),
                ("target", rec.target_weights),
                ("closing", rec.closing_weights),
                ("gross", rec.gross_relatives),
                ("overnight", rec.overnight_gross_relatives),
            ):
                row[f"{prefix}_{label}"] = float(values[i])
            row[f"weight_change_{label}"] = float(rec.target_weights[i] - rec.pre_trade_weights[i])
            if i:
                row[f"trade_value_{label}"] = float(rec.trade_values[i - 1])
        rows.append(row)
    return pd.DataFrame(rows)


def _arrays(frame: pd.DataFrame, n: int):
    labels = ["cash", *[f"asset_{i}" for i in range(1, n + 1)]]
    return tuple(
        frame[[f"{prefix}_{a}" for a in labels]].to_numpy()
        for prefix in ("target", "gross", "overnight")
    )


def path_metrics(path):
    values = performance_metrics(path.wealth, path.target_weights, path.turnover)
    values["total_transaction_cost"] = float(path.costs.sum())
    return values


def audit_run_artifacts(output: Path, config: dict) -> dict:
    """Refuse to publish a run that lacks paper-relevant reproducibility artifacts."""
    exact = [
        "config.json",
        "metadata.json",
        "source.zip",
        "dependencies.json",
        "input_provenance.json",
        "training_identity.json",
        "training_status.json",
        "training_summary.json",
        "resolved_model_parameters.json",
        "progress.csv",
        "final_checkpoint.zip",
        "model.zip",
        "evaluation_status.json",
        "metrics.json",
        "validation_trajectory.csv",
        "validation_trajectory.parquet",
        "trajectory.csv",
        "trajectory.parquet",
        "rl_steps.csv",
        "rl_steps.parquet",
        "rl_actions.csv",
        "rl_actions.parquet",
        "rl_observations.csv",
        "rl_observations.parquet",
        "policy_observations.csv",
        "policy_observations.parquet",
        "target_weights.csv",
        "target_weights.parquet",
        "weight_changes.csv",
        "weight_changes.parquet",
        "trade_values.csv",
        "trade_values.parquet",
        "trades.csv",
        "trades.parquet",
        "data_dictionary.csv",
        "data_dictionary.parquet",
        "controls/main/summary.json",
        "controls/main/full_ppo.csv",
        "controls/main/full_ppo.parquet",
        "controls/main/cost_sensitivity.csv",
        "controls/main/cost_sensitivity.parquet",
    ]
    if config.get("kind") != "synthetic":
        exact.append("training_scaler.json")
    if config.get("kind") == "synthetic":
        exact.extend(["synthetic_market_metrics.csv", "synthetic_market_metrics.parquet"])
    missing = [name for name in exact if not (output / name).is_file()]
    sessions = sorted((output / "learning").glob("session_*"))
    session_missing = []
    for session in sessions:
        for pattern in ("progress.csv", "progress.json", "log.txt"):
            if not (session / pattern).is_file():
                session_missing.append((session / pattern).relative_to(output).as_posix())
        if not list(session.glob("*tfevents*")):
            session_missing.append(f"{session.relative_to(output).as_posix()}/*tfevents*")
    checkpoints = sorted((output / "checkpoints").glob("step_*"))
    checkpoint_missing = []
    for checkpoint in checkpoints:
        for name in ("model.zip", "state.pkl", "checkpoint.json"):
            if not (checkpoint / name).is_file():
                checkpoint_missing.append((checkpoint / name).relative_to(output).as_posix())
    if missing or session_missing or checkpoint_missing or not sessions or not checkpoints:
        raise RuntimeError(
            "paper artifact audit failed: "
            f"missing={missing}; session_missing={session_missing}; "
            f"checkpoint_missing={checkpoint_missing}; sessions={len(sessions)}; "
            f"checkpoints={len(checkpoints)}"
        )
    training = pd.read_csv(output / "progress.csv")
    validation = pd.read_parquet(output / "validation_trajectory.parquet")
    test = pd.read_parquet(output / "trajectory.parquet")
    actions = pd.read_parquet(output / "rl_actions.parquet")
    observations = pd.read_parquet(output / "rl_observations.parquet")
    audit = {
        "status": "complete",
        "paper_ready": True,
        "training_detail_granularity": "one row per PPO rollout/update",
        "evaluation_detail_granularity": (
            "one row per environment step plus long-form actions and observations"
        ),
        "training_rollout_rows": len(training),
        "validation_steps": len(validation),
        "test_steps": len(test),
        "test_action_rows": len(actions),
        "test_observation_rows": len(observations),
        "checkpoint_count": len(checkpoints),
        "training_session_count": len(sessions),
        "console_metrics_enabled": bool(config.get("training", {}).get("console_metrics", False)),
        "progress_bar_enabled": bool(config.get("training", {}).get("progress_bar", False)),
        "text_log_saved": True,
        "csv_log_saved": True,
        "json_log_saved": True,
        "tensorboard_log_saved": True,
        "full_training_step_observations_saved": False,
        "full_training_step_observations_required": False,
        "figures_regenerable_from_saved_tables": True,
    }
    write_json(audit, output / "artifact_audit.json")
    return audit


def evaluate_controls(
    frame: pd.DataFrame,
    validation: pd.DataFrame,
    config: dict,
    output: Path,
    split_name: str = "main",
) -> dict:
    frame = frame.reset_index(drop=True)
    n = len(config["tickers"])
    target, gross, overnight = _arrays(frame, n)
    _, valid_gross, valid_overnight = _arrays(validation, n)
    capital = float(config.get("initial_wealth", 1_000_000))
    rate = float(config.get("transaction_cost_bps", 10))

    def replay(w, g=gross, o=overnight, cost=rate):
        return evaluate_target_path(w, g, cost, initial_wealth=capital, overnight_gross_relatives=o)

    paths = {"full_ppo": replay(target)}
    paths["initial_weight_rebalance"] = replay(np.repeat(target[:1], len(target), axis=0))
    for k in (1, 5, 20, 63):
        if k <= len(frame):
            paths[f"freeze_{k}"] = freeze_k_path(
                target, gross, k, rate, overnight_gross_relatives=overnight, initial_wealth=capital
            )
    equal = np.tile(np.r_[0.0, np.full(n, 1 / n)], (len(target), 1))
    paths["equal_weight_hold"] = freeze_k_path(
        equal, gross, 1, rate, overnight_gross_relatives=overnight, initial_wealth=capital
    )
    cash = np.zeros_like(target)
    cash[:, 0] = 1
    paths["cash"] = replay(cash)
    # Match initial PPO cash, but distribute its risky exposure equally.
    exposure_equal = equal * (1 - target[0, 0])
    exposure_equal[:, 0] = target[0, 0]
    paths["initial_exposure_equal_weight_hold"] = freeze_k_path(
        exposure_equal, gross, 1, rate, overnight_gross_relatives=overnight, initial_wealth=capital
    )
    if n == 1:
        scores = {}
        for e in (0.0, 0.25, 0.5, 0.75, 1.0):
            w = np.tile([1 - e, e], (len(validation), 1))
            scores[e] = replay(w, valid_gross, valid_overnight).net_log_wealth
        chosen = max(scores, key=scores.get)
        paths["validation_fixed_exposure"] = replay(np.tile([1 - chosen, chosen], (len(frame), 1)))
        mean = float(target[:, 1].mean())
        paths["ex_post_mean_exposure"] = replay(np.tile([1 - mean, mean], (len(frame), 1)))
    attribution = two_by_two_attribution(
        target, gross, rate, overnight_gross_relatives=overnight, initial_wealth=capital
    )
    effects = {
        key: float(value)
        for key, value in attribution.items()
        if isinstance(value, (float, int, np.floating))
    }
    paths.update({key: value for key, value in attribution.items() if hasattr(value, "wealth")})
    if not np.allclose(paths["full_ppo"].wealth[1:], frame["wealth_after"], rtol=1e-10):
        raise AssertionError("Replay does not reproduce evaluated policy")
    destination = output / "controls" / split_name
    destination.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        data = pd.DataFrame(
            {
                "decision_date": frame["decision_date"],
                "execution_date": frame["execution_date"],
                "wealth_before": path.wealth[:-1],
                "wealth_after": path.wealth[1:],
                "transaction_cost": path.costs,
                "turnover": path.turnover,
            }
        )
        for i, ticker in enumerate(["CASH", *config["tickers"]]):
            data[f"target_{ticker}"] = path.target_weights[:, i]
            data[f"pre_trade_{ticker}"] = path.pre_trade_weights[:, i]
            data[f"closing_{ticker}"] = path.closing_weights[:, i]
            if i:
                data[f"trade_value_{ticker}"] = path.trade_values[:, i - 1]
        data["reward"] = np.log(path.wealth[1:] / path.wealth[:-1])
        save_table(data, destination / name)
    sensitivity = []
    for bps in (0, 5, 10, 25):
        for name, w in (("full_ppo", target), ("freeze_1", target), ("equal_weight_hold", equal)):
            path = (
                replay(w, cost=bps)
                if name == "full_ppo"
                else freeze_k_path(
                    w, gross, 1, bps, overnight_gross_relatives=overnight, initial_wealth=capital
                )
            )
            sensitivity.append(
                {
                    "strategy": name,
                    "cost_bps": bps,
                    "interpretation": "fixed_decisions_no_retraining",
                    **path_metrics(path),
                }
            )
            save_table(
                pd.DataFrame(
                    {
                        "decision_date": frame["decision_date"],
                        "execution_date": frame["execution_date"],
                        "wealth_before": path.wealth[:-1],
                        "wealth_after": path.wealth[1:],
                        "cost": path.costs,
                        "turnover": path.turnover,
                    }
                ),
                destination / "cost_paths" / f"{name}_{bps}bps",
            )
    save_table(pd.DataFrame(sensitivity), destination / "cost_sensitivity")
    effects["dynamic_trading_after_first_allocation"] = (
        paths["full_ppo"].net_log_wealth - paths["freeze_1"].net_log_wealth
    )
    results = {
        "metrics": {k: path_metrics(v) for k, v in paths.items()},
        "effects": effects,
        "attribution_is_ex_post": True,
    }
    if n == 1:
        results["validation_selected_exposure"] = chosen
        effects["single_asset_timing"] = (
            paths["full_ppo"].net_log_wealth - paths["ex_post_mean_exposure"].net_log_wealth
        )
    write_json(results, destination / "summary.json")
    return results


def run_experiment(config: dict, seed: int) -> Path:
    from dtasrl.provenance import run_directory

    try:
        return _run_experiment(config, seed)
    except (Exception, KeyboardInterrupt) as exc:
        output = run_directory(config, seed)
        status = output / "evaluation_status.json"
        if status.exists() and json.loads(status.read_text())["status"] == "running":
            write_json({"status": "interrupted", "reason": str(exc)}, status)
        raise


def _run_experiment(config: dict, seed: int) -> Path:
    started = time.perf_counter()
    if str(config.get("algorithm", "ppo")).lower() != "ppo":
        raise ValueError("The portfolio attribution study is a PPO-only protocol")
    if config.get("kind") == "synthetic":
        from dtasrl.data.inputs import load_synthetic_inputs
        from dtasrl.envs.calibration import make_calibration_train_env

        prepared, data_hash = load_synthetic_inputs(config)
        train_env = make_calibration_train_env(config)
    else:
        prepared, data_hash = prepare_experiment_data(config)
        train_env = make_env(prepared["train"], config, training=True)
    config = dict(config)
    config["_data_hash"] = data_hash
    output, _ = initialise_run(config, seed, data_hash)
    snapshot_code(output)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output / "run.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    write_json(
        prepared.get("provenance", prepared.get("metadata", {})), output / "input_provenance.json"
    )
    if "scaler" in prepared:
        write_json(prepared["scaler"], output / "training_scaler.json")
    status_file = output / "training_status.json"
    trained = status_file.exists() and json.loads(status_file.read_text())["status"] == "complete"
    setup_seconds = time.perf_counter() - started
    if config.get("_resume") and trained:
        if (output / "manifest.json").exists():
            raise FileExistsError("Completed evaluation must not be overwritten")
        from dtasrl.training import PPO

        model = PPO.load(output / "final_checkpoint.zip", device="cpu")
    else:
        model = train_model(config, seed, train_env, output)
    model.save(output / "model")
    evaluation_start = time.perf_counter()
    write_json({"status": "running"}, output / "evaluation_status.json")
    validation = evaluate_model(model, prepared["validation"], config)
    save_table(validation, output / "validation_trajectory")
    write_target_weight_run_exports(output / "validation", validation, prepared["validation"])
    test = evaluate_model(model, prepared["test"], config)
    save_table(test, output / "trajectory")
    write_target_weight_run_exports(output, test, prepared["test"])
    main = test
    blocks = []
    if config.get("kind") != "synthetic":
        dates = pd.to_datetime(test["execution_date"])
        main = test.loc[dates <= "2025-12-31"]
        for name, start, end in (
            ("main", "2020-01-01", "2025-12-31"),
            ("2020_2021", "2020-01-01", "2021-12-31"),
            ("2022_2023", "2022-01-01", "2023-12-31"),
            ("2024_2025", "2024-01-01", "2025-12-31"),
            ("extension_2026", "2026-01-01", "2026-08-31"),
        ):
            part = test.loc[dates.between(start, end)]
            if len(part):
                blocks.append(
                    {
                        "period": name,
                        "steps": len(part),
                        "net_log_wealth": float(part["reward"].sum()),
                        "continuous_holdings": True,
                    }
                )
    controls = evaluate_controls(main, validation, config, output)
    if config.get("kind") == "synthetic":
        evaluate_synthetic_paths(model, config, output)
    save_table(pd.DataFrame(blocks), output / "period_metrics")
    result = {
        "status": "complete",
        "stage": config.get("stage"),
        "seed": seed,
        "data_hash": data_hash,
        "actual_transitions": int(model.num_timesteps),
        "total_wall_seconds": time.perf_counter() - started,
        "setup_seconds": setup_seconds,
        "evaluation_export_seconds": time.perf_counter() - evaluation_start,
        "resumed_evaluation": bool(config.get("_resume") and trained),
        "controls": controls,
        "formal_inference_allowed": False if config.get("stage") == "benchmark" else None,
    }
    write_json(result, output / "metrics.json")
    write_json({"status": "complete"}, output / "evaluation_status.json")
    audit_run_artifacts(output, config)
    inventory(output)
    return output


def evaluate_synthetic_paths(model, config: dict, output: Path) -> None:
    """Independent common test markets; no policy fitting or selection on these paths."""
    from dtasrl.data.inputs import load_synthetic_market
    from dtasrl.envs.calibration import calibration_settings

    settings = calibration_settings(config)
    count = int(config.get("evaluation", {}).get("test_paths", 100))
    if count < 1:
        raise ValueError("evaluation.test_paths must be positive")
    rows = []
    folder = output / "synthetic_test_paths"
    folder.mkdir(exist_ok=True)
    for market_seed in range(settings.test_market_seed, settings.test_market_seed + count):
        data = load_synthetic_market(config, market_seed)
        frame = evaluate_model(model, data, config)
        save_table(frame, folder / f"market_{market_seed}" / "trajectory")
        write_target_weight_run_exports(folder / f"market_{market_seed}", frame, data)
        target, gross, overnight = _arrays(frame, 1)
        risky = data["reference_risky_weight"][:-1]
        paths = {
            "full_ppo": target,
            "known_signal_rule_not_cost_optimal": np.column_stack([1 - risky, risky]),
            "cash": np.tile([1.0, 0.0], (len(frame), 1)),
            "buy_and_hold": np.tile([0.0, 1.0], (len(frame), 1)),
            "ex_post_mean_exposure": np.tile(target.mean(axis=0), (len(frame), 1)),
        }
        for name, weights in paths.items():
            kwargs = dict(
                initial_wealth=float(config.get("initial_wealth", 1_000_000)),
                overnight_gross_relatives=overnight,
            )
            cost = float(config.get("transaction_cost_bps", 10))
            path = (
                freeze_k_path(weights, gross, 1, cost, **kwargs)
                if name == "buy_and_hold"
                else evaluate_target_path(weights, gross, cost, **kwargs)
            )
            rows.append(
                {
                    "market_seed": market_seed,
                    "strategy": name,
                    "net_log_wealth": path.net_log_wealth,
                    "mean_risky_exposure": float(path.target_weights[:, 1].mean()),
                    "total_transaction_cost": float(path.costs.sum()),
                }
            )
            path_frame = pd.DataFrame(
                {
                    "step": frame["step"],
                    "wealth_before": path.wealth[:-1],
                    "wealth_after": path.wealth[1:],
                    "cost": path.costs,
                    "turnover": path.turnover,
                    "target_CASH": path.target_weights[:, 0],
                    "target_SYNTH": path.target_weights[:, 1],
                }
            )
            path_frame["pre_trade_CASH"] = path.pre_trade_weights[:, 0]
            path_frame["pre_trade_SYNTH"] = path.pre_trade_weights[:, 1]
            path_frame["closing_CASH"] = path.closing_weights[:, 0]
            path_frame["closing_SYNTH"] = path.closing_weights[:, 1]
            path_frame["trade_value_SYNTH"] = path.trade_values[:, 0]
            save_table(path_frame, folder / f"market_{market_seed}" / "controls" / name)
    save_table(pd.DataFrame(rows), output / "synthetic_market_metrics")
