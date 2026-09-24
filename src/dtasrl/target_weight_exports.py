from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dtasrl.storage import save_table


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_table(frame, path.with_suffix(""))


def _date_value(value: Any) -> Any:
    if isinstance(value, (np.datetime64, pd.Timestamp)):
        return pd.Timestamp(value).date().isoformat()
    return value.item() if isinstance(value, np.generic) else value


def write_target_weight_run_exports(
    run_path: Path,
    trajectory: pd.DataFrame,
    split_data: dict[str, Any],
) -> dict[str, str]:
    """Write explicit RL state/action/reward files for one evaluated run.

    These are target-weight decisions. No field in these files represents share quantities.
    """

    tickers = list(split_data["tickers"])
    labels = ["CASH", *tickers]
    steps = trajectory[
        [
            "step",
            "decision_date",
            "next_observation_date",
            "reward",
            "wealth_before",
            "wealth_after",
            "turnover",
            "transaction_cost",
        ]
    ].copy()
    for optional in ("execution_date", "pre_trade_wealth", "training_reward"):
        if optional in trajectory:
            steps[optional] = trajectory[optional].to_numpy()
    steps["wealth_log_change"] = np.log(steps["wealth_after"] / steps["wealth_before"])
    steps["reward_reconciles"] = np.isclose(steps["reward"], steps["wealth_log_change"], atol=1e-12)

    action_rows: list[dict[str, Any]] = []
    for _, row in trajectory.iterrows():
        for index, label in enumerate(labels):
            suffix = "cash" if index == 0 else f"asset_{index}"
            action_rows.append(
                {
                    "step": int(row["step"]),
                    "decision_date": row["decision_date"],
                    "next_observation_date": row["next_observation_date"],
                    "asset": label,
                    "raw_action_logit": float(row[f"raw_action_{suffix}"]),
                    "environment_input_logit": float(row[f"raw_action_{suffix}"]),
                    "actor_mean_logit": float(row.get(f"actor_mean_{suffix}", np.nan)),
                    "pre_trade_weight": float(row[f"pre_trade_{suffix}"]),
                    "decision_weight": float(
                        row.get(f"decision_weight_{suffix}", row[f"pre_trade_{suffix}"])
                    ),
                    "target_weight": float(row[f"target_{suffix}"]),
                    "closing_weight": float(row[f"closing_{suffix}"]),
                    "weight_change": float(row[f"weight_change_{suffix}"]),
                    "gross_relative": float(row[f"gross_{suffix}"]),
                    "overnight_gross_relative": float(row.get(f"overnight_{suffix}", 1.0)),
                    "execution_date": row.get("execution_date", row["decision_date"]),
                    "signed_trade_value": float(row.get(f"trade_value_{suffix}", np.nan)),
                }
            )
    actions = pd.DataFrame(action_rows)

    dates = np.asarray(split_data["dates"])
    features = np.asarray(split_data["features"], dtype=float)
    feature_names = list(split_data.get("feature_names", ["feature"]))
    observation_tickers = list(split_data.get("observation_tickers", tickers))
    expected_width = len(feature_names) * len(observation_tickers)
    if features.shape[1] != expected_width:
        raise ValueError(
            f"feature width {features.shape[1]} does not match "
            f"{len(feature_names)} features x {len(observation_tickers)} observed tickers"
        )
    feature_cube = features[:-1].reshape(
        len(features) - 1, len(feature_names), len(observation_tickers)
    )
    observation_rows: list[dict[str, Any]] = []
    for step_index in range(len(trajectory)):
        for ticker_index, ticker in enumerate(observation_tickers):
            row = {
                "step": step_index,
                "decision_date": _date_value(dates[step_index]),
                "ticker": ticker,
            }
            for feature_index, feature_name in enumerate(feature_names):
                row[feature_name] = float(feature_cube[step_index, feature_index, ticker_index])
            observation_rows.append(row)
    observations = pd.DataFrame(observation_rows)
    full_observations = [c for c in trajectory if c.startswith("observation_")]
    if full_observations:
        _write_csv(trajectory[["step", "decision_date", *full_observations]],
                   run_path / "policy_observations.csv")

    target = trajectory[["decision_date"]].copy()
    delta = trajectory[["decision_date"]].copy()
    trade_value = pd.DataFrame(
        {"date": trajectory.get("execution_date", trajectory["decision_date"])}
    )
    for index, label in enumerate(labels):
        suffix = "cash" if index == 0 else f"asset_{index}"
        target[label] = trajectory[f"target_{suffix}"].to_numpy()
        delta[label] = trajectory[f"weight_change_{suffix}"].to_numpy()
        if index > 0 and f"trade_value_{suffix}" in trajectory:
            trade_value[label] = trajectory[f"trade_value_{suffix}"].to_numpy()

    dictionary = pd.DataFrame(
        [
            ("rl_steps.csv", "reward", "Net log wealth change for the decision interval."),
            ("rl_steps.csv", "turnover", "Absolute risky trade value divided by pre-trade wealth."),
            (
                "rl_actions.csv",
                "raw_action_logit",
                "Legacy name for clipped environment-input logit.",
            ),
            (
                "rl_actions.csv",
                "actor_mean_logit",
                "Unclipped Gaussian actor mean; NaN in older runs.",
            ),
            ("rl_actions.csv", "pre_trade_weight", "Weight immediately before rebalancing."),
            ("rl_actions.csv", "target_weight", "Long-only weight after softmax and halt rules."),
            (
                "rl_actions.csv",
                "weight_change",
                "Target weight minus pre-trade weight; not shares.",
            ),
            ("rl_actions.csv", "closing_weight", "Weight after the next-period price relative."),
            ("rl_observations.csv", "feature columns", "Causal features known at decision time."),
            ("target_weights.csv", "asset columns", "Date-by-asset target-weight matrix."),
            ("weight_changes.csv", "asset columns", "Date-by-asset signed weight-change matrix."),
            (
                "trade_values.csv",
                "ticker columns",
                "Execution date by signed traded currency amount, not shares.",
            ),
        ],
        columns=["file", "field", "definition"],
    )
    files = {
        "rl_steps": run_path / "rl_steps.csv",
        "rl_actions": run_path / "rl_actions.csv",
        "rl_observations": run_path / "rl_observations.csv",
        "target_weights": run_path / "target_weights.csv",
        "weight_changes": run_path / "weight_changes.csv",
        "data_dictionary": run_path / "data_dictionary.csv",
    }
    for frame, key in (
        (steps, "rl_steps"),
        (actions, "rl_actions"),
        (observations, "rl_observations"),
        (target, "target_weights"),
        (delta, "weight_changes"),
        (dictionary, "data_dictionary"),
    ):
        _write_csv(frame, files[key])
    if len(trade_value.columns) > 1:
        files["trade_values"] = run_path / "trade_values.csv"
        _write_csv(trade_value, files["trade_values"])
        trades = actions.loc[actions["asset"] != "CASH"].copy()
        trades["side"] = np.where(
            trades["signed_trade_value"] > 1e-8,
            "BUY",
            np.where(trades["signed_trade_value"] < -1e-8, "SELL", "HOLD"),
        )
        files["trades"] = run_path / "trades.csv"
        _write_csv(trades, files["trades"])
    return {key: str(path) for key, path in files.items()}
