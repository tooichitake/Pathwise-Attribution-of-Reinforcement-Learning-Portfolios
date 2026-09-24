"""Local tables and figures shared by the four current notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter

from dtasrl.config import PROJECT_ROOT, load_config, public_config
from dtasrl.provenance import run_directory, sha256_file, write_json
from dtasrl.storage import inventory, save_table, verify_inventory

GROUPS = {
    "01_synthetic": ("synthetic_high", "synthetic_zero"),
    "02_single_asset": ("single_msft", "single_jpm", "single_jnj"),
    "03_multi_asset": ("multi_asset",),
}
PALETTE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#6F4E7C",
    "#949494",
    "#000000",
)
BACKGROUND = "#F7F8FA"
GRID = "#D9DEE7"
TEXT = "#263238"
EXPORT_DPI = 320
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.dpi": 130,
        "figure.facecolor": "white",
        "axes.facecolor": BACKGROUND,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.75,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    }
)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pretty(label: str) -> str:
    acronyms = {"ppo": "PPO", "msft": "MSFT", "jpm": "JPM", "jnj": "JNJ", "cash": "Cash"}
    return " ".join(acronyms.get(word.lower(), word.title()) for word in str(label).split("_"))


def figure(
    frame,
    directory,
    name,
    *,
    title,
    ylabel,
    x,
    columns,
    kind="line",
    percent=False,
):
    directory.mkdir(parents=True, exist_ok=True)
    save_table(frame, directory / name)
    fig, ax = plt.subplots(figsize=(11.2, 5.4), layout="constrained")
    if kind == "heatmap":
        values = frame[columns].to_numpy().T
        bound = max(float(np.nanmax(np.abs(values))), 1e-12)
        image = ax.imshow(
            values,
            aspect="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
            interpolation="nearest",
        )
        ax.set_yticks(range(len(columns)), [_pretty(c) for c in columns])
        positions = np.unique(np.linspace(0, len(frame) - 1, min(7, len(frame))).astype(int))
        try:
            labels = pd.to_datetime(frame[x]).astype(str).iloc[positions]
        except (TypeError, ValueError):
            labels = frame[x].astype(str).iloc[positions]
        ax.set_xticks(positions, labels, rotation=25, ha="right")
        ax.grid(False)
        fig.colorbar(image, ax=ax, label=ylabel)
    elif kind == "bar":
        frame.set_index(x)[columns].plot.bar(
            ax=ax, rot=22, color=PALETTE[: len(columns)], width=0.78
        )
        ax.axhline(0, color=TEXT, linewidth=0.8)
    elif kind == "barh":
        values = frame[columns[0]].to_numpy()
        colors = np.where(values >= 0, "#009E73", "#D55E00")
        ax.barh([_pretty(v) for v in frame[x]], values, color=colors, alpha=0.9)
        ax.axvline(0, color=TEXT, linewidth=0.9)
        ax.invert_yaxis()
    elif kind == "area":
        values = frame[columns].to_numpy().T
        ax.stackplot(
            frame[x],
            values,
            labels=[_pretty(c) for c in columns],
            colors=PALETTE[: len(columns)],
            alpha=0.9,
            linewidth=0.2,
        )
        ax.set_ylim(0, 1)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=min(4, len(columns)), fontsize=8
        )
    else:
        for index, column in enumerate(columns):
            highlight = column == "full_ppo" or index == 0
            ax.plot(
                frame[x],
                frame[column],
                label=_pretty(column),
                color=PALETTE[index % len(PALETTE)],
                linewidth=2.5 if highlight else 1.35,
                alpha=1 if highlight else 0.78,
            )
        if len(columns) > 1:
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.14),
                fontsize=8,
                ncol=min(4, len(columns)),
            )
        if pd.api.types.is_datetime64_any_dtype(frame[x]):
            locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel(_pretty(x))
    ax.set_ylabel(ylabel)
    ax.margins(x=0.01)
    fig.text(
        0.995,
        0.004,
        "DTASRL research artifact",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#7A8491",
    )
    fig.savefig(directory / f"{name}.png", dpi=EXPORT_DPI)
    plt.close(fig)
    write_json(
        {
            "title": title,
            "ylabel": ylabel,
            "x": x,
            "columns": columns,
            "kind": kind,
            "percent": percent,
            "palette": list(PALETTE),
            "export_dpi": EXPORT_DPI,
            "generator": "dtasrl.study_reports.figure",
            "generator_sha256": sha256_file(Path(__file__)),
        },
        directory / f"{name}.json",
    )
    return directory / f"{name}.png"


def status_figure(status: pd.DataFrame, directory: Path) -> Path:
    """Compact condition-by-seed progress map for human-readable notebooks."""
    directory.mkdir(parents=True, exist_ok=True)
    save_table(status, directory / "run_status")
    states = ["failed", "interrupted", "running", "not_started", "complete"]
    score = {name: index for index, name in enumerate(states)}
    matrix = status.assign(value=status.status.map(score).fillna(0)).pivot(
        index="condition", columns="seed", values="value"
    )
    cmap = LinearSegmentedColormap.from_list(
        "study_status", ["#D55E00", "#E69F00", "#56B4E9", "#D9DEE7", "#009E73"], N=5
    )
    fig, ax = plt.subplots(figsize=(7.8, max(2.7, 0.62 * len(matrix))), layout="constrained")
    ax.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=4)
    ax.set_xticks(range(len(matrix.columns)), [str(v) for v in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), [_pretty(v) for v in matrix.index])
    ax.set_xlabel("Seed")
    ax.set_ylabel("")
    ax.set_title("Formal experiment progress", loc="left", pad=12)
    ax.grid(False)
    for row, condition in enumerate(matrix.index):
        for col, seed in enumerate(matrix.columns):
            value = status.loc[
                (status.condition == condition) & (status.seed == seed), "status"
            ].iloc[0]
            ax.text(
                col,
                row,
                value.replace("_", "\n"),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value in {"complete", "failed"} else TEXT,
            )
    path = directory / "run_status.png"
    fig.savefig(path, dpi=EXPORT_DPI)
    plt.close(fig)
    write_json(
        {
            "states": states,
            "palette": list(PALETTE),
            "export_dpi": EXPORT_DPI,
            "generator_sha256": sha256_file(Path(__file__)),
        },
        directory / "run_status.json",
    )
    return path


def data_report():
    config = load_config(PROJECT_ROOT / "configs/experiments/multi_asset.yaml")
    meta = _json(PROJECT_ROOT / config["data"]["snapshot"])
    out = PROJECT_ROOT / "outputs/00_data"
    rows = []
    for name, layer in meta["layers"].items():
        csv, pq = PROJECT_ROOT / layer["csv"], PROJECT_ROOT / layer["parquet"]
        assert sha256_file(csv) == layer["csv_sha256"]
        assert sha256_file(pq) == layer["parquet_sha256"]
        stored = pd.read_parquet(pq)
        parsed = pd.read_csv(csv, parse_dates=["date"], float_precision="round_trip")
        pd.testing.assert_frame_equal(stored, parsed, check_dtype=False, check_exact=True)
        rows.append(
            {
                "layer": name,
                "rows": len(stored),
                "first_date": str(stored.date.min()),
                "last_date": str(stored.date.max()),
                "csv": str(csv),
                "parquet": str(pq),
                "sha256": layer["parquet_sha256"],
                "row_equality": True,
            }
        )
    inputs = PROJECT_ROOT / config["data"]["frozen_inputs"]
    verify_inventory(inputs)
    panel = pd.read_parquet(PROJECT_ROOT / meta["layers"]["processed"]["parquet"])
    assert panel.groupby("date").ticker.nunique().eq(30).all()
    assert panel.date.min() == pd.Timestamp("2008-03-19")
    assert panel.date.max() == pd.Timestamp("2026-08-31")
    coverage = (
        panel.groupby("ticker")
        .agg(rows=("date", "size"), first_date=("date", "min"), last_date=("date", "max"))
        .reset_index()
    )
    missing = (
        panel[list(meta["feature_order"])]
        .isna()
        .sum()
        .rename_axis("feature")
        .reset_index(name="missing_rows")
    )
    save_table(pd.DataFrame(rows), out / "input_inventory")
    save_table(coverage, out / "coverage")
    save_table(missing, out / "feature_missingness")
    write_json(meta, out / "source_metadata.json")
    write_json(_json(inputs / "scaler.json"), out / "training_scaler.json")
    prices = panel.loc[panel.ticker.isin(config["tickers"])].pivot(
        index="date", columns="ticker", values="close"
    )
    prices = prices / prices.iloc[0]
    pics = [
        figure(
            prices.reset_index(),
            out / "figures",
            "prices",
            title="Adjusted prices of the three study stocks",
            ylabel="Price / first price",
            x="date",
            columns=config["tickers"],
        ),
        figure(
            missing,
            out / "figures",
            "missingness",
            title="Feature warm-up missing values",
            ylabel="Rows",
            x="feature",
            columns=["missing_rows"],
            kind="bar",
        ),
    ]
    inventory(out)
    return {
        "inputs": pd.DataFrame(rows),
        "coverage": coverage,
        "missing": missing,
        "figures": pics,
        "out": out,
    }


def run_figures(run, destination):
    config = _json(run / "config.json")
    trajectory = pd.read_parquet(run / "trajectory.parquet")
    files, metrics = [], _json(run / "controls/main/summary.json")
    wealth = pd.DataFrame({"execution_date": trajectory.execution_date})
    # Control paths cover the main period; the full 2026 extension remains in the run ledger.
    controls = []
    for name in (
        "full_ppo",
        "freeze_1",
        "freeze_5",
        "freeze_20",
        "freeze_63",
        "equal_weight_hold",
        "validation_fixed_exposure",
        "ex_post_mean_exposure",
        "initial_weight_rebalance",
    ):
        path = run / "controls/main" / f"{name}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            controls.append(
                frame.set_index("execution_date").wealth_after.rename(name)
                / frame.wealth_before.iloc[0]
            )
    wealth = pd.concat(controls, axis=1).reset_index()
    files.append(
        figure(
            wealth,
            destination,
            "wealth",
            title=f"Net wealth paths | {config['stage']}",
            ylabel="Wealth / initial wealth",
            x="execution_date",
            columns=list(wealth.columns[1:]),
        )
    )
    weights = pd.read_parquet(run / "target_weights.parquet")
    files.append(
        figure(
            weights,
            destination,
            "weights",
            title="Policy target weights",
            ylabel="Portfolio weight",
            x="decision_date",
            columns=["CASH", *config["tickers"]],
            kind="area",
            percent=True,
        )
    )
    changes = pd.read_parquet(run / "weight_changes.parquet")
    files.append(
        figure(
            changes,
            destination,
            "weight_changes",
            title="Executed weight changes by asset",
            ylabel="Weight change",
            x="decision_date",
            columns=config["tickers"],
            kind="heatmap",
        )
    )
    turnover = trajectory[["execution_date", "turnover"]].copy()
    turnover["21_day_mean"] = turnover.turnover.rolling(21, min_periods=1).mean()
    files.append(
        figure(
            turnover,
            destination,
            "turnover",
            title="Daily turnover and 21-day mean",
            ylabel="Share of pre-trade wealth",
            x="execution_date",
            columns=["turnover", "21_day_mean"],
            percent=True,
        )
    )
    effects = pd.DataFrame(
        [{"effect": k, "net_log_wealth_difference": v} for k, v in metrics["effects"].items()]
    )
    if len(effects):
        files.append(
            figure(
                effects,
                destination,
                "effects",
                title="Pathwise contributions | ex-post attribution",
                ylabel="Net log wealth difference",
                x="effect",
                columns=["net_log_wealth_difference"],
                kind="barh",
            )
        )
    two = []
    for path in sorted((run / "controls/main").glob("*.parquet")):
        if any(key in path.stem for key in ("dynamic", "mean_exposure_mean_composition")):
            frame = pd.read_parquet(path)
            if "wealth_after" in frame:
                two.append(
                    frame.set_index("execution_date").wealth_after.rename(path.stem)
                    / frame.wealth_before.iloc[0]
                )
    if two:
        f = pd.concat(two, axis=1).reset_index()
        files.append(
            figure(
                f,
                destination,
                "attribution_paths",
                title="Exposure and composition replay paths",
                ylabel="Wealth / initial wealth",
                x="execution_date",
                columns=list(f.columns[1:]),
            )
        )
    progress = run / "progress.csv"
    if progress.exists():
        f = pd.read_csv(progress)
        columns = [c for c in ("train/value_loss", "train/policy_gradient_loss") if c in f]
        if columns:
            f = f.dropna(subset=columns).drop_duplicates("time/total_timesteps", keep="last")
            files.append(
                figure(
                    f[["time/total_timesteps", *columns]],
                    destination,
                    "training",
                    title="PPO optimization diagnostics",
                    ylabel="Logged loss",
                    x="time/total_timesteps",
                    columns=columns,
                )
            )
    if config["kind"] == "synthetic":
        market = pd.read_parquet(run / "synthetic_market_metrics.parquet")
        summary = market.groupby("strategy", as_index=False).net_log_wealth.mean()
        files.append(
            figure(
                summary,
                destination,
                "synthetic_comparison",
                title="Mean across fixed synthetic test markets",
                ylabel="Net log wealth",
                x="strategy",
                columns=["net_log_wealth"],
                kind="bar",
            )
        )
        obs = pd.read_parquet(run / "policy_observations.parquet")
        sample = trajectory.head(150)[["step", "target_asset_1"]].copy()
        sample["observed_signal"] = obs.iloc[: len(sample), -1].to_numpy()
        sample["reference_risky_weight"] = (
            (sample.observed_signal > 0).astype(float)
            if config["synthetic"]["signal_strength"]
            else 0.0
        )
        files.append(
            figure(
                sample,
                destination,
                "signal_and_exposure",
                title="Observed signal and agent exposure",
                ylabel="Signal or risky weight",
                x="step",
                columns=["observed_signal", "target_asset_1", "reference_risky_weight"],
            )
        )
    inventory(destination)
    return files


def experiment_report(group):
    out = PROJECT_ROOT / "outputs" / group
    (out / "runs").mkdir(parents=True, exist_ok=True)
    rows, summaries, detail_pics = [], [], []
    configs = []
    for condition in GROUPS[group]:
        config = load_config(PROJECT_ROOT / "configs/experiments" / f"{condition}.yaml")
        configs.append(public_config(config))
        verify_inventory(PROJECT_ROOT / config["data"]["frozen_inputs"])
        for seed in (101, 202, 303):
            run = run_directory(config, seed)
            complete = (run / "manifest.json").exists() and (run / "metrics.json").exists()
            state = "complete" if complete else "not_started"
            if not complete and (run / "training_status.json").exists():
                state = _json(run / "training_status.json")["status"]
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "stage": config["stage"],
                    "status": state,
                    "run": run.relative_to(PROJECT_ROOT).as_posix(),
                }
            )
            if complete:
                verify_inventory(run)
                result = _json(run / "metrics.json")
                if result["actual_transitions"] != 5_001_216:
                    raise ValueError("A primary result has an incomplete step budget")
                for strategy, values in result["controls"]["metrics"].items():
                    summaries.append(
                        {"condition": condition, "seed": seed, "strategy": strategy, **values}
                    )
                detail_pics.extend(run_figures(run, out / "figures" / condition / str(seed)))
    status = pd.DataFrame(rows)
    summary = pd.DataFrame(
        summaries,
        columns=None if summaries else ["condition", "seed", "strategy", "net_log_wealth"],
    )
    save_table(status, out / "status")
    save_table(summary, out / "summary")
    overview = out / "figures" / "overview"
    pics = [status_figure(status, overview)]
    if len(summary):
        preferred = [
            "full_ppo",
            "equal_weight_hold",
            "validation_fixed_exposure",
            "ex_post_mean_exposure",
            "initial_weight_rebalance",
            "freeze_1",
            "freeze_63",
        ]
        aggregate = (
            summary.loc[summary.strategy.isin(preferred)]
            .groupby(["condition", "strategy"], as_index=False)
            .net_log_wealth.mean()
        )
        if len(aggregate):
            pivot = aggregate.pivot(index="condition", columns="strategy", values="net_log_wealth")
            ordered = [name for name in preferred if name in pivot]
            plot_data = pivot.reindex(columns=ordered).reset_index()
            pics.append(
                figure(
                    plot_data,
                    overview,
                    "strategy_comparison",
                    title="Mean net log wealth across completed seeds",
                    ylabel="Net log wealth",
                    x="condition",
                    columns=ordered,
                    kind="bar",
                )
            )
    write_json(configs, out / "configurations.json")
    write_json(
        {
            "expected_primary_runs": len(rows),
            "completed_primary_runs": int(status.status.eq("complete").sum()),
            "primary_results_available": bool(len(summaries)),
            "detail_directory": "runs",
            "figures_directory": "figures",
            "inference": "descriptive only",
        },
        out / "report_manifest.json",
    )
    inventory(out)
    return {
        "status": status,
        "summary": summary,
        "figures": pics,
        "detail_figures": detail_pics,
        "out": out,
    }


def control_example(group):
    from dtasrl.attribution import evaluate_target_path, freeze_k_path

    out = PROJECT_ROOT / "outputs" / group / "worked_example"
    targets = np.array([[0.2, 0.5, 0.3], [0.5, 0.2, 0.3], [0.1, 0.7, 0.2], [0.3, 0.4, 0.3]])
    returns = np.array([[1, 1.2, 0.9], [1, 0.95, 1.1], [1, 1.05, 1.02], [1, 0.9, 1.12]])
    paths = {
        "hand_chosen_dynamic": evaluate_target_path(targets, returns, 10, 1000),
        "freeze_1": freeze_k_path(targets, returns, 1, 10, initial_wealth=1000),
        "initial_weight_rebalance": evaluate_target_path(
            np.tile(targets[0], (4, 1)), returns, 10, 1000
        ),
    }
    summary, wealth = [], pd.DataFrame({"step": np.arange(5)})
    for name, path in paths.items():
        wealth[name] = path.wealth
        summary.append(
            {
                "strategy": name,
                "initial_wealth": 1000,
                "final_wealth": path.wealth[-1],
                "total_cost": path.costs.sum(),
                "net_log_wealth": path.net_log_wealth,
            }
        )
        ledger = pd.DataFrame(
            {
                "step": range(1, 5),
                "wealth_before": path.wealth[:-1],
                "wealth_after": path.wealth[1:],
                "cost": path.costs,
                "trade_A": path.trade_values[:, 0],
                "trade_B": path.trade_values[:, 1],
            }
        )
        save_table(ledger, out / name)
    pic = figure(
        wealth,
        out,
        "comparison",
        title="Worked example only | no trained model",
        ylabel="Wealth",
        x="step",
        columns=list(paths),
    )
    table = pd.DataFrame(summary)
    save_table(table, out / "summary")
    inventory(out)
    return table, pic
