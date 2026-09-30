"""Data, training, and base-plan entry points for pathwise portfolio attribution."""

from pathlib import Path
from typing import Annotated

import typer

from dtasrl.config import load_config

app = typer.Typer(help="Pathwise attribution of reinforcement-learning portfolios")
data_app = typer.Typer()
experiment_app = typer.Typer()
study_app = typer.Typer()
app.add_typer(data_app, name="data")
app.add_typer(experiment_app, name="experiment")
app.add_typer(study_app, name="study")


@data_app.command("build")
def data_build(config: Annotated[Path, typer.Option(exists=True, dir_okay=False)]):
    from dtasrl.data.panel import build_market_snapshot

    typer.echo(build_market_snapshot(load_config(config)))


@data_app.command("prepare")
def data_prepare():
    """Freeze scalers and inputs for the base 18-task plan; see Notebook 00 for larger universes."""
    from dtasrl.study import prepare_inputs

    typer.echo(prepare_inputs())


@experiment_app.command("run")
def experiment_run(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    seed: Annotated[int, typer.Option(min=0)],
    resume: Annotated[bool, typer.Option()] = False,
):
    from dtasrl.experiments.portfolio import run_experiment

    loaded = load_config(config)
    if loaded.get("study") != "ppo_attribution" or loaded.get("backend") != "sbx_jax":
        raise typer.BadParameter("Only the current PPO attribution SBX protocol is supported")
    loaded["_resume"] = resume
    typer.echo(run_experiment(loaded, seed))


@study_app.command("plan")
def study_plan(output: Annotated[Path | None, typer.Option()] = None):
    """Plan the 18 synthetic, single-stock, and 3-stock models without training."""
    from dtasrl.study import build_plan

    plan = build_plan(output)
    typer.echo({"tasks": plan["task_count"], "models_started": plan["models_started"]})


@study_app.command("summarize")
def study_summarize(plan: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None):
    """Summarize the base 18-task plan; larger universes have separate notebook reports."""
    from dtasrl.study import summarize

    result = summarize(plan)
    typer.echo(
        {"ready": result["ready"], "complete": int(result["status"].eligible_for_summary.sum())}
    )


if __name__ == "__main__":
    app()
