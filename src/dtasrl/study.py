"""Preparation and read-only reporting for the base 18-model attribution plan.

The complete study also includes three 30-stock and three 100-stock models,
prepared and reported independently by their joint-portfolio notebooks.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

from dtasrl.config import (
    PROJECT_ROOT,
    canonical_json,
    config_hash,
    dump_yaml,
    load_config,
    public_config,
)
from dtasrl.data.inputs import freeze_real_inputs, freeze_synthetic_inputs
from dtasrl.provenance import run_directory, sha256_file, source_tree_sha256, utc_now, write_json
from dtasrl.storage import inventory, save_table, verify_inventory
from dtasrl.study_reports import control_example, experiment_report

CONDITIONS = (
    "synthetic_high",
    "synthetic_zero",
    "single_msft",
    "single_jpm",
    "single_jnj",
    "multi_asset",
)
SEEDS = (101, 202, 303)
STEPS = 5_001_216

# These three source-tree identities were audited after the primary runs.
# The public snapshots retain the executable src/ tree while omitting internal
# drafting and test files. Never infer equivalence from a hash prefix.
EQUIVALENT_SOURCE_VARIANTS = {
    "7c5430a4cff5413f1a02a7d4cb800c815ec23ed2d5487bc8540462158076d320": (
        "914d69802c126d706a01d36ca14567521ed529df0730ce79e7aff27374f5fad7",
        "fcc5bbabf0b08d5c4e833e3f7c9ceb01e8680108c91e975822728d9e64e3c07d",
    ),
    "a1d8b9433d5c3680db62de6ca5c55dd652f0d6c5995de46214822e51e7dac8e2": (
        "ef3668ebacd72629f289f88539ad7cef6769fd5719f6677dfa9c25253d2dd72d",
        "f6e5eb2e0a24e9a0f8347e64c51bbf10cb13f11d13ec30f26ac59815a1ff1cd3",
    ),
    "e827dce08ca97a22852598dcc9266e942b979d811847aea8a859149d95fcc748": (
        "606570f942a93607e54b622c88789f16c3e680ef56ea71ae4fbd4e4fcd875005",
        "86c109544003b1b51aee1f776596ed53fb68f1195482356bc2ba4b0c14bf4416",
    ),
}
def _equivalent_source_snapshot(run: Path, actual_hash: str, expected_hash: str) -> bool:
    """Validate an archived source tree against an explicitly audited identity."""
    if expected_hash not in EQUIVALENT_SOURCE_VARIANTS:
        return False
    expected_variant = EQUIVALENT_SOURCE_VARIANTS.get(actual_hash)
    if expected_variant is None:
        return False
    try:
        with zipfile.ZipFile(run / "source.zip") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                return False
            sources = sorted(
                name for name in names if name.startswith("src/") and name.endswith(".py")
            )
            if len(sources) != 22 or "src/dtasrl/training.py" not in sources:
                return False
            training_hash = hashlib.sha256(
                archive.read("src/dtasrl/training.py")
            ).hexdigest()
            source_tree_hash = hashlib.sha256(
                b"".join(
                    name.removeprefix("src/").encode() + b"\0" + archive.read(name) + b"\0"
                    for name in sources
                )
            ).hexdigest()
    except (OSError, zipfile.BadZipFile, KeyError):
        return False
    return training_hash == expected_variant[0] and source_tree_hash == actual_hash


def prepare_inputs() -> dict:
    """Freeze real/synthetic inputs and bind the six base-condition configs to artifacts."""
    config_dir = PROJECT_ROOT / "configs/experiments"
    snapshot = "data/processed/metadata.json"
    multi = load_config(config_dir / "multi_asset.yaml")
    multi["data"]["snapshot"] = snapshot
    real = freeze_real_inputs(multi)
    bindings: dict[str, str] = {}
    for condition in CONDITIONS:
        path = config_dir / f"{condition}.yaml"
        config = load_config(path)
        config["backend"] = "sbx_jax"
        if config["kind"] == "synthetic":
            frozen = freeze_synthetic_inputs(config)
            config["data"] = {"frozen_inputs": frozen.relative_to(PROJECT_ROOT).as_posix()}
        else:
            frozen = real
            config["data"] = {
                "snapshot": snapshot,
                "frozen_inputs": real.relative_to(PROJECT_ROOT).as_posix(),
            }
        dump_yaml(public_config(config), path)
        verify_inventory(frozen)
        bindings[condition] = frozen.relative_to(PROJECT_ROOT).as_posix()
    report = {"prepared_at_utc": utc_now(), "snapshot": snapshot, "bindings": bindings}
    write_json(report, PROJECT_ROOT / "outputs/00_data/preparation.json")
    return report


def build_plan(output: Path | None = None) -> dict:
    """Write the fixed 18-task plan without executing any learner."""
    output = output or PROJECT_ROOT / "outputs/study_plan"
    if output.exists() and (output / "plan.json").exists():
        existing = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        if existing.get("execution_authorized"):
            raise FileExistsError("An execution-authorized plan cannot be overwritten")
        if existing.get("task_count") != 18 or existing.get("models_started") != 0:
            raise ValueError("Existing plan is not the fixed 18-task study plan")
        # The planned source hash is part of the formal evidence. Rebuilding a
        # plan after reporting-code edits must not silently rebind completed
        # training runs to a different source tree.
        return existing
    output.mkdir(parents=True, exist_ok=True)
    source_hash = source_tree_sha256()
    sources: dict[str, dict] = {}
    tasks: list[dict] = []
    for condition in CONDITIONS:
        path = PROJECT_ROOT / "configs/experiments" / f"{condition}.yaml"
        config = load_config(path)
        if not (
            config["backend"] == "sbx_jax"
            and config["algorithm"] == "ppo"
            and config["stage"] == "primary"
            and config["total_timesteps"] == STEPS
        ):
            raise ValueError(f"Invalid primary configuration: {path}")
        folder = PROJECT_ROOT / config["data"]["frozen_inputs"]
        verify_inventory(folder)
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        data_hash = metadata.get("data_hash", metadata.get("data_sha256"))
        sources[condition] = {
            "frozen_inputs": folder.relative_to(PROJECT_ROOT).as_posix(),
            "metadata_sha256": sha256_file(folder / "metadata.json"),
            "inventory_sha256": sha256_file(folder / "manifest.json"),
            "data_hash": data_hash,
        }
        for seed in SEEDS:
            command = [
                "python",
                "-m",
                "dtasrl.cli",
                "experiment",
                "run",
                "--config",
                path.relative_to(PROJECT_ROOT).as_posix(),
                "--seed",
                str(seed),
            ]
            tasks.append(
                {
                    "task_id": f"{condition}_seed_{seed}",
                    "condition": condition,
                    "seed": seed,
                    "status": "planned_not_started",
                    "transitions": STEPS,
                    "config_file": path.relative_to(PROJECT_ROOT).as_posix(),
                    "config_file_sha256": sha256_file(path),
                    "config_hash": config_hash(config),
                    "data_hash": data_hash,
                    "source_sha256": source_hash,
                    "command_argv": command,
                    "config": public_config(config),
                }
            )
    identity = [
        {key: task[key] for key in ("task_id", "config_hash", "data_hash", "source_sha256")}
        for task in tasks
    ]
    plan = {
        "schema_version": 3,
        "created_at_utc": utc_now(),
        "status": "planned_not_started",
        "execution_authorized": False,
        "models_started": 0,
        "task_count": len(tasks),
        "total_transitions": len(tasks) * STEPS,
        "seeds": list(SEEDS),
        "source_sha256": source_hash,
        "plan_sha256": hashlib.sha256(canonical_json(identity).encode()).hexdigest(),
        "sources": sources,
        "tasks": tasks,
    }
    write_json(plan, output / "plan.json")
    flat = [
        {k: v for k, v in task.items() if k not in ("config", "command_argv")} for task in tasks
    ]
    save_table(pd.DataFrame(flat), output / "plan")
    inventory(output)
    return plan


def _validate_complete_task(task: dict) -> tuple[bool, str, Path]:
    config_path = PROJECT_ROOT / task["config_file"]
    config = load_config(config_path)
    run = run_directory(config, task["seed"])
    if (
        sha256_file(config_path) != task["config_file_sha256"]
        or config_hash(config) != task["config_hash"]
    ):
        return False, "configuration_mismatch", run
    if not (run / "manifest.json").exists():
        return False, "not_started" if not run.exists() else "incomplete", run
    try:
        verify_inventory(run)
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        evaluation = json.loads((run / "evaluation_status.json").read_text(encoding="utf-8"))
        training = json.loads((run / "training_status.json").read_text(encoding="utf-8"))
        if metrics["actual_transitions"] != STEPS or training["status"] != "complete":
            return False, "incomplete_step_budget", run
        if evaluation["status"] != "complete" or not (run / "final_checkpoint.zip").is_file():
            return False, "incomplete_evaluation", run
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        if metadata["data_hash"] != task["data_hash"]:
            return False, "data_mismatch", run
        if (
            metadata["config_hash"] != task["config_hash"]
            or metadata["seed"] != task["seed"]
            or metadata["stage"] != "primary"
        ):
            return False, "run_identity_mismatch", run
        source_hash = metadata["source_tree_sha256"]
        if source_hash != task["source_sha256"]:
            if not _equivalent_source_snapshot(run, source_hash, task["source_sha256"]):
                return False, "run_identity_mismatch", run
            return True, "complete_equivalent_source", run
    except (FileNotFoundError, KeyError, ValueError):
        return False, "invalid_artifacts", run
    return True, "complete", run


def summarize(plan_path: Path | None = None) -> dict:
    """Summarize only complete, hash-matched runs; never train or resume a model."""
    plan_path = plan_path or PROJECT_ROOT / "outputs/study_plan/plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["task_count"] != 18 or plan["models_started"] != 0:
        raise ValueError("Unexpected study plan")
    reports = {
        group: experiment_report(group)
        for group in ("01_synthetic", "02_single_asset", "03_multi_asset")
    }
    control_example("01_synthetic")
    control_example("03_multi_asset")
    rows = []
    for task in plan["tasks"]:
        complete, status, run = _validate_complete_task(task)
        rows.append(
            {
                "task_id": task["task_id"],
                "condition": task["condition"],
                "seed": task["seed"],
                "status": status,
                "eligible_for_summary": complete,
                "run": run.relative_to(PROJECT_ROOT).as_posix(),
            }
        )
    status = pd.DataFrame(rows)
    out = PROJECT_ROOT / "outputs/03_multi_asset/cross_experiment"
    save_table(status, out / "status")
    ready = bool(status.eligible_for_summary.all())
    write_json(
        {
            "ready": ready,
            "complete_runs": int(status.eligible_for_summary.sum()),
            "required_runs": 18,
            "formal_inference_available": False,
            "inference": "descriptive only; three seeds share one real-market history",
            "note": (
                "Aggregate metrics require all 18 plan-matched or explicitly audited "
                "equivalent-source runs; scientific inference remains descriptive."
            ),
        },
        out / "readiness.json",
    )
    if ready:
        effects = []
        for task in plan["tasks"]:
            config = load_config(PROJECT_ROOT / task["config_file"])
            run = run_directory(config, task["seed"])
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            for name, value in metrics["controls"]["effects"].items():
                effects.append(
                    {
                        "condition": task["condition"],
                        "seed": task["seed"],
                        "effect": name,
                        "net_log_wealth_difference": value,
                    }
                )
        save_table(pd.DataFrame(effects), out / "effects")
    inventory(out)
    for report in reports.values():
        inventory(report["out"])
    return {"ready": ready, "status": status, "output": out}
