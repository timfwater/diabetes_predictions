#!/usr/bin/env python3
"""
Composable pipeline runner.

Every stage is an independently runnable step. Run one, run several, or run a
named preset - the presets are just lists of steps, so nothing is hardcoded
into control flow any more.

    # See what would happen, touch nothing, spend nothing
    python -m src.run_pipeline --steps split,feature_select,apply --dry-run

    # Run a single stage
    python -m src.run_pipeline --steps feature_select

    # Named preset (equivalent to the old PIPELINE_MODE)
    python -m src.run_pipeline --preset prepare_selected

    # Override any config value inline
    python -m src.run_pipeline --steps split --set split.strategy=grouped

    # List everything available
    python -m src.run_pipeline --list
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ENV_OVERRIDES, cfg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- step model
@dataclass
class Step:
    name: str
    script: str
    description: str
    # Extra env applied only to this step (used where a script's default input
    # differs from what this stage should consume)
    env: dict = field(default_factory=dict)
    args: list = field(default_factory=list)
    # Optional callable returning a list of warnings to show before running
    preflight: Optional[Callable[[], list]] = None
    aws_cost: str = "none"   # none | low | high  - shown in dry-run


def _fs_preflight() -> list:
    warnings = []
    method = cfg.get("feature_selection.method")
    if method == "cumulative_importance":
        warnings.append(
            f"method=cumulative_importance -> using cum_importance="
            f"{cfg.get('feature_selection.cum_importance')}; top_k is ignored"
        )
    else:
        warnings.append(
            f"method=top_k -> using top_k={cfg.get('feature_selection.top_k')}; "
            "cum_importance is ignored"
        )
    return warnings


def _split_preflight() -> list:
    strategy = cfg.get("split.strategy")
    if strategy == "stratified":
        return [
            "split.strategy=stratified - encounters from the same patient can "
            "land on both sides of the split (known leakage; phase 2 fix)"
        ]
    return [f"split.strategy={strategy} on column {cfg.get('split.group_col')}"]


def build_steps() -> dict[str, Step]:
    """
    Built lazily, AFTER --set overrides are applied to the environment.

    This must not be a module-level dict: cfg.get() would then run at import
    time, baking in pre-override values while the printed config showed the
    post-override ones. Silent divergence between the plan and the run.
    """
    return {
    "engineer": Step(
        name="engineer",
        script="src/data_engineering.py",
        description="Raw CSV -> cleaned, encoded full dataset",
        aws_cost="low",
    ),
    "split": Step(
        name="split",
        script="src/split_train_test.py",
        description="Full -> train/test split",
        preflight=_split_preflight,
        aws_cost="low",
    ),
    "feature_select": Step(
        name="feature_select",
        script="src/feature_selection.py",
        description="Rank features on TRAIN only -> selected_features.csv",
        env={"FILTERED_INPUT_FILE": cfg.get("data.files.train")},
        preflight=_fs_preflight,
        aws_cost="low",
    ),
    "apply_features": Step(
        name="apply_features",
        script="src/apply_selected_features.py",
        description="Project train/test onto the selected feature list",
        aws_cost="low",
    ),
    "tune_xgb": Step(
        name="tune_xgb",
        script="src/run_tuning_xgb.py",
        description="SageMaker HPO for XGBoost",
        env={"FILTERED_INPUT_FILE": cfg.get("data.files.train_selected")},
        aws_cost="high",
    ),
    "tune_nn": Step(
        name="tune_nn",
        script="src/run_tuning_nn.py",
        description="SageMaker HPO for the neural network",
        env={"FILTERED_INPUT_FILE": cfg.get("data.files.train_selected")},
        aws_cost="high",
    ),
    "deploy_xgb": Step(
        name="deploy_xgb",
        script="src/deploy_best_xgb.py",
        description="Register best XGB model and create/update endpoint",
        aws_cost="high",
    ),
    "deploy_nn": Step(
        name="deploy_nn",
        script="src/deploy_best_nn.py",
        description="Register best NN model and create/update endpoint",
        aws_cost="high",
    ),
    "predict": Step(
        name="predict",
        script="src/predict_from_both.py",
        description="Batch score the test set -> 03_scored/*_with_predictions.csv",
        args=[
            "--input-key", cfg.s3_key("engineered", "test_selected"),
            "--label-col", str(cfg.get("data.label_col")),
        ],
        aws_cost="low",
    ),
    "evaluate": Step(
        name="evaluate",
        script="src/diabetes_eval_export.py",
        description="Metrics, curves, cost analysis -> 04_eval/",
        aws_cost="none",
    ),
    }


PRESETS: dict[str, list] = {
    "prepare_selected": ["split", "feature_select", "apply_features"],
    "prepare_full": ["engineer", "split", "feature_select", "apply_features"],
    "tune_both": ["tune_xgb", "tune_nn"],
    "deploy_both": ["deploy_xgb", "deploy_nn"],
    "score_and_eval": ["predict", "evaluate"],
    "full_experiment": [
        "engineer", "split", "feature_select", "apply_features",
        "tune_xgb", "tune_nn", "deploy_xgb", "deploy_nn",
        "predict", "evaluate",
    ],
}


# ------------------------------------------------------------------ helpers
def _apply_overrides(pairs: list) -> None:
    """--set dotted.path=value  ->  export the mapped environment variable."""
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects dotted.path=value, got: {pair}")
        dotted, value = pair.split("=", 1)
        env_name = ENV_OVERRIDES.get(dotted.strip())
        if not env_name:
            raise SystemExit(
                f"No environment mapping for '{dotted}'. "
                f"Add it to ENV_OVERRIDES in src/config.py."
            )
        os.environ[env_name] = value.strip()
        print(f"  override: {dotted} = {value.strip()}  (via ${env_name})")


def _config_env() -> dict:
    """
    Export every resolved config value under the env var name the scripts
    already read.

    This is a migration bridge. Scripts that have not yet been converted to
    cfg.get() still call os.getenv() directly, and previously got their values
    by sourcing fargate_deployment/config.env. Exporting here means the YAML
    is the single source of truth for refactored and unrefactored scripts
    alike. Once every script reads from cfg, this can be deleted.
    """
    env = {}
    for dotted, var in ENV_OVERRIDES.items():
        value = cfg.get(dotted)
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, list):
            value = ",".join(str(v) for v in value)
        env[var] = str(value)

    # feature_selection.py checks FS_CUM_IMPORTANCE before FS_TOP_K, so
    # exporting both would silently pin it to cumulative importance. Export
    # only the one the configured method actually uses.
    if cfg.get("feature_selection.method") == "cumulative_importance":
        env.pop("FS_TOP_K", None)
    else:
        env.pop("FS_CUM_IMPORTANCE", None)

    return env


def _step_env(step: Step) -> dict:
    env = os.environ.copy()
    # Config first, then the caller's shell wins, then step-specific values.
    merged = _config_env()
    merged.update(env)
    merged.update({k: str(v) for k, v in step.env.items() if v is not None})
    return merged


def _eval_args(step: Step) -> list:
    """The exporter takes explicit --data/--out; build them from config."""
    if step.name != "evaluate":
        return step.args
    bucket = cfg.get("storage.bucket")
    scored = cfg.prefix("scored")
    evalp = cfg.prefix("eval")
    import time
    stamp = time.strftime("%Y%m%d-%H%M%S")
    basename = Path(cfg.get("data.files.test_selected")).stem
    return [
        "--data", f"s3://{bucket}/{scored}/{basename}_with_predictions.csv",
        "--out", f"s3://{bucket}/{evalp}/runs/{stamp}",
        "--topk", str(cfg.get("evaluation.topk")),
    ]


def _resolve_steps(args, steps: dict) -> list:
    if args.preset:
        if args.preset not in PRESETS:
            raise SystemExit(
                f"Unknown preset '{args.preset}'. Available: {', '.join(PRESETS)}"
            )
        return PRESETS[args.preset]
    names = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [n for n in names if n not in steps]
    if unknown:
        raise SystemExit(
            f"Unknown step(s): {', '.join(unknown)}. Available: {', '.join(steps)}"
        )
    return names


def _print_listing(steps: dict) -> None:
    print("\nAVAILABLE STEPS")
    print("-" * 72)
    for step in steps.values():
        flag = {"none": "     ", "low": "  $  ", "high": " $$$ "}[step.aws_cost]
        print(f"  {step.name:<16}{flag} {step.description}")
    print("\nPRESETS")
    print("-" * 72)
    for name, steps in PRESETS.items():
        print(f"  {name:<20} {' -> '.join(steps)}")
    print("\n  $ = incurs AWS charges,  $$$ = launches training/endpoint resources\n")


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Composable diabetes pipeline runner")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--steps", help="comma-separated step names")
    group.add_argument("--preset", help="named sequence of steps")
    ap.add_argument("--list", action="store_true", help="list steps and presets")
    ap.add_argument("--dry-run", action="store_true",
                    help="show resolved config and planned steps without executing")
    ap.add_argument("--set", action="append", default=[], metavar="path=value",
                    help="override a config value (repeatable)")
    ap.add_argument("--show-config", action="store_true", help="print resolved config")
    args = ap.parse_args()

    if args.set:
        print("\nApplying overrides:")
        _apply_overrides(args.set)

    # Built only now, so overrides are reflected in every step's args and env
    steps = build_steps()

    if args.list:
        _print_listing(steps)
        return 0

    if args.show_config or args.dry_run:
        print(cfg.describe())

    if not args.steps and not args.preset:
        if args.show_config:
            return 0
        ap.error("one of --steps or --preset is required (or use --list)")

    names = _resolve_steps(args, steps)

    print("=" * 68)
    print("EXECUTION PLAN" + ("  [DRY RUN - nothing will run]" if args.dry_run else ""))
    print("=" * 68)
    for i, name in enumerate(names, 1):
        step = steps[name]
        extra = _eval_args(step)
        print(f"\n{i}. {step.name}  ({step.aws_cost} cost)")
        print(f"     {step.description}")
        print(f"     $ python {step.script}" + ("".join(f" {a}" for a in extra)))
        for key, val in step.env.items():
            print(f"     env: {key}={val}")
        if step.preflight:
            for warning in step.preflight():
                print(f"     note: {warning}")

    if args.dry_run:
        print("\nDry run complete. No AWS calls were made.\n")
        return 0

    print("\n" + "=" * 68 + "\n")
    for i, name in enumerate(names, 1):
        step = steps[name]
        print(f"\n>>> [{i}/{len(names)}] {step.name}")
        cmd = [sys.executable, str(ROOT / step.script)] + _eval_args(step)
        code = subprocess.run(cmd, env=_step_env(step), cwd=ROOT).returncode
        if code != 0:
            print(f"\nStep '{step.name}' failed (exit {code}). Stopping.")
            return code
        print(f"<<< {step.name} complete")

    print("\nPipeline complete.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
