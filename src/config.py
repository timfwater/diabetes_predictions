#!/usr/bin/env python3
"""
Central configuration loader for the diabetes readmissions pipeline.

Resolution order (highest priority first):
    1. Environment variable  (e.g. BUCKET=foo)
    2. config.infra.yaml     (gitignored infrastructure identifiers)
    3. config.yaml           (checked-in defaults)

Usage:
    from src.config import cfg

    bucket = cfg.get("storage.bucket")
    folds  = cfg.get("tuning.kfolds")
    key    = cfg.s3_key("engineered", "train")        # 02_engineered/prepared_...csv
    uri    = cfg.s3_uri("engineered", "train")        # s3://bucket/02_engineered/...

Every script keeps working with its existing env vars because each dotted path
is registered with the env var name the scripts already use.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

# --------------------------------------------------------------------------
# Map dotted config paths -> the environment variable that overrides them.
# These names match what the existing scripts already read, so setting any of
# them on the command line behaves exactly as it does today.
# --------------------------------------------------------------------------
ENV_OVERRIDES = {
    "aws.region": "AWS_REGION",
    "storage.bucket": "BUCKET",
    "storage.raw_prefix": "RAW_PREFIX",
    "storage.engineered_prefix": "PREFIX",
    "storage.scored_prefix": "OUTPUT_PREFIX",
    "storage.eval_prefix": "EVAL_PREFIX",
    "storage.input_file": "INPUT_FILE",
    "data.label_col": "LABEL_COL",
    "data.files.selected_features": "SELECTED_FEATURES_FILE",
    "split.test_size": "TEST_SIZE",
    "split.seed": "SPLIT_SEED",
    "split.strategy": "SPLIT_STRATEGY",
    "split.group_col": "SPLIT_GROUP_COL",
    "feature_selection.method": "FS_METHOD",
    "feature_selection.cum_importance": "FS_CUM_IMPORTANCE",
    "feature_selection.top_k": "FS_TOP_K",
    "feature_selection.mode": "FS_MODE",
    "tuning.kfolds": "KFOLDS",
    "tuning.max_jobs": "HPO_MAX_JOBS",
    "tuning.max_parallel": "HPO_MAX_PARALLEL",
    "tuning.eval_metric": "EVAL_METRIC",
    "tuning.objective_metric": "OBJECTIVE_METRIC",
    "tuning.xgb.instance_type": "XGB_INSTANCE_TYPE",
    "tuning.xgb.output_prefix": "XGB_OUTPUT_PREFIX",
    "tuning.xgb.use_scale_pos_weight": "XGB_USE_SPW",
    "tuning.nn.instance_type": "NN_INSTANCE_TYPE",
    "tuning.nn.output_prefix": "NN_OUTPUT_PREFIX",
    "deploy.instance_type": "DEPLOY_INSTANCE_TYPE",
    "deploy.xgb_endpoint": "ENDPOINT",
    "deploy.nn_endpoint": "ENDPOINT_NN",
    "deploy.wait_for_tuning": "WAIT_FOR_TUNING",
    "deploy.tuning_job_file": "TUNING_JOB_FILE",
    "deploy.tuning_job_name": "TUNING_JOB_NAME",
    "predict.batch_size": "BATCH_SIZE",
    "predict.run_mode": "RUN_MODE",
    "evaluation.topk": "EVAL_TOPK",
    "evaluation.cost.program_cost_per_enrollee": "COST_PROGRAM",
    "evaluation.cost.readmission_cost": "COST_READMISSION",
    "evaluation.cost.program_efficacy": "COST_EFFICACY",
    # Infrastructure (from config.infra.yaml, never checked in)
    "infra.account_id": "AWS_ACCOUNT_ID",
    "infra.sagemaker_role": "SAGEMAKER_TRAINING_ROLE",
    "infra.task_role": "TASK_ROLE",
    "infra.task_execution_role": "TASK_EXECUTION_ROLE",
    "infra.ecs_cluster": "ECS_CLUSTER_NAME",
    "infra.task_family": "TASK_FAMILY",
    "infra.subnet_ids": "FARGATE_SUBNET_IDS",
    "infra.security_group_ids": "FARGATE_SECURITY_GROUP_IDS",
    "infra.assign_public_ip": "ASSIGN_PUBLIC_IP",
    "infra.ecr_repo_name": "ECR_REPO_NAME",
    "infra.log_group": "LOG_GROUP",
    "infra.log_stream_prefix": "LOG_STREAM_PREFIX",
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(raw: str, reference: Any) -> Any:
    """Cast an env-var string to match the type of the YAML default."""
    if isinstance(reference, bool):
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"Cannot interpret {raw!r} as a boolean")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(raw)
    if isinstance(reference, float):
        return float(raw)
    if isinstance(reference, list):
        return [p.strip() for p in raw.split(",") if p.strip()]
    return raw


class Config:
    def __init__(self, path: Optional[str] = None, infra_path: Optional[str] = None):
        root = Path(__file__).resolve().parent.parent

        self.path = Path(path or os.getenv("CONFIG_FILE") or root / "config.yaml")
        if not self.path.exists():
            raise SystemExit(f"Config file not found: {self.path}")

        with open(self.path) as fh:
            self._data = yaml.safe_load(fh) or {}

        # Optional infrastructure overlay (gitignored)
        self.infra_path = Path(
            infra_path or os.getenv("CONFIG_INFRA_FILE") or root / "config.infra.yaml"
        )
        if self.infra_path.exists():
            with open(self.infra_path) as fh:
                infra = yaml.safe_load(fh) or {}
            self._data.setdefault("infra", {}).update(infra.get("infra", infra))

        self._env_used: dict[str, str] = {}

    # ---------------------------------------------------------------- access
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                node = default
                break
            node = node[part]

        env_name = ENV_OVERRIDES.get(dotted)
        if env_name:
            raw = os.getenv(env_name)
            if raw is not None and raw != "":
                self._env_used[dotted] = env_name
                return _coerce(raw, node)
        return node

    def require(self, dotted: str) -> Any:
        val = self.get(dotted)
        if val is None or val == "":
            env_name = ENV_OVERRIDES.get(dotted, "")
            hint = f" (or set {env_name})" if env_name else ""
            raise SystemExit(f"Required config value missing: {dotted}{hint}")
        return val

    # ------------------------------------------------------------ s3 helpers
    _PREFIXES = {
        "raw": "storage.raw_prefix",
        "engineered": "storage.engineered_prefix",
        "scored": "storage.scored_prefix",
        "eval": "storage.eval_prefix",
    }

    def prefix(self, stage: str) -> str:
        if stage not in self._PREFIXES:
            raise KeyError(f"Unknown stage {stage!r}; expected one of {list(self._PREFIXES)}")
        return self.get(self._PREFIXES[stage])

    def s3_key(self, stage: str, file_alias: str) -> str:
        """Build an S3 key from a stage prefix and a data.files alias."""
        filename = self.get(f"data.files.{file_alias}")
        if filename is None:
            filename = file_alias  # allow raw filenames too
        return f"{self.prefix(stage)}/{filename}"

    def s3_uri(self, stage: str, file_alias: str) -> str:
        return f"s3://{self.get('storage.bucket')}/{self.s3_key(stage, file_alias)}"

    # -------------------------------------------------------------- reporting
    def resolved(self) -> dict:
        """Flatten to dotted-path -> value, for dry-run display."""
        out: dict[str, Any] = {}

        def walk(node: Any, trail: str = "") -> None:
            if isinstance(node, dict):
                for key, val in node.items():
                    walk(val, f"{trail}.{key}" if trail else key)
            else:
                out[trail] = self.get(trail, node)

        walk(self._data)
        return out

    def describe(self, mask_infra: bool = True) -> str:
        lines = [
            "=" * 68,
            "RESOLVED CONFIGURATION",
            f"  config file : {self.path}",
            f"  infra file  : {self.infra_path if self.infra_path.exists() else '(none)'}",
            "=" * 68,
        ]
        section = None
        for dotted, value in sorted(self.resolved().items()):
            top = dotted.split(".")[0]
            if top != section:
                section = top
                lines.append(f"\n[{section}]")
            if mask_infra and top == "infra" and value:
                shown = f"{str(value)[:6]}...{str(value)[-4:]}" if len(str(value)) > 12 else "***"
            else:
                shown = value
            marker = f"   <- ${self._env_used[dotted]}" if dotted in self._env_used else ""
            lines.append(f"  {dotted:<52} = {shown}{marker}")
        lines.append("")
        return "\n".join(lines)


# Module-level singleton used by every script
cfg = Config()


if __name__ == "__main__":
    print(cfg.describe())
    print("Sample resolved paths:")
    for stage, alias in [
        ("engineered", "full"),
        ("engineered", "train_selected"),
        ("engineered", "test_selected"),
        ("scored", "test_selected"),
    ]:
        print(f"  {stage:<12} {alias:<16} -> {cfg.s3_uri(stage, alias)}")
