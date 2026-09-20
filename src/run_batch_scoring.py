#!/usr/bin/env python3
"""
Launch the two no-endpoint batch-scoring SageMaker Processing jobs: XGBoost
fold-average and NN fold-average, each built from its own image
(Dockerfile.xgb-scorer / Dockerfile.nn-scorer). Pure inference against
already-trained fold models - no training, no endpoints.

One-time (or after any code change to the scoring path):
    bash sagemaker_processing/build_and_push_scorers.sh

Then:
    python -m src.run_batch_scoring                    # xgb, then nn, logs streamed, merged
    python -m src.run_batch_scoring --which xgb         # just xgb
    python -m src.run_batch_scoring --which nn          # just nn
    python -m src.run_batch_scoring --merge-only        # re-merge existing per-model outputs
    python -m src.run_batch_scoring --no-wait           # launch and return; no log streaming, no merge

Why per-model output files, then a merge step: predict_from_both.py writes
whichever columns it computed and sets the other model's column to None. Two
separate Processing jobs writing straight to the shared
"<test>_with_predictions.csv" key would have the second job silently erase
the first job's column. Each job here gets its own "_xgb"/"_nn" suffixed
key; once both exist, they're merged (row-aligned - both jobs score the
exact same, unshuffled input file) into the canonical key that
run_pipeline.py's "evaluate" step reads.
"""
import argparse
import sys
from pathlib import Path

import boto3
from sagemaker import Session
from sagemaker.processing import Processor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import cfg  # noqa: E402
from src.predict_from_both import _s3_read_csv, _s3_write_csv  # noqa: E402

AWS_REGION = cfg.get("aws.region")
BUCKET = cfg.get("storage.bucket")
ACCOUNT_ID = cfg.get("infra.account_id")
ECR_REPO_NAME = cfg.get("infra.ecr_repo_name")
ROLE = cfg.get("infra.sagemaker_role")
ECR_URI = f"{ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO_NAME}" if ACCOUNT_ID and ECR_REPO_NAME else None

OUTPUT_PREFIX = cfg.prefix("scored")
INPUT_KEY = cfg.s3_key("engineered", "test_selected")
LABEL_COL = cfg.get("data.label_col")
BASENAME = Path(cfg.get("data.files.test_selected")).stem  # e.g. prepared_diabetes_test_selected

CANONICAL_KEY = f"{OUTPUT_PREFIX}/{BASENAME}_with_predictions.csv"

JOBS = {
    "xgb": {
        "image": f"{ECR_URI}:xgb-scorer" if ECR_URI else None,
        "run_mode": "xgb",
        "source_env": "XGB_SOURCE",
        "source_val": str(cfg.get("predict.xgb_source") or "local_folds"),
        "out_key": f"{OUTPUT_PREFIX}/{BASENAME}_with_predictions_xgb.csv",
        "prob_col": "xgb_prob",
    },
    "nn": {
        "image": f"{ECR_URI}:nn-scorer" if ECR_URI else None,
        "run_mode": "nn",
        "source_env": "NN_SOURCE",
        "source_val": str(cfg.get("predict.nn_source") or "local_folds"),
        "out_key": f"{OUTPUT_PREFIX}/{BASENAME}_with_predictions_nn.csv",
        "prob_col": "nn_prob",
    },
}


def run_one(which: str, instance_type: str, wait: bool) -> str:
    spec = JOBS[which]
    if not spec["image"]:
        raise SystemExit("❌ infra.account_id / infra.ecr_repo_name not resolved "
                          "(check config.infra.yaml).")

    session = Session(boto_session=boto3.Session(region_name=AWS_REGION))
    processor = Processor(
        image_uri=spec["image"],
        role=ROLE,
        instance_count=1,
        instance_type=instance_type,
        base_job_name=f"diabetes-{which}-scoring",
        sagemaker_session=session,
        env={
            "RUN_MODE": spec["run_mode"],
            spec["source_env"]: spec["source_val"],
        },
    )
    print(f"\n🚀 Launching {which.upper()} scoring job")
    print(f"   image:      {spec['image']}")
    print(f"   instance:   {instance_type}")
    print(f"   input:      s3://{BUCKET}/{INPUT_KEY}")
    print(f"   output:     s3://{BUCKET}/{spec['out_key']}")
    processor.run(
        arguments=["--input-key", INPUT_KEY, "--label-col", LABEL_COL,
                   "--out-key", spec["out_key"]],
        wait=wait,
        logs=wait,
    )
    job_name = processor.jobs[-1].describe()["ProcessingJobName"]
    print(f"✅ {which.upper()} job: {job_name}")
    return job_name


def merge_outputs() -> str:
    """Combine the xgb-only and nn-only outputs (same input, same row order)
    into the canonical <test>_with_predictions.csv that 'evaluate' reads."""
    xgb_key, nn_key = JOBS["xgb"]["out_key"], JOBS["nn"]["out_key"]
    print(f"\n🔗 Merging:\n   s3://{BUCKET}/{xgb_key}\n   s3://{BUCKET}/{nn_key}")
    xgb_df = _s3_read_csv(BUCKET, xgb_key)
    nn_df = _s3_read_csv(BUCKET, nn_key)
    if len(xgb_df) != len(nn_df):
        raise SystemExit(f"❌ Row count mismatch: xgb output has {len(xgb_df)} rows, "
                          f"nn output has {len(nn_df)}. Refusing to merge row-aligned.")
    merged = xgb_df.copy()
    merged["nn_prob"] = nn_df["nn_prob"].values
    _s3_write_csv(BUCKET, CANONICAL_KEY, merged)
    print(f"✅ Wrote merged predictions to s3://{BUCKET}/{CANONICAL_KEY}")
    return CANONICAL_KEY


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["xgb", "nn", "both"], default="both")
    ap.add_argument("--instance-type", default="ml.m5.large")
    ap.add_argument("--no-wait", action="store_true",
                     help="launch and return immediately instead of streaming logs "
                          "(skips the merge step - run --merge-only once both finish)")
    ap.add_argument("--merge-only", action="store_true",
                     help="skip launching jobs; just merge existing xgb/nn output files")
    args = ap.parse_args()

    if args.merge_only:
        merge_outputs()
        return 0

    if not ROLE:
        raise SystemExit("❌ infra.sagemaker_role not set (config.infra.yaml).")

    targets = ["xgb", "nn"] if args.which == "both" else [args.which]
    for which in targets:
        run_one(which, args.instance_type, wait=not args.no_wait)

    if args.which == "both" and not args.no_wait:
        merge_outputs()
    elif args.no_wait:
        print("\n⏳ Launched without waiting - jobs are running in the background.")
        print("   Check status in the SageMaker Processing console, then run:")
        print("   python -m src.run_batch_scoring --merge-only")
    else:
        print(f"\nℹ️  Ran '{args.which}' only. Its output is at "
              f"s3://{BUCKET}/{JOBS[args.which]['out_key']}.")
        print("   Run the other model, then merge with:")
        print("   python -m src.run_batch_scoring --merge-only")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
