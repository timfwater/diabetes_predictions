
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter
from sagemaker.xgboost.estimator import XGBoost
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='diabetes_processed_data/k')
    parser.add_argument('--train_file', type=str, default='train_proto.data')
    parser.add_argument('--validation_file', type=str, default='validation_proto.data')
    return parser.parse_args()

def main():
    args = parse_args()

    role = get_execution_role()
    session = sagemaker.Session()
    region = session.boto_region_name

    xgb_image_uri = sagemaker.image_uris.retrieve("xgboost", region, version="1.5-1")

    estimator = XGBoost(
        entry_point=None,
        image_uri=xgb_image_uri,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        output_path=f's3://{args.bucket}/{args.prefix}/xgb_output',
        sagemaker_session=session,
        base_job_name="xgb-diabetes-tune",
        hyperparameters={
            "objective": "binary:logistic",
            "num_round": 100
        }
    )

    hyperparameter_ranges = {
        "eta": ContinuousParameter(0.01, 0.3),
        "max_depth": IntegerParameter(3, 10),
        "min_child_weight": IntegerParameter(1, 10),
        "subsample": ContinuousParameter(0.5, 1.0),
    }

    objective_metric_name = "validation:auc"

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name=objective_metric_name,
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=[{'Name': 'validation:auc', 'Regex': 'auc:([0-9\.]+)'}],
        max_jobs=10,
        max_parallel_jobs=2,
        objective_type='Maximize'
    )

    s3_input_train = TrainingInput(f's3://{args.bucket}/{args.prefix}/{args.train_file}', content_type='application/x-recordio-protobuf')
    s3_input_val = TrainingInput(f's3://{args.bucket}/{args.prefix}/{args.validation_file}', content_type='application/x-recordio-protobuf')

    tuner.fit({'train': s3_input_train, 'validation': s3_input_val})

if __name__ == "__main__":
    main()
