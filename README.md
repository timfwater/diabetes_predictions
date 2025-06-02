# Waterman_Projects
Data science work in healthcare and other domains

Diabetes Predictions (6 files):

1. Data_Engineering.ipynb: Script contains all code for cleaning, engineering, and otherwise preparing data for machine learning.

2. Feature_selection.ipynb: Script combs through cleaned output of script #1, and reduces the dimensionality of the dataset, retaining only features of relatively high predictive value, thereby improving the predictive power of our eventual model. The curated features are then used for the later machine learning scripts.

3. Model_Tuning_Deployment_Predictions.ipynb: Script Contains hyperparameter tuning job, and training of an finalized models from the optimized hyperparameters. Also contains the deployment of this model, used for batch processing of the holdout test dataset as well as real-time-inference.
   
4. Evaluation_on_Test_Data.ipynb: Script contains the assessment of the test-set predictions across multiple prediction thresholds, optimizing cost-savings, ROC AUC, and readmission-reductions, through different model iterations.

Lambda_function_script_facilitating_inference_calls.py: Script used to facilitate back-end real-time inference calls alongside the deployed model and an API Gateway.

Diabetes_Inputs.csv: (contains raw input data for the above "1. Feature_Engineering.ipynb" file).
