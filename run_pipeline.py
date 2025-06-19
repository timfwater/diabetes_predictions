import subprocess

steps = [
    "data_engineering.py",
    "feature_selection.py",
    "run_tuning.py",
    "deploy_best_xgb.py",
    "predict_from_endpoint.py"
]

print("🚀 Starting full pipeline...\n")

for step in steps:
    print(f"🔧 Running: {step}")
    result = subprocess.run(["python", f"preprocessing/{step}"])
    if result.returncode != 0:
        print(f"❌ Step failed: {step}")
        break
    print(f"✅ Completed: {step}\n")

print("🏁 Pipeline complete.")
