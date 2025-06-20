import subprocess

print("🔍 Installed pandas version:")
result = subprocess.run(["pip", "show", "pandas"], capture_output=True, text=True)
print(result.stdout)

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
    result = subprocess.run(["python", f"preprocessing/{step}"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"⚠️ STDERR for {step}:\n{result.stderr}")
    if result.returncode != 0:
        print(f"❌ Step failed: {step} with return code {result.returncode}")
        break
    print(f"✅ Completed: {step}\n")

print("🏁 Pipeline complete.")
