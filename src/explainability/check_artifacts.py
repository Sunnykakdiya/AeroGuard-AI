
from pathlib import Path
import joblib

ARTIFACTS = Path("artifacts")

files = {
    "Best model": ARTIFACTS / "models" / "best_model_FD001.pkl",
    "Random Forest": ARTIFACTS / "models" / "randomforest_FD001.pkl",
    "XGBoost": ARTIFACTS / "models" / "xgboost_FD001.pkl",
    "Feature columns": ARTIFACTS / "preprocessors" / "feature_columns_FD001.pkl",
    "Scaler": ARTIFACTS / "preprocessors" / "scaler_FD001.pkl",
    "Selected sensors": ARTIFACTS / "preprocessors" / "selected_sensors_FD001.pkl",
}

for name, path in files.items():
    print(f"\n--- {name} ---")

    if not path.exists():
        print(f"NOT FOUND: {path}")
        continue

    artifact = joblib.load(path)

    if name in ("Best model", "Random Forest", "XGBoost"):
        print("Class:", type(artifact).__name__)
        print("Module:", type(artifact).__module__)

        if hasattr(artifact, "n_features_in_"):
            print("Expected features:", artifact.n_features_in_)

    elif name == "Feature columns":
        print("Total features:", len(artifact))
        print("First 15:", artifact[:15])

    elif name == "Scaler":
        print("Class:", type(artifact).__name__)

        if hasattr(artifact, "n_features_in_"):
            print("Expected features:", artifact.n_features_in_)

    elif name == "Selected sensors":
        print("Total sensors:", len(artifact))
        print("Sensors:", artifact)
