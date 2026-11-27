
from pathlib import Path

from src.data.ingestion import load_engine_data
from src.inference.predict import predict_latest_rul


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"

# Load the existing NASA test sensor readings.
test_data = load_engine_data(
    file_path=DATA_DIR / "test_FD001.txt",
    dataset_name="FD001",
)

print("Test data shape:", test_data.shape)
print("Test engines:", test_data["engine_id"].nunique())

# Predict the latest RUL for all engines.
predictions = predict_latest_rul(test_data)

print("\nPRODUCTION INFERENCE RESULTS")
print(predictions.to_string(index=False))

# Save predictions.
REPORT_DIR.mkdir(parents=True, exist_ok=True)

output_path = (
    REPORT_DIR /
    "production_inference_smoke_test_FD001.csv"
)

predictions.to_csv(
    output_path,
    index=False,
)

print("\nPredictions saved to:", output_path)
print("\nInference test completed successfully.")
