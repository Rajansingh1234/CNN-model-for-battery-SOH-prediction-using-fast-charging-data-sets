"""Predict test SOH, calculate metrics, and save result tables."""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from step01_config  import PER_FILE_METRICS_FILENAME, PREDICTIONS_FILENAME


def regression_metrics(true, predicted):
    """Return MAE, RMSE and R-squared for one collection of samples."""
    return {
        "MAE": mean_absolute_error(true, predicted),
        "RMSE": np.sqrt(mean_squared_error(true, predicted)),
        "R2": r2_score(true, predicted) if len(true) > 1 else np.nan,
    }


def evaluate_model(
    model_path,
    scaled,
    cycle_test,
    file_test,
    data_folder,
    number_of_test_files,
):
    """Load the best model and evaluate it on the untouched test split."""
    best_model = tf.keras.models.load_model(model_path)
    predicted_scaled = best_model.predict(
        {
            "sequence_input": scaled["x_test"],
            "physical_input": scaled["p_test"],
        }
    ).flatten()
    predicted_soh = predicted_scaled * scaled["y_std"] + scaled["y_mean"]
    true_soh = scaled["y_test"]

    overall = regression_metrics(true_soh, predicted_soh)
    print("\n" + "=" * 70)
    print("FINAL MULTI-CELL TEST RESULTS")
    print("=" * 70)
    print("Test files:", number_of_test_files)
    print("Test cycles:", len(true_soh))
    print("MAE:", overall["MAE"], "%")
    print("RMSE:", overall["RMSE"], "%")
    print("R²:", overall["R2"])

    results = pd.DataFrame(
        {
            "source_file": file_test,
            "cycle_index": cycle_test,
            "true_SOH": true_soh,
            "predicted_SOH": predicted_soh,
            "absolute_error": np.abs(true_soh - predicted_soh),
        }
    )
    result_path = os.path.join(data_folder, PREDICTIONS_FILENAME)
    results.to_csv(result_path, index=False)
    print("\nPredictions saved to:")
    print(result_path)

    per_file_rows = []
    for filename in np.unique(file_test):
        mask = file_test == filename
        metrics = regression_metrics(true_soh[mask], predicted_soh[mask])
        per_file_rows.append(
            {
                "file": filename,
                "cycles": int(np.sum(mask)),
                **metrics,
            }
        )

    per_file = pd.DataFrame(per_file_rows)
    print("\n" + "=" * 70)
    print("PER-BATTERY TEST PERFORMANCE")
    print("=" * 70)
    print(per_file.to_string(index=False))

    per_file_path = os.path.join(data_folder, PER_FILE_METRICS_FILENAME)
    per_file.to_csv(per_file_path, index=False)
    print("\nPer-file metrics saved to:")
    print(per_file_path)

    return {
        "true_soh": true_soh,
        "predicted_soh": predicted_soh,
        "cycles": cycle_test,
        "files": file_test,
        "overall_metrics": overall,
        "per_file_metrics": per_file,
    }

