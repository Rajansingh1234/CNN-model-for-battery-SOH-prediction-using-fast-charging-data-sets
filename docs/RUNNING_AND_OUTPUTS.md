# Running the project and understanding outputs

## Before running

1. Put the nine Python files in one folder.
2. Install the libraries from `requirements.txt`.
3. Set `DATA_FOLDER` in `step01_config.py`.
4. Check that the JSON structure matches [DATA_FORMAT.md](DATA_FORMAT.md).

## Run the complete workflow

```powershell
python step09_main.py
```

The program prints the assigned filenames, usable cycle counts, input shapes,
SOH ranges, model summary, epoch progress and final evaluation metrics.

## Generated files

Files are saved inside the configured data folder.

| File | Contents |
|---|---|
| `dataset_file_split.csv` | Battery-file assignment to train, validation or test |
| `best_multi_cell_soh_cnn.keras` | Model with the lowest validation loss |
| `cnn_test_predictions.csv` | True and predicted SOH for every test cycle |
| `cnn_test_per_file_metrics.csv` | MAE, RMSE and R² for each test battery |

## Plots

The code displays:

1. training and validation Huber loss by epoch;
2. true SOH versus predicted SOH;
3. the distribution of absolute SOH errors;
4. true and predicted SOH over cycle number for every test battery.

## Metric interpretation

| Metric | Interpretation |
|---|---|
| MAE | Average absolute error in SOH percentage points; lower is better |
| RMSE | Similar to MAE but penalizes large errors more strongly; lower is better |
| R² | Explained variation; values closer to 1 indicate a better fit |

For example, an MAE of `2.0` means predictions differ from true SOH by an
average of two percentage points.

## Running individual modules

The eight supporting modules mainly define functions and are imported by
`step09_main.py`. Running one directly usually only loads its definitions.

Functions can be explored in a Python console or notebook. For example:

```python
from step05_model import build_model

model = build_model(
    n_points=400,
    n_sequence_features=4,
    n_physical_features=9,
)
model.summary()
```

Training and evaluation require data produced by earlier stages, so use
`step09_main.py` for an ordinary complete experiment.

