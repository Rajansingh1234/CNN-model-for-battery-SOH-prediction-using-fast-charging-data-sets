# Guide to the nine Python files

## 1. `step01_config.py`

Contains all values that may need adjustment: dataset path, random seed, split
fractions, voltage window, interpolation length, SOH filters, training settings
and output filenames.

## 2. `step02_data_split.py`

Finds JSON files, shuffles them with the fixed random seed, and assigns whole
files to training, validation and testing. It saves the assignment as
`dataset_file_split.csv` for reproducibility.

## 3. `step03_preprocessing.py`

Reads and validates JSON files, selects charging rows, removes invalid voltage
samples, fills missing current and temperature values, sorts the curves by
voltage, removes duplicate voltages and interpolates every accepted cycle onto a
common 400-point voltage grid.

It processes one file with `prepare_single_file()` and a complete split with
`prepare_file_group()`.

## 4. `step04_feature_engineering.py`

Calculates the reference capacity and SOH target. It creates the four sequential
features and the nine physical features for every accepted cycle.

## 5. `step05_model.py`

Defines the two-input neural network. The charging-curve branch contains three
Conv1D blocks. A small dense branch processes the physical features. The two
representations are joined and passed through dense layers to produce one SOH
value.

## 6. `step06_train.py`

Fits the sequence and physical-feature scalers using training data only. It also
normalizes the SOH target, builds the model, configures callbacks, trains the
network and saves the checkpoint with the lowest validation loss.

## 7. `step07_evaluate.py`

Loads the best model, predicts the test cycles, converts predictions back to SOH
percent, calculates MAE, RMSE and R², and saves overall predictions and
per-battery metrics.

## 8. `step08_plotting.py`

Creates the training-history plot, true-versus-predicted scatter plot, absolute
error histogram, and a cycle-by-cycle SOH plot for every test battery.

## 9. `step09_main.py`

Controls the complete workflow. This is the normal entry point:

```powershell
python step09_main.py
```

The other files mostly define functions, so running them directly normally
produces no output.

## Import relationships

```text
step09_main
 ├── step01_config
 ├── step02_data_split ─────── step01_config
 ├── step03_preprocessing ─── step01_config
 │                            step04_feature_engineering
 ├── step06_train ─────────── step01_config
 │                            step05_model
 ├── step07_evaluate ──────── step01_config
 └── step08_plotting
```

If a Python file is renamed, every import that refers to that filename must also
be updated. Do not include `.py` in an import statement.

