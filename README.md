# Multi-Battery CNN for State-of-Health Estimation

This project estimates battery state of health (SOH) from charging current and
temperature measurements. It uses a two-input neural network:

- a one-dimensional CNN learns patterns from charging curves;
- a dense branch processes nine physical summary features;
- both branches are combined to predict SOH as a percentage.

SOH is calculated from discharge capacity:

```text
SOH (%) = cycle discharge capacity / reference capacity × 100
```

The reference capacity is the median valid discharge capacity from cycles 1–10.

## Project structure

```text
battery_soh_cnn_modular/
├── step01_config.py
├── step02_data_split.py
├── step03_preprocessing.py
├── step04_feature_engineering.py
├── step05_model.py
├── step06_train.py
├── step07_evaluate.py
├── step08_plotting.py
├── step09_main.py
├── requirements.txt
├── .gitignore
├── README.md
└── docs/
    ├── INSTALLATION.md
    ├── DATA_FORMAT.md
    ├── CODE_GUIDE.md
    ├── MODEL_ARCHITECTURE.md
    ├── RUNNING_AND_OUTPUTS.md
    └── TROUBLESHOOTING.md
```

## Quick start

1. Create and activate a Python virtual environment.
2. Install the required libraries:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Open `step01_config.py` and set `DATA_FOLDER` to the directory containing
   the battery JSON files.
4. Run the complete workflow:

   ```powershell
   python step09_main.py
   ```

Keep all nine Python files in the same folder. The other modules define reusable
functions; `step09_main.py` calls them in the correct order.

## Workflow

```text
JSON battery files
        ↓
File-level train/validation/test split
        ↓
Charging-data cleaning and interpolation
        ↓
Four sequential features + nine physical features
        ↓
Training-only standardization
        ↓
Two-input 1D CNN training
        ↓
Test prediction, metrics, CSV files and plots
```

The split is performed at file level. If one JSON file represents one battery,
all cycles from that battery remain in the same split, reducing leakage between
training and testing.

## Model inputs

Each usable charging cycle produces:

- a `400 × 4` sequence containing normalized current, temperature rise,
  `dI/dV`, and `dT/dV`;
- nine physical values summarizing temperature and current behaviour.

The target is one SOH percentage for that cycle. See
[MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md) for the complete layout.

## Documentation

- [Installation and libraries](docs/INSTALLATION.md)
- [Expected JSON data](docs/DATA_FORMAT.md)
- [Explanation of each Python file](docs/CODE_GUIDE.md)
- [Features and CNN architecture](docs/MODEL_ARCHITECTURE.md)
- [Running the project and understanding outputs](docs/RUNNING_AND_OUTPUTS.md)
- [Common problems](docs/TROUBLESHOOTING.md)

## Main results

The program reports MAE, RMSE and R² for the complete test set and for each test
battery. It also saves test predictions, file-level metrics, the best model, and
the reproducible dataset split.

## Research note

Record the random seed, dataset version, voltage window, file assignments and
software environment when reporting thesis results. Performance should be
reported on the untouched test batteries after model selection is complete.

