"""Configuration for the multi-battery CNN SOH project."""

DATA_FOLDER = r"C:\Users\rajan\Desktop\DATA\JSON_file"

# Reproducibility
RANDOM_SEED = 42

# Data split (performed at JSON-file/battery level)
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.20
TEST_FRACTION = 0.10

# Preprocessing
REMOVE_CYCLE_ZERO = True
REFERENCE_START = 1
REFERENCE_END = 10
V_MIN = 3.10
V_MAX = 3.55
N_POINTS = 400
MIN_VALID_SOH = 50.0
MAX_VALID_SOH = 110.0
MAX_SOH_JUMP = 8.0
MIN_CURVE_POINTS = 20

# Model training
LEARNING_RATE = 1e-4
EPOCHS = 70
BATCH_SIZE = 32

# Output filenames
SPLIT_FILENAME = "dataset_file_split.csv"
MODEL_FILENAME = "best_multi_cell_soh_cnn.keras"
PREDICTIONS_FILENAME = "cnn_test_predictions.csv"
PER_FILE_METRICS_FILENAME = "cnn_test_per_file_metrics.csv"

