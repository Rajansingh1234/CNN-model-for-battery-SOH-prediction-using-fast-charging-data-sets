"""Find JSON files and split whole battery files into data groups."""

import os
import random

import numpy as np
import pandas as pd

from step01_config import (
    RANDOM_SEED,
    SPLIT_FILENAME,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
)


def find_json_files(data_folder):
    """Return all JSON paths in deterministic filename order."""
    json_files = sorted(
        os.path.join(data_folder, filename)
        for filename in os.listdir(data_folder)
        if filename.lower().endswith(".json")
    )

    print("=" * 70)
    print("JSON FILE SEARCH")
    print("=" * 70)
    print("Total JSON files found:", len(json_files))

    if len(json_files) < 3:
        raise ValueError(
            "At least 3 JSON files are needed for train/validation/test splitting."
        )
    return json_files


def split_json_files(json_files, data_folder):
    """Shuffle and split files, then record the assignment in a CSV file."""
    files = list(json_files)
    random.Random(RANDOM_SEED).shuffle(files)
    n_files = len(files)

    n_train = max(1, int(np.floor(TRAIN_FRACTION * n_files)))
    n_val = max(1, int(np.floor(VAL_FRACTION * n_files)))
    n_test = n_files - n_train - n_val

    if n_test < 1:
        raise ValueError("Not enough files for a test set.")

    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    # This also makes it visible if future fractions do not total 1.0.
    expected_test = TEST_FRACTION * n_files
    if abs(n_test - expected_test) > 1.0:
        print("Note: test size contains all files remaining after train/validation split.")

    groups = {
        "TRAIN": train_files,
        "VALIDATION": val_files,
        "TEST": test_files,
    }
    print("\nFile split:")
    for name, group in groups.items():
        print(f"{name.title()} files:", len(group))
        print(f"\n{name} FILES")
        for path in group:
            print(os.path.basename(path))

    rows = [
        {"filename": os.path.basename(path), "split": split_name}
        for split_name, group in (
            ("train", train_files),
            ("validation", val_files),
            ("test", test_files),
        )
        for path in group
    ]
    split_path = os.path.join(data_folder, SPLIT_FILENAME)
    pd.DataFrame(rows).to_csv(split_path, index=False)
    print("\nDataset split saved to:")
    print(split_path)

    return train_files, val_files, test_files


def create_file_splits(data_folder):
    """Convenience function used by main.py."""
    return split_json_files(find_json_files(data_folder), data_folder)

