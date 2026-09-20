"""Run the complete multi-battery CNN SOH workflow."""

import random

import numpy as np
import tensorflow as tf

from step01_config import DATA_FOLDER, RANDOM_SEED
from step02_data_split import create_file_splits
from step03_preprocessing import prepare_file_group
from step06_train import scale_datasets, train_model
from step07_evaluate import evaluate_model
from step08_plotting import plot_all_results


def print_dataset_summary(train_set, val_set, test_set):
    print("\n" + "=" * 70)
    print("FINAL PREPROCESSED DATA")
    print("=" * 70)
    for name, dataset in (
        ("TRAIN", train_set),
        ("VALIDATION", val_set),
        ("TEST", test_set),
    ):
        sequence, physical, targets, _, _ = dataset
        print(f"{name}:", sequence.shape, physical.shape, targets.shape)

    print("\nSOH ranges:")
    for name, dataset in (
        ("Train", train_set),
        ("Validation", val_set),
        ("Test", test_set),
    ):
        targets = dataset[2]
        print(f"{name}:", targets.min(), "-", targets.max())


def main():
    # Reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Split complete JSON files so cycles from one battery stay together.
    train_files, val_files, test_files = create_file_splits(DATA_FOLDER)

    # Each tuple contains: sequence, physical features, SOH, cycles, filenames.
    train_set = prepare_file_group(train_files, "TRAINING DATA")
    val_set = prepare_file_group(val_files, "VALIDATION DATA")
    test_set = prepare_file_group(test_files, "TEST DATA")
    print_dataset_summary(train_set, val_set, test_set)

    # Only the first three tuple items are needed for scaling/training.
    scaled = scale_datasets(
        train_set[:3],
        val_set[:3],
        test_set[:3],
    )
    history, model_path = train_model(scaled, DATA_FOLDER)

    evaluation = evaluate_model(
        model_path=model_path,
        scaled=scaled,
        cycle_test=test_set[3],
        file_test=test_set[4],
        data_folder=DATA_FOLDER,
        number_of_test_files=len(test_files),
    )
    plot_all_results(history, evaluation)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

