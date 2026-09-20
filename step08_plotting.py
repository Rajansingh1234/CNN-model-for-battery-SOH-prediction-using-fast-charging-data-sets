"""Plots for training behaviour and final test performance."""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.title("Multi-cell CNN Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_true_vs_predicted(true_soh, predicted_soh):
    plt.figure(figsize=(7, 7))
    plt.scatter(true_soh, predicted_soh, alpha=0.5)
    minimum = min(true_soh.min(), predicted_soh.min())
    maximum = max(true_soh.max(), predicted_soh.max())
    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
    plt.xlabel("True SOH (%)")
    plt.ylabel("Predicted SOH (%)")
    plt.title("All Test Cells: True vs Predicted SOH")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_error_distribution(true_soh, predicted_soh):
    absolute_error = np.abs(true_soh - predicted_soh)
    plt.figure(figsize=(9, 5))
    plt.hist(absolute_error, bins=40)
    plt.xlabel("Absolute SOH Error (%)")
    plt.ylabel("Number of cycles")
    plt.title("SOH Prediction Error Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_each_test_battery(cycles, true_soh, predicted_soh, source_files):
    for filename in np.unique(source_files):
        mask = source_files == filename
        file_cycles = cycles[mask]
        file_true = true_soh[mask]
        file_predicted = predicted_soh[mask]
        order = np.argsort(file_cycles)

        plt.figure(figsize=(11, 5))
        plt.plot(file_cycles[order], file_true[order], label="True SOH")
        plt.plot(
            file_cycles[order], file_predicted[order], label="Predicted SOH"
        )
        plt.xlabel("Cycle index")
        plt.ylabel("SOH (%)")
        plt.title(filename)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_all_results(history, evaluation):
    """Create the same four groups of plots as the original script."""
    plot_training_history(history)
    plot_true_vs_predicted(
        evaluation["true_soh"], evaluation["predicted_soh"]
    )
    plot_error_distribution(
        evaluation["true_soh"], evaluation["predicted_soh"]
    )
    plot_each_test_battery(
        evaluation["cycles"],
        evaluation["true_soh"],
        evaluation["predicted_soh"],
        evaluation["files"],
    )

