"""Scale the data, train the CNN, and save its best validation checkpoint."""

import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from step01_config  import BATCH_SIZE, EPOCHS, MODEL_FILENAME
from step05_model  import build_model


def scale_datasets(train_data, val_data, test_data):
    """Fit every scaler on training data only and transform all splits."""
    x_train, p_train, y_train = train_data
    x_val, p_val, y_val = val_data
    x_test, p_test, y_test = test_data

    n_sequence_features = x_train.shape[2]
    sequence_scaler = StandardScaler()
    sequence_scaler.fit(x_train.reshape(-1, n_sequence_features))

    def scale_sequence(values):
        return sequence_scaler.transform(
            values.reshape(-1, n_sequence_features)
        ).reshape(values.shape)

    physical_scaler = StandardScaler()
    p_train_scaled = physical_scaler.fit_transform(p_train)
    p_val_scaled = physical_scaler.transform(p_val)
    p_test_scaled = physical_scaler.transform(p_test)

    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-8:
        raise ValueError("Training SOH standard deviation is too small.")

    print("\nTarget normalization:")
    print("Training SOH mean:", y_mean)
    print("Training SOH std:", y_std)

    return {
        "x_train": scale_sequence(x_train),
        "x_val": scale_sequence(x_val),
        "x_test": scale_sequence(x_test),
        "p_train": p_train_scaled,
        "p_val": p_val_scaled,
        "p_test": p_test_scaled,
        "y_train": (y_train - y_mean) / y_std,
        "y_val": (y_val - y_mean) / y_std,
        "y_test": y_test,
        "y_mean": y_mean,
        "y_std": y_std,
        "sequence_scaler": sequence_scaler,
        "physical_scaler": physical_scaler,
    }


def train_model(scaled, data_folder):
    """Build and train the model, returning its history and checkpoint path."""
    cnn_model = build_model(
        n_points=scaled["x_train"].shape[1],
        n_sequence_features=scaled["x_train"].shape[2],
        n_physical_features=scaled["p_train"].shape[1],
    )
    cnn_model.summary()

    model_path = os.path.join(data_folder, MODEL_FILENAME)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=30,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = cnn_model.fit(
        {
            "sequence_input": scaled["x_train"],
            "physical_input": scaled["p_train"],
        },
        scaled["y_train"],
        validation_data=(
            {
                "sequence_input": scaled["x_val"],
                "physical_input": scaled["p_val"],
            },
            scaled["y_val"],
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )
    return history, model_path

