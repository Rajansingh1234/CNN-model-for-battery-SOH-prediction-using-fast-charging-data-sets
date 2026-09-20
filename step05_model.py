"""Define and compile the two-input 1D CNN for SOH regression."""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.regularizers import l2

from step01_config  import LEARNING_RATE


def build_model(n_points, n_sequence_features, n_physical_features):
    """Build the CNN curve branch, physical branch, and regression head."""
    sequence_input = Input(
        shape=(n_points, n_sequence_features), name="sequence_input"
    )

    x = Conv1D(32, 9, padding="same", kernel_regularizer=l2(1e-4))(
        sequence_input
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.10)(x)

    x = Conv1D(64, 7, padding="same", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.15)(x)

    x = Conv1D(96, 5, padding="same", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    average_features = GlobalAveragePooling1D()(x)
    maximum_features = GlobalMaxPooling1D()(x)
    cnn_features = Concatenate()([average_features, maximum_features])

    physical_input = Input(
        shape=(n_physical_features,), name="physical_input"
    )
    p = Dense(32, activation="relu", kernel_regularizer=l2(1e-4))(
        physical_input
    )
    p = BatchNormalization()(p)
    p = Dropout(0.20)(p)

    combined = Concatenate()([cnn_features, p])
    combined = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(
        combined
    )
    combined = BatchNormalization()(combined)
    combined = Dropout(0.35)(combined)
    combined = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(
        combined
    )
    combined = Dropout(0.25)(combined)
    combined = Dense(32, activation="relu", kernel_regularizer=l2(1e-4))(
        combined
    )

    output = Dense(1, activation="linear", name="SOH")(combined)
    cnn_model = Model(
        inputs=[sequence_input, physical_input], outputs=output
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, clipnorm=1.0
    )
    cnn_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=0.5),
        metrics=["mae"],
    )
    return cnn_model

