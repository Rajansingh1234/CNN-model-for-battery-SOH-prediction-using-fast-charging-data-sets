# Features and model architecture

## Target

The target for each cycle is:

```text
SOH (%) = discharge capacity / median early-cycle reference capacity × 100
```

## Sequential CNN input

Each charging curve is interpolated at 400 equally spaced voltage values between
3.10 V and 3.55 V. Four values are supplied at every voltage point:

| Channel | Meaning |
|---|---|
| Normalized current | Current divided by the largest absolute current in the cycle |
| Temperature rise | Temperature relative to the first voltage point |
| `dI/dV` | Change in normalized current with voltage |
| `dT/dV` | Change in temperature rise with voltage |

The input shape for one cycle is therefore `400 × 4`.

## Nine physical inputs

| # | Feature | Meaning |
|---:|---|---|
| 1 | Total temperature rise | Final temperature minus initial temperature |
| 2 | Mean temperature | Average across the voltage window |
| 3 | Maximum temperature | Highest value in the voltage window |
| 4 | Mean current | Average charging current |
| 5 | Current standard deviation | Amount of current variation |
| 6 | Initial current | Current at the beginning of the window |
| 7 | Final current | Current at the end of the window |
| 8 | Current-voltage area | Numerical integral of current with voltage |
| 9 | Temperature-voltage area | Integral of temperature rise with voltage |

The current-voltage area is a curve-shape feature. It is not charge capacity,
because capacity requires current to be integrated with respect to time.

## Network layout

```text
Sequence input: 400 × 4             Physical input: 9
          │                                  │
Conv1D(32, kernel 9)                    Dense(32)
BatchNorm → ReLU                     BatchNorm → Dropout
MaxPool → Dropout                           │
          │                                  │
Conv1D(64, kernel 7)                         │
BatchNorm → ReLU                            │
MaxPool → Dropout                           │
          │                                  │
Conv1D(96, kernel 5)                         │
BatchNorm → ReLU                            │
          │                                  │
Global average + max pooling                 │
          └──────────────┬───────────────────┘
                         │
                      Concatenate
                         │
             Dense(128) → BatchNorm → Dropout
                         │
                  Dense(64) → Dropout
                         │
                     Dense(32)
                         │
               Linear output: scaled SOH
```

Conv1D layers learn local shapes within the charging curves. Global pooling
summarizes the detected shapes, while dense layers combine them with the
physical measurements.

## Training controls

- Adam optimizer with gradient clipping;
- Huber loss to reduce the influence of large outliers;
- L2 regularization and dropout to limit overfitting;
- early stopping based on validation loss;
- automatic learning-rate reduction;
- checkpointing of the best validation model.

Sequence features, physical features and the SOH target are standardized using
training-set statistics only. This prevents validation and test information from
entering model training.

