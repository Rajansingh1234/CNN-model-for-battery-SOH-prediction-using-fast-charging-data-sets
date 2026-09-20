# Expected JSON data format

The program expects a folder containing at least three `.json` files. Each file
should represent one battery or another independent experimental unit.

Set that folder in `step01_config.py`:

```python
DATA_FOLDER = r"C:\path\to\your\JSON_file"
```

## Required top-level sections

Each JSON file must contain:

```json
{
  "cycles_interpolated": {},
  "summary": {}
}
```

## `cycles_interpolated`

The following arrays are required and must describe corresponding measurement
rows:

| Field | Meaning |
|---|---|
| `cycle_index` | Cycle number for every measurement |
| `step_type` | Operating step; charging rows must contain `"charge"` |
| `voltage` | Cell voltage |
| `current` | Charging current |

One of these temperature fields must also exist:

- `temperature`
- `temperature_c`
- `cell_temperature`

Example:

```json
{
  "cycles_interpolated": {
    "cycle_index": [1, 1, 1, 2, 2, 2],
    "step_type": ["charge", "charge", "charge", "charge", "charge", "charge"],
    "voltage": [3.10, 3.30, 3.55, 3.10, 3.30, 3.55],
    "current": [1.5, 1.4, 0.5, 1.5, 1.3, 0.4],
    "temperature": [25.0, 25.4, 26.0, 25.1, 25.6, 26.3]
  }
}
```

The real curves need substantially more measurements than this shortened
example. The code requires at least 20 usable voltage points and coverage of the
configured voltage window.

## `summary`

The summary must contain:

| Field | Meaning |
|---|---|
| `cycle_index` | Cycle number |
| `discharge_capacity` | Measured discharge capacity for that cycle |

Example:

```json
{
  "summary": {
    "cycle_index": [1, 2, 3, 4],
    "discharge_capacity": [2.01, 2.00, 1.99, 1.98]
  }
}
```

At least three valid capacity values must exist between the configured reference
cycles, which are cycles 1–10 by default.

## Cycle acceptance rules

A charging cycle is used only when it:

- has a matching discharge-capacity target;
- contains enough valid voltage, current and temperature values;
- covers the complete voltage interval from 3.10 V to 3.55 V;
- has an SOH between 50% and 110%;
- does not exceed the configured isolated SOH-jump limit.

Missing current and temperature values are filled by linear interpolation when
possible. Invalid voltage rows and duplicate voltage values are removed.

