"""Load, clean, interpolate, and assemble battery-cycle samples."""

import json
import os

import numpy as np
import pandas as pd

from step01_config  import (
    MAX_SOH_JUMP,
    MIN_CURVE_POINTS,
    N_POINTS,
    REMOVE_CYCLE_ZERO,
    V_MAX,
    V_MIN,
)
from step04_feature_engineering  import (
    build_soh_dictionary,
    create_cycle_features,
    valid_soh,
)


def fill_nan(signal):
    """Fill gaps by linear interpolation in both directions."""
    series = pd.Series(np.asarray(signal, dtype=float))
    return series.interpolate(method="linear", limit_direction="both").to_numpy()


def _read_and_validate_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if "cycles_interpolated" not in data:
        raise ValueError("Does not contain cycles_interpolated")
    if "summary" not in data:
        raise ValueError("Does not contain summary")

    cycles = data["cycles_interpolated"]
    for key in ("cycle_index", "step_type", "voltage", "current"):
        if key not in cycles:
            raise ValueError(f"Missing {key}")

    temperature_key = next(
        (
            key
            for key in ("temperature", "temperature_c", "cell_temperature")
            if key in cycles
        ),
        None,
    )
    if temperature_key is None:
        raise ValueError("Temperature field not found")
    return cycles, data["summary"], temperature_key


def _extract_charge_dataframe(cycles, temperature_key):
    cycle_index = np.asarray(cycles["cycle_index"], dtype=int)
    step_type = np.asarray(cycles["step_type"])
    voltage = np.asarray(cycles["voltage"], dtype=float)
    current = np.asarray(cycles["current"], dtype=float)
    temperature = np.asarray(cycles[temperature_key], dtype=float)

    charge_mask = step_type == "charge"
    frame = pd.DataFrame(
        {
            "cycle_index": cycle_index[charge_mask],
            "voltage": voltage[charge_mask],
            "current": current[charge_mask],
            "temperature": temperature[charge_mask],
        }
    )
    if REMOVE_CYCLE_ZERO:
        frame = frame[frame["cycle_index"] != 0].copy()
    return frame


def _clean_and_interpolate_cycle(frame, voltage_grid):
    """Return current and temperature on the shared voltage grid."""
    voltage = frame["voltage"].to_numpy(dtype=float)
    current = frame["current"].to_numpy(dtype=float)
    temperature = frame["temperature"].to_numpy(dtype=float)

    valid_voltage = np.isfinite(voltage)
    voltage = voltage[valid_voltage]
    current = current[valid_voltage]
    temperature = temperature[valid_voltage]
    if len(voltage) < MIN_CURVE_POINTS:
        return None

    current = fill_nan(current)
    temperature = fill_nan(temperature)
    valid = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(temperature)
    voltage, current, temperature = (
        voltage[valid],
        current[valid],
        temperature[valid],
    )
    if len(voltage) < MIN_CURVE_POINTS:
        return None

    order = np.argsort(voltage)
    voltage, current, temperature = (
        voltage[order],
        current[order],
        temperature[order],
    )
    unique_voltage, unique_indices = np.unique(voltage, return_index=True)
    unique_current = current[unique_indices]
    unique_temperature = temperature[unique_indices]

    if len(unique_voltage) < MIN_CURVE_POINTS:
        return None
    if unique_voltage.min() > V_MIN or unique_voltage.max() < V_MAX:
        return None

    return (
        np.interp(voltage_grid, unique_voltage, unique_current),
        np.interp(voltage_grid, unique_voltage, unique_temperature),
    )


def prepare_single_file(file_path, verbose=False):
    """Convert one battery JSON file into cycle-level model samples."""
    filename = os.path.basename(file_path)
    try:
        cycles, summary, temperature_key = _read_and_validate_json(file_path)
        charge_data = _extract_charge_dataframe(cycles, temperature_key)
        if charge_data.empty:
            raise ValueError("No charging data")
        soh_by_cycle = build_soh_dictionary(summary)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        print(f"Could not prepare {filename}: {error}")
        return None

    voltage_grid = np.linspace(V_MIN, V_MAX, N_POINTS)
    sequences, physical, targets, cycle_numbers, filenames = [], [], [], [], []
    previous_valid_soh = None

    for cycle in np.sort(charge_data["cycle_index"].unique()):
        if cycle not in soh_by_cycle:
            continue
        soh = soh_by_cycle[cycle]
        if not valid_soh(soh):
            continue
        if (
            previous_valid_soh is not None
            and abs(soh - previous_valid_soh) > MAX_SOH_JUMP
        ):
            continue

        cycle_frame = charge_data[charge_data["cycle_index"] == cycle]
        interpolated = _clean_and_interpolate_cycle(cycle_frame, voltage_grid)
        if interpolated is None:
            continue

        current, temperature = interpolated
        features = create_cycle_features(voltage_grid, current, temperature)
        if features is None:
            continue

        sequence_features, physical_features = features
        sequences.append(sequence_features)
        physical.append(physical_features)
        targets.append(soh)
        cycle_numbers.append(cycle)
        filenames.append(filename)
        previous_valid_soh = soh

    if not sequences:
        print(f"{filename}: no usable cycles")
        return None

    result = {
        "sequence": np.asarray(sequences, dtype=np.float32),
        "physical": np.asarray(physical, dtype=np.float32),
        "y": np.asarray(targets, dtype=np.float32),
        "cycle": np.asarray(cycle_numbers, dtype=int),
        "file": np.asarray(filenames),
    }
    if verbose:
        print(filename, "usable cycles:", len(result["y"]))
    return result


def prepare_file_group(files, group_name):
    """Process and combine every battery file assigned to one data split."""
    print("\n" + "=" * 70)
    print("PREPARING", group_name)
    print("=" * 70)

    collected = {key: [] for key in ("sequence", "physical", "y", "cycle", "file")}
    successful_files = 0
    failed_files = 0

    for index, file_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}]", os.path.basename(file_path))
        result = prepare_single_file(file_path)
        if result is None:
            failed_files += 1
            continue
        successful_files += 1
        for key in collected:
            collected[key].append(result[key])

    if not collected["sequence"]:
        raise ValueError(f"No usable data in {group_name}")

    combined = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    print("\nCompleted:", group_name)
    print("Successful files:", successful_files)
    print("Rejected files:", failed_files)
    print("Total cycles:", len(combined["y"]))
    print("Sequence shape:", combined["sequence"].shape)
    print("Physical shape:", combined["physical"].shape)
    print("SOH range:", combined["y"].min(), "to", combined["y"].max())

    return (
        combined["sequence"],
        combined["physical"],
        combined["y"],
        combined["cycle"],
        combined["file"],
    )

