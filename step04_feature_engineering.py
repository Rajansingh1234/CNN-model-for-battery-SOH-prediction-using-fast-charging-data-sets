"""Create SOH targets, CNN sequences, and physical summary features."""

import numpy as np

from step01_config  import (
    MAX_VALID_SOH,
    MIN_VALID_SOH,
    REFERENCE_END,
    REFERENCE_START,
)


def build_soh_dictionary(summary):
    """Calculate SOH using the median valid capacity from early cycles."""
    if "cycle_index" not in summary or "discharge_capacity" not in summary:
        raise ValueError("Summary capacity unavailable")

    summary_cycles = np.asarray(summary["cycle_index"], dtype=int)
    capacities = np.asarray(summary["discharge_capacity"], dtype=float)
    capacity_by_cycle = dict(zip(summary_cycles, capacities))

    reference_values = [
        capacity_by_cycle[cycle]
        for cycle in range(REFERENCE_START, REFERENCE_END + 1)
        if cycle in capacity_by_cycle
        and np.isfinite(capacity_by_cycle[cycle])
        and capacity_by_cycle[cycle] > 0
    ]
    if len(reference_values) < 3:
        raise ValueError("Insufficient early reference capacities")

    reference_capacity = np.median(reference_values)
    return {
        cycle: (capacity / reference_capacity) * 100.0
        for cycle, capacity in capacity_by_cycle.items()
        if np.isfinite(capacity) and capacity > 0
    }


def valid_soh(soh):
    """Check the physically plausible SOH range."""
    return MIN_VALID_SOH <= soh <= MAX_VALID_SOH


def create_cycle_features(voltage_grid, current, temperature):
    """Create the four sequence features and nine physical features."""
    temperature_rise = temperature - temperature[0]

    current_scale = np.max(np.abs(current))
    if current_scale < 1e-8:
        return None
    normalized_current = current / current_scale

    current_gradient = np.gradient(normalized_current, voltage_grid)
    temperature_gradient = np.gradient(temperature_rise, voltage_grid)

    # Clip isolated derivative spikes at the 99th absolute percentile.
    current_limit = np.percentile(np.abs(current_gradient), 99)
    temperature_limit = np.percentile(np.abs(temperature_gradient), 99)
    if current_limit > 0:
        current_gradient = np.clip(
            current_gradient, -current_limit, current_limit
        )
    if temperature_limit > 0:
        temperature_gradient = np.clip(
            temperature_gradient, -temperature_limit, temperature_limit
        )

    sequence_features = np.column_stack(
        [
            normalized_current,
            temperature_rise,
            current_gradient,
            temperature_gradient,
        ]
    ).astype(np.float32)

    physical_features = np.asarray(
        [
            temperature[-1] - temperature[0],
            np.mean(temperature),
            np.max(temperature),
            np.mean(current),
            np.std(current),
            current[0],
            current[-1],
            np.trapezoid(current, voltage_grid),
            np.trapezoid(temperature_rise, voltage_grid),
        ],
        dtype=np.float32,
    )

    if not np.all(np.isfinite(sequence_features)):
        return None
    if not np.all(np.isfinite(physical_features)):
        return None
    return sequence_features, physical_features

