"""Plot minute-level sensor measurements from a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SENSORS = ("co2", "temp", "hum")


def load_minute_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV rows and return one value per sensor and minute.

    If several readings for a sensor fall within one minute, their mean is
    plotted. Minutes without readings are retained as NaN so Matplotlib shows
    visible gaps instead of connecting across missing data.
    """
    data = pd.read_csv(csv_path)
    required_columns = {"ts", "value", "sensor"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            "CSV must contain columns ts, value, sensor; missing: "
            + ", ".join(sorted(missing_columns))
        )
    if data.empty:
        raise ValueError("CSV file contains no measurement rows")

    data["ts"] = pd.to_datetime(data["ts"], utc=True, errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data["sensor"] = data["sensor"].astype("string").str.strip().str.lower()

    invalid_rows = data["ts"].isna() | data["value"].isna()
    if invalid_rows.any():
        raise ValueError(
            f"CSV has {int(invalid_rows.sum())} row(s) with invalid timestamp or value"
        )

    unknown_sensors = sorted(set(data["sensor"].dropna()) - set(SENSORS))
    if unknown_sensors:
        raise ValueError("Unknown sensor(s): " + ", ".join(unknown_sensors))

    data["minute"] = data["ts"].dt.floor("min")
    minute_values = data.pivot_table(
        index="minute", columns="sensor", values="value", aggfunc="mean"
    )
    minute_index = pd.date_range(
        start=data["minute"].min(),
        end=data["minute"].max(),
        freq="1min",
        tz="UTC",
    )
    return minute_values.reindex(index=minute_index, columns=SENSORS)


def plot_csv(csv_path: Path, output_path: Path | None = None) -> None:
    """Plot all three sensor series, optionally saving the figure."""
    minute_data = load_minute_data(csv_path)
    fig, axes = plt.subplots(
        len(SENSORS), 1, figsize=(12, 8), sharex=True, constrained_layout=True
    )
    colors = {"co2": "#2b6cb0", "temp": "#dd6b20", "hum": "#2f855a"}

    for sensor, axis in zip(SENSORS, axes, strict=True):
        values = minute_data[sensor]
        axis.plot(minute_data.index, values, color=colors[sensor], linewidth=1.2)
        axis.set_title(sensor.upper())
        axis.set_ylabel("Value")
        axis.grid(True, alpha=0.3)
        # Mark absent minute slots explicitly, including leading/trailing gaps
        # for a sensor within the full dataset's time range.


    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(f"Sensor measurements by minute — {csv_path.name}")

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        print(f"Chart saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot co2, temp, and hum readings from a CSV at one-minute steps."
    )
    parser.add_argument("csv_file", type=Path, help="Input CSV with ts,value,sensor columns")
    parser.add_argument(
        "-o", "--output", type=Path, help="Save the chart to this image path instead of opening it"
    )
    args = parser.parse_args()
    try:
        plot_csv(args.csv_file, args.output)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
