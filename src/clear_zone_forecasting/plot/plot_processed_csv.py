"""Plot wide-format sensor data produced by :mod:`resample_csv`."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SENSORS = ("co2", "temp", "hum")
COLORS = {"co2": "#2b6cb0", "temp": "#dd6b20", "hum": "#2f855a"}


def load_processed_data(csv_path: Path) -> pd.DataFrame:
    """Read processed sensor data and an optional ``segment_id`` column."""
    data = pd.read_csv(csv_path)
    required = {"ts", *SENSORS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "Processed CSV is missing required column(s): "
            + ", ".join(sorted(missing))
        )
    if data.empty:
        raise ValueError("CSV file contains no data rows")

    data["ts"] = pd.to_datetime(data["ts"], utc=True, errors="coerce")
    if data["ts"].isna().any():
        raise ValueError("CSV contains invalid timestamps")
    for sensor in SENSORS:
        data[sensor] = pd.to_numeric(data[sensor], errors="coerce")

    data = data.sort_values("ts").set_index("ts")
    columns = [*SENSORS]
    if "segment_id" in data.columns:
        data["segment_id"] = pd.to_numeric(data["segment_id"], errors="coerce")
        columns.append("segment_id")
    return data.loc[:, columns]


def plot_processed_csv(csv_path: Path, output_path: Path | None = None) -> None:
    """Plot each sensor from a processed CSV; NaNs remain visible as gaps."""
    data = load_processed_data(csv_path)
    fig, axes = plt.subplots(
        len(SENSORS), 1, figsize=(12, 8), sharex=True, constrained_layout=True
    )
    for sensor, axis in zip(SENSORS, axes, strict=True):
        axis.plot(data.index, data[sensor], color=COLORS[sensor], linewidth=1.2)
        axis.set_title(sensor.upper())
        axis.set_ylabel("Value")
        axis.grid(True, alpha=0.3)

    if "segment_id" in data.columns:
        segment_ids = data["segment_id"].dropna().unique()
        for segment_id in sorted(segment_ids):
            segment = data.loc[data["segment_id"] == segment_id]
            if segment.empty:
                continue
            midpoint = segment.index[len(segment) // 2]
            for axis in axes:
                axis.axvspan(
                    segment.index[0],
                    segment.index[-1],
                    color="#718096",
                    alpha=0.08,
                    linewidth=0,
                )
                axis.text(
                    midpoint,
                    0.95,
                    f"Segment {int(segment_id)}",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#4a5568",
                )

    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(f"Processed sensor measurements — {csv_path.name}")

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        print(f"Chart saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot co2, temp, and hum from a processed sensor CSV."
    )
    parser.add_argument(
        "csv_file", type=Path, help="Processed CSV with ts,co2,temp,hum columns"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Save chart to this image path instead of opening a window",
    )
    args = parser.parse_args()
    try:
        plot_processed_csv(args.csv_file, args.output)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
