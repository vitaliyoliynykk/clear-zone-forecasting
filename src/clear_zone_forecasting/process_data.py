"""Resample long-format sensor CSV data onto a one-minute time grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SENSORS = ("co2", "temp", "hum")
MIN_SEGMENT_LENGTH = 180


def clean_csv(input_path: Path) -> pd.DataFrame:
    """Read and clean long-format sensor data from a CSV file.

    Duplicate timestamps are removed, keeping the first reading. Rows where
    CO2 is zero or humidity exceeds 100 are treated as anomalies and removed.
    """
    data = pd.read_csv(input_path)
    required = {"ts", "value", "sensor"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError("CSV is missing required column(s): " + ", ".join(sorted(missing)))
    if data.empty:
        raise ValueError("CSV file contains no measurement rows")

    data["ts"] = pd.to_datetime(data["ts"], utc=True, errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data["sensor"] = data["sensor"].astype("string").str.strip().str.lower()
    invalid = data["ts"].isna() | data["value"].isna() | data["sensor"].isna()

    if invalid.any():
        raise ValueError(f"CSV contains {int(invalid.sum())} invalid row(s)")

    unknown = sorted(set(data["sensor"]) - set(SENSORS))
    if unknown:
        raise ValueError("Unknown sensor(s): " + ", ".join(unknown))

    before = len(data)
    duplicate_timestamps = data["ts"].duplicated(keep="first")
    duplicate_count = int(duplicate_timestamps.sum())
    data = data.loc[~duplicate_timestamps].copy()

    co2_anomalies = data["sensor"].eq("co2") & data["value"].eq(0)
    humidity_anomalies = data["sensor"].eq("hum") & data["value"].gt(100)
    co2_count = int(co2_anomalies.sum())
    humidity_count = int(humidity_anomalies.sum())
    data = data.loc[~(co2_anomalies | humidity_anomalies)].copy()

    print("CSV cleaning report:")
    print(f"  Rows read: {before}")
    print(f"  Duplicate timestamps removed: {duplicate_count}")
    print(f"  CO2 zero anomalies removed: {co2_count}")
    print(f"  Humidity > 100 anomalies removed: {humidity_count}")
    print(f"  Rows remaining: {len(data)}")

    if data.empty:
        raise ValueError("CSV contains no rows after cleaning")

    return data


def resample_csv(data: pd.DataFrame) -> pd.DataFrame:
    """Write one row per minute with a separate column for each sensor.

    Readings within the same minute are averaged. Empty minutes are filled
    using time-based linear interpolation between readings.
    """
    data["minute"] = data["ts"].dt.floor("min")
    wide = data.pivot_table(
        index="minute", columns="sensor", values="value", aggfunc="mean"
    )
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq="1min", tz="UTC")

    result = wide.reindex(index=grid, columns=SENSORS).interpolate(
        method="time", limit_area="inside", limit=15
    )
    result.index.name = "ts"

    return result


def segment_data(data: pd.DataFrame) -> pd.DataFrame:
    """Keep uninterrupted data runs of at least 180 rows and label each run.

    A row is considered missing when all sensor values are empty. Each
    contiguous run of nonmissing, one-minute rows becomes a separate segment.
    """
    if data.empty:
        result = data.copy()
        result["segment_id"] = pd.Series(dtype="int64")
        print("Segments created: 0")
        return result

    ordered = data.sort_index().copy()
    has_data = ordered.loc[:, SENSORS].notna().any(axis=1)
    consecutive_minute = ordered.index.to_series().diff().eq(pd.Timedelta(minutes=1))
    starts_segment = has_data & (~has_data.shift(fill_value=False) | ~consecutive_minute)
    run_ids = starts_segment.cumsum()

    segments: list[pd.DataFrame] = []
    segment_lengths: list[int] = []
    segment_id = 1
    for _, run in ordered.loc[has_data].groupby(run_ids[has_data], sort=False):
        if len(run) < MIN_SEGMENT_LENGTH:
            continue
        run = run.copy()
        run["segment_id"] = segment_id
        segments.append(run)
        segment_lengths.append(len(run))
        segment_id += 1

    if segments:
        result = pd.concat(segments)
    else:
        result = ordered.iloc[0:0].copy()
        result["segment_id"] = pd.Series(index=result.index, dtype="int64")

    print(f"Segments created: {segment_id - 1}")
    for index, length in enumerate(segment_lengths, start=1):
        print(f"  Segment {index}: {length} entries")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resample ts,value,sensor CSV measurements onto a one-minute grid."
    )
    parser.add_argument("input_csv", type=Path, help="Raw input CSV")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV (default: <input-name>-processed.csv beside the input)",
    )
    args = parser.parse_args()
    output_path = args.output or args.input_csv.with_name(
        f"{args.input_csv.stem}-processed.csv"
    )
    try:
        cleaned = clean_csv(args.input_csv)
        resampled = resample_csv(cleaned)
        result = segment_data(resampled)

        result.to_csv(output_path, date_format="%Y-%m-%dT%H:%M:%SZ", na_rep="")
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
