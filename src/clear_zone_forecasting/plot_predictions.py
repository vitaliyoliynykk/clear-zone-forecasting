"""Plot raw CO2 measurements alongside rolling model forecasts."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clear_zone_forecasting.process_data import clean_csv, resample_csv, segment_data


SENSOR_FEATURES = ["co2", "temp", "hum"]
FEATURES = SENSOR_FEATURES + ["hour_sin", "hour_cos", "day_sin", "day_cos"]
WINDOW_SIZE = 180
FORECAST_HORIZONS = (60, 120, 180)
FORECAST_STEP_MINUTES = 60


def prepare_features(data: pd.DataFrame, scaler) -> pd.DataFrame:
    """Apply the training scaler and create the model's local-time features."""
    result = data.copy()
    result[SENSOR_FEATURES] = scaler.transform(result[SENSOR_FEATURES])

    local_time = result.index.tz_convert("Europe/Kyiv")
    hour = local_time.hour + local_time.minute / 60
    day = local_time.day_of_week 
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["day_sin"] = np.sin(2 * np.pi * day / 7)
    result["day_cos"] = np.cos(2 * np.pi * day / 7)
    return result


def predict_next_three(
    data: pd.DataFrame, model: keras.Model, scaler
) -> list[float]:
    """Predict CO₂ at 60, 120, and 180 minutes from exactly 180 rows.

    The returned values are de-normalised to the original CO₂ units (ppm).
    """
    if len(data) != WINDOW_SIZE:
        raise ValueError(f"Expected exactly {WINDOW_SIZE} rows, received {len(data)}")

    missing = set(SENSOR_FEATURES).difference(data.columns)
    if missing:
        raise ValueError(f"Missing required sensor columns: {', '.join(sorted(missing))}")

    features = prepare_features(data.sort_index(), scaler)
    window = features[FEATURES].to_numpy(dtype=np.float32)
    if not np.isfinite(window).all():
        raise ValueError("Input window contains missing or non-finite values")

    prediction = model.predict(window[np.newaxis, ...], verbose=0)[0]
    co2_index = scaler.feature_names_in_.tolist().index("co2")
    prediction_ppm = prediction * scaler.scale_[co2_index] + scaler.mean_[co2_index]
    return [float(value) for value in prediction_ppm]


def create_forecasts(
    data: pd.DataFrame, model: keras.Model, scaler
) -> dict[int, tuple[list[pd.Timestamp], list[float]]]:
    """Forecast each model horizon from a new input window every 15 minutes."""
    
    forecasts = {
        horizon: ([], [])
        for horizon in FORECAST_HORIZONS
    }
    for _, segment in data.groupby("segment_id", sort=False):
        segment = segment.sort_index()

        for end in range(WINDOW_SIZE, len(segment) + 1, FORECAST_STEP_MINUTES):
            window = segment.iloc[end - WINDOW_SIZE : end]
            try:
                prediction_ppm = predict_next_three(window, model, scaler)
            except ValueError:
                continue
            origin = segment.index[end - 1]
            for index, horizon in enumerate(FORECAST_HORIZONS):
                times, co2_values = forecasts[horizon]
                times.append(origin + pd.Timedelta(minutes=horizon))
                co2_values.append(prediction_ppm[index])

    return forecasts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot raw CO2 readings and 15-minute-step rolling forecasts."
    )
    parser.add_argument("input_csv", type=Path, help="Raw ts,value,sensor CSV")
    parser.add_argument("--model", type=Path, default=Path("models/co2_lstm_v1.keras"))
    parser.add_argument("--scaler", type=Path, default=Path("models/co2_lstm_v1_scaler.pkl"))
    parser.add_argument("-o", "--output", type=Path, help="Save plot to this image path")
    args = parser.parse_args()

    raw = clean_csv(args.input_csv)
    minute_data = segment_data(resample_csv(raw))
    if minute_data.empty:
        parser.error("No continuous data segments are available for forecasting")

    with args.scaler.open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    model = keras.models.load_model(args.model)
    forecasts = create_forecasts(minute_data, model, scaler)

    measurements = minute_data["co2"].dropna()
    plt.figure(figsize=(14, 6))
    plt.plot(measurements.index, measurements, label="Raw CO₂", color="black", alpha=0.7)
    for horizon, (times, values) in forecasts.items():
        if times:
            order = np.argsort(times)
            sorted_times = [times[index] for index in order]
            sorted_values = [values[index] for index in order]
            plt.plot(
                sorted_times,
                sorted_values,
                marker=".",
                markersize=3,
                linewidth=1,
                label=f"{horizon}-minute forecast (15-minute steps)",
            )

    plt.xlabel("Time (UTC)")
    plt.ylabel("CO₂ (ppm)")
    plt.title("Raw CO₂ measurements and rolling forecasts")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=150)
        print(f"Plot saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
