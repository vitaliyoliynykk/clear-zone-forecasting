import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import keras
from keras import Model, layers
import matplotlib.pyplot as plt
import tensorflow as tf
import tf2onnx

from sklearn.preprocessing import StandardScaler

SENSOR_FEATURES = ["co2", "temp", "hum"]

TIME_FEATURES = [
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]
FEATURES = SENSOR_FEATURES + TIME_FEATURES
WINDOW_SIZE = 180
FORECAST_HORIZONS = [60, 120, 180]  # one-minute rows


def build_model() -> keras.Model:
    inputs = keras.Input(shape=(WINDOW_SIZE, len(FEATURES)), name="sensor_window")
    x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(len(FORECAST_HORIZONS), name="co2_forecasts")(x)

    model = keras.Model(inputs, outputs, name="co2_1d_cnn")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def build_lstm_model(name: str) -> keras.Model:
    inputs = keras.Input(shape=(WINDOW_SIZE, len(FEATURES)), name="sensor_window")
    x = layers.LSTM(32, return_sequences=True, dropout=0.05)(inputs)
    x = layers.LSTM(16, dropout=0.05)(x)
    x = layers.Dense(8, activation="relu")(x)
    outputs = layers.Dense(len(FORECAST_HORIZONS), name="co2_forecasts")(x)

    model = keras.Model(inputs, outputs, name=name)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def report_horizon_metrics(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    co2_std: float,
) -> None:
    predictions = model.predict(x, verbose=0)

    print("Model:")
    for index, horizon in enumerate(FORECAST_HORIZONS):
        errors_scaled = y[:, index] - predictions[:, index]
        mse_scaled = float(np.mean(np.square(errors_scaled)))
        rmse_scaled = float(np.sqrt(mse_scaled))
        mse_ppm2 = mse_scaled * co2_std**2
        rmse_ppm = rmse_scaled * co2_std
        print(
            f"{horizon}-minute horizon - "
            f"MSE: {mse_scaled:.5f} (scaled), {mse_ppm2:.2f} ppm²; "
            f"RMSE: {rmse_scaled:.5f} (scaled), {rmse_ppm:.2f} ppm"
        )


def report_persistence_baseline(
    x: np.ndarray,
    y: np.ndarray,
    co2_std: float,
) -> None:
    """Report metrics for predicting future CO₂ as the last observed CO₂."""
    co2_feature_index = FEATURES.index("co2")
    last_co2 = x[:, -1, co2_feature_index]
    predictions = np.repeat(last_co2[:, np.newaxis], len(FORECAST_HORIZONS), axis=1)

    print("Persistence baseline:")
    for index, horizon in enumerate(FORECAST_HORIZONS):
        errors_scaled = y[:, index] - predictions[:, index]
        mse_scaled = float(np.mean(np.square(errors_scaled)))
        rmse_scaled = float(np.sqrt(mse_scaled))
        mse_ppm2 = mse_scaled * co2_std**2
        rmse_ppm = rmse_scaled * co2_std
        print(
            f"{horizon}-minute horizon - "
            f"MSE: {mse_scaled:.5f} (scaled), {mse_ppm2:.2f} ppm²; "
            f"RMSE: {rmse_scaled:.5f} (scaled), {rmse_ppm:.2f} ppm"
        )


def make_windows(df: pd.DataFrame, stride = 10) -> tuple[np.ndarray, np.ndarray]:
    """Create windows without crossing segment boundaries.
    """
    windows: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for _, segment in df.groupby("segment_id", sort=False):
        values = segment[FEATURES].to_numpy(dtype=np.float32)
        co2 = segment["co2"].to_numpy(dtype=np.float32)
        max_horizon = max(FORECAST_HORIZONS)
        for end in range(WINDOW_SIZE, len(segment) - max_horizon + 1, stride):
            windows.append(values[end - WINDOW_SIZE : end])
            targets.append(co2[end + np.asarray(FORECAST_HORIZONS) - 1])

    if not windows:
        return (
            np.empty((0, WINDOW_SIZE, len(FEATURES)), dtype=np.float32),
            np.empty((0, len(FORECAST_HORIZONS)), dtype=np.float32),
        )

    x = np.stack(windows).astype(np.float32)
    y = np.stack(targets).astype(np.float32)

    valid = np.isfinite(x).all(axis=(1, 2)) & np.isfinite(y).all(axis=1)
    return x[valid], y[valid]


def read_file() -> pd.DataFrame:
    df = pd.read_parquet("data/processed/living-6H.parquet").reset_index()

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["segment_id", "ts"])

    return df

def get_temporal_split(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start = df["ts"].min()
    end = df["ts"].max()

    duration = end - start

    train_end = start + duration * train_ratio
    val_end = start + duration * (train_ratio + val_ratio)

    print(train_end, val_end)

    train_df = df[df["ts"] < train_end].copy()

    val_df = df[
        (df["ts"] >= train_end)
        & (df["ts"] < val_end)
    ].copy()

    test_df = df[df["ts"] >= val_end].copy()

    for name, subset in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        print(name, len(subset))

    return train_df, val_df, test_df

def prepare_features(df, scaler):
    df = df.copy()

    # Нормалізація показників сенсорів
    df[SENSOR_FEATURES] = scaler.transform(
        df[SENSOR_FEATURES]
    )

    # Часові ознаки
    timestamp = pd.to_datetime(df["ts"], utc=True)
    local_time = timestamp.dt.tz_convert("Europe/Kyiv")

    hour = local_time.dt.hour + local_time.dt.minute / 60

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    day = local_time.dt.day_of_week

    df["day_sin"] = np.sin(2 * np.pi * day / 7)
    df["day_cos"] = np.cos(2 * np.pi * day / 7)

    return df

def save_mode_metadata(
    model: Model,
    x: np.ndarray,
    y: np.ndarray,
    std: float,
    unit: str,
    target_value: str
) -> None:
    predictions = model.predict(x, verbose=0)

    metrics_by_horizon = {}

    for index, horizon in enumerate(FORECAST_HORIZONS):
        errors = (y[:, index] - predictions[:, index]) * std
        metrics_by_horizon[f"{horizon}_minutes"] = {
            "mae": float(np.mean(np.abs(errors))),
            "mse": float(np.mean(np.square(errors))),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        }

    metadata = {
        "model_name": model.name,
        "window_size": WINDOW_SIZE,
        "forecast_horizons_minutes": FORECAST_HORIZONS,
        "features": FEATURES,
        "num_params": model.count_params(),
        "test_metrics": metrics_by_horizon,
        "unit": unit,
        "target_value": target_value
    }

    print

    with open(artifacts_dir / f"{model.name}_metadata.json", "w") as f:
        json.dump(metadata, f)


df = read_file()
train_df, val_df, test_df = get_temporal_split(df)



scaler = StandardScaler()
scaler.fit(train_df[SENSOR_FEATURES])

train_df = prepare_features(train_df, scaler)
val_df = prepare_features(val_df, scaler)
test_df = prepare_features(test_df, scaler)



X_train, y_train = make_windows(train_df)
X_val, y_val = make_windows(val_df)
X_test, y_test = make_windows(test_df)

print(X_train.shape, y_train.shape)
print("X_train:", np.isfinite(X_train).all(), "NaN:", np.isnan(X_train).sum())
print("y_train:", np.isfinite(y_train).all(), "NaN:", np.isnan(y_train).sum())
print("X_val:", np.isfinite(X_val).all(), "NaN:", np.isnan(X_val).sum())
print("y_val:", np.isfinite(y_val).all(), "NaN:", np.isnan(y_val).sum())


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"TensorFlow GPU devices: {[gpu.name for gpu in gpus]}")
    else:
        print("TensorFlow GPU not found; training will use CPU.")

    model = build_lstm_model("co2_lstm_v1")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=128,
        shuffle=True,
    )

    artifacts_dir = Path("models")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Re-trace LSTM layers through Keras' portable implementation for ONNX.
    # Otherwise TensorFlow may emit CudnnRNNV3, which ONNX Runtime cannot load.
    for layer in model.layers:
        if isinstance(layer, layers.LSTM):
            layer.use_cudnn = False

    model.save(artifacts_dir / f"{model.name}.keras")
    onnx_path = artifacts_dir / f"{model.name}.onnx"
    tf2onnx.convert.from_keras(
        model,
        input_signature=(
            tf.TensorSpec(
                (None, WINDOW_SIZE, len(FEATURES)),
                tf.float32,
                name="sensor_window",
            ),
        ),
        opset=15,
        output_path=str(onnx_path),
    )

    with (artifacts_dir / f"{model.name}_scaler.pkl").open("wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    scaler_data = {
        "features": scaler.feature_names_in_.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }

    with open(artifacts_dir / f"{model.name}_scaler.json", "w") as f:
        json.dump(scaler_data, f)

    print(f"Keras model, ONNX model, and scaler saved to {artifacts_dir.resolve()}")

    co2_std = scaler.scale_[scaler.feature_names_in_.tolist().index("co2")]
    print("Saving model metadata and reporting metrics...")
    save_mode_metadata(model, X_test, y_test, co2_std, 'ppm', 'co2')
    print("Saved")
    report_persistence_baseline(X_test, y_test, co2_std)
    report_horizon_metrics(model, X_test, y_test, co2_std)

    plt.plot(history.epoch, history.history["loss"], label="loss")
    plt.plot(history.epoch, history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

