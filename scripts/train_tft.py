"""Обучение Temporal Fusion Transformer для прогноза продаж топлива.

Скрипт обучает настоящую TFT-модель через `pytorch-forecasting`.
Чтобы обучение на обычном CPU не занимало слишком много времени, используется
компактный файл `_задание/5stations_data.csv`: 5 АЗС, один год почасовых данных.

Что сохраняется после запуска:

- `artifacts/models/tft_total_fuel_sales.ckpt` - веса лучшей модели;
- `artifacts/models/tft_config.json` - параметры запуска;
- `artifacts/metrics/tft_metrics.json` - MAE, MSE, RMSE, R2 на backtest;
- `artifacts/forecasts/tft_backtest_forecast.csv` - факт/прогноз на декабре;
- `artifacts/forecasts/tft_future_forecast.csv` - прогноз на будущие 7 дней.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "_задание"
ARTIFACTS_DIR = ROOT / "artifacts"

TARGET = "total_fuel_sales"
MODEL_VERSION = "tft_total_fuel_sales_v1"

MAX_ENCODER_LENGTH = 48
MAX_PREDICTION_LENGTH = 24
FUTURE_DAYS = 1

CATEGORICAL_COLUMNS = [
    "station_id",
    "station_name",
    "road_type",
    "direction",
    "settlement_size",
    "weather_condition",
    "season",
    "ad_channel",
    "holiday_name",
]

KNOWN_REAL_COLUMNS = [
    "time_idx",
    "hour",
    "day_of_week",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
    "is_holiday",
    "is_rush_hour",
    "is_night",
    "temperature",
    "precipitation_mm",
    "visibility_km",
    "wind_speed_ms",
    "is_snow",
    "is_rain",
    "is_fog",
    "promotion_fuel_active",
    "promotion_shop_active",
    "promotion_cafe_active",
    "ad_active",
    "competitor_price_AI92",
    "competitor_price_AI95",
    "competitor_price_DT",
    "price_AI92",
    "price_AI95",
    "price_AI98",
    "price_DT_EURO",
    "price_DT_TANEKO",
    "price_DT_SUMMER",
    "price_DT_WINTER",
]

UNKNOWN_REAL_COLUMNS = [
    TARGET,
    "total_traffic",
    "traffic_Passengers_cars",
    "traffic_Truck_short",
    "traffic_Truck",
    "traffic_Truck_long",
    "traffic_Transporter",
    "traffic_Undefined",
]


def ensure_dirs() -> None:
    """Создает каталоги для весов, метрик и прогнозов."""

    for path in [
        ARTIFACTS_DIR / "models",
        ARTIFACTS_DIR / "metrics",
        ARTIFACTS_DIR / "forecasts",
        ARTIFACTS_DIR / "logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Загружает 5-станционный датасет и готовит типы для TFT."""

    df = pd.read_csv(DATA_DIR / "5stations_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Для локального CPU берем последние 6 месяцев: это ускоряет обучение,
    # но сохраняет сезонность, недельные и суточные паттерны.
    df = df[df["timestamp"] >= "2023-07-01"].copy()
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    # TFT требует целочисленный индекс времени без пропусков внутри группы.
    df["time_idx"] = df.groupby("station_id").cumcount()

    # Все категориальные признаки переводим в строки и заполняем пропуски.
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("нет").astype(str)

    # Числовые признаки приводим к float, чтобы не ловить ошибки типов.
    numeric_columns = list(set(KNOWN_REAL_COLUMNS + UNKNOWN_REAL_COLUMNS))
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    df["time_idx"] = df["time_idx"].astype(int)

    return df


def build_datasets(df: pd.DataFrame) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, pd.DataFrame]:
    """Создает train/validation наборы для TFT.

    Train: июль-ноябрь 2023.
    Validation/backtest: декабрь 2023.
    """

    validation_start = int(df[df["timestamp"] >= "2023-12-01"]["time_idx"].min())
    training_cutoff = validation_start - 1

    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=TARGET,
        group_ids=["station_id"],
        min_encoder_length=24,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["station_id", "station_name", "road_type", "direction", "settlement_size"],
        time_varying_known_categoricals=["weather_condition", "season", "ad_channel", "holiday_name"],
        time_varying_known_reals=KNOWN_REAL_COLUMNS,
        time_varying_unknown_reals=UNKNOWN_REAL_COLUMNS,
        target_normalizer=GroupNormalizer(groups=["station_id"], transformation="softplus"),
        categorical_encoders={col: NaNLabelEncoder(add_nan=True) for col in CATEGORICAL_COLUMNS},
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation_frame = df[
        (df["time_idx"] >= validation_start - MAX_ENCODER_LENGTH)
        & (df["time_idx"] < validation_start + MAX_PREDICTION_LENGTH)
    ].copy()

    validation = TimeSeriesDataSet.from_dataset(
        training,
        validation_frame,
        predict=True,
        stop_randomization=True,
    )

    return training, validation, df


def train_model(training: TimeSeriesDataSet, validation: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """Обучает компактную TFT-модель и возвращает лучшую версию."""

    reusable_checkpoint = ARTIFACTS_DIR / "models" / "tft_total_fuel_sales.ckpt"
    if reusable_checkpoint.exists():
        return TemporalFusionTransformer.load_from_checkpoint(str(reusable_checkpoint))

    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")

    train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ARTIFACTS_DIR / "models",
        filename="tft_total_fuel_sales",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=2, mode="min")
    logger = CSVLogger(save_dir=ARTIFACTS_DIR / "logs", name="tft")

    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        gradient_clip_val=0.1,
        limit_train_batches=20,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=10,
    )

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=8,
        attention_head_size=2,
        dropout=0.15,
        hidden_continuous_size=4,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=2,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = checkpoint_callback.best_model_path
    if not best_path:
        best_path = str(ARTIFACTS_DIR / "models" / "tft_total_fuel_sales.ckpt")
        trainer.save_checkpoint(best_path)
    else:
        target_path = ARTIFACTS_DIR / "models" / "tft_total_fuel_sales.ckpt"
        target_path.write_bytes(Path(best_path).read_bytes())
        best_path = str(target_path)

    return TemporalFusionTransformer.load_from_checkpoint(best_path)


def prediction_to_frame(prediction, df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """Преобразует Prediction object из pytorch-forecasting в обычный DataFrame."""

    output = prediction.output.detach().cpu().numpy()
    index = prediction.index.reset_index(drop=True)

    rows = []
    for sample_id in range(output.shape[0]):
        station_id = str(index.loc[sample_id, "station_id"])
        decoder_start = int(index.loc[sample_id, "time_idx"])

        for step in range(output.shape[1]):
            time_idx = decoder_start + step
            match = df[(df["station_id"] == station_id) & (df["time_idx"] == time_idx)]
            if match.empty:
                continue
            source_row = match.iloc[0]
            rows.append(
                {
                    "timestamp": source_row["timestamp"],
                    "station_id": station_id,
                    "station_name": source_row["station_name"],
                    "target": target_name,
                    "target_label": "Продажи топлива TFT",
                    "actual": float(source_row[TARGET]),
                    "prediction": float(output[sample_id, step]),
                    "model_version": MODEL_VERSION,
                }
            )

    result = pd.DataFrame(rows)
    return result.drop_duplicates(["timestamp", "station_id", "target"]).sort_values(["timestamp", "station_id"])


def calculate_metrics(backtest: pd.DataFrame) -> dict:
    """Считает метрики TFT на backtest-прогнозе."""

    mse = mean_squared_error(backtest["actual"], backtest["prediction"])
    return {
        "model": "Temporal Fusion Transformer",
        "model_version": MODEL_VERSION,
        "target": TARGET,
        "target_label": "Продажи топлива TFT",
        "mae": float(mean_absolute_error(backtest["actual"], backtest["prediction"])),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(backtest["actual"], backtest["prediction"])),
        "rows": int(len(backtest)),
    }


def make_future_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Создает сценарный будущий датасет на 7 дней после конца истории.

    Для будущих известных признаков берем последнюю неделю 2023 года и сдвигаем
    ее на 2024-01-01...2024-01-07. Это сценарный прогноз: погода, акции, цены
    и трафик предполагаются похожими на последнюю наблюдаемую неделю.
    """

    history_parts = []
    future_parts = []
    max_time_idx = int(df["time_idx"].max())
    future_hours = FUTURE_DAYS * 24

    for station_id, station_df in df.groupby("station_id"):
        station_df = station_df.sort_values("time_idx")
        encoder_history = station_df.tail(MAX_ENCODER_LENGTH).copy()
        template = station_df.tail(future_hours).copy()

        last_timestamp = station_df["timestamp"].max()
        template["timestamp"] = pd.date_range(
            last_timestamp + pd.Timedelta(hours=1),
            periods=future_hours,
            freq="h",
        )
        template["time_idx"] = np.arange(max_time_idx + 1, max_time_idx + future_hours + 1)
        template["hour"] = template["timestamp"].dt.hour.astype(float)
        template["day_of_week"] = template["timestamp"].dt.dayofweek.astype(float)
        template["week_of_year"] = template["timestamp"].dt.isocalendar().week.astype(float)
        template["month"] = template["timestamp"].dt.month.astype(float)
        template["quarter"] = template["timestamp"].dt.quarter.astype(float)
        template["is_weekend"] = template["timestamp"].dt.dayofweek.isin([5, 6]).astype(float)
        template["is_holiday"] = template["timestamp"].dt.strftime("%m-%d").isin(["01-01", "01-02", "01-03"]).astype(float)
        template["holiday_name"] = np.where(template["is_holiday"] == 1, "Новогодние праздники", "нет")
        template["season"] = "winter"
        template["is_rush_hour"] = template["hour"].isin([7, 8, 9, 17, 18, 19]).astype(float)
        template["is_night"] = ((template["hour"] >= 22) | (template["hour"] <= 5)).astype(float)

        # Целевое значение будущего неизвестно. Для predict=True оно нужно
        # только как техническое поле, поэтому ставим 0.
        template[TARGET] = 0.0

        history_parts.append(encoder_history)
        future_parts.append(template)

    return pd.concat(history_parts + future_parts, ignore_index=True).sort_values(["station_id", "time_idx"])


def make_future_forecast(model: TemporalFusionTransformer, training: TimeSeriesDataSet, df: pd.DataFrame) -> pd.DataFrame:
    """Строит прогноз на будущие 7 дней после 2023-12-31."""

    future_df = make_future_frame(df)
    prediction = model.predict(
        future_df,
        mode="prediction",
        return_index=True,
        trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False, "logger": False},
    )

    output = prediction.output.detach().cpu().numpy()
    index = prediction.index.reset_index(drop=True)

    rows = []
    for sample_id in range(output.shape[0]):
        station_id = str(index.loc[sample_id, "station_id"])
        station_name = future_df[future_df["station_id"] == station_id]["station_name"].iloc[0]
        start_idx = int(index.loc[sample_id, "time_idx"])
        for step in range(output.shape[1]):
            time_idx = start_idx + step
            match = future_df[(future_df["station_id"] == station_id) & (future_df["time_idx"] == time_idx)]
            if match.empty:
                continue
            rows.append(
                {
                    "timestamp": match.iloc[0]["timestamp"],
                    "station_id": station_id,
                    "station_name": station_name,
                    "target": TARGET,
                    "target_label": "Будущий прогноз топлива TFT",
                    "prediction": float(output[sample_id, step]),
                    "model_version": MODEL_VERSION,
                }
            )

    return pd.DataFrame(rows).drop_duplicates(["timestamp", "station_id", "target"]).sort_values(["timestamp", "station_id"])


def save_config(metrics: dict) -> None:
    """Сохраняет параметры обучения для отчета и воспроизводимости."""

    config = {
        "model_version": MODEL_VERSION,
        "source_data": "_задание/5stations_data.csv",
        "target": TARGET,
        "max_encoder_length": MAX_ENCODER_LENGTH,
        "max_prediction_length": MAX_PREDICTION_LENGTH,
        "future_days": FUTURE_DAYS,
        "metrics": metrics,
    }
    with (ARTIFACTS_DIR / "models" / "tft_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)


def main() -> None:
    """Полный запуск обучения TFT и экспорта артефактов."""

    ensure_dirs()
    df = load_data()
    training, validation, prepared_df = build_datasets(df)
    model = train_model(training, validation)

    val_loader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)
    prediction = model.predict(
        val_loader,
        mode="prediction",
        return_index=True,
        trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False, "logger": False},
    )
    backtest = prediction_to_frame(prediction, prepared_df, TARGET)
    metrics = calculate_metrics(backtest)
    future = make_future_forecast(model, training, prepared_df)

    backtest.to_csv(ARTIFACTS_DIR / "forecasts" / "tft_backtest_forecast.csv", index=False)
    future.to_csv(ARTIFACTS_DIR / "forecasts" / "tft_future_forecast.csv", index=False)
    with (ARTIFACTS_DIR / "metrics" / "tft_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    save_config(metrics)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
