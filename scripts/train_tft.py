"""Обучение Temporal Fusion Transformer для прогноза топлива и товаров.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

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

TARGET_CONFIGS = [
    {
        "target": "total_fuel_sales",
        "target_label": "Продажи топлива TFT",
        "future_target_label": "Будущий прогноз топлива TFT",
        "model_version": "tft_total_fuel_sales",
        "data_start": "2023-07-01",
        "max_encoder_length": 48,
        "max_prediction_length": 168,
        "learning_rate": 0.03,
        "hidden_size": 8,
        "attention_head_size": 2,
        "hidden_continuous_size": 4,
        "dropout": 0.15,
        "normalizer_method": "standard",
        "normalizer_transformation": "softplus",
        "max_epochs": 8,
    },
    {
        "target": "shop_total_revenue",
        "target_label": "Выручка магазина TFT",
        "future_target_label": "Будущий прогноз выручки магазина TFT",
        "model_version": "tft_shop_total_revenue",
        "data_start": None,
        "max_encoder_length": 168,
        "max_prediction_length": 168,
        "learning_rate": 0.01,
        "hidden_size": 16,
        "attention_head_size": 4,
        "hidden_continuous_size": 8,
        "dropout": 0.2,
        "normalizer_method": "robust",
        "normalizer_transformation": "log1p",
        "max_epochs": 16,
    },
]
TARGET_COLUMNS = [config["target"] for config in TARGET_CONFIGS]

FUTURE_DAYS = 7
DEFAULT_MAX_EPOCHS = 8
DEFAULT_BATCH_SIZE = 128

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

BASE_UNKNOWN_REAL_COLUMNS = [
    "total_traffic",
    "traffic_Passengers_cars",
    "traffic_Truck_short",
    "traffic_Truck",
    "traffic_Truck_long",
    "traffic_Transporter",
    "traffic_Undefined",
]


def allow_checkpoint_globals() -> None:
    """Разрешает безопасную загрузку классов из checkpoint в torch 2.6+."""

    safe_add = getattr(torch.serialization, "add_safe_globals", None)
    if safe_add is None:
        return

    safe_classes = [GroupNormalizer, NaNLabelEncoder, QuantileLoss, pd.DataFrame, pd.Series, pd.Index]
    try:
        safe_add(safe_classes)
    except Exception:
        # Если версия torch не поддерживает часть классов, оставляем fallback
        # на обычную загрузку checkpoint.
        pass


def ensure_dirs() -> None:
    """Создает каталоги для весов, метрик и прогнозов."""

    for path in [
        ARTIFACTS_DIR / "models",
        ARTIFACTS_DIR / "metrics",
        ARTIFACTS_DIR / "forecasts",
        ARTIFACTS_DIR / "logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_data(data_start: str | None) -> pd.DataFrame:
    """Загружает 5-станционный датасет и готовит типы для TFT."""

    df = pd.read_csv(DATA_DIR / "5stations_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if data_start:
        df = df[df["timestamp"] >= data_start].copy()
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    df["time_idx"] = df.groupby("station_id").cumcount()

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("нет").astype(str)

    # Числовые признаки приводим к float, чтобы не ловить ошибки типов.
    numeric_columns = list(set(KNOWN_REAL_COLUMNS + TARGET_COLUMNS + BASE_UNKNOWN_REAL_COLUMNS))
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    df["time_idx"] = df["time_idx"].astype(int)

    return df


def build_datasets(
    df: pd.DataFrame,
    target: str,
    *,
    max_encoder_length: int,
    max_prediction_length: int,
    normalizer_method: str,
    normalizer_transformation: str,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, pd.DataFrame]:
    """Создает train/validation наборы для TFT.

    Train: июль-ноябрь 2023.
    Validation/backtest: декабрь 2023.
    """

    validation_start = int(df[df["timestamp"] >= "2023-12-01"]["time_idx"].min())
    training_cutoff = validation_start - 1

    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=target,
        group_ids=["station_id"],
        min_encoder_length=min(24, max_encoder_length),
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["station_id", "station_name", "road_type", "direction", "settlement_size"],
        time_varying_known_categoricals=["weather_condition", "season", "ad_channel", "holiday_name"],
        time_varying_known_reals=KNOWN_REAL_COLUMNS,
        time_varying_unknown_reals=[target] + BASE_UNKNOWN_REAL_COLUMNS,
        target_normalizer=GroupNormalizer(
            groups=["station_id"],
            method=normalizer_method,
            transformation=normalizer_transformation,
        ),
        categorical_encoders={col: NaNLabelEncoder(add_nan=True) for col in CATEGORICAL_COLUMNS},
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation_frame = df[
        (df["time_idx"] >= validation_start - max_encoder_length)
        & (df["time_idx"] < validation_start + max_prediction_length)
    ].copy()

    validation = TimeSeriesDataSet.from_dataset(
        training,
        validation_frame,
        predict=True,
        stop_randomization=True,
    )

    return training, validation, df


def train_model(
    training: TimeSeriesDataSet,
    validation: TimeSeriesDataSet,
    *,
    target: str,
    model_version: str,
    learning_rate: float,
    hidden_size: int,
    attention_head_size: int,
    hidden_continuous_size: int,
    dropout: float,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume_from_checkpoint: bool = True,
) -> TemporalFusionTransformer:
    """Обучает TFT-модель"""

    reusable_checkpoint = ARTIFACTS_DIR / "models" / f"{model_version}.ckpt"

    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")
    allow_checkpoint_globals()

    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ARTIFACTS_DIR / "models",
        filename=model_version,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=2, mode="min")
    logger = CSVLogger(save_dir=ARTIFACTS_DIR / "logs", name="tft")

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=10,
    )

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=2,
    )

    ckpt_path = str(reusable_checkpoint) if reusable_checkpoint.exists() and resume_from_checkpoint else None
    if ckpt_path:
        print(f"Продолжаю обучение из {ckpt_path}")

    def fit_once(current_model: TemporalFusionTransformer, resume_path: str | None) -> None:
        trainer.fit(
            current_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_path,
        )

    try:
        fit_once(model, ckpt_path)
    except Exception as exc:
        if ckpt_path is None:
            raise
        print(f"Не удалось восстановить checkpoint: {exc}")
        print("Перезапускаю обучение с нуля без checkpoint.")
        fresh_model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            loss=QuantileLoss(),
            reduce_on_plateau_patience=2,
        )
        fit_once(fresh_model, None)
        model = fresh_model

    best_path = checkpoint_callback.best_model_path
    if not best_path:
        best_path = str(ARTIFACTS_DIR / "models" / f"{model_version}.ckpt")
        trainer.save_checkpoint(best_path)
    else:
        target_path = ARTIFACTS_DIR / "models" / f"{model_version}.ckpt"
        target_path.write_bytes(Path(best_path).read_bytes())
        best_path = str(target_path)

    return TemporalFusionTransformer.load_from_checkpoint(best_path)


def prediction_to_frame(
    prediction,
    df: pd.DataFrame,
    target_name: str,
    target_label: str,
    model_version: str,
) -> pd.DataFrame:
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
                    "target_label": target_label,
                    "actual": float(source_row[target_name]),
                    "prediction": float(output[sample_id, step]),
                    "model_version": model_version,
                }
            )

    result = pd.DataFrame(rows)
    return result.drop_duplicates(["timestamp", "station_id", "target"]).sort_values(["timestamp", "station_id"])


def calculate_metrics(backtest: pd.DataFrame, target: str, target_label: str, model_version: str) -> dict:
    """Считает метрики TFT на backtest-прогнозе."""

    mse = mean_squared_error(backtest["actual"], backtest["prediction"])
    return {
        "model": "Temporal Fusion Transformer",
        "model_version": model_version,
        "target": target,
        "target_label": target_label,
        "mae": float(mean_absolute_error(backtest["actual"], backtest["prediction"])),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(backtest["actual"], backtest["prediction"])),
        "rows": int(len(backtest)),
    }


def make_future_frame(df: pd.DataFrame, target: str, *, max_encoder_length: int) -> pd.DataFrame:
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
        encoder_history = station_df.tail(max_encoder_length).copy()
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
        template[target] = 0.0

        history_parts.append(encoder_history)
        future_parts.append(template)

    return pd.concat(history_parts + future_parts, ignore_index=True).sort_values(["station_id", "time_idx"])


def make_future_forecast(
    model: TemporalFusionTransformer,
    training: TimeSeriesDataSet,
    df: pd.DataFrame,
    target: str,
    target_label: str,
    future_target_label: str,
    model_version: str,
    *,
    max_encoder_length: int,
) -> pd.DataFrame:
    """Строит прогноз на будущие 7 дней после 2023-12-31."""

    future_df = make_future_frame(df, target, max_encoder_length=max_encoder_length)
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
                    "target": target,
                    "target_label": future_target_label,
                    "prediction": float(output[sample_id, step]),
                    "model_version": model_version,
                }
            )

    return pd.DataFrame(rows).drop_duplicates(["timestamp", "station_id", "target"]).sort_values(["timestamp", "station_id"])


def save_config(
    metrics: dict,
    *,
    max_epochs: int,
    batch_size: int,
    resume_from_checkpoint: bool,
    target: str,
    model_version: str,
    max_encoder_length: int,
    max_prediction_length: int,
    target_config: dict,
) -> None:
    """Сохраняет параметры обучения для отчета и воспроизводимости."""

    config = {
        "model_version": model_version,
        "source_data": "_задание/5stations_data.csv",
        "target": target,
        "max_encoder_length": max_encoder_length,
        "max_prediction_length": max_prediction_length,
        "future_days": FUTURE_DAYS,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "resume_from_checkpoint": resume_from_checkpoint,
        "learning_rate": target_config["learning_rate"],
        "hidden_size": target_config["hidden_size"],
        "attention_head_size": target_config["attention_head_size"],
        "hidden_continuous_size": target_config["hidden_continuous_size"],
        "dropout": target_config["dropout"],
        "normalizer_method": target_config["normalizer_method"],
        "normalizer_transformation": target_config["normalizer_transformation"],
        "metrics": metrics,
    }
    with (ARTIFACTS_DIR / "models" / "tft_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    """Разбирает параметры запуска обучения."""

    parser = argparse.ArgumentParser(description="Обучение TFT для прогноза продаж топлива")
    parser.add_argument("--epochs", type=int, default=DEFAULT_MAX_EPOCHS, help="Максимальное число эпох")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Размер batch")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Не продолжать обучение из существующего чекпоинта",
    )
    return parser.parse_args()


def main() -> None:
    """Полный запуск обучения TFT и экспорта артефактов."""

    args = parse_args()
    ensure_dirs()
    all_metrics = []

    for target_config in TARGET_CONFIGS:
        target = target_config["target"]
        target_label = target_config["target_label"]
        future_target_label = target_config["future_target_label"]
        model_version = target_config["model_version"]
        data_start = target_config["data_start"]
        max_encoder_length = target_config["max_encoder_length"]
        max_prediction_length = target_config["max_prediction_length"]
        learning_rate = target_config["learning_rate"]
        hidden_size = target_config["hidden_size"]
        attention_head_size = target_config["attention_head_size"]
        hidden_continuous_size = target_config["hidden_continuous_size"]
        dropout = target_config["dropout"]
        normalizer_method = target_config["normalizer_method"]
        normalizer_transformation = target_config["normalizer_transformation"]
        target_max_epochs = max(args.epochs, int(target_config.get("max_epochs", args.epochs)))

        df = load_data(data_start)
        training, validation, prepared_df = build_datasets(
            df,
            target,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            normalizer_method=normalizer_method,
            normalizer_transformation=normalizer_transformation,
        )
        model = train_model(
            training,
            validation,
            target=target,
            model_version=model_version,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            hidden_continuous_size=hidden_continuous_size,
            dropout=dropout,
            max_epochs=target_max_epochs,
            batch_size=args.batch_size,
            resume_from_checkpoint=not args.no_resume,
        )

        val_loader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)
        prediction = model.predict(
            val_loader,
            mode="prediction",
            return_index=True,
            trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False, "logger": False},
        )
        backtest = prediction_to_frame(prediction, prepared_df, target, target_label, model_version)
        metrics = calculate_metrics(backtest, target, target_label, model_version)
        future = make_future_forecast(
            model,
            training,
            prepared_df,
            target,
            target_label,
            future_target_label,
            model_version,
            max_encoder_length=max_encoder_length,
        )

        backtest.to_csv(ARTIFACTS_DIR / "forecasts" / f"{model_version}_backtest_forecast.csv", index=False)
        future.to_csv(ARTIFACTS_DIR / "forecasts" / f"{model_version}_future_forecast.csv", index=False)
        with (ARTIFACTS_DIR / "metrics" / f"{model_version}_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)
        save_config(
            metrics,
            max_epochs=target_max_epochs,
            batch_size=args.batch_size,
            resume_from_checkpoint=not args.no_resume,
            target=target,
            model_version=model_version,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            target_config=target_config,
        )
        all_metrics.append(metrics)

    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
