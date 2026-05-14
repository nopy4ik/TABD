"""Обучение базовых моделей и выгрузка артефактов для Dash.

Этот скрипт нужен как первый полностью рабочий ML-пайплайн проекта.
Он не заменяет TFT, но проверяет весь поток данных:

1. читает исходный `5stations_data.csv`;
2. делит данные по времени на train/test;
3. обучает быстрые модели для продаж топлива и товаров;
4. считает метрики качества;
5. сохраняет прогнозы, важность признаков и рекомендации.

После починки окружения TFT файл `scripts/train_tft.py` должен сохранить
артефакты в таком же формате, чтобы Dash продолжил работать без переписывания.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "_задание"
ARTIFACTS_DIR = ROOT / "artifacts"

# В baseline обучаем две основные цели, которые нужны в dashboard:
# продажи топлива и выручка сопутствующих товаров.
TARGETS = {
    "total_fuel_sales": "Продажи топлива",
    "shop_total_revenue": "Выручка магазина",
}
FUTURE_DAYS = 7

# Категориальные признаки кодируются OrdinalEncoder.
# Для baseline этого достаточно, а TFT позже будет использовать свои энкодеры.
CATEGORICAL_FEATURES = [
    "station_id",
    "road_type",
    "direction",
    "settlement_size",
    "weather_condition",
    "season",
    "ad_channel",
    "holiday_name",
]

# Числовые признаки оставляем в исходном виде.
# Исходные CSV не меняются: все преобразования выполняются только в памяти.
NUMERIC_FEATURES = [
    "distance_to_city_km",
    "total_pumps",
    "shop_area_m2",
    "has_car_wash",
    "has_cafe",
    "has_shop",
    "competitors_within_5km",
    "corporate_customer_ratio",
    "staff_engagement_score",
    "customer_loyalty_score",
    "temperature",
    "precipitation_mm",
    "visibility_km",
    "wind_speed_ms",
    "is_snow",
    "is_rain",
    "is_fog",
    "traffic_Passengers_cars",
    "traffic_Truck_short",
    "traffic_Truck",
    "traffic_Truck_long",
    "traffic_Transporter",
    "traffic_Undefined",
    "total_traffic",
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
    "hour",
    "day_of_week",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
    "is_holiday",
    "is_rush_hour",
    "is_night",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def ensure_dirs() -> None:
    """Создает каталоги для артефактов, если их еще нет."""

    for path in [
        ARTIFACTS_DIR / "forecasts",
        ARTIFACTS_DIR / "metrics",
        ARTIFACTS_DIR / "recommendations",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Загружает исходные данные и приводит служебные поля к нужным типам."""

    df = pd.read_csv(DATA_DIR / "5stations_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Категориальные пропуски заменяем строкой `none`, чтобы энкодер видел
    # отдельную категорию, а не NaN.
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("none").astype(str)

    return df.sort_values(["timestamp", "station_id"]).reset_index(drop=True)


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит датасет по времени: обучение до декабря, тест на декабре."""

    train = df[df["timestamp"] < "2023-12-01"].copy()
    test = df[df["timestamp"] >= "2023-12-01"].copy()
    return train, test


def build_model() -> Pipeline:
    """Собирает sklearn Pipeline: кодирование признаков + регрессор."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )

    # HistGradientBoostingRegressor выбран как быстрый baseline для большого
    # почасового датасета. Он обучается заметно быстрее RandomForest.
    regressor = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", regressor)])


def calculate_metrics(actual: pd.Series, prediction: np.ndarray) -> dict:
    """Считает основные метрики регрессии для отчета и dashboard."""

    mse = mean_squared_error(actual, prediction)
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, prediction)),
    }


def export_feature_importance(model: Pipeline, test: pd.DataFrame, target: str) -> pd.DataFrame:
    """Оценивает важность признаков через permutation importance.

    Берем ограниченную выборку из test, чтобы расчет не занимал слишком много
    времени. Для отчета нам важен порядок факторов, а не идеальная точность
    оценки важности.
    """

    sample = test.sample(min(5000, len(test)), random_state=42)
    result = permutation_importance(
        model,
        sample[FEATURES],
        sample[target],
        n_repeats=3,
        random_state=42,
        n_jobs=-1,
    )
    importance = pd.DataFrame({"feature": FEATURES, "importance": result.importances_mean})
    importance["target"] = target
    importance["target_label"] = TARGETS[target]
    return importance.sort_values("importance", ascending=False)


def build_recommendations(forecast: pd.DataFrame) -> pd.DataFrame:
    """Формирует простые рекомендации по станциям на основе ошибок прогноза."""

    fuel_forecast = forecast[forecast["target"] == "total_fuel_sales"].copy()
    station_errors = (
        fuel_forecast.assign(abs_error=lambda x: (x["actual"] - x["prediction"]).abs())
        .groupby(["station_id", "station_name"], as_index=False)
        .agg(
            actual_mean=("actual", "mean"),
            prediction_mean=("prediction", "mean"),
            mae=("abs_error", "mean"),
        )
    )
    station_errors["gap"] = station_errors["prediction_mean"] - station_errors["actual_mean"]

    # Рекомендации здесь rule-based. После TFT можно заменить их на выводы из
    # прогноза, ошибок, важности признаков и сценарного анализа.
    station_errors["recommendation"] = np.select(
        [
            station_errors["gap"] < -10,
            station_errors["gap"] > 10,
            station_errors["mae"] > station_errors["mae"].median(),
        ],
        [
            "Проверить причины просадки спроса: акции, рекламу, цены и трафик в проблемные часы",
            "Проверить достаточность запасов топлива и персонала в часы повышенного спроса",
            "Разобрать станцию отдельно: ошибка прогноза выше медианы сети",
        ],
        default="Станция прогнозируется стабильно, поддерживать текущую стратегию",
    )
    return station_errors.sort_values("mae", ascending=False)


def make_future_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Создает сценарные будущие признаки на 7 дней после 2023-12-31.

    Это не изменение исходных данных. Мы берем последнюю наблюдаемую неделю как
    шаблон поведения и переносим ее на 2024-01-01...2024-01-07.
    """

    future_parts = []
    future_hours = FUTURE_DAYS * 24
    for station_id, station_df in df.groupby("station_id"):
        station_df = station_df.sort_values("timestamp")
        template = station_df.tail(future_hours).copy()
        last_timestamp = station_df["timestamp"].max()
        template["timestamp"] = pd.date_range(
            last_timestamp + pd.Timedelta(hours=1),
            periods=future_hours,
            freq="h",
        )
        template["hour"] = template["timestamp"].dt.hour
        template["day_of_week"] = template["timestamp"].dt.dayofweek
        template["week_of_year"] = template["timestamp"].dt.isocalendar().week.astype(int)
        template["month"] = template["timestamp"].dt.month
        template["quarter"] = template["timestamp"].dt.quarter
        template["is_weekend"] = template["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)
        template["is_holiday"] = template["timestamp"].dt.strftime("%m-%d").isin(["01-01", "01-02", "01-03"]).astype(int)
        template["holiday_name"] = np.where(template["is_holiday"] == 1, "Новогодние праздники", "нет")
        template["season"] = "winter"
        template["is_rush_hour"] = template["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        template["is_night"] = ((template["hour"] >= 22) | (template["hour"] <= 5)).astype(int)
        future_parts.append(template)

    future = pd.concat(future_parts, ignore_index=True)
    for col in CATEGORICAL_FEATURES:
        future[col] = future[col].fillna("none").astype(str)
    return future


def train_target(train: pd.DataFrame, test: pd.DataFrame, future: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Обучает одну модель для одной целевой переменной."""

    model = build_model()
    model.fit(train[FEATURES], train[target])
    predictions = model.predict(test[FEATURES])
    future_predictions = model.predict(future[FEATURES])

    metrics = calculate_metrics(test[target], predictions)
    metrics.update(
        {
            "model": "HistGradientBoostingRegressor baseline",
            "target": target,
            "target_label": TARGETS[target],
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        }
    )

    forecast = test[
        [
            "timestamp",
            "station_id",
            "station_name",
            target,
            "total_traffic",
            "ad_active",
            "promotion_fuel_active",
            "promotion_shop_active",
            "hour",
            "day_of_week",
        ]
    ].copy()
    forecast = forecast.rename(columns={target: "actual"})
    forecast["prediction"] = predictions
    forecast["target"] = target
    forecast["target_label"] = TARGETS[target]
    forecast["model_version"] = "baseline_hgb_v2"

    future_forecast = future[["timestamp", "station_id", "station_name"]].copy()
    future_forecast["prediction"] = future_predictions
    future_forecast["target"] = target
    future_forecast["target_label"] = TARGETS[target]
    future_forecast["model_version"] = "baseline_hgb_v2"

    importance = export_feature_importance(model, test, target)
    return forecast, future_forecast, metrics, importance


def main() -> None:
    """Запускает весь baseline-пайплайн."""

    ensure_dirs()
    df = load_data()
    train, test = split_by_time(df)
    future = make_future_frame(df)

    forecasts = []
    future_forecasts = []
    metrics = []
    importances = []

    for target in TARGETS:
        forecast, future_forecast, target_metrics, importance = train_target(train, test, future, target)
        forecasts.append(forecast)
        future_forecasts.append(future_forecast)
        metrics.append(target_metrics)
        importances.append(importance)

    forecast_df = pd.concat(forecasts, ignore_index=True)
    future_forecast_df = pd.concat(future_forecasts, ignore_index=True)
    importance_df = pd.concat(importances, ignore_index=True)
    recommendations = build_recommendations(forecast_df)

    forecast_df.to_csv(ARTIFACTS_DIR / "forecasts" / "baseline_forecast.csv", index=False)
    future_forecast_df.to_csv(ARTIFACTS_DIR / "forecasts" / "baseline_future_forecast.csv", index=False)
    importance_df.to_csv(ARTIFACTS_DIR / "metrics" / "baseline_feature_importance.csv", index=False)
    recommendations.to_csv(
        ARTIFACTS_DIR / "recommendations" / "station_recommendations.csv",
        index=False,
    )
    with (ARTIFACTS_DIR / "metrics" / "baseline_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
