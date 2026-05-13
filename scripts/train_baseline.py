"""Train a baseline model and export artifacts for the Dash dashboard."""

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

TARGET = "total_fuel_sales"

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


def ensure_dirs() -> None:
    for path in [
        ARTIFACTS_DIR / "forecasts",
        ARTIFACTS_DIR / "metrics",
        ARTIFACTS_DIR / "recommendations",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "detailed_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("none").astype(str)
    return df.sort_values(["timestamp", "station_id"]).reset_index(drop=True)


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["timestamp"] < "2023-12-01"].copy()
    test = df[df["timestamp"] >= "2023-12-01"].copy()
    return train, test


def build_model() -> Pipeline:
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
    regressor = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", regressor)])


def export_feature_importance(model: Pipeline, test: pd.DataFrame) -> pd.DataFrame:
    sample = test.sample(min(5000, len(test)), random_state=42)
    features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    result = permutation_importance(
        model,
        sample[features],
        sample[TARGET],
        n_repeats=3,
        random_state=42,
        n_jobs=-1,
    )
    importance = pd.DataFrame({"feature": features, "importance": result.importances_mean})
    return importance.sort_values("importance", ascending=False)


def build_recommendations(forecast: pd.DataFrame) -> pd.DataFrame:
    station_errors = (
        forecast.assign(abs_error=lambda x: (x["actual"] - x["prediction"]).abs())
        .groupby(["station_id", "station_name"], as_index=False)
        .agg(
            actual_mean=("actual", "mean"),
            prediction_mean=("prediction", "mean"),
            mae=("abs_error", "mean"),
        )
    )
    station_errors["gap"] = station_errors["prediction_mean"] - station_errors["actual_mean"]
    station_errors["recommendation"] = np.select(
        [
            station_errors["gap"] < -10,
            station_errors["gap"] > 10,
            station_errors["mae"] > station_errors["mae"].median(),
        ],
        [
            "Проверить факторы просадки спроса и усилить акции/рекламу в проблемные часы",
            "Проверить достаточность запасов топлива и персонала в часы повышенного спроса",
            "Разобрать станцию отдельно: ошибка прогноза выше медианы сети",
        ],
        default="Станция прогнозируется стабильно, поддерживать текущую стратегию",
    )
    return station_errors.sort_values("mae", ascending=False)


def main() -> None:
    ensure_dirs()
    df = load_data()
    train, test = split_by_time(df)

    model = build_model()
    model.fit(train[CATEGORICAL_FEATURES + NUMERIC_FEATURES], train[TARGET])
    predictions = model.predict(test[CATEGORICAL_FEATURES + NUMERIC_FEATURES])

    mse = mean_squared_error(test[TARGET], predictions)
    metrics = {
        "model": "HistGradientBoostingRegressor baseline",
        "target": TARGET,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mae": float(mean_absolute_error(test[TARGET], predictions)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(test[TARGET], predictions)),
    }

    forecast = test[
        [
            "timestamp",
            "station_id",
            "station_name",
            TARGET,
            "total_traffic",
            "ad_active",
            "promotion_fuel_active",
            "hour",
            "day_of_week",
        ]
    ].copy()
    forecast = forecast.rename(columns={TARGET: "actual"})
    forecast["prediction"] = predictions
    forecast["model_version"] = "baseline_hgb_v1"

    importance = export_feature_importance(model, test)
    recommendations = build_recommendations(forecast)

    forecast.to_csv(ARTIFACTS_DIR / "forecasts" / "baseline_forecast.csv", index=False)
    importance.to_csv(ARTIFACTS_DIR / "metrics" / "baseline_feature_importance.csv", index=False)
    recommendations.to_csv(
        ARTIFACTS_DIR / "recommendations" / "station_recommendations.csv",
        index=False,
    )
    with (ARTIFACTS_DIR / "metrics" / "baseline_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
