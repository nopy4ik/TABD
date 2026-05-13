#!/usr/bin/env python3
"""Build compact Power BI-friendly aggregates from the raw CSV files."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import median
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "_задание"
OUT_DIR = ROOT / "derived"

FUEL_COLS = [
    "sales_AI92",
    "sales_AI95",
    "sales_AI98",
    "sales_DT_EURO",
    "sales_DT_TANEKO",
    "sales_DT_SUMMER",
    "sales_DT_WINTER",
]

SHOP_COLS = [
    "shop_напитки",
    "shop_закуски",
    "shop_автотовары",
    "shop_кофе",
    "shop_табак",
]

TRAFFIC_COLS = [
    "traffic_Passengers_cars",
    "traffic_Truck_short",
    "traffic_Truck",
    "traffic_Truck_long",
    "traffic_Transporter",
    "traffic_Undefined",
]


def to_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def to_int(value: str) -> int:
    return int(round(to_float(value)))


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        yield from csv.DictReader(fh)


def load_station_metadata(path: Path):
    stations = {}
    for row in read_csv(path):
        stations[row["station_id"]] = row
    return stations


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_date_row(date_key: str, sample_row: dict) -> dict:
    date_obj = datetime.strptime(date_key, "%Y-%m-%d")
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    return {
        "date": date_key,
        "year": date_obj.year,
        "quarter": to_int(sample_row.get("quarter", 0)),
        "month": date_obj.month,
        "day": date_obj.day,
        "day_of_week": date_obj.weekday(),
        "day_name": day_names[date_obj.weekday()],
        "week_of_year": date_obj.isocalendar().week,
        "season": sample_row.get("season", ""),
        "is_weekend": 1 if date_obj.weekday() >= 5 else 0,
        "is_holiday": to_int(sample_row.get("is_holiday", 0)),
        "holiday_name": sample_row.get("holiday_name", ""),
    }


def build_assets():
    ensure_out_dir()

    metadata = load_station_metadata(SOURCE_DIR / "stations_metadata.csv")
    station_stats = defaultdict(lambda: defaultdict(float))
    daily_stats = defaultdict(lambda: defaultdict(float))
    hourly_stats = defaultdict(lambda: defaultdict(float))
    date_samples = {}
    max_date = None

    total_rows = 0

    for row in read_csv(SOURCE_DIR / "detailed_data.csv"):
        total_rows += 1
        station_id = row["station_id"]
        date_key = row["timestamp"][:10]
        hour_key = row["hour"]
        date_samples.setdefault(date_key, row)
        date_obj = datetime.strptime(date_key, "%Y-%m-%d")
        if max_date is None or date_obj > max_date:
            max_date = date_obj

        station_bucket = station_stats[station_id]
        daily_bucket = daily_stats[date_key]
        hourly_bucket = hourly_stats[hour_key]

        for bucket in (station_bucket, daily_bucket, hourly_bucket):
            bucket["rows"] += 1
            bucket["total_fuel_sales"] += to_float(row["total_fuel_sales"])
            bucket["shop_total_revenue"] += to_float(row["shop_total_revenue"])
            bucket["total_traffic"] += to_float(row["total_traffic"])
            bucket["temperature"] += to_float(row["temperature"])
            bucket["precipitation_mm"] += to_float(row["precipitation_mm"])
            bucket["ad_active"] += to_float(row["ad_active"])
            bucket["promotion_fuel_active"] += to_float(row["promotion_fuel_active"])
            bucket["promotion_shop_active"] += to_float(row["promotion_shop_active"])
            bucket["promotion_cafe_active"] += to_float(row["promotion_cafe_active"])
            bucket["price_gap_ai92"] += to_float(row["price_AI92"]) - to_float(row["competitor_price_AI92"])
            bucket["price_gap_ai95"] += to_float(row["price_AI95"]) - to_float(row["competitor_price_AI95"])
            bucket["price_gap_dt"] += to_float(row["price_DT_WINTER"]) - to_float(row["competitor_price_DT"])

        for col in FUEL_COLS:
            station_bucket[col] += to_float(row[col])
            daily_bucket[col] += to_float(row[col])
            hourly_bucket[col] += to_float(row[col])

        for col in SHOP_COLS:
            station_bucket[col] += to_float(row[col])
            daily_bucket[col] += to_float(row[col])
            hourly_bucket[col] += to_float(row[col])

        for col in TRAFFIC_COLS:
            station_bucket[col] += to_float(row[col])
            daily_bucket[col] += to_float(row[col])
            hourly_bucket[col] += to_float(row[col])

    station_rows_base = []
    station_rows = []
    for station_id, meta in sorted(metadata.items(), key=lambda item: int(item[0])):
        stats = station_stats.get(station_id, {})
        rows = stats.get("rows", 0.0) or 1.0
        avg_total_fuel_sales = round(stats.get("total_fuel_sales", 0.0) / rows, 2)
        avg_shop_total_revenue = round(stats.get("shop_total_revenue", 0.0) / rows, 2)
        avg_total_traffic = round(stats.get("total_traffic", 0.0) / rows, 2)
        avg_temperature = round(stats.get("temperature", 0.0) / rows, 2)
        avg_precipitation_mm = round(stats.get("precipitation_mm", 0.0) / rows, 2)
        promo_fuel_share = round(stats.get("promotion_fuel_active", 0.0) / rows, 4)
        promo_shop_share = round(stats.get("promotion_shop_active", 0.0) / rows, 4)
        promo_cafe_share = round(stats.get("promotion_cafe_active", 0.0) / rows, 4)
        ad_active_share = round(stats.get("ad_active", 0.0) / rows, 4)
        avg_price_gap_ai92 = round(stats.get("price_gap_ai92", 0.0) / rows, 3)
        avg_price_gap_ai95 = round(stats.get("price_gap_ai95", 0.0) / rows, 3)
        avg_price_gap_dt = round(stats.get("price_gap_dt", 0.0) / rows, 3)
        out = dict(meta)
        out.update(
            {
                "observations": int(stats.get("rows", 0.0)),
                "avg_total_fuel_sales": avg_total_fuel_sales,
                "avg_shop_total_revenue": avg_shop_total_revenue,
                "avg_total_traffic": avg_total_traffic,
                "avg_temperature": avg_temperature,
                "avg_precipitation_mm": avg_precipitation_mm,
                "promo_fuel_share": promo_fuel_share,
                "promo_shop_share": promo_shop_share,
                "promo_cafe_share": promo_cafe_share,
                "ad_active_share": ad_active_share,
                "avg_price_gap_ai92": avg_price_gap_ai92,
                "avg_price_gap_ai95": avg_price_gap_ai95,
                "avg_price_gap_dt": avg_price_gap_dt,
            }
        )
        station_rows.append(out)
        station_rows_base.append(
            {
                "station_id": station_id,
                "station_name": meta.get("station_name", ""),
                "avg_total_fuel_sales": avg_total_fuel_sales,
                "avg_shop_total_revenue": avg_shop_total_revenue,
                "avg_total_traffic": avg_total_traffic,
                "avg_price_gap_ai92": avg_price_gap_ai92,
                "avg_price_gap_ai95": avg_price_gap_ai95,
                "avg_price_gap_dt": avg_price_gap_dt,
                "promo_fuel_share": promo_fuel_share,
                "promo_shop_share": promo_shop_share,
                "promo_cafe_share": promo_cafe_share,
                "ad_active_share": ad_active_share,
                "has_cafe": to_int(meta.get("has_cafe", 0)),
                "has_shop": to_int(meta.get("has_shop", 0)),
                "has_car_wash": to_int(meta.get("has_car_wash", 0)),
                "has_tire_service": to_int(meta.get("has_tire_service", 0)),
                "distance_to_city_km": to_float(meta.get("distance_to_city_km", 0)),
                "total_pumps": to_int(meta.get("total_pumps", 0)),
            }
        )

    sales_median = median([row["avg_total_fuel_sales"] for row in station_rows_base]) if station_rows_base else 0.0
    traffic_median = median([row["avg_total_traffic"] for row in station_rows_base]) if station_rows_base else 0.0

    recommendation_rows = []
    for row in station_rows_base:
        actions = []
        priority = 0
        if row["avg_total_fuel_sales"] <= sales_median:
            priority += 2
            actions.append("Слабые продажи топлива относительно медианы сети")
        if row["avg_price_gap_ai92"] < 0 or row["avg_price_gap_ai95"] < 0 or row["avg_price_gap_dt"] < 0:
            priority += 2
            actions.append("Цена ниже рынка по одной или нескольким позициям")
        if row["ad_active_share"] < 0.25:
            priority += 1
            actions.append("Низкая рекламная активность")
        if row["promo_fuel_share"] < 0.25:
            priority += 1
            actions.append("Мало акций на топливо")
        if row["avg_total_traffic"] >= traffic_median and row["has_cafe"] == 0:
            priority += 1
            actions.append("Есть смысл рассмотреть кафе")
        if row["avg_total_traffic"] >= traffic_median and row["has_shop"] == 0:
            priority += 1
            actions.append("Есть смысл усилить магазин")
        if row["has_car_wash"] == 0 and row["avg_total_traffic"] >= traffic_median:
            priority += 1
            actions.append("Проверить окупаемость автомойки")
        if row["has_tire_service"] == 0 and row["distance_to_city_km"] > 10:
            priority += 1
            actions.append("Рассмотреть сервис шиномонтажа на трассовых точках")

        if not actions:
            actions.append("Станция выглядит сбалансированной, держать текущую стратегию")

        recommendation_rows.append(
            {
                "station_id": row["station_id"],
                "station_name": row["station_name"],
                "priority_score": priority,
                "avg_total_fuel_sales": row["avg_total_fuel_sales"],
                "avg_shop_total_revenue": row["avg_shop_total_revenue"],
                "avg_total_traffic": row["avg_total_traffic"],
                "avg_price_gap_ai92": row["avg_price_gap_ai92"],
                "avg_price_gap_ai95": row["avg_price_gap_ai95"],
                "avg_price_gap_dt": row["avg_price_gap_dt"],
                "ad_active_share": row["ad_active_share"],
                "promo_fuel_share": row["promo_fuel_share"],
                "promo_shop_share": row["promo_shop_share"],
                "promo_cafe_share": row["promo_cafe_share"],
                "recommended_actions": " | ".join(actions),
            }
        )

    daily_rows = []
    for date_key in sorted(daily_stats):
        stats = daily_stats[date_key]
        rows = stats.get("rows", 0.0) or 1.0
        daily_rows.append(
            {
                "date": date_key,
                "observations": int(stats.get("rows", 0.0)),
                "total_fuel_sales": round(stats.get("total_fuel_sales", 0.0), 2),
                "shop_total_revenue": round(stats.get("shop_total_revenue", 0.0), 2),
                "total_traffic": round(stats.get("total_traffic", 0.0), 2),
                "avg_temperature": round(stats.get("temperature", 0.0) / rows, 2),
                "avg_precipitation_mm": round(stats.get("precipitation_mm", 0.0) / rows, 2),
                "promo_fuel_share": round(stats.get("promotion_fuel_active", 0.0) / rows, 4),
                "promo_shop_share": round(stats.get("promotion_shop_active", 0.0) / rows, 4),
                "promo_cafe_share": round(stats.get("promotion_cafe_active", 0.0) / rows, 4),
                "ad_active_share": round(stats.get("ad_active", 0.0) / rows, 4),
                "avg_price_gap_ai92": round(stats.get("price_gap_ai92", 0.0) / rows, 3),
                "avg_price_gap_ai95": round(stats.get("price_gap_ai95", 0.0) / rows, 3),
                "avg_price_gap_dt": round(stats.get("price_gap_dt", 0.0) / rows, 3),
            }
        )

    date_rows = [
        build_date_row(date_key, sample_row)
        for date_key, sample_row in sorted(date_samples.items())
    ]

    forecast_rows = []
    if max_date is not None:
        for horizon_day in range(1, 8):
            forecast_date = (max_date + timedelta(days=horizon_day)).strftime("%Y-%m-%d")
            for station_id, meta in sorted(metadata.items(), key=lambda item: int(item[0])):
                forecast_rows.append(
                    {
                        "forecast_date": forecast_date,
                        "horizon_day": horizon_day,
                        "station_id": station_id,
                        "station_name": meta.get("station_name", ""),
                        "target": "",
                        "forecast": "",
                        "lower_bound": "",
                        "upper_bound": "",
                        "model_version": "",
                    }
                )

    hourly_rows = []
    for hour_key in sorted(hourly_stats, key=lambda x: int(x)):
        stats = hourly_stats[hour_key]
        rows = stats.get("rows", 0.0) or 1.0
        hourly_rows.append(
            {
                "hour": int(hour_key),
                "observations": int(stats.get("rows", 0.0)),
                "total_fuel_sales": round(stats.get("total_fuel_sales", 0.0), 2),
                "shop_total_revenue": round(stats.get("shop_total_revenue", 0.0), 2),
                "total_traffic": round(stats.get("total_traffic", 0.0), 2),
                "avg_temperature": round(stats.get("temperature", 0.0) / rows, 2),
                "avg_precipitation_mm": round(stats.get("precipitation_mm", 0.0) / rows, 2),
                "promo_fuel_share": round(stats.get("promotion_fuel_active", 0.0) / rows, 4),
                "promo_shop_share": round(stats.get("promotion_shop_active", 0.0) / rows, 4),
                "promo_cafe_share": round(stats.get("promotion_cafe_active", 0.0) / rows, 4),
                "ad_active_share": round(stats.get("ad_active", 0.0) / rows, 4),
                "avg_price_gap_ai92": round(stats.get("price_gap_ai92", 0.0) / rows, 3),
                "avg_price_gap_ai95": round(stats.get("price_gap_ai95", 0.0) / rows, 3),
                "avg_price_gap_dt": round(stats.get("price_gap_dt", 0.0) / rows, 3),
            }
        )

    write_csv(OUT_DIR / "station_kpis.csv", list(station_rows[0].keys()) if station_rows else [], station_rows)
    write_csv(OUT_DIR / "daily_kpis.csv", list(daily_rows[0].keys()) if daily_rows else [], daily_rows)
    write_csv(OUT_DIR / "hourly_kpis.csv", list(hourly_rows[0].keys()) if hourly_rows else [], hourly_rows)
    write_csv(OUT_DIR / "date_dimension.csv", list(date_rows[0].keys()) if date_rows else [], date_rows)
    write_csv(OUT_DIR / "station_recommendations.csv", list(recommendation_rows[0].keys()) if recommendation_rows else [], recommendation_rows)
    write_csv(OUT_DIR / "forecast_template.csv", list(forecast_rows[0].keys()) if forecast_rows else [], forecast_rows)

    print(f"Processed rows: {total_rows}")
    print(f"Wrote: {OUT_DIR / 'station_kpis.csv'}")
    print(f"Wrote: {OUT_DIR / 'daily_kpis.csv'}")
    print(f"Wrote: {OUT_DIR / 'hourly_kpis.csv'}")
    print(f"Wrote: {OUT_DIR / 'date_dimension.csv'}")
    print(f"Wrote: {OUT_DIR / 'station_recommendations.csv'}")
    print(f"Wrote: {OUT_DIR / 'forecast_template.csv'}")


if __name__ == "__main__":
    build_assets()
