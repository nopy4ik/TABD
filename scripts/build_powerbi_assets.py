#!/usr/bin/env python3
"""Build compact Power BI-friendly aggregates from the raw CSV files."""

from __future__ import annotations

import csv
from collections import defaultdict
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

PRICE_GAPS = [
    ("AI92", "price_AI92", "competitor_price_AI92"),
    ("AI95", "price_AI95", "competitor_price_AI95"),
    ("DT", "price_DT_WINTER", "competitor_price_DT"),
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


def build_assets():
    ensure_out_dir()

    metadata = load_station_metadata(SOURCE_DIR / "stations_metadata.csv")
    station_stats = defaultdict(lambda: defaultdict(float))
    daily_stats = defaultdict(lambda: defaultdict(float))
    hourly_stats = defaultdict(lambda: defaultdict(float))

    total_rows = 0

    for row in read_csv(SOURCE_DIR / "detailed_data.csv"):
        total_rows += 1
        station_id = row["station_id"]
        date_key = row["timestamp"][:10]
        hour_key = row["hour"]

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

    station_rows = []
    for station_id, meta in sorted(metadata.items(), key=lambda item: int(item[0])):
        stats = station_stats.get(station_id, {})
        rows = stats.get("rows", 0.0) or 1.0
        out = dict(meta)
        out.update(
            {
                "observations": int(stats.get("rows", 0.0)),
                "avg_total_fuel_sales": round(stats.get("total_fuel_sales", 0.0) / rows, 2),
                "avg_shop_total_revenue": round(stats.get("shop_total_revenue", 0.0) / rows, 2),
                "avg_total_traffic": round(stats.get("total_traffic", 0.0) / rows, 2),
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
        station_rows.append(out)

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

    print(f"Processed rows: {total_rows}")
    print(f"Wrote: {OUT_DIR / 'station_kpis.csv'}")
    print(f"Wrote: {OUT_DIR / 'daily_kpis.csv'}")
    print(f"Wrote: {OUT_DIR / 'hourly_kpis.csv'}")


if __name__ == "__main__":
    build_assets()

