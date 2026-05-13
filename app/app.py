"""Dash dashboard for TFT analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from dash import Dash, Input, Output, dcc, html
    import plotly.express as px
except ImportError as exc:
    raise SystemExit("Install dashboard dependencies: python -m pip install -r requirements.txt") from exc


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "_задание"
ARTIFACTS_DIR = ROOT / "artifacts"


def load_source() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "detailed_data.csv", parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["day_name"] = df["day_of_week"].map(
        {
            0: "Понедельник",
            1: "Вторник",
            2: "Среда",
            3: "Четверг",
            4: "Пятница",
            5: "Суббота",
            6: "Воскресенье",
        }
    )
    return df


def load_metrics() -> dict:
    path = ARTIFACTS_DIR / "metrics" / "baseline_metrics.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


source_df = load_source()
forecast_df = load_optional_csv(ARTIFACTS_DIR / "forecasts" / "baseline_forecast.csv")
importance_df = load_optional_csv(ARTIFACTS_DIR / "metrics" / "baseline_feature_importance.csv")
recommendations_df = load_optional_csv(ARTIFACTS_DIR / "recommendations" / "station_recommendations.csv")
metrics = load_metrics()

app = Dash(__name__)
server = app.server

station_options = [
    {"label": name, "value": station_id}
    for station_id, name in source_df[["station_id", "station_name"]].drop_duplicates().sort_values("station_id").values
]


def metric_card(title: str, value: object) -> html.Div:
    return html.Div(
        [
            html.Div(title, className="metric-title"),
            html.Div("-" if value is None else value, className="metric-value"),
        ],
        className="metric-card",
    )


app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("TFT-анализ сети АЗС"),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="station-filter",
                            options=station_options,
                            value=station_options[0]["value"],
                            clearable=False,
                        )
                    ],
                    className="toolbar",
                ),
            ],
            className="header",
        ),
        html.Div(
            [
                metric_card("MAE", round(metrics.get("mae", 0), 3) if metrics else None),
                metric_card("MSE", round(metrics.get("mse", 0), 3) if metrics else None),
                metric_card("RMSE", round(metrics.get("rmse", 0), 3) if metrics else None),
                metric_card("R2", round(metrics.get("r2", 0), 3) if metrics else None),
            ],
            className="metrics",
        ),
        dcc.Tabs(
            [
                dcc.Tab(label="Аналитика", children=[dcc.Graph(id="sales-trend"), dcc.Graph(id="traffic-sales")]),
                dcc.Tab(label="Прогноз", children=[dcc.Graph(id="forecast-graph")]),
                dcc.Tab(label="Признаки", children=[dcc.Graph(id="importance-graph")]),
                dcc.Tab(label="Рекомендации", children=[html.Div(id="recommendations-table")]),
            ]
        ),
    ],
    className="page",
)


@app.callback(
    Output("sales-trend", "figure"),
    Output("traffic-sales", "figure"),
    Output("forecast-graph", "figure"),
    Output("importance-graph", "figure"),
    Output("recommendations-table", "children"),
    Input("station-filter", "value"),
)
def update_dashboard(station_id: int):
    station_data = source_df[source_df["station_id"] == station_id].copy()
    daily = (
        station_data.groupby("date", as_index=False)
        .agg(total_fuel_sales=("total_fuel_sales", "sum"), shop_total_revenue=("shop_total_revenue", "sum"))
    )
    sales_fig = px.line(
        daily,
        x="date",
        y=["total_fuel_sales", "shop_total_revenue"],
        title="Динамика продаж топлива и выручки магазина",
    )

    traffic_hourly = (
        station_data.groupby("hour", as_index=False)
        .agg(total_fuel_sales=("total_fuel_sales", "mean"), total_traffic=("total_traffic", "mean"))
    )
    traffic_fig = px.bar(
        traffic_hourly,
        x="hour",
        y=["total_fuel_sales", "total_traffic"],
        barmode="group",
        title="Средние продажи и трафик по часам",
    )

    if forecast_df.empty:
        forecast_fig = px.line(title="Сначала запустите scripts/train_baseline.py или scripts/train_tft.py")
    else:
        station_forecast = forecast_df[forecast_df["station_id"] == station_id].copy()
        station_forecast["timestamp"] = pd.to_datetime(station_forecast["timestamp"])
        forecast_fig = px.line(
            station_forecast,
            x="timestamp",
            y=["actual", "prediction"],
            title="Actual vs forecast на тестовом периоде",
        )

    if importance_df.empty:
        importance_fig = px.bar(title="Важность признаков появится после обучения")
    else:
        top_importance = importance_df.head(20)
        importance_fig = px.bar(
            top_importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Top-20 важных признаков baseline-модели",
        )
        importance_fig.update_layout(yaxis={"categoryorder": "total ascending"})

    if recommendations_df.empty:
        recommendations = html.Div("Рекомендации появятся после обучения модели.")
    else:
        row = recommendations_df[recommendations_df["station_id"] == station_id].head(1)
        recommendations = html.Table(
            [
                html.Thead(html.Tr([html.Th(col) for col in row.columns])),
                html.Tbody([html.Tr([html.Td(row.iloc[0][col]) for col in row.columns])]) if not row.empty else html.Tbody(),
            ],
            className="data-table",
        )

    return sales_fig, traffic_fig, forecast_fig, importance_fig, recommendations


if __name__ == "__main__":
    app.run(debug=False)
