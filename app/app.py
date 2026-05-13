"""Интерактивный Dash-dashboard для анализа сети АЗС.

Приложение читает исходные данные из `_задание/` и артефакты обучения из
`artifacts/`. Поэтому порядок работы такой:

1. запустить `python scripts/train_baseline.py`;
2. запустить `python app/app.py`;
3. открыть `http://127.0.0.1:8050/`.

Когда появится полноценная TFT-модель, она должна сохранить прогнозы и метрики
в те же файлы, и dashboard автоматически покажет новые результаты.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from dash import Dash, Input, Output, dash_table, dcc, html
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as exc:
    raise SystemExit("Установите зависимости: python -m pip install -r requirements.txt") from exc


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "_задание"
ARTIFACTS_DIR = ROOT / "artifacts"

FUEL_COLUMNS = [
    "sales_AI92",
    "sales_AI95",
    "sales_AI98",
    "sales_DT_EURO",
    "sales_DT_TANEKO",
    "sales_DT_SUMMER",
    "sales_DT_WINTER",
]

SHOP_COLUMNS = [
    "shop_напитки",
    "shop_закуски",
    "shop_автотовары",
    "shop_кофе",
    "shop_табак",
]

DAY_NAMES = {
    0: "Понедельник",
    1: "Вторник",
    2: "Среда",
    3: "Четверг",
    4: "Пятница",
    5: "Суббота",
    6: "Воскресенье",
}


def load_source() -> pd.DataFrame:
    """Загружает исходный CSV и добавляет поля только для визуализации."""

    df = pd.read_csv(DATA_DIR / "detailed_data.csv", parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["day_name"] = df["day_of_week"].map(DAY_NAMES)
    return df


def load_metrics() -> pd.DataFrame:
    """Загружает метрики модели; поддерживает старый и новый формат JSON."""

    path = ARTIFACTS_DIR / "metrics" / "baseline_metrics.json"
    if not path.exists():
        return pd.DataFrame()

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Первая версия сохраняла один dict, новая версия сохраняет список dict.
    if isinstance(data, dict):
        data = [data]
    return pd.DataFrame(data)


def load_optional_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Читает CSV, если файл существует, иначе возвращает пустой DataFrame."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates)


def build_metric_card(title: str, value: object, suffix: str = "") -> html.Div:
    """Создает компактную карточку с одним числовым показателем."""

    if value is None or value == "":
        display_value = "-"
    elif isinstance(value, float):
        display_value = f"{value:,.3f}".replace(",", " ")
    else:
        display_value = str(value)

    return html.Div(
        [
            html.Div(title, className="metric-title"),
            html.Div(f"{display_value}{suffix}", className="metric-value"),
        ],
        className="metric-card",
    )


def empty_figure(title: str) -> go.Figure:
    """Возвращает пустой график с понятной подписью."""

    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white")
    return fig


source_df = load_source()
forecast_df = load_optional_csv(
    ARTIFACTS_DIR / "forecasts" / "baseline_forecast.csv",
    parse_dates=["timestamp"],
)
importance_df = load_optional_csv(ARTIFACTS_DIR / "metrics" / "baseline_feature_importance.csv")
recommendations_df = load_optional_csv(ARTIFACTS_DIR / "recommendations" / "station_recommendations.csv")
metrics_df = load_metrics()

app = Dash(__name__)
server = app.server

station_options = [
    {"label": name, "value": int(station_id)}
    for station_id, name in source_df[["station_id", "station_name"]]
    .drop_duplicates()
    .sort_values("station_id")
    .values
]


app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H1("TFT-анализ сети АЗС"),
                        html.P("Аналитика, прогнозы, метрики модели и рекомендации по станциям."),
                    ]
                ),
                html.Div(
                    [
                        html.Label("АЗС"),
                        dcc.Dropdown(
                            id="station-filter",
                            options=station_options,
                            value=station_options[0]["value"],
                            clearable=False,
                        ),
                    ],
                    className="toolbar",
                ),
            ],
            className="header",
        ),
        html.Div(id="summary-cards", className="metrics"),
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Аналитика",
                    children=[
                        html.Div(
                            [
                                dcc.Graph(id="daily-sales-graph"),
                                dcc.Graph(id="fuel-mix-graph"),
                                dcc.Graph(id="shop-mix-graph"),
                                dcc.Graph(id="promo-weather-graph"),
                            ],
                            className="grid",
                        )
                    ],
                ),
                dcc.Tab(
                    label="Прогноз",
                    children=[
                        html.Div(
                            [
                                dcc.Graph(id="fuel-forecast-graph"),
                                dcc.Graph(id="shop-forecast-graph"),
                                dcc.Graph(id="forecast-error-graph"),
                            ],
                            className="grid",
                        )
                    ],
                ),
                dcc.Tab(
                    label="Метрики модели",
                    children=[
                        html.Div(
                            [
                                dcc.Graph(id="metrics-bar-graph"),
                                dcc.Graph(id="station-error-graph"),
                            ],
                            className="grid",
                        )
                    ],
                ),
                dcc.Tab(
                    label="Признаки",
                    children=[
                        html.Div(
                            [
                                dcc.Graph(id="importance-graph"),
                                dcc.Graph(id="traffic-hour-graph"),
                            ],
                            className="grid",
                        )
                    ],
                ),
                dcc.Tab(
                    label="Рекомендации",
                    children=[
                        html.Div(id="recommendations-panel", className="panel"),
                        html.Div(id="recommendations-table", className="panel"),
                    ],
                ),
            ]
        ),
    ],
    className="page",
)


@app.callback(
    Output("summary-cards", "children"),
    Output("daily-sales-graph", "figure"),
    Output("fuel-mix-graph", "figure"),
    Output("shop-mix-graph", "figure"),
    Output("promo-weather-graph", "figure"),
    Output("fuel-forecast-graph", "figure"),
    Output("shop-forecast-graph", "figure"),
    Output("forecast-error-graph", "figure"),
    Output("metrics-bar-graph", "figure"),
    Output("station-error-graph", "figure"),
    Output("importance-graph", "figure"),
    Output("traffic-hour-graph", "figure"),
    Output("recommendations-panel", "children"),
    Output("recommendations-table", "children"),
    Input("station-filter", "value"),
)
def update_dashboard(station_id: int):
    """Перестраивает все графики при выборе другой АЗС."""

    station_data = source_df[source_df["station_id"] == station_id].copy()
    station_name = station_data["station_name"].iloc[0]

    total_fuel = station_data["total_fuel_sales"].sum()
    total_shop = station_data["shop_total_revenue"].sum()
    avg_traffic = station_data["total_traffic"].mean()
    promo_share = station_data["promotion_fuel_active"].mean() * 100

    cards = [
        build_metric_card("Продажи топлива", total_fuel),
        build_metric_card("Выручка магазина", total_shop, " руб."),
        build_metric_card("Средний трафик/час", avg_traffic),
        build_metric_card("Доля топливных акций", promo_share, "%"),
    ]

    daily = (
        station_data.groupby("date", as_index=False)
        .agg(
            total_fuel_sales=("total_fuel_sales", "sum"),
            shop_total_revenue=("shop_total_revenue", "sum"),
            total_traffic=("total_traffic", "sum"),
        )
    )
    daily_sales_fig = px.line(
        daily,
        x="date",
        y=["total_fuel_sales", "shop_total_revenue"],
        title=f"Динамика продаж и товаров: {station_name}",
        template="plotly_white",
    )

    fuel_mix = station_data[FUEL_COLUMNS].sum().reset_index()
    fuel_mix.columns = ["fuel_type", "sales"]
    fuel_mix_fig = px.pie(
        fuel_mix,
        names="fuel_type",
        values="sales",
        title="Структура продаж по видам топлива",
        hole=0.35,
    )

    shop_mix = station_data[SHOP_COLUMNS].sum().reset_index()
    shop_mix.columns = ["category", "revenue"]
    shop_mix_fig = px.bar(
        shop_mix,
        x="category",
        y="revenue",
        title="Выручка по категориям сопутствующих товаров",
        template="plotly_white",
    )

    promo_weather = (
        station_data.groupby(["weather_condition", "promotion_fuel_active"], as_index=False)
        .agg(total_fuel_sales=("total_fuel_sales", "mean"))
    )
    promo_weather["promotion_fuel_active"] = promo_weather["promotion_fuel_active"].map(
        {0: "Без акции", 1: "Акция активна"}
    )
    promo_weather_fig = px.bar(
        promo_weather,
        x="weather_condition",
        y="total_fuel_sales",
        color="promotion_fuel_active",
        barmode="group",
        title="Средние продажи топлива по погоде и акциям",
        template="plotly_white",
    )

    if forecast_df.empty:
        fuel_forecast_fig = empty_figure("Сначала запустите scripts/train_baseline.py")
        shop_forecast_fig = empty_figure("Сначала запустите scripts/train_baseline.py")
        forecast_error_fig = empty_figure("Ошибки прогноза появятся после обучения")
    else:
        station_forecast = forecast_df[forecast_df["station_id"] == station_id].copy()
        fuel_forecast = station_forecast[station_forecast["target"] == "total_fuel_sales"]
        shop_forecast = station_forecast[station_forecast["target"] == "shop_total_revenue"]

        fuel_forecast_fig = px.line(
            fuel_forecast,
            x="timestamp",
            y=["actual", "prediction"],
            title="Прогноз продаж топлива: факт против прогноза",
            template="plotly_white",
        )
        shop_forecast_fig = px.line(
            shop_forecast,
            x="timestamp",
            y=["actual", "prediction"],
            title="Прогноз выручки магазина: факт против прогноза",
            template="plotly_white",
        )

        error_data = station_forecast.copy()
        error_data["abs_error"] = (error_data["actual"] - error_data["prediction"]).abs()
        error_data = (
            error_data.groupby(["target_label", "hour"], as_index=False)
            .agg(abs_error=("abs_error", "mean"))
        )
        forecast_error_fig = px.line(
            error_data,
            x="hour",
            y="abs_error",
            color="target_label",
            title="Средняя абсолютная ошибка по часам",
            template="plotly_white",
        )

    if metrics_df.empty:
        metrics_bar_fig = empty_figure("Метрики появятся после обучения")
    else:
        metrics_long = metrics_df.melt(
            id_vars=["target_label"],
            value_vars=["mae", "mse", "rmse", "r2"],
            var_name="metric",
            value_name="value",
        )
        metrics_bar_fig = px.bar(
            metrics_long,
            x="metric",
            y="value",
            color="target_label",
            barmode="group",
            title="Метрики качества модели",
            template="plotly_white",
        )

    if recommendations_df.empty:
        station_error_fig = empty_figure("Ошибки по станциям появятся после обучения")
        recommendation_panel = html.Div("Рекомендации появятся после обучения модели.")
        recommendation_table = html.Div()
    else:
        station_error_fig = px.bar(
            recommendations_df.sort_values("mae", ascending=False),
            x="station_name",
            y="mae",
            title="MAE прогноза топлива по станциям",
            template="plotly_white",
        )
        station_error_fig.update_layout(xaxis_tickangle=-45)

        selected_row = recommendations_df[recommendations_df["station_id"] == station_id].head(1)
        if selected_row.empty:
            recommendation_panel = html.Div("Для выбранной станции рекомендация не найдена.")
        else:
            row = selected_row.iloc[0]
            recommendation_panel = html.Div(
                [
                    html.H3(f"Рекомендация для {row['station_name']}"),
                    html.P(row["recommendation"]),
                    html.Div(
                        [
                            build_metric_card("MAE станции", float(row["mae"])),
                            build_metric_card("Средний факт", float(row["actual_mean"])),
                            build_metric_card("Средний прогноз", float(row["prediction_mean"])),
                            build_metric_card("Смещение прогноза", float(row["gap"])),
                        ],
                        className="metrics",
                    ),
                ]
            )

        recommendation_table = dash_table.DataTable(
            data=recommendations_df.to_dict("records"),
            columns=[{"name": col, "id": col} for col in recommendations_df.columns],
            page_size=10,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "fontSize": 13, "padding": "8px"},
        )

    if importance_df.empty:
        importance_fig = empty_figure("Важность признаков появится после обучения")
    else:
        top_importance = importance_df[importance_df["target"] == "total_fuel_sales"].head(20)
        importance_fig = px.bar(
            top_importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Top-20 важных признаков для прогноза топлива",
            template="plotly_white",
        )
        importance_fig.update_layout(yaxis={"categoryorder": "total ascending"})

    traffic_hour = (
        station_data.groupby("hour", as_index=False)
        .agg(total_fuel_sales=("total_fuel_sales", "mean"), total_traffic=("total_traffic", "mean"))
    )
    traffic_hour_fig = px.bar(
        traffic_hour,
        x="hour",
        y=["total_fuel_sales", "total_traffic"],
        barmode="group",
        title="Средние продажи и трафик по часам",
        template="plotly_white",
    )

    return (
        cards,
        daily_sales_fig,
        fuel_mix_fig,
        shop_mix_fig,
        promo_weather_fig,
        fuel_forecast_fig,
        shop_forecast_fig,
        forecast_error_fig,
        metrics_bar_fig,
        station_error_fig,
        importance_fig,
        traffic_hour_fig,
        recommendation_panel,
        recommendation_table,
    )


if __name__ == "__main__":
    app.run(debug=False)
