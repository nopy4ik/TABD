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

FUEL_LABELS = {
    "sales_AI92": "АИ-92",
    "sales_AI95": "АИ-95",
    "sales_AI98": "АИ-98",
    "sales_DT_EURO": "ДТ Евро+",
    "sales_DT_TANEKO": "ДТ ТАНЕКО",
    "sales_DT_SUMMER": "ДТ летнее",
    "sales_DT_WINTER": "ДТ зимнее",
}

SHOP_LABELS = {
    "shop_напитки": "Напитки",
    "shop_закуски": "Закуски",
    "shop_автотовары": "Автотовары",
    "shop_кофе": "Кофе",
    "shop_табак": "Табак",
}

FEATURE_LABELS = {
    "total_traffic": "Суммарный трафик",
    "hour": "Час суток",
    "day_of_week": "День недели",
    "temperature": "Температура",
    "promotion_fuel_active": "Акция на топливо",
    "promotion_shop_active": "Акция на товары",
    "promotion_cafe_active": "Акция на кафе",
    "ad_active": "Реклама активна",
    "price_AI92": "Цена АИ-92",
    "price_AI95": "Цена АИ-95",
    "price_DT_WINTER": "Цена ДТ зимнее",
    "competitor_price_AI92": "Цена конкурента АИ-92",
    "competitor_price_AI95": "Цена конкурента АИ-95",
    "competitor_price_DT": "Цена конкурента ДТ",
    "traffic_Passengers_cars": "Трафик легковых",
    "traffic_Truck": "Трафик грузовых",
    "traffic_Truck_long": "Трафик тяжелых грузовых",
    "is_weekend": "Выходной день",
    "is_holiday": "Праздник",
    "is_rush_hour": "Час пик",
    "is_night": "Ночное время",
}

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
    """Загружает метрики baseline и TFT; поддерживает разные JSON-форматы."""

    rows = []
    for path in [
        ARTIFACTS_DIR / "metrics" / "baseline_metrics.json",
        ARTIFACTS_DIR / "metrics" / "tft_metrics.json",
    ]:
        if not path.exists():
            continue

        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Baseline сохраняет список, TFT сохраняет один словарь.
        if isinstance(data, dict):
            rows.append(data)
        else:
            rows.extend(data)

    return pd.DataFrame(rows)


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
future_baseline_df = load_optional_csv(
    ARTIFACTS_DIR / "forecasts" / "baseline_future_forecast.csv",
    parse_dates=["timestamp"],
)
tft_backtest_df = load_optional_csv(
    ARTIFACTS_DIR / "forecasts" / "tft_backtest_forecast.csv",
    parse_dates=["timestamp"],
)
tft_future_df = load_optional_csv(
    ARTIFACTS_DIR / "forecasts" / "tft_future_forecast.csv",
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
                                dcc.Graph(id="future-fuel-graph"),
                                dcc.Graph(id="future-shop-graph"),
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
    Output("future-fuel-graph", "figure"),
    Output("future-shop-graph", "figure"),
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
        daily.rename(
            columns={
                "total_fuel_sales": "Продажи топлива",
                "shop_total_revenue": "Выручка магазина",
            }
        ),
        x="date",
        y=["Продажи топлива", "Выручка магазина"],
        title=f"Динамика продаж и товаров: {station_name}",
        template="plotly_white",
    )

    fuel_mix = station_data[FUEL_COLUMNS].sum().reset_index()
    fuel_mix.columns = ["fuel_type", "sales"]
    fuel_mix["fuel_type"] = fuel_mix["fuel_type"].map(FUEL_LABELS)
    fuel_mix_fig = px.pie(
        fuel_mix,
        names="fuel_type",
        values="sales",
        title="Структура продаж по видам топлива",
        hole=0.35,
    )

    shop_mix = station_data[SHOP_COLUMNS].sum().reset_index()
    shop_mix.columns = ["category", "revenue"]
    shop_mix["category"] = shop_mix["category"].map(SHOP_LABELS)
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

    station_forecast = forecast_df[forecast_df["station_id"] == station_id].copy() if not forecast_df.empty else pd.DataFrame()
    station_tft_backtest = (
        tft_backtest_df[tft_backtest_df["station_id"].astype(str) == str(station_id)].copy()
        if not tft_backtest_df.empty
        else pd.DataFrame()
    )
    station_tft_future = (
        tft_future_df[tft_future_df["station_id"].astype(str) == str(station_id)].copy()
        if not tft_future_df.empty
        else pd.DataFrame()
    )
    station_future_baseline = (
        future_baseline_df[future_baseline_df["station_id"] == station_id].copy()
        if not future_baseline_df.empty
        else pd.DataFrame()
    )

    if forecast_df.empty and station_tft_backtest.empty:
        fuel_forecast_fig = empty_figure("Сначала запустите scripts/train_baseline.py")
        shop_forecast_fig = empty_figure("Сначала запустите scripts/train_baseline.py")
        future_fuel_fig = empty_figure("Будущий прогноз появится после обучения")
        future_shop_fig = empty_figure("Будущий прогноз появится после обучения")
        forecast_error_fig = empty_figure("Ошибки прогноза появятся после обучения")
    else:
        if not station_tft_backtest.empty:
            fuel_forecast = station_tft_backtest
            fuel_title = "Проверка TFT на декабре: факт против прогноза топлива"
        else:
            fuel_forecast = station_forecast[station_forecast["target"] == "total_fuel_sales"]
            fuel_title = "Проверка baseline на декабре: факт против прогноза топлива"
        shop_forecast = station_forecast[station_forecast["target"] == "shop_total_revenue"]

        fuel_forecast_fig = px.line(
            fuel_forecast.rename(columns={"actual": "Факт", "prediction": "Прогноз"}),
            x="timestamp",
            y=["Факт", "Прогноз"],
            title=fuel_title,
            template="plotly_white",
        )
        shop_forecast_fig = px.line(
            shop_forecast.rename(columns={"actual": "Факт", "prediction": "Прогноз"}),
            x="timestamp",
            y=["Факт", "Прогноз"],
            title="Проверка baseline на декабре: факт против прогноза выручки магазина",
            template="plotly_white",
        )

        if not station_tft_future.empty:
            future_fuel_fig = px.line(
                station_tft_future,
                x="timestamp",
                y="prediction",
                title="Будущий прогноз продаж топлива TFT после 2023-12-31",
                labels={"timestamp": "Дата и время", "prediction": "Прогноз"},
                template="plotly_white",
            )
        else:
            future_fuel = station_future_baseline[station_future_baseline["target"] == "total_fuel_sales"]
            future_fuel_fig = px.line(
                future_fuel,
                x="timestamp",
                y="prediction",
                title="Будущий прогноз продаж топлива baseline после 2023-12-31",
                labels={"timestamp": "Дата и время", "prediction": "Прогноз"},
                template="plotly_white",
            )

        future_shop = station_future_baseline[station_future_baseline["target"] == "shop_total_revenue"]
        future_shop_fig = px.line(
            future_shop,
            x="timestamp",
            y="prediction",
            title="Будущий прогноз выручки магазина baseline после 2023-12-31",
            labels={"timestamp": "Дата и время", "prediction": "Прогноз"},
            template="plotly_white",
        )

        error_data = pd.concat(
            [
                station_forecast,
                station_tft_backtest.assign(target_label="Продажи топлива TFT")
                if not station_tft_backtest.empty
                else pd.DataFrame(),
            ],
            ignore_index=True,
        )
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
        recommendation_columns = {
            "station_id": "ID АЗС",
            "station_name": "Название АЗС",
            "actual_mean": "Средний факт",
            "prediction_mean": "Средний прогноз",
            "mae": "MAE",
            "gap": "Смещение",
            "recommendation": "Рекомендация",
        }
        station_error_fig = px.bar(
            recommendations_df.sort_values("mae", ascending=False),
            x="station_name",
            y="mae",
            title="MAE прогноза топлива по станциям",
            labels={"station_name": "АЗС", "mae": "MAE"},
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

        recommendation_display = recommendations_df.rename(columns=recommendation_columns)
        recommendation_table = dash_table.DataTable(
            data=recommendation_display.to_dict("records"),
            columns=[{"name": col, "id": col} for col in recommendation_display.columns],
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
        top_importance = top_importance.copy()
        top_importance["feature_label"] = top_importance["feature"].map(FEATURE_LABELS).fillna(top_importance["feature"])
        importance_fig = px.bar(
            top_importance,
            x="importance",
            y="feature_label",
            orientation="h",
            title="Top-20 важных признаков для прогноза топлива",
            labels={"importance": "Важность", "feature_label": "Признак"},
            template="plotly_white",
        )
        importance_fig.update_layout(yaxis={"categoryorder": "total ascending"})

    traffic_hour = (
        station_data.groupby("hour", as_index=False)
        .agg(total_fuel_sales=("total_fuel_sales", "mean"), total_traffic=("total_traffic", "mean"))
    )
    traffic_hour_fig = px.bar(
        traffic_hour.rename(columns={"total_fuel_sales": "Продажи топлива", "total_traffic": "Трафик"}),
        x="hour",
        y=["Продажи топлива", "Трафик"],
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
        future_fuel_fig,
        future_shop_fig,
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
