"""Интерактивный Dash-dashboard для анализа сети АЗС.

Приложение читает исходные данные из `_задание/` и артефакты обучения из
`artifacts/`. Поэтому порядок работы такой:

1. запустить `python scripts/train_tft.py`;
2. запустить `python app/app.py`;
3. открыть `http://127.0.0.1:8050/`.

Dashboard показывает именно TFT-прогнозы по топливу и товарам, а также
метрики качества, ошибки и рекомендации по станциям.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
    "distance_to_city_km": "Расстояние до города",
    "total_pumps": "Количество колонок",
    "shop_area_m2": "Площадь магазина, м2",
    "has_car_wash": "Есть автомойка",
    "has_cafe": "Есть кафе",
    "has_shop": "Есть магазин",
    "competitors_within_5km": "Конкуренты в 5 км",
    "corporate_customer_ratio": "Доля корпоративных клиентов",
    "staff_engagement_score": "Вовлеченность персонала",
    "customer_loyalty_score": "Лояльность клиентов",
    "total_traffic": "Суммарный трафик",
    "hour": "Час суток",
    "day_of_week": "День недели",
    "week_of_year": "Неделя года",
    "month": "Месяц",
    "quarter": "Квартал",
    "is_weekend": "Выходной день",
    "is_holiday": "Праздник",
    "is_rush_hour": "Час пик",
    "is_night": "Ночное время",
    "temperature": "Температура",
    "precipitation_mm": "Осадки, мм",
    "visibility_km": "Видимость, км",
    "wind_speed_ms": "Скорость ветра, м/с",
    "is_snow": "Снег",
    "is_rain": "Дождь",
    "is_fog": "Туман",
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
    "traffic_Transporter": "Трафик фургонов",
    "traffic_Undefined": "Прочий трафик",
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

    df = pd.read_csv(DATA_DIR / "5stations_data.csv", parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["day_name"] = df["day_of_week"].map(DAY_NAMES)
    return df


def load_metrics() -> pd.DataFrame:
    """Загружает метрики TFT; поддерживает разные JSON-форматы."""

    rows = []
    for path in [
        ARTIFACTS_DIR / "metrics" / "tft_total_fuel_sales_metrics.json",
        ARTIFACTS_DIR / "metrics" / "tft_shop_total_revenue_metrics.json",
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


def attach_model_family(metrics: pd.DataFrame) -> pd.DataFrame:
    """Добавляет столбец с семейством модели для фильтрации в дашборде."""

    if metrics.empty:
        return metrics

    result = metrics.copy()
    result["model_family"] = "tft"
    return result


def load_optional_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Читает CSV, если файл существует, иначе возвращает пустой DataFrame."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates)


def build_metric_card(title: str, value: object, suffix: str = "") -> html.Div:
    """Создает компактную карточку с одним числовым показателем."""

    display_value = format_metric_value(value, suffix)

    return html.Div(
        [
            html.Div(title, className="metric-title"),
            html.Div(display_value, className="metric-value"),
        ],
        className="metric-card",
    )


def format_metric_value(value: object, suffix: str = "") -> str:
    """Форматирует метрики в понятный для человека вид."""

    if value is None or value == "":
        return "-"

    if pd.isna(value):
        return "-"

    if isinstance(value, (int,)) and not isinstance(value, bool):
        formatted = f"{value:,}".replace(",", " ")
    elif isinstance(value, float):
        abs_value = abs(value)
        if suffix == "%":
            formatted = f"{value:.1f}"
        elif abs_value >= 10000:
            formatted = f"{value:,.0f}"
        elif abs_value >= 1000:
            formatted = f"{value:,.1f}"
        elif abs_value >= 100:
            formatted = f"{value:,.1f}"
        elif abs_value >= 10:
            formatted = f"{value:,.2f}"
        else:
            formatted = f"{value:,.3f}"
        formatted = formatted.replace(",", " ")
    else:
        formatted = str(value)

    return f"{formatted}{suffix}"


def build_badge(label: str, value: str) -> html.Div:
    """Создает короткий информационный бейдж для шапки."""

    return html.Div(
        [
            html.Span(label, className="badge-label"),
            html.Span(value, className="badge-value"),
        ],
        className="info-badge",
    )


def resolve_model_family(choice: str) -> str:
    """Выбирает фактическую модель для отображения."""

    return "tft"


def family_label(family: str) -> str:
    """Возвращает человекочитаемое имя модели."""

    return "TFT"


def pick_model_metrics(choice: str) -> pd.DataFrame:
    """Возвращает метрики для выбранного семейства модели."""

    if metrics_df.empty:
        return metrics_df

    return metrics_df.copy()


def pick_fuel_forecasts(choice: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Возвращает backtest и future прогнозы для выбранного семейства."""

    if not tft_backtest_df.empty:
        backtest = tft_backtest_df[tft_backtest_df["target"].astype(str) == "total_fuel_sales"].copy()
        future = tft_future_df[tft_future_df["target"].astype(str) == "total_fuel_sales"].copy()
        return backtest, future, "TFT"
    return pd.DataFrame(), pd.DataFrame(), "TFT"


def pick_shop_forecasts(choice: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Возвращает backtest и future прогнозы для магазина."""

    if not tft_shop_backtest_df.empty:
        backtest = tft_shop_backtest_df[tft_shop_backtest_df["target"].astype(str) == "shop_total_revenue"].copy()
        future = tft_shop_future_df[tft_shop_future_df["target"].astype(str) == "shop_total_revenue"].copy()
        return backtest, future, "TFT"
    return pd.DataFrame(), pd.DataFrame(), "TFT"


def build_station_recommendations(fuel_forecast: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Считает простые рекомендации по станциям из выбранного прогноза."""

    if fuel_forecast.empty or {"actual", "prediction"} - set(fuel_forecast.columns):
        return pd.DataFrame()

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
    station_errors["gap_pct"] = np.where(
        station_errors["actual_mean"].abs() > 1e-9,
        station_errors["gap"] / station_errors["actual_mean"].abs() * 100,
        0.0,
    )
    station_errors["model_name"] = model_name
    station_errors["recommendation"] = np.select(
        [
            station_errors["gap_pct"] > 5,
            station_errors["gap_pct"] < -5,
        ],
        [
            "Стоит увеличить закупку: модель ожидает спрос выше факта",
            "Можно уменьшить закупку: модель ожидает спрос ниже факта",
        ],
        default="Объем можно оставить без изменений",
    )
    return station_errors.sort_values("gap_pct", key=lambda s: s.abs(), ascending=False)


def build_action_phrase(gap: float, item_label: str) -> str:
    """Возвращает короткую фразу для рекомендации по закупке."""

    if gap > 0:
        return f"Стоит увеличить закупку: по {item_label.lower()} прогноз выше факта."
    if gap < 0:
        return f"Можно сократить закупку: по {item_label.lower()} прогноз ниже факта."
    return f"Объем {item_label.lower()} можно оставить без изменений."


def build_action_recommendations(forecast: pd.DataFrame, target_name: str, item_label: str, model_name: str) -> pd.DataFrame:
    """Считает понятные рекомендации для топлива или магазина."""

    if forecast.empty or {"actual", "prediction"} - set(forecast.columns):
        return pd.DataFrame()

    rows = (
        forecast.assign(abs_error=lambda x: (x["actual"] - x["prediction"]).abs())
        .groupby(["station_id", "station_name"], as_index=False)
        .agg(
            actual_mean=("actual", "mean"),
            prediction_mean=("prediction", "mean"),
            mae=("abs_error", "mean"),
        )
    )
    rows["target"] = target_name
    rows["target_label"] = item_label
    rows["model_name"] = model_name
    rows["gap"] = rows["prediction_mean"] - rows["actual_mean"]
    rows["gap_pct"] = np.where(rows["actual_mean"].abs() > 1e-9, rows["gap"] / rows["actual_mean"].abs() * 100, 0.0)
    rows["recommendation"] = rows["gap"].map(lambda value: build_action_phrase(float(value), item_label))
    return rows.sort_values("gap_pct", key=lambda s: s.abs(), ascending=False)


def build_metric_section(title: str, metrics_row: pd.Series | None) -> html.Div:
    """Создает блок с карточками MAE/MSE/RMSE/R2."""

    if metrics_row is None or metrics_row.empty:
        return html.Div([html.H3(title), html.P("Метрики появятся после обучения модели.")], className="metric-section")

    return html.Div(
        [
            html.H3(title),
            html.Div(
                [
                    build_metric_card("MAE", float(metrics_row["mae"])),
                    build_metric_card("MSE", float(metrics_row["mse"])),
                    build_metric_card("RMSE", float(metrics_row["rmse"])),
                    build_metric_card("R2", float(metrics_row["r2"])),
                ],
                className="metrics",
            ),
        ],
        className="metric-section",
    )


def empty_figure(title: str) -> go.Figure:
    """Возвращает пустой график с понятной подписью."""

    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white")
    return fig


metrics_df = attach_model_family(load_metrics())
source_df = load_source()
DATA_START = source_df["timestamp"].min().date().isoformat()
DATA_END = source_df["timestamp"].max().date().isoformat()
MODEL_STATUS_TEXT = "TFT для топлива и товаров"
tft_backtest_df = load_optional_csv(
    ARTIFACTS_DIR / "forecasts" / "tft_total_fuel_sales_backtest_forecast.csv",
    parse_dates=["timestamp"],
)
tft_future_df = load_optional_csv(
    ARTIFACTS_DIR / "forecasts" / "tft_total_fuel_sales_future_forecast.csv",
    parse_dates=["timestamp"],
)
tft_shop_backtest_df = load_optional_csv(
    ARTIFACTS_DIR / "forecasts" / "tft_shop_total_revenue_backtest_forecast.csv",
    parse_dates=["timestamp"],
)
tft_shop_future_df = load_optional_csv(
    ARTIFACTS_DIR / "forecasts" / "tft_shop_total_revenue_future_forecast.csv",
    parse_dates=["timestamp"],
)
importance_df = load_optional_csv(ARTIFACTS_DIR / "metrics" / "tft_feature_importance.csv")

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
                        html.Div("TABD / TFT", className="eyebrow"),
                        html.H1("Аналитический дашборд сети АЗС"),
                        html.P(
                            "Показывает продажи топлива и магазина, влияние рекламы, погоды и трафика, "
                            "а также прогнозы, качество модели и рекомендации по станциям."
                        ),
                        html.Div(
                            [
                                build_badge("АЗС", f"{len(station_options)} станций"),
                                build_badge("Период данных", f"{DATA_START} - {DATA_END}"),
                                build_badge("Модель", MODEL_STATUS_TEXT),
                            ],
                            className="header-badges",
                        ),
                    ],
                    className="header-copy",
                ),
                html.Div(
                    [
                        html.Label("Выбор АЗС"),
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
                    className="app-tab",
                    selected_className="app-tab--selected",
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
                    className="app-tab",
                    selected_className="app-tab--selected",
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
                    label="Качество и признаки",
                    className="app-tab",
                    selected_className="app-tab--selected",
                    children=[
                        html.Div(id="quality-panel", className="panel"),
                    ],
                ),
                dcc.Tab(
                    label="Рекомендации",
                    className="app-tab",
                    selected_className="app-tab--selected",
                    children=[
                        html.Div(id="recommendations-panel", className="panel"),
                        html.Div(id="recommendations-table", className="panel"),
                    ],
                ),
            ],
            className="app-tabs",
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
    Output("quality-panel", "children"),
    Output("recommendations-panel", "children"),
    Output("recommendations-table", "children"),
    Input("station-filter", "value"),
)
def update_dashboard(station_id: int):
    """Перестраивает все графики при выборе другой АЗС."""

    station_data = source_df[source_df["station_id"] == station_id].copy()
    station_name = station_data["station_name"].iloc[0]
    selected_model_label = "TFT"

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
        labels={"date": "Дата", "value": "Значение", "variable": "Показатель"},
        template="plotly_white",
    )
    daily_sales_fig.update_layout(hovermode="x unified")

    fuel_mix = station_data[FUEL_COLUMNS].sum().reset_index()
    fuel_mix.columns = ["fuel_type", "sales"]
    fuel_mix["fuel_type"] = fuel_mix["fuel_type"].map(FUEL_LABELS)
    fuel_mix_fig = px.pie(
        fuel_mix,
        names="fuel_type",
        values="sales",
        title="Структура продаж по видам топлива",
        hole=0.35,
        labels={"fuel_type": "Вид топлива", "sales": "Продажи"},
    )

    shop_mix = station_data[SHOP_COLUMNS].sum().reset_index()
    shop_mix.columns = ["category", "revenue"]
    shop_mix["category"] = shop_mix["category"].map(SHOP_LABELS)
    shop_mix_fig = px.bar(
        shop_mix,
        x="category",
        y="revenue",
        title="Выручка по категориям сопутствующих товаров",
        labels={"category": "Категория", "revenue": "Выручка, руб."},
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
        labels={
            "weather_condition": "Погода",
            "total_fuel_sales": "Средние продажи топлива",
            "promotion_fuel_active": "Топливная акция",
        },
        template="plotly_white",
    )

    fuel_backtest, fuel_future, fuel_model_label = pick_fuel_forecasts("tft")
    shop_backtest, shop_future, shop_model_label = pick_shop_forecasts("tft")
    station_fuel_backtest = (
        fuel_backtest[fuel_backtest["station_id"].astype(str) == str(station_id)].copy()
        if not fuel_backtest.empty
        else pd.DataFrame()
    )
    station_fuel_future = (
        fuel_future[fuel_future["station_id"].astype(str) == str(station_id)].copy()
        if not fuel_future.empty
        else pd.DataFrame()
    )
    station_shop_backtest = (
        shop_backtest[shop_backtest["station_id"].astype(str) == str(station_id)].copy()
        if not shop_backtest.empty
        else pd.DataFrame()
    )
    station_shop_future = (
        shop_future[shop_future["station_id"].astype(str) == str(station_id)].copy()
        if not shop_future.empty
        else pd.DataFrame()
    )

    if station_fuel_backtest.empty:
        fuel_forecast_fig = empty_figure("Факт и прогноз топлива появятся после обучения")
        future_fuel_fig = empty_figure("Будущий прогноз топлива появится после обучения")
    else:
        fuel_forecast_fig = px.line(
            station_fuel_backtest.rename(columns={"actual": "Факт", "prediction": "Прогноз"}),
            x="timestamp",
            y=["Факт", "Прогноз"],
            title=f"Проверка {selected_model_label}: факт против прогноза топлива",
            labels={"timestamp": "Дата и время", "value": "Значение", "variable": "Показатель"},
            template="plotly_white",
        )
        if station_fuel_future.empty:
            future_fuel_fig = empty_figure("Будущий прогноз топлива недоступен")
        else:
            future_fuel_fig = px.line(
                station_fuel_future,
                x="timestamp",
                y="prediction",
                title=f"Будущий прогноз продаж топлива: {selected_model_label}",
                labels={"timestamp": "Дата и время", "prediction": "Прогноз"},
                template="plotly_white",
            )

    if station_shop_backtest.empty:
        shop_forecast_fig = empty_figure("Факт и прогноз товаров появятся после обучения")
    else:
        shop_forecast_fig = px.line(
            station_shop_backtest.rename(columns={"actual": "Факт", "prediction": "Прогноз"}),
            x="timestamp",
            y=["Факт", "Прогноз"],
            title=f"Проверка {shop_model_label}: факт против прогноза выручки магазина",
            labels={"timestamp": "Дата и время", "value": "Значение", "variable": "Показатель"},
            template="plotly_white",
        )

    if station_shop_future.empty:
        future_shop_fig = empty_figure("Будущий прогноз товаров недоступен")
    else:
        future_shop_fig = px.line(
            station_shop_future,
            x="timestamp",
            y="prediction",
            title=f"Будущий прогноз выручки магазина: {shop_model_label}",
            labels={"timestamp": "Дата и время", "prediction": "Прогноз"},
            template="plotly_white",
        )

    error_frames = []
    if not station_fuel_backtest.empty:
        error_frames.append(station_fuel_backtest.assign(target_label=f"Топливо / {selected_model_label}"))
    if not station_shop_backtest.empty:
        error_frames.append(station_shop_backtest.assign(target_label="Товары / TFT"))
    if error_frames:
        error_data = pd.concat(error_frames, ignore_index=True)
        error_data["date"] = pd.to_datetime(error_data["timestamp"]).dt.date
        error_data["abs_error"] = (error_data["actual"] - error_data["prediction"]).abs()
        error_data = error_data.groupby(["date", "target_label"], as_index=False).agg(abs_error=("abs_error", "mean"))
        forecast_error_fig = px.line(
            error_data,
            x="date",
            y="abs_error",
            color="target_label",
            title="Средняя абсолютная ошибка по дням",
            labels={"date": "Дата", "abs_error": "Средняя абсолютная ошибка", "target_label": "Направление"},
            template="plotly_white",
        )
    else:
        forecast_error_fig = empty_figure("Ошибки прогноза появятся после обучения")

    selected_metrics = pick_model_metrics("tft")
    fuel_metrics_row = None
    if not selected_metrics.empty:
        selected_fuel_metrics = selected_metrics[selected_metrics["target"] == "total_fuel_sales"]
        if not selected_fuel_metrics.empty:
            fuel_metrics_row = selected_fuel_metrics.iloc[0]

    shop_metrics_row = None
    if not selected_metrics.empty:
        selected_shop_metrics = selected_metrics[selected_metrics["target"] == "shop_total_revenue"]
        if not selected_shop_metrics.empty:
            shop_metrics_row = selected_shop_metrics.iloc[0]

    traffic_hour = (
        station_data.groupby("hour", as_index=False)
        .agg(total_fuel_sales=("total_fuel_sales", "mean"), total_traffic=("total_traffic", "mean"))
    )
    traffic_hour_fig = px.line(
        traffic_hour.rename(columns={"total_fuel_sales": "Продажи топлива", "total_traffic": "Трафик"}),
        x="hour",
        y=["Продажи топлива", "Трафик"],
        title="Средние продажи и трафик по часам",
        labels={"hour": "Час", "value": "Значение", "variable": "Показатель"},
        template="plotly_white",
    )

    fuel_recommendations = build_action_recommendations(
        station_fuel_backtest,
        "total_fuel_sales",
        "Топливо",
        selected_model_label,
    )
    shop_recommendations = build_action_recommendations(
        station_shop_backtest,
        "shop_total_revenue",
        "Товары",
        shop_model_label,
    )
    non_empty_recommendations = [df for df in [fuel_recommendations, shop_recommendations] if not df.empty]
    combined_recommendations = pd.concat(non_empty_recommendations, ignore_index=True) if non_empty_recommendations else pd.DataFrame()

    if combined_recommendations.empty:
        station_gap_fig = empty_figure("Отклонение прогноза по станциям появится после обучения")
        recommendation_panel = html.Div("Рекомендации появятся после обучения модели.")
        recommendation_table = html.Div()
    else:
        station_gap_fig = px.bar(
            combined_recommendations.sort_values("gap", key=lambda s: s.abs(), ascending=False),
            x="station_name",
            y="gap",
            color="target_label",
            barmode="group",
            title="Отклонение прогноза по станциям",
            labels={"station_name": "АЗС", "gap": "Разница", "target_label": "Направление"},
            template="plotly_white",
        )
        station_gap_fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        station_gap_fig.update_layout(xaxis_tickangle=-45)

        def build_recommendation_block(title: str, rows: pd.DataFrame) -> html.Div:
            if rows.empty:
                return html.Div([html.H3(title), html.P("Нет данных для выбранной станции.")], className="panel")

            row = rows.iloc[0]
            return html.Div(
                [
                    html.H3(title),
                    html.P(row["recommendation"]),
                    html.Div(
                        [
                            build_metric_card("Средний факт", float(row["actual_mean"])),
                            build_metric_card("Средний прогноз", float(row["prediction_mean"])),
                            build_metric_card("Разница", float(row["gap"])),
                            build_metric_card("Разница, %", float(row["gap_pct"]), "%"),
                        ],
                        className="metrics",
                    ),
                ],
                className="panel",
            )

        selected_fuel = (
            fuel_recommendations[fuel_recommendations["station_id"] == station_id].head(1)
            if not fuel_recommendations.empty and "station_id" in fuel_recommendations.columns
            else pd.DataFrame()
        )
        selected_shop = (
            shop_recommendations[shop_recommendations["station_id"] == station_id].head(1)
            if not shop_recommendations.empty and "station_id" in shop_recommendations.columns
            else pd.DataFrame()
        )
        recommendation_panel = html.Div(
            [
                build_recommendation_block("Топливо", selected_fuel),
                build_recommendation_block("Товары", selected_shop),
            ]
        )

        recommendation_display = combined_recommendations.rename(
            columns={
                "station_id": "ID АЗС",
                "station_name": "Название АЗС",
                "target_label": "Направление",
                "actual_mean": "Средний факт",
                "prediction_mean": "Средний прогноз",
                "gap": "Разница",
                "gap_pct": "Разница, %",
                "recommendation": "Что делать",
                "model_name": "Модель",
            }
        )
        recommendation_table = dash_table.DataTable(
            data=recommendation_display.to_dict("records"),
            columns=[{"name": col, "id": col} for col in recommendation_display.columns],
            page_size=10,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "fontSize": 13, "padding": "8px"},
        )

    quality_panel = html.Div(
        [
            html.H3("Качество модели и признаки"),
            html.P("Метрики показаны карточками, чтобы не смешивать шкалы. Ниже показаны ошибки и поведение спроса."),
            html.Div(
                [
                    build_metric_section("Топливо", fuel_metrics_row),
                    build_metric_section("Товары", shop_metrics_row),
                ],
                className="metrics",
            ),
            dcc.Graph(figure=traffic_hour_fig),
            dcc.Graph(figure=station_gap_fig),
        ]
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
        quality_panel,
        recommendation_panel,
        recommendation_table,
    )


if __name__ == "__main__":
    app.run(debug=False)
