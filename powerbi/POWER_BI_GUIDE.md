# Power BI Dashboard Guide

## Цель

Показать, как меняются продажи топлива и сопутствующих товаров под влиянием:

- рекламы;
- трафика по типам транспорта;
- акций;
- цен конкурентов;
- погоды;
- календарных факторов;
- особенностей конкретной АЗС.

## Что загрузить в Power BI

1. `detailed_data.csv` как факт-таблицу.
2. `stations_metadata.csv` как справочник станций.
3. `derived/station_kpis.csv` как витрину для ранжирования АЗС.
4. `derived/daily_kpis.csv` как витрину для трендов.
5. `derived/hourly_kpis.csv` как витрину для часовых паттернов.
6. `derived/date_dimension.csv` как календарную таблицу.
7. `derived/station_recommendations.csv` как таблицу рекомендаций.
8. `derived/forecast_template.csv` как заготовку под будущие прогнозы TFT.

## Модель

- `detailed_data` - центральная таблица.
- `stations_metadata` - dimension по `station_id`.
- `date_dimension` - отдельная календарная таблица.
- `station_recommendations` - витрина для action list по станциям.
- `forecast_template` - заглушка для будущих forecast rows.

## DAX меры

```DAX
Total Fuel Sales = SUM(detailed_data[total_fuel_sales])
Fuel Sales AI92 = SUM(detailed_data[sales_AI92])
Fuel Sales AI95 = SUM(detailed_data[sales_AI95])
Fuel Sales AI98 = SUM(detailed_data[sales_AI98])
Fuel Sales DT = SUM(detailed_data[sales_DT_EURO]) + SUM(detailed_data[sales_DT_TANEKO]) + SUM(detailed_data[sales_DT_SUMMER]) + SUM(detailed_data[sales_DT_WINTER])
Shop Revenue = SUM(detailed_data[shop_total_revenue])
Total Traffic = SUM(detailed_data[total_traffic])
Avg Fuel Sales per Hour = AVERAGE(detailed_data[total_fuel_sales])
Promo Fuel Share = AVERAGE(detailed_data[promotion_fuel_active])
Promo Shop Share = AVERAGE(detailed_data[promotion_shop_active])
Promo Cafe Share = AVERAGE(detailed_data[promotion_cafe_active])
Ad Active Share = AVERAGE(detailed_data[ad_active])
Fuel Sales per Traffic = DIVIDE([Total Fuel Sales], [Total Traffic])
Shop Revenue per Traffic = DIVIDE([Shop Revenue], [Total Traffic])
Price Gap AI92 = AVERAGEX(detailed_data, detailed_data[price_AI92] - detailed_data[competitor_price_AI92])
Price Gap AI95 = AVERAGEX(detailed_data, detailed_data[price_AI95] - detailed_data[competitor_price_AI95])
Price Gap DT = AVERAGEX(detailed_data, detailed_data[price_DT_WINTER] - detailed_data[competitor_price_DT])
```

## Страницы dashboard

### 1. Executive Summary

Что добавить:

- карточки KPI: `Total Fuel Sales`, `Shop Revenue`, `Total Traffic`, `Ad Active Share`, `Promo Fuel Share`;
- line chart: `timestamp` vs `Total Fuel Sales`;
- stacked area: продажи по типам топлива;
- bar chart: топ-АЗС по `Total Fuel Sales`;
- slicers: дата, АЗС, `road_type`, `direction`, `season`, `weather_condition`, `ad_channel`.

Что показать:

- общий объем продаж;
- вклад рекламы и акций;
- сезонные пики;
- лучшие и худшие станции.

### 2. Fuel Mix

Что добавить:

- clustered column chart по типам топлива;
- matrix по станциям и типам топлива;
- heatmap по `hour` x `day_of_week`;
- small multiples по сезону.

Что показать:

- какие виды топлива тянут выручку;
- как меняется микс по времени суток и сезону;
- где есть перекос в сторону отдельных типов топлива.

### 3. Demand Drivers

Что добавить:

- scatter plot: `competitor_price_ratio_*` или `Price Gap` vs `Total Fuel Sales`;
- decomposition tree для `Total Fuel Sales`;
- bar chart по `ad_channel`;
- bar chart по типам транспорта;
- table с условиями погоды и продажами.

Что показать:

- насколько спрос чувствителен к ценам конкурентов;
- какие каналы рекламы работают лучше;
- какой транспорт формирует основную загрузку.

### 4. Station Performance

Что добавить:

- ranking table по `station_name`;
- bar chart по `stations_metadata[road_type]`;
- bar chart по `direction`;
- bar chart по `settlement_size`;
- KPI по сервисам: `has_cafe`, `has_shop`, `has_car_wash`, `has_hotel`.

Что показать:

- какие станции сильнее и почему;
- какие локации лучше монетизируются;
- какие сервисы связаны с более высокой выручкой.

### 5. Recommendations

Что добавить:

- карточки с rule-based рекомендациями;
- table со станциями, где `Price Gap` отрицательный и продажи слабые;
- table со станциями, где `Ad Active Share` низкий, но спрос проседает;
- table с точками, где есть потенциал для кафе/магазина.

Что показать:

- где нужна ценовая корректировка;
- где стоит усилить рекламу;
- где имеет смысл развивать сопутствующие услуги.

### 6. Forecast

Что добавить:

- line chart actual vs forecast;
- band chart с `lower_bound` и `upper_bound`;
- table по ошибкам прогноза;
- slicer по горизонту прогноза и станции;
- KPI по качеству прогноза.

Что показать:

- насколько хорошо TFT повторяет фактические продажи;
- где модель ошибается сильнее всего;
- какие станции требуют отдельной перенастройки.

## Ручные шаги в Power BI

1. Импортируй CSV-файлы.
2. Создай календарную таблицу.
3. Настрой отношения между таблицами.
4. Добавь меры DAX.
5. Собери визуалы по страницам выше.
6. Протестируй фильтры и drill-down.
7. Подключи таблицу прогноза, когда она появится.
8. Экспортируй `.pbix` или `.pbit`.

## Если будет готов TFT

Добавь в модель ещё одну таблицу:

- `forecast.csv` с колонками `timestamp`, `station_id`, `target`, `forecast`, `lower_bound`, `upper_bound`.

После этого на отдельной странице можно добавить:

- line chart actual vs forecast;
- confidence band;
- таблицу ошибок прогноза;
- KPI по MAE / MAPE / RMSE.

## Рекомендации по использованию витрин

- `station_recommendations.csv` удобно вывести в таблицу с условным форматированием по `priority_score`.
- `date_dimension.csv` лучше сделать основной календарной таблицей, а не строить дату прямо из `timestamp`.
- `hourly_kpis.csv` хорошо подходит для графика суточного профиля спроса.
- `daily_kpis.csv` лучше использовать для трендов и сезонности.
- `forecast_template.csv` уже имеет форму, удобную для последующей подстановки фактического прогноза TFT.
