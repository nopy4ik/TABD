# Data Model

## Источники

- `detailed_data.csv` - основная факт-таблица с часовыми наблюдениями.
- `stations_metadata.csv` - справочник АЗС со статикой.
- `5stations_data.csv` и `5stations_metadata.csv` - компактные наборы для проверки и демонстрации.

## Логическая схема

### Fact

`detailed_data.csv`

Ключевые поля:

- `timestamp`
- `station_id`
- `total_traffic`
- `total_fuel_sales`
- `shop_total_revenue`
- `promotion_fuel_active`
- `promotion_shop_active`
- `promotion_cafe_active`
- `ad_active`
- `ad_channel`
- `competitor_price_*`
- `price_*`
- `weather_condition`
- `season`
- `hour`, `day_of_week`, `week_of_year`, `month`, `quarter`

### Dimension

`stations_metadata.csv`

Ключевые поля:

- `station_id`
- `station_name`
- `road_type`
- `direction`
- `settlement_size`
- `distance_to_city_km`
- `total_pumps`
- `shop_area_m2`
- `has_car_wash`
- `has_tire_service`
- `has_cafe`
- `has_hotel`
- `has_shop`
- `customer_loyalty_score`
- `staff_quality_score`
- `corporate_customer_ratio`
- `staff_engagement_score`

## Связи

- `detailed_data[station_id]` -> `stations_metadata[station_id]`
- Календарная таблица -> `detailed_data[timestamp]` по дате

## Почему это удобно

- факт-таблица хранит временной ряд и позволяет строить прогнозы;
- справочник АЗС помогает резать показатели по типу дороги, направлению, удаленности и сервисам;
- отдельная календарная таблица упрощает time intelligence;
- сводные витрины из `derived/` ускоряют сборку dashboard в Power BI.

