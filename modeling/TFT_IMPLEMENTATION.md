# TFT implementation plan

## Цель модели

Обучить Temporal Fusion Transformer для почасового прогноза продаж сети АЗС. Модель должна использовать временные, погодные, рекламные, ценовые, трафиковые и статические признаки станции.

## Целевые переменные

Основная цель первой версии:

- `total_fuel_sales`

Дополнительные цели для следующих запусков:

- `sales_AI92`
- `sales_AI95`
- `sales_AI98`
- `sales_DT_EURO`
- `sales_DT_TANEKO`
- `sales_DT_SUMMER`
- `sales_DT_WINTER`
- `shop_total_revenue`

## Признаки

Static categoricals:

- `station_id`
- `road_type`
- `direction`
- `settlement_size`

Static reals:

- `distance_to_city_km`
- `total_pumps`
- `shop_area_m2`
- `has_car_wash`
- `has_cafe`
- `has_shop`
- `competitors_within_5km`
- `corporate_customer_ratio`
- `staff_engagement_score`
- `customer_loyalty_score`

Time-varying known categoricals:

- `season`
- `weather_condition`
- `ad_channel`
- `holiday_name`

Time-varying known reals:

- `hour`
- `day_of_week`
- `week_of_year`
- `month`
- `quarter`
- `is_weekend`
- `is_holiday`
- `is_rush_hour`
- `is_night`
- `temperature`
- `precipitation_mm`
- `visibility_km`
- `wind_speed_ms`
- `is_snow`
- `is_rain`
- `is_fog`
- `promotion_fuel_active`
- `promotion_shop_active`
- `promotion_cafe_active`
- `ad_active`
- `competitor_price_AI92`
- `competitor_price_AI95`
- `competitor_price_DT`
- `price_AI92`
- `price_AI95`
- `price_AI98`
- `price_DT_EURO`
- `price_DT_TANEKO`
- `price_DT_SUMMER`
- `price_DT_WINTER`

Time-varying observed reals:

- `traffic_Passengers_cars`
- `traffic_Truck_short`
- `traffic_Truck`
- `traffic_Truck_long`
- `traffic_Transporter`
- `traffic_Undefined`
- `total_traffic`
- целевая переменная и лаги целевой переменной

## Split

Данные почасовые за 2023 год. Разбивка должна идти по времени, одинаково для всех станций:

- train: январь-сентябрь;
- validation: октябрь-ноябрь;
- test: декабрь.

Такой split сохраняет временной порядок и проверяет прогноз на будущем периоде.

## Метрики

Обязательные метрики:

- `MAE`
- `MSE`
- `RMSE`
- `R2`

Дополнительно можно добавить:

- `MAPE`, если нет проблем с нулевыми значениями;
- ошибки по станциям;
- ошибки по часам суток;
- ошибки по типам топлива.

## Артефакты

Сохраняем:

- checkpoint модели;
- JSON-конфиг;
- прогнозы на test/horizon;
- метрики;
- важность признаков или variable selection scores;
- рекомендации для dashboard.

## Dashboard

Dash должен показывать:

- обзор сети АЗС и фильтры по станции/периоду;
- динамику `total_fuel_sales`, топлива по типам и `shop_total_revenue`;
- влияние рекламы, акций, трафика, погоды и цен конкурентов;
- actual vs forecast;
- метрики модели;
- важность признаков;
- рекомендации по станциям.
