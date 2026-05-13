# TFT implementation plan

## Текущее ограничение

В этой среде нет `torch`, `pandas`, `numpy` и `pytorch_forecasting`, поэтому полноценное обучение TFT здесь не запускается.

## Что нужно сделать в Python-среде

1. Установить зависимости:
   - `torch`
   - `pytorch-forecasting`
   - `lightning`
   - `pandas`
   - `numpy`
2. Сформировать train/validation/test split по времени.
3. Обучить отдельные модели:
   - по `total_fuel_sales`;
   - по каждой фракции топлива;
   - по `shop_total_revenue`.
4. Сохранить checkpoint и конфигурацию.
5. Сделать скрипт загрузки уже обученной модели.
6. Экспортировать прогноз в CSV для Power BI.

## Минимальная схема обучения

- `target`: `total_fuel_sales`
- дополнительные цели:
  - `sales_AI92`
  - `sales_AI95`
  - `sales_AI98`
  - `sales_DT_EURO`
  - `sales_DT_TANEKO`
  - `sales_DT_SUMMER`
  - `sales_DT_WINTER`
  - `shop_total_revenue`
- static covariates:
  - тип дороги;
  - направление;
  - удаленность;
  - число колонок;
  - сервисы;
  - базовые цены;
  - customer / staff scores.
- time-varying known inputs:
  - календарь;
  - сезон;
  - праздники;
  - час;
  - день недели;
  - погода;
  - рекламные флаги;
  - акции;
  - цены конкурентов.
- time-varying observed inputs:
  - трафик по типам;
  - фактические продажи;
  - выручка магазина.

## Рекомендуемый split

- train: первые 70 процентов времени;
- validation: следующие 15 процентов;
- test: последние 15 процентов;
- для каждой станции использовать один и тот же временной срез.

## Конфигурация модели

- `max_encoder_length`: 30-60 часов;
- `max_prediction_length`: 24-72 часа;
- `hidden_size`: 16-64;
- `attention_head_size`: 4-8;
- `dropout`: 0.1-0.3;
- `loss`: QuantileLoss или MSE для baseline.

## Артефакты, которые стоит сохранить

- `tft_checkpoint.ckpt`
- `tft_config.json`
- `feature_mapping.json`
- `forecast.csv`
- `metrics.json`

## Входы для TFT

- статические признаки станции;
- погодные признаки;
- трафик по типам транспорта;
- акции и реклама;
- цены конкурентов;
- собственные цены;
- календарные признаки.

## Выходы

- прогноз `total_fuel_sales`;
- прогноз по типам топлива;
- прогноз `shop_total_revenue`;
- интервалы неопределенности;
- feature importance / variable selection scores.

## Что показать в dashboard после обучения

- actual vs forecast;
- forecast horizon;
- ошибки прогноза;
- влияние признаков;
- рекомендации по станциям и временным слотам.

## Как связать с Power BI

1. Сохрани `forecast.csv` с полями:
   - `timestamp`
   - `station_id`
   - `target`
   - `forecast`
   - `lower_bound`
   - `upper_bound`
   - `model_version`
2. Загрузи `forecast.csv` в Power BI.
3. Свяжи его по `station_id` и дате с календарной таблицей.
4. Покажи actual vs forecast на отдельной странице.
