# TABD: TFT анализ сети АЗС

В репозитории собраны исходные данные задания, описание модели данных, план dashboard для Power BI и вспомогательные артефакты для подготовки витрины.

## Что здесь есть

- `_задание/` - исходные CSV, PDF и DOCX из задания.
- `powerbi/` - спецификация dashboard, список визуализаций и DAX-мер.
- `modeling/` - план обучения TFT и список того, что нужно будет сделать вручную в Python-среде.
- `notebooks/` - Colab-шаблон для обучения TFT и сохранения модели.
- `scripts/` - утилита для подготовки агрегированных таблиц под Power BI.
- `derived/` - автоматически сгенерированные сводные CSV для импорта в Power BI.

## Кратко по заданию

Нужно сделать TFT-анализ сети АЗС, объяснить продажи топлива и сопутствующих товаров, учесть рекламу, трафик, акции, цены конкурентов и погодные/временные факторы, а затем собрать это в Power BI dashboard.

## Что я уже подготовил

- разобрал текст задания и описание данных;
- выделил главную факт-таблицу `detailed_data.csv` и таблицу статических атрибутов `stations_metadata.csv`;
- подготовил план dashboard по страницам;
- добавил скрипт для генерации сводных таблиц и уже сгенерировал `derived/`;
- создал Colab-ноутбук для обучения TFT: `notebooks/tft_training_colab.ipynb`

```bash
python3 scripts/build_powerbi_assets.py
```

## Как собирать Power BI

1. Импортируй `derived/station_kpis.csv` и `derived/daily_kpis.csv` как вспомогательные витрины.
2. Импортируй `stations_metadata.csv` как справочник АЗС.
3. Импортируй `detailed_data.csv` как факт-таблицу.
4. Создай связь `detailed_data[station_id] -> stations_metadata[station_id]`.
5. Добавь календарную таблицу в Power BI и свяжи её с `detailed_data[timestamp]` по дате.
6. Собери страницы dashboard по `powerbi/POWER_BI_GUIDE.md`.

## Что останется сделать вручную

- обучить TFT в Python-среде с `torch` и `pytorch-forecasting`;
- открыть `notebooks/tft_training_colab.ipynb` в Colab и запустить обучение;
- сохранить checkpoint и файл для повторной загрузки модели;
- выгрузить прогнозы в CSV и подключить их к Power BI;
- собрать `.pbix`/`.pbit` в Power BI Desktop;
- опубликовать dashboard в нужный workspace.
