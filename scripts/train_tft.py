"""Заготовка точки входа для обучения настоящей TFT-модели.

Сейчас проект уже имеет рабочий baseline-пайплайн в `scripts/train_baseline.py`.
Этот файл оставлен как место для полноценной реализации Temporal Fusion
Transformer через `pytorch-forecasting`.

Почему здесь пока нет запуска обучения:
- локальное окружение должно стабильно импортировать `torch`, `lightning`,
  `pytorch_forecasting`;
- после этого нужно собрать `TimeSeriesDataSet`, обучить
  `TemporalFusionTransformer` и сохранить артефакты в форматах dashboard.

Dashboard уже ожидает такие файлы:
- `artifacts/forecasts/baseline_forecast.csv`;
- `artifacts/metrics/baseline_metrics.json`;
- `artifacts/metrics/baseline_feature_importance.csv`;
- `artifacts/recommendations/station_recommendations.csv`.

Когда TFT будет готова, можно либо заменить baseline-файлы, либо сохранить
TFT-файлы рядом и переключить dashboard на них.
"""

from __future__ import annotations


def check_tft_environment() -> None:
    """Проверяет, что зависимости для TFT импортируются без ошибок."""

    try:
        import torch  # noqa: F401
        import lightning  # noqa: F401
        import pytorch_forecasting  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Не хватает зависимостей для TFT. Запустите: "
            "python -m pip install -r requirements.txt"
        ) from exc
    except OSError as exc:
        raise SystemExit(
            "PyTorch установлен, но не смог загрузить системные DLL. "
            "Нужно переустановить CPU-версию PyTorch или проверить Visual C++ Runtime."
        ) from exc


def main() -> None:
    """Пока выполняет только проверку окружения и объясняет следующий шаг."""

    check_tft_environment()
    raise SystemExit(
        "Окружение TFT импортируется. Следующий шаг разработки: "
        "реализовать TimeSeriesDataSet и обучение TemporalFusionTransformer."
    )


if __name__ == "__main__":
    main()
