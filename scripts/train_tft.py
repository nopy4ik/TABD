"""Entry point for TFT training.

Install dependencies from requirements.txt before running this script.
The full implementation will use pytorch-forecasting after the baseline
pipeline and dashboard artifacts are verified.
"""

from __future__ import annotations


def main() -> None:
    try:
        import pytorch_forecasting  # noqa: F401
        import lightning  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "TFT dependencies are not installed yet. Run: "
            "python -m pip install -r requirements.txt"
        ) from exc

    raise SystemExit("TFT training implementation is the next step after baseline validation.")


if __name__ == "__main__":
    main()
