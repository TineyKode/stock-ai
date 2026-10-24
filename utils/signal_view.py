"""Labels and backtest lines shown on the dashboard. No network and no Streamlit."""
import json
import os

import pandas as pd


def position_label(probability, threshold):
    """Long when the model is confident enough. Otherwise the book is in cash."""
    if probability >= threshold:
        return "Long"
    return "Cash"


def format_bar_date(value):
    """Calendar date of the bar the signal used."""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def last_bar_date(dates):
    """Latest date in a Kenya file. The chart window can be shorter than the file."""
    parsed = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
    if parsed.empty:
        return None
    return parsed.max().strftime("%Y-%m-%d")


_UNIVERSE_KEYS = (
    "n_tickers",
    "median_sharpe",
    "median_sharpe_buyhold",
    "median_total_return",
    "median_total_return_buyhold",
    "median_max_drawdown",
    "median_max_drawdown_buyhold",
    "beat_buyhold",
)


def _complete_summary(summary):
    return bool(summary) and all(key in summary for key in _UNIVERSE_KEYS)


def universe_metrics(summary):
    """Three median cards. Comparisons stay text so a minus sign is not a down arrow."""
    if not _complete_summary(summary):
        return []
    return [
        {
            "label": "Median Sharpe",
            "value": f"{summary['median_sharpe']:.3f}",
            "detail": f"Buy and hold {summary['median_sharpe_buyhold']:.3f}",
        },
        {
            "label": "Median return",
            "value": f"{summary['median_total_return']:.2%}",
            "detail": f"Buy and hold {summary['median_total_return_buyhold']:.2%}",
        },
        {
            "label": "Median max drawdown",
            "value": f"{summary['median_max_drawdown']:.2%}",
            "detail": f"Buy and hold {summary['median_max_drawdown_buyhold']:.2%}",
        },
    ]


def beat_line(summary):
    """How many names beat buy and hold. Empty when the backtest is missing."""
    if not _complete_summary(summary):
        return ""
    return f"{int(summary['beat_buyhold'])} of {int(summary['n_tickers'])} names beat buy and hold"


def universe_lines(summary):
    """Plain-text form of the median cards."""
    metrics = universe_metrics(summary)
    if not metrics:
        return []
    lines = []
    for item in metrics:
        benchmark = item["detail"].removeprefix("Buy and hold ")
        lines.append(f"{item['label']} {item['value']} against {benchmark} for buy and hold")
    lines.append(beat_line(summary))
    return lines


def load_technical_summary(path):
    """Read the technical summary. A missing file means the backtest has not been run."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload.get("technical_summary") or {}
    return summary if isinstance(summary, dict) else {}
