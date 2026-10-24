"""Dashboard signal labels. No Streamlit and no network."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.signal_view import (
    beat_line,
    format_bar_date,
    last_bar_date,
    load_technical_summary,
    position_label,
    universe_lines,
    universe_metrics,
)


def test_position_is_long_only_above_the_threshold():
    assert position_label(0.55, 0.55) == "Long"
    assert position_label(0.54, 0.55) == "Cash"


def test_bar_date_is_a_calendar_day():
    assert format_bar_date("2024-11-03T00:00:00") == "2024-11-03"
    assert format_bar_date("not-a-date") is None


def test_kenya_last_date_is_the_file_end_not_the_chart_window():
    assert last_bar_date(["2024-01-02", "2024-06-01", "2020-03-01"]) == "2024-06-01"


def test_universe_lines_use_the_saved_medians(tmp_path):
    path = tmp_path / "backtest_universe.json"
    path.write_text(json.dumps({
        "technical_summary": {
            "n_tickers": 40,
            "median_sharpe": 0.0,
            "median_sharpe_buyhold": 0.72,
            "median_total_return": -0.0028,
            "median_total_return_buyhold": 0.8138,
            "median_max_drawdown": -0.158,
            "median_max_drawdown_buyhold": -0.3337,
            "beat_buyhold": 7,
        }
    }), encoding="utf-8")
    lines = universe_lines(load_technical_summary(str(path)))
    assert lines[0].startswith("Median Sharpe 0.000 against 0.720")
    assert "81.38%" in lines[1]
    assert "-15.80%" in lines[2]
    assert lines[3] == "7 of 40 names beat buy and hold"


def test_missing_backtest_file_has_no_lines(tmp_path):
    assert load_technical_summary(str(tmp_path / "missing.json")) == {}
    assert universe_lines({}) == []
    assert universe_metrics({}) == []
    assert beat_line({}) == ""


def test_universe_metrics_keep_the_comparison_as_text():
    summary = {
        "n_tickers": 40,
        "median_sharpe": 0.0,
        "median_sharpe_buyhold": 0.72,
        "median_total_return": -0.0028,
        "median_total_return_buyhold": 0.8138,
        "median_max_drawdown": -0.158,
        "median_max_drawdown_buyhold": -0.3337,
        "beat_buyhold": 7,
    }
    metrics = universe_metrics(summary)
    assert [item["label"] for item in metrics] == [
        "Median Sharpe",
        "Median return",
        "Median max drawdown",
    ]
    assert metrics[0]["value"] == "0.000"
    assert metrics[0]["detail"] == "Buy and hold 0.720"
    assert metrics[1]["value"] == "-0.28%"
    assert metrics[2]["detail"] == "Buy and hold -33.37%"
    assert not metrics[2]["detail"].startswith("-")
    assert beat_line(summary) == "7 of 40 names beat buy and hold"
