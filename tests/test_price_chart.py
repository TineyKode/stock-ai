"""Price window and chart spec. No Streamlit and no network."""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.price_chart import (
    load_saved_prices,
    normalize_yahoo,
    price_chart,
    price_span,
    price_window,
)


def test_window_reports_the_file_end_before_it_shortens_the_chart():
    frame = pd.DataFrame({
        "Date": ["2024-01-02", "2024-06-01", "2024-12-31"],
        "Close": [10.0, 12.0, 11.0],
    })
    file_end, window = price_window(frame, 1)
    assert file_end == "2024-12-31"
    assert list(window["Close"]) == [11.0]


def test_span_is_the_move_across_the_visible_rows():
    frame = pd.DataFrame({"Close": [10.0, 12.5]})
    assert price_span(frame)["change"] == 0.25
    assert price_span(pd.DataFrame({"Close": []})) is None


def test_yahoo_multiindex_keeps_high_and_low():
    columns = pd.MultiIndex.from_tuples([("Close", "AAPL"), ("High", "AAPL"), ("Low", "AAPL")])
    raw = pd.DataFrame([[10.0, 11.0, 9.0]], columns=columns)
    raw.index.name = "Date"
    flat = normalize_yahoo(raw)
    assert list(flat.columns) == ["Date", "Close", "High", "Low"]


def test_saved_file_filters_to_one_ticker(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("Date,Ticker,Close\n2024-01-02,SCOM,10\n2024-01-02,EQTY,5\n", encoding="utf-8")
    saved = load_saved_prices(str(path), "SCOM")
    assert list(saved["Close"]) == [10]


def test_chart_can_zoom_and_names_the_close():
    frame = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=4),
        "Close": [10.0, 11.0, 10.5, 12.0],
        "High": [11.0, 12.0, 11.0, 13.0],
        "Low": [9.0, 10.0, 10.0, 11.0],
    })
    spec = price_chart(frame, style="Area", include_zero=False).to_dict()
    raw = json.dumps(spec)
    assert "Close" in raw
    params = spec.get("params") or []
    assert any(item.get("bind") == "scales" for item in params)
    assert price_chart(pd.DataFrame(columns=["Date", "Close"])) is None
