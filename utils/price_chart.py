"""Price window and chart spec. No Streamlit and no network."""
import os

import altair as alt
import pandas as pd

from utils.signal_view import last_bar_date

_PRICE_COLUMNS = ("Date", "Close", "High", "Low")
_LINE = "#38BDF8"
_POINT = "#F8FAFC"
_GRID = "#1E293B"
_LABEL = "#94A3B8"


def normalize_yahoo(frame):
    """Flatten a Yahoo download to Date plus Close, and High and Low when present."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Date", "Close"])
    work = frame.copy()
    if isinstance(work.columns, pd.MultiIndex):
        work.columns = work.columns.get_level_values(0)
    work = work.reset_index()
    if "Date" not in work.columns and "index" in work.columns:
        work = work.rename(columns={"index": "Date"})
    keep = [name for name in _PRICE_COLUMNS if name in work.columns]
    return work[keep]


def load_saved_prices(path, ticker):
    """Close series for one ticker from a processed file."""
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["Date", "Close"])
    hist = pd.read_csv(path)
    if "Ticker" not in hist.columns:
        return pd.DataFrame(columns=["Date", "Close"])
    hist = hist[hist["Ticker"] == ticker]
    keep = [name for name in _PRICE_COLUMNS if name in hist.columns]
    if "Date" not in keep or "Close" not in keep:
        return pd.DataFrame(columns=["Date", "Close"])
    return hist[keep]


def price_window(frame, n_days):
    """Latest calendar date of the full file, then the rows the chart shows.

    The end date is taken before the window is shortened.
    """
    if frame is None or frame.empty or "Date" not in frame.columns or "Close" not in frame.columns:
        return None, pd.DataFrame(columns=["Date", "Close"])
    work = frame.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date", "Close"]).sort_values("Date")
    file_end = last_bar_date(work["Date"])
    if n_days is not None:
        work = work.tail(int(n_days))
    return file_end, work


def price_span(frame):
    """First close, last close, and the change across the visible window."""
    if frame is None or frame.empty or "Close" not in frame.columns:
        return None
    closes = frame["Close"].dropna()
    if closes.empty:
        return None
    start = float(closes.iloc[0])
    end = float(closes.iloc[-1])
    change = None if start == 0 else (end - start) / start
    return {"start": start, "end": end, "change": change}


def price_chart(frame, style="Line", include_zero=False):
    """Hover, pan, and zoom chart. Returns None when there is nothing to draw."""
    if frame is None or frame.empty:
        return None
    plot = frame.copy()
    if "Date" not in plot.columns:
        plot = plot.reset_index()
    plot["Date"] = pd.to_datetime(plot["Date"], errors="coerce")
    plot = plot.dropna(subset=["Date", "Close"]).sort_values("Date")
    if plot.empty:
        return None

    y_scale = alt.Scale(zero=bool(include_zero), nice=True)
    hover = alt.selection_point(
        name="hover_date",
        nearest=True,
        on="pointerover",
        fields=["Date"],
        empty=False,
    )
    tip = [
        alt.Tooltip("Date:T", title="Date", format="%Y-%m-%d"),
        alt.Tooltip("Close:Q", title="Close", format=",.2f"),
    ]
    base = alt.Chart(plot)
    layers = []
    if {"High", "Low"}.issubset(plot.columns) and plot[["High", "Low"]].notna().any().any():
        layers.append(
            base.mark_area(color=_LINE, opacity=0.16).encode(
                x=alt.X("Date:T", title=None),
                y=alt.Y("Low:Q", title=None, scale=y_scale),
                y2="High:Q",
            )
        )
    if style == "Area":
        layers.append(
            base.mark_area(color=_LINE, opacity=0.28).encode(
                x=alt.X("Date:T", title=None),
                y=alt.Y("Close:Q", title=None, scale=y_scale),
            )
        )
    layers.append(
        base.mark_line(color=_LINE, strokeWidth=2).encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Close:Q", title=None, scale=y_scale),
            tooltip=tip,
        )
    )
    layers.append(
        base.mark_rule(color="#334155").encode(
            x=alt.X("Date:T"),
            opacity=alt.condition(hover, alt.value(0.8), alt.value(0)),
        )
    )
    layers.append(
        base.mark_circle(size=70, color=_POINT).encode(
            x=alt.X("Date:T"),
            y=alt.Y("Close:Q", scale=y_scale),
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=tip,
        ).add_params(hover)
    )
    latest = alt.Chart(plot.tail(1)).mark_circle(size=80, color=_POINT).encode(
        x=alt.X("Date:T"),
        y=alt.Y("Close:Q", scale=y_scale),
        tooltip=tip,
    )
    layers.append(latest)
    chart = alt.layer(*layers).resolve_scale(y="shared").properties(height=380)
    chart = chart.interactive()
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=_LABEL,
            titleColor=_LABEL,
            gridColor=_GRID,
            domainColor="#334155",
        )
    )
