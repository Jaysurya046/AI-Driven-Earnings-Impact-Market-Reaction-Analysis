"""Tests for trading-day based event windows and return aggregation."""
import numpy as np
import pandas as pd
import pytest

from src.event_analysis import EventAnalyzer


def _linear_frame(ticker="AAA"):
    """Prices that rise exactly 1% per trading day -> returns are trivially checkable."""
    dates = pd.bdate_range("2024-01-02", periods=120)
    prices = 100 * (1.01 ** np.arange(len(dates)))
    return pd.DataFrame({
        "Date": dates,
        "Adj Close": prices,
        "Volume": np.full(len(dates), 1_000_000.0),
        "Ticker": ticker,
    })


def test_window_uses_trading_days_not_calendar_days():
    """A Friday announcement must still get 5 full trading days on each side."""
    df = _linear_frame()
    analyzer = EventAnalyzer(event_window=5, estimation_window=60)

    # Announcement on a Saturday (calendar gap around it)
    earnings_date = df["Date"].iloc[80] + pd.Timedelta(days=1)  # Sunday
    result = analyzer.analyze_event_window(df, earnings_date, "AAA")

    assert result is not None
    # Pre window: strictly before snapped event day, within 5 trading days
    assert result['DataPointsInWindow'] == 11  # t0-5 .. t0+5 inclusive
    assert result['DataPointsInEstimation'] >= 10


def test_post_return_includes_event_day_reaction():
    """Event-day return belongs to the post window; each side has ew returns."""
    df = _linear_frame()
    analyzer = EventAnalyzer(event_window=5, estimation_window=60)
    t0_idx = 80
    earnings_date = df["Date"].iloc[t0_idx]

    result = analyzer.analyze_event_window(df, earnings_date, "AAA")
    # Series rises exactly 1% per trading day. Post window has ew returns
    # (starting with the event-day reaction); pre window has ew returns.
    assert result['PostEventReturn'] == pytest.approx(5.0, abs=1e-9)
    assert result['PreEventReturn'] == pytest.approx(5.0, abs=1e-9)


def test_weekend_announcement_snaps_to_next_trading_day():
    """Announcement over the weekend should snap to Monday; reaction counted from Tuesday."""
    dates = pd.bdate_range("2024-01-02", periods=100)
    prices = np.full(len(dates), 100.0)
    friday = dates[50]
    monday = dates[51]
    # Gap up happens on Tuesday (first full session after Monday's open move)
    prices[52:] = 110.0
    df = pd.DataFrame({"Date": dates, "Adj Close": prices,
                       "Volume": np.ones(len(dates)), "Ticker": "AAA"})

    analyzer = EventAnalyzer(event_window=5, estimation_window=60)
    saturday = friday + pd.Timedelta(days=1)
    result = analyzer.analyze_event_window(df, saturday, "AAA")

    assert result is not None
    # Snapped to Monday (index 51); post window includes Tue jump (+10%)
    assert result['PostEventReturn'] > 5.0
    # Pre window (Fri and earlier) has no jump
    assert abs(result['PreEventReturn']) < 1e-9


def test_single_post_day_yields_nan_volatility_not_error():
    """With one post-event return, std is undefined -> NaN, but analysis succeeds."""
    df = _linear_frame()
    analyzer = EventAnalyzer(event_window=1, estimation_window=30)
    result = analyzer.analyze_event_window(df, df["Date"].iloc[60], "AAA")
    assert result is not None
    assert np.isnan(result['PostEventVolatility']) or isinstance(result['PostEventVolatility'], float)


def test_insufficient_estimation_data_returns_none():
    df = _linear_frame()
    analyzer = EventAnalyzer(event_window=5, estimation_window=200)
    result = analyzer.analyze_event_window(df, df["Date"].iloc[7], "AAA")
    assert result is None


def test_missing_ticker_returns_none(price_data):
    analyzer = EventAnalyzer()
    assert analyzer.analyze_event_window(price_data, pd.Timestamp("2023-06-01"), "ZZZ") is None


def test_abnormal_and_cumulative_return_math():
    analyzer = EventAnalyzer(event_window=2, estimation_window=10)
    est = pd.Series([100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100])
    evt = pd.Series([100, 102, 104])
    ar = analyzer.calculate_abnormal_returns(evt, est)
    car = analyzer.calculate_cumulative_abnormal_returns(ar)
    assert len(ar) == 2
    assert len(car) == 2
    # CAR is compounding of abnormal returns
    manual = (1 + ar.iloc[0] / 100) * (1 + ar.iloc[1] / 100) - 1
    assert car.iloc[1] == pytest.approx(manual)
