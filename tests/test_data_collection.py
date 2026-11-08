"""Tests for hardened data collection and schema handling."""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.data_collection import DataCollector, retry_call


def test_retry_call_succeeds_first_try():
    mock = MagicMock(return_value="success")
    result = retry_call(mock, max_retries=3, initial_delay=0.01)
    assert result == "success"
    assert mock.call_count == 1


def test_retry_call_recovers_after_failures():
    mock = MagicMock(side_effect=[ValueError("error 1"), ConnectionError("error 2"), "recovered"])
    result = retry_call(mock, max_retries=3, initial_delay=0.01, backoff_factor=1.0)
    assert result == "recovered"
    assert mock.call_count == 3


def test_retry_call_raises_after_max_retries():
    mock = MagicMock(side_effect=ConnectionError("fail forever"))
    with pytest.raises(ConnectionError, match="fail forever"):
        retry_call(mock, max_retries=2, initial_delay=0.01)
    assert mock.call_count == 2


def test_clean_price_dataframe_handles_multiindex():
    collector = DataCollector()
    dates = pd.bdate_range("2024-01-01", periods=5)
    # Simulate yfinance multi-index columns: (Metric, Ticker)
    tuples = [
        ('Date', ''),
        ('Open', 'AAPL'),
        ('High', 'AAPL'),
        ('Low', 'AAPL'),
        ('Close', 'AAPL'),
        ('Adj Close', 'AAPL'),
        ('Volume', 'AAPL'),
    ]
    cols = pd.MultiIndex.from_tuples(tuples)
    raw_df = pd.DataFrame(
        [
            [dates[0], 100, 102, 99, 101, 101, 1000],
            [dates[1], 101, 103, 100, 102, 102, 1100],
            [dates[2], 102, 104, 101, 103, 103, 1200],
            [dates[3], 103, 105, 102, 104, 104, 1300],
            [dates[4], 104, 106, 103, 105, 105, 1400],
        ],
        columns=cols
    )

    cleaned = collector._clean_price_dataframe(raw_df, "AAPL")
    assert list(cleaned.columns) == ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker']
    assert cleaned['Ticker'].iloc[0] == "AAPL"
    assert len(cleaned) == 5
    assert not isinstance(cleaned.columns, pd.MultiIndex)


def test_clean_price_dataframe_fallback_adj_close():
    collector = DataCollector()
    dates = pd.bdate_range("2024-01-01", periods=3)
    raw_df = pd.DataFrame({
        'Date': dates,
        'Open': [10, 11, 12],
        'High': [12, 13, 14],
        'Low': [9, 10, 11],
        'Close': [11, 12, 13],
        'Volume': [100, 100, 100],
    })
    cleaned = collector._clean_price_dataframe(raw_df, "XYZ")
    assert 'Adj Close' in cleaned.columns
    assert (cleaned['Adj Close'] == cleaned['Close']).all()


def test_clean_earnings_dataframe_filters_future_and_nan_dates():
    collector = DataCollector()
    now = pd.Timestamp.now()
    past_date = now - pd.Timedelta(days=90)
    future_date = now + pd.Timedelta(days=30)

    # Typical yfinance earnings_dates index format
    raw_df = pd.DataFrame(
        {
            'EPS Estimate': [1.2, 1.5, 2.0, 1.0],
            'Reported EPS': [1.3, np.nan, 2.1, 0.9],
            'Surprise(%)': [8.3, np.nan, 5.0, -10.0],
        },
        index=[past_date, future_date, pd.NaT, past_date - pd.Timedelta(days=90)]
    )

    cleaned = collector._clean_earnings_dataframe(raw_df, "AAPL")
    assert len(cleaned) == 2
    assert (cleaned['Date'] <= now).all()
    assert not cleaned['Date'].isna().any()
    assert 'Surprise_Percentage' in cleaned.columns
    assert 'EPS' in cleaned.columns
    assert 'Ticker' in cleaned.columns


def test_parallel_fetch_multiple_stocks(monkeypatch):
    """Verify fetch_multiple_stocks runs concurrently across worker threads and preserves ordering"""
    collector = DataCollector(max_workers=3)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]

    def _mock_price(ticker):
        return pd.DataFrame({
            'Date': [pd.Timestamp('2023-01-01')],
            'Open': [100.0], 'High': [105.0], 'Low': [95.0], 'Close': [102.0],
            'Adj Close': [102.0], 'Volume': [1000], 'Ticker': [ticker]
        })

    def _mock_earnings(ticker):
        return pd.DataFrame({
            'Date': [pd.Timestamp('2023-01-15')],
            'EPS': [1.5],
            'Surprise_Percentage': [3.5],
            'Ticker': [ticker]
        })

    monkeypatch.setattr(collector, "fetch_stock_price_data", _mock_price)
    monkeypatch.setattr(collector, "fetch_earnings_dates", _mock_earnings)

    # Test concurrent execution
    price_df, earnings_df = collector.fetch_multiple_stocks(tickers, max_workers=2)
    assert len(price_df) == 4
    assert len(earnings_df) == 4
    assert price_df['Ticker'].tolist() == tickers
    assert earnings_df['Ticker'].tolist() == tickers

    # Test serial execution fallback (max_workers=1)
    price_serial, earnings_serial = collector.fetch_multiple_stocks(tickers, max_workers=1)
    assert price_serial['Ticker'].tolist() == tickers
    assert earnings_serial['Ticker'].tolist() == tickers


