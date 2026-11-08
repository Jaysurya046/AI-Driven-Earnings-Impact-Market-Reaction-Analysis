"""Shared fixtures for the earnings-analysis test suite."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the project root importable regardless of where pytest is invoked
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_ticker_frame(ticker: str, n_days: int = 250, seed: int = 0) -> pd.DataFrame:
    """Synthetic trading-day price series with a jump on the event day."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n_days)))
    volume = rng.integers(1_000_000, 5_000_000, n_days).astype(float)

    # Deterministic reaction so labels are predictable in tests
    event_idx = n_days - 30  # leave post-event room
    prices[event_idx + 1:] *= 1.05  # +5% jump after the announcement
    volume[event_idx + 1:] *= 2.0

    return pd.DataFrame({
        "Date": dates,
        "Open": prices,
        "High": prices * 1.01,
        "Low": prices * 0.99,
        "Close": prices,
        "Adj Close": prices,
        "Volume": volume,
        "Ticker": ticker,
    })


@pytest.fixture
def price_data() -> pd.DataFrame:
    frames = [_make_ticker_frame(t, seed=i) for i, t in enumerate(["AAA", "BBB"])]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def earnings_data(price_data) -> pd.DataFrame:
    """One earnings date per ticker, taken from the synthetic event day."""
    rows = []
    for ticker in ["AAA", "BBB"]:
        dates = price_data.loc[price_data["Ticker"] == ticker, "Date"]
        event_date = dates.iloc[len(dates) - 30]
        rows.append({
            "Date": event_date,
            "EPS": 1.5,
            "Surprise_Percentage": 3.0,
            "Ticker": ticker,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def event_analysis_df(price_data, earnings_data) -> pd.DataFrame:
    from src.event_analysis import EventAnalyzer

    analyzer = EventAnalyzer(event_window=5, estimation_window=60)
    df = analyzer.analyze_multiple_events(price_data, earnings_data)
    assert not df.empty, "fixture failed to produce event analysis"
    return df
