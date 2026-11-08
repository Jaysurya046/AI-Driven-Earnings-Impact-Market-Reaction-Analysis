"""
Event-Based Analysis Module
Computes returns, abnormal returns, and volatility around earnings events
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventAnalyzer:
    """Analyzes stock price movements around earnings events"""
    
    def __init__(self, event_window: int = 5, estimation_window: int = 60):
        """
        Initialize EventAnalyzer
        
        Args:
            event_window: Days before/after event to analyze (e.g., 5 = -5 to +5 days)
            estimation_window: Days for calculating baseline metrics (default: -60 to -10)
        """
        self.event_window = event_window
        self.estimation_window = estimation_window
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate daily returns
        
        Args:
            prices: Series of adjusted close prices
            
        Returns:
            Series of daily returns
        """
        return prices.pct_change() * 100
    
    @staticmethod
    def _find_event_index(dates: pd.Series, earnings_date: pd.Timestamp) -> int:
        """
        Locate the trading-day index of an earnings announcement.

        Announcements usually fall on weekends/holidays or after the close,
        so we snap to the nearest available trading day (the first trading
        day on/after the announcement date; if none exists, the last one
        before it). Returns -1 when no data is usable.
        """
        dates = pd.to_datetime(dates).reset_index(drop=True)
        candidates = np.where(dates.values >= np.datetime64(earnings_date))[0]
        if len(candidates) > 0:
            return int(candidates[0])
        candidates = np.where(dates.values <= np.datetime64(earnings_date))[0]
        if len(candidates) == 0:
            return -1
        return int(candidates[-1])
    
    def calculate_abnormal_returns(
        self,
        event_prices: pd.DataFrame,
        estimation_prices: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate abnormal returns using mean-adjusted model
        
        Args:
            event_prices: Prices during event window
            estimation_prices: Prices during estimation window
            
        Returns:
            Series of abnormal returns
        """
        # Calculate expected return from estimation period
        estimation_returns = self.calculate_returns(estimation_prices)
        expected_return = estimation_returns.mean()
        
        # Calculate actual returns during event (drop the leading NaN from
        # pct_change so downstream sums/CAR are not silently NaN)
        actual_returns = self.calculate_returns(event_prices).dropna()

        # Abnormal returns = Actual - Expected
        abnormal_returns = actual_returns - expected_return

        return abnormal_returns

    def calculate_cumulative_abnormal_returns(self, abnormal_returns: pd.Series) -> pd.Series:
        """
        Calculate cumulative abnormal returns (CAR)

        Args:
            abnormal_returns: Series of abnormal returns

        Returns:
            Series of cumulative abnormal returns
        """
        return (1 + abnormal_returns / 100).cumprod() - 1
    
    def analyze_event_window(
        self,
        price_data: pd.DataFrame,
        earnings_date: pd.Timestamp,
        ticker: str
    ) -> Optional[Dict]:
        """
        Comprehensive analysis of stock around earnings event
        
        Args:
            price_data: DataFrame with Date and Adj Close columns
            earnings_date: Date of earnings announcement
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with event analysis metrics
        """
        try:
            # Filter data for ticker
            ticker_data = price_data[price_data['Ticker'] == ticker].copy()
            ticker_data['Date'] = pd.to_datetime(ticker_data['Date'])
            ticker_data = ticker_data.sort_values('Date').reset_index(drop=True)

            if ticker_data.empty:
                logger.warning(f"No price data for {ticker}")
                return None

            # Snap the announcement to a trading day and use *positional*
            # (trading-day) offsets instead of calendar-day windows. Calendar
            # days silently shrink/expand windows across weekends & holidays.
            t0 = self._find_event_index(ticker_data['Date'], earnings_date)
            if t0 < 0:
                logger.warning(f"No trading data near {earnings_date} for {ticker}")
                return None

            ew = self.event_window
            event_start_i = max(t0 - ew, 0)
            event_end_i = min(t0 + ew, len(ticker_data) - 1)
            estimation_start_i = max(t0 - ew - self.estimation_window, 0)
            estimation_end_i = max(t0 - ew - 1, -1)

            event_window_data = ticker_data.iloc[event_start_i:event_end_i + 1]
            estimation_window_data = ticker_data.iloc[
                estimation_start_i:estimation_end_i + 1
            ]

            if len(event_window_data) == 0 or len(estimation_window_data) < 10:
                logger.warning(f"Insufficient data for {ticker} on {earnings_date}")
                return None

            # Pre/post windows are strictly before/after the event day.
            # Returns are computed on the *contiguous* series and then sliced
            # positionally, so every return (including the event-day one) is
            # attributed to exactly one side and none are dropped at boundaries.
            pre_event = ticker_data.iloc[event_start_i:t0]
            post_event = ticker_data.iloc[t0 + 1:event_end_i + 1]

            window_returns = self.calculate_returns(
                ticker_data.iloc[max(event_start_i - 1, 0):event_end_i + 1]['Adj Close']
            )
            # window_returns[k] corresponds to trading row event_start_i-1+k
            offset = t0 - (event_start_i - 1)
            pre_returns = window_returns.iloc[max(offset - ew, 0):offset].dropna()
            post_returns = window_returns.iloc[offset:offset + ew].dropna()

            # Calculate abnormal returns
            abnormal_returns = self.calculate_abnormal_returns(
                event_window_data['Adj Close'],
                estimation_window_data['Adj Close']
            )
            car = self.calculate_cumulative_abnormal_returns(abnormal_returns)

            # Calculate volatility
            pre_volatility = pre_returns.std() if len(pre_returns) > 1 else np.nan
            post_volatility = post_returns.std() if len(post_returns) > 1 else np.nan
            estimation_volatility = self.calculate_returns(estimation_window_data['Adj Close']).std()

            # Calculate volume changes
            pre_volume = pre_event['Volume'].mean() if len(pre_event) > 0 else np.nan
            post_volume = post_event['Volume'].mean() if len(post_event) > 0 else np.nan
            est_volume = estimation_window_data['Volume'].mean()

            results = {
                'Ticker': ticker,
                'EarningsDate': earnings_date,
                'PreEventReturn': float(pre_returns.sum()) if len(pre_returns) else np.nan,
                'PostEventReturn': float(post_returns.sum()) if len(post_returns) else np.nan,
                'AbnormalReturnCAR': car.iloc[-1] * 100 if len(car) > 0 else np.nan,
                'PreEventVolatility': pre_volatility,
                'PostEventVolatility': post_volatility,
                'EstimationVolatility': estimation_volatility,
                'PreEventAvgVolume': pre_volume,
                'PostEventAvgVolume': post_volume,
                'EstimationAvgVolume': est_volume,
                'VolumeSpikeRatio': (post_volume / est_volume) if (est_volume and est_volume > 0) else np.nan,
                'DataPointsInWindow': len(event_window_data),
                'DataPointsInEstimation': len(estimation_window_data)
            }

            return results
            
        except Exception as e:
            logger.error(f"Error analyzing event for {ticker} on {earnings_date}: {str(e)}")
            return None
    
    def analyze_multiple_events(
        self,
        price_data: pd.DataFrame,
        earnings_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze multiple earnings events
        
        Args:
            price_data: DataFrame with historical price data
            earnings_data: DataFrame with earnings dates
            
        Returns:
            DataFrame with analysis results for all events
        """
        results = []

        for _, row in earnings_data.iterrows():
            ticker = row['Ticker']
            earnings_date = pd.to_datetime(row['Date'])

            analysis = self.analyze_event_window(price_data, earnings_date, ticker)
            if analysis:
                analysis['EPS'] = row.get('EPS', np.nan)
                # Accept both naming conventions used across the codebase
                surprise = row.get('SurprisePercentage',
                                   row.get('Surprise_Percentage', np.nan))
                analysis['SurprisePercentage'] = surprise
                results.append(analysis)

        return pd.DataFrame(results)
