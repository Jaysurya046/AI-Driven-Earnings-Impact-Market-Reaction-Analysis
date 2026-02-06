"""
Event-Based Analysis Module
Computes returns, abnormal returns, and volatility around earnings events
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
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
        
        # Calculate actual returns during event
        actual_returns = self.calculate_returns(event_prices)
        
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
    ) -> Dict:
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
            ticker_data = ticker_data.sort_values('Date')
            
            # Define date ranges
            event_start = earnings_date - pd.Timedelta(days=self.event_window)
            event_end = earnings_date + pd.Timedelta(days=self.event_window)
            estimation_start = earnings_date - pd.Timedelta(days=self.estimation_window + self.event_window)
            estimation_end = event_start
            
            # Extract price windows
            event_window_data = ticker_data[
                (ticker_data['Date'] >= event_start) & 
                (ticker_data['Date'] <= event_end)
            ]
            estimation_window_data = ticker_data[
                (ticker_data['Date'] >= estimation_start) & 
                (ticker_data['Date'] <= estimation_end)
            ]
            
            if len(event_window_data) == 0 or len(estimation_window_data) == 0:
                logger.warning(f"Insufficient data for {ticker} on {earnings_date}")
                return None
            
            # Calculate returns
            pre_event = ticker_data[
                (ticker_data['Date'] < earnings_date) & 
                (ticker_data['Date'] >= event_start)
            ]
            post_event = ticker_data[
                (ticker_data['Date'] > earnings_date) & 
                (ticker_data['Date'] <= event_end)
            ]
            
            pre_returns = self.calculate_returns(pre_event['Adj Close'])
            post_returns = self.calculate_returns(post_event['Adj Close'])
            
            # Calculate abnormal returns
            abnormal_returns = self.calculate_abnormal_returns(
                event_window_data['Adj Close'],
                estimation_window_data['Adj Close']
            )
            car = self.calculate_cumulative_abnormal_returns(abnormal_returns)
            
            # Calculate volatility
            pre_volatility = pre_returns.std() if len(pre_returns) > 0 else np.nan
            post_volatility = post_returns.std() if len(post_returns) > 0 else np.nan
            estimation_volatility = self.calculate_returns(estimation_window_data['Adj Close']).std()
            
            # Calculate volume changes
            pre_volume = pre_event['Volume'].mean() if len(pre_event) > 0 else np.nan
            post_volume = post_event['Volume'].mean() if len(post_event) > 0 else np.nan
            est_volume = estimation_window_data['Volume'].mean()
            
            results = {
                'Ticker': ticker,
                'EarningsDate': earnings_date,
                'PreEventReturn': pre_returns.sum(),
                'PostEventReturn': post_returns.sum(),
                'AbnormalReturnCAR': car.iloc[-1] * 100 if len(car) > 0 else np.nan,
                'PreEventVolatility': pre_volatility,
                'PostEventVolatility': post_volatility,
                'EstimationVolatility': estimation_volatility,
                'PreEventAvgVolume': pre_volume,
                'PostEventAvgVolume': post_volume,
                'EstimationAvgVolume': est_volume,
                'VolumeSpikeRatio': (post_volume / est_volume) if est_volume > 0 else np.nan,
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
                analysis['SurprisePercentage'] = row.get('Surprise_Percentage', np.nan)
                results.append(analysis)
        
        return pd.DataFrame(results)
