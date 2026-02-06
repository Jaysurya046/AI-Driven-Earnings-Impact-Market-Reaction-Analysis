"""
Data Collection Module
Fetches historical stock price data and earnings announcement information
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """Collects historical stock price and earnings data"""
    
    def __init__(self, lookback_years: int = 5):
        """
        Initialize DataCollector
        
        Args:
            lookback_years: Years of historical data to fetch
        """
        self.lookback_years = lookback_years
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_years * 365)
    
    def fetch_stock_price_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetch historical stock price data from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            DataFrame with columns: Date, Open, Close, High, Low, Volume, Adj Close
        """
        try:
            logger.info(f"Fetching stock data for {ticker}...")
            stock_data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            
            stock_data = stock_data.reset_index()
            stock_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            stock_data['Ticker'] = ticker
            
            logger.info(f"Successfully fetched {len(stock_data)} trading days for {ticker}")
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_earnings_dates(self, ticker: str) -> pd.DataFrame:
        """
        Fetch earnings announcement dates for a stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with earnings dates and information
        """
        try:
            logger.info(f"Fetching earnings dates for {ticker}...")
            stock = yf.Ticker(ticker)
            
            earnings_dates = stock.earnings_dates
            if earnings_dates is not None and len(earnings_dates) > 0:
                earnings_df = earnings_dates.reset_index()
                earnings_df.columns = ['Date', 'EPS', 'Surprise_Percentage']
                earnings_df['Ticker'] = ticker
                earnings_df['Date'] = pd.to_datetime(earnings_df['Date'])
                
                logger.info(f"Found {len(earnings_df)} earnings dates for {ticker}")
                return earnings_df
            else:
                logger.warning(f"No earnings dates found for {ticker}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching earnings for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_stocks(self, tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch data for multiple stocks
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Tuple of (price_data_df, earnings_data_df)
        """
        all_price_data = []
        all_earnings_data = []
        
        for ticker in tickers:
            price_data = self.fetch_stock_price_data(ticker)
            earnings_data = self.fetch_earnings_dates(ticker)
            
            if not price_data.empty:
                all_price_data.append(price_data)
            
            if not earnings_data.empty:
                all_earnings_data.append(earnings_data)
        
        price_df = pd.concat(all_price_data, ignore_index=True) if all_price_data else pd.DataFrame()
        earnings_df = pd.concat(all_earnings_data, ignore_index=True) if all_earnings_data else pd.DataFrame()
        
        return price_df, earnings_df
    
    def save_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save DataFrame to CSV file
        
        Args:
            df: DataFrame to save
            filepath: Path to save CSV file
        """
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
