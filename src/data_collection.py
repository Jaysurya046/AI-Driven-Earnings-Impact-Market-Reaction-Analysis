"""
Data Collection Module
Fetches historical stock price data and earnings announcement information
"""

import concurrent.futures
import logging
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_call(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    description: str = "API call",
) -> T:
    """
    Execute a callable with retry logic and exponential backoff.
    
    Args:
        func: Zero-arg callable to execute
        max_retries: Maximum number of attempts
        initial_delay: Delay in seconds before the first retry
        backoff_factor: Multiplier applied to delay on each subsequent retry
        description: Description of the call for logging
        
    Returns:
        Result of the callable
        
    Raises:
        Exception: The last exception encountered if all attempts fail
    """
    delay = initial_delay
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    f"{description} failed on attempt {attempt}/{max_retries}: {e}. Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"{description} failed after {max_retries} attempts: {e}")
    if last_exception:
        raise last_exception
    raise RuntimeError(f"{description} failed after {max_retries} attempts")


class DataCollector:
    """Collects historical stock price and earnings data"""
    
    def __init__(self, lookback_years: int = 5, max_retries: int = 3, max_workers: int = 5):
        """
        Initialize DataCollector
        
        Args:
            lookback_years: Years of historical data to fetch
            max_retries: Number of retries for API requests
            max_workers: Maximum concurrent worker threads for multi-ticker data collection
        """
        self.lookback_years = lookback_years
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_years * 365)
    
    def _clean_price_dataframe(self, raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Normalize and harden price data returned from yfinance.
        Handles MultiIndex columns, missing 'Adj Close', and timezone-aware dates.
        """
        if raw_df is None or raw_df.empty:
            return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker'])

        df = raw_df.copy()

        # Handle MultiIndex columns (common in newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for col in df.columns:
                # Typically level 0 is metric name ('Open', 'Close', etc.)
                col_name = str(col[0]).strip()
                new_cols.append(col_name)
            df.columns = new_cols

        # Ensure Date is a regular column
        if 'Date' not in df.columns:
            df = df.reset_index()
            # If the index name was not Date, rename the first column
            if 'Date' not in df.columns:
                first_col = df.columns[0]
                df = df.rename(columns={first_col: 'Date'})

        # Flatten column names to string and standardize
        col_map = {}
        for c in df.columns:
            c_str = str(c).strip()
            for std_col in ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                if c_str.lower() == std_col.lower() or c_str.lower() == std_col.lower().replace(' ', ''):
                    col_map[c] = std_col
                    break
        df = df.rename(columns=col_map)

        # Fallback for Adj Close if missing
        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            df['Adj Close'] = df['Close']

        # Ensure required columns are present
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Normalize Date column to tz-naive pd.Timestamp
        df['Date'] = pd.to_datetime(df['Date'])
        if hasattr(df['Date'].dt, 'tz') and df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)

        df['Ticker'] = ticker
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker']]

    def fetch_stock_price_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetch historical stock price data from Yahoo Finance with retries
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume, Ticker
        """
        try:
            logger.info(f"Fetching stock data for {ticker}...")

            def _download():
                return yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False
                )

            stock_data = retry_call(
                _download,
                max_retries=self.max_retries,
                description=f"yf.download({ticker})"
            )
            
            cleaned_df = self._clean_price_dataframe(stock_data, ticker)
            logger.info(f"Successfully fetched {len(cleaned_df)} trading days for {ticker}")
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker'])

    def _clean_earnings_dataframe(self, raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Normalize and harden earnings data returned from yfinance.
        Filters future dates, handles various column layouts, and drops NaNs.
        """
        if raw_df is None or raw_df.empty:
            return pd.DataFrame(columns=['Date', 'EPS', 'Surprise_Percentage', 'Ticker'])

        df = raw_df.copy()

        # If earnings dates are in index, reset index
        if 'Date' not in df.columns and 'Earnings Date' not in df.columns:
            df = df.reset_index()
            # The reset index column is usually index 0 ('index' or 'Earnings Date')
            first_col = df.columns[0]
            if first_col != 'Date':
                df = df.rename(columns={first_col: 'Date'})

        # Map possible column names
        col_map = {}
        for c in df.columns:
            c_str = str(c).strip().lower()
            if 'date' in c_str:
                col_map[c] = 'Date'
            elif c_str in ['reported eps', 'eps', 'reported_eps']:
                col_map[c] = 'EPS'
            elif 'surprise' in c_str:
                col_map[c] = 'Surprise_Percentage'

        df = df.rename(columns=col_map)

        if 'Date' not in df.columns:
            logger.warning(f"Could not identify Date column in earnings data for {ticker}")
            return pd.DataFrame(columns=['Date', 'EPS', 'Surprise_Percentage', 'Ticker'])

        # Ensure standard columns exist
        if 'EPS' not in df.columns:
            df['EPS'] = np.nan
        if 'Surprise_Percentage' not in df.columns:
            df['Surprise_Percentage'] = np.nan

        # Convert Date to timezone-naive datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if hasattr(df['Date'].dt, 'tz') and df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)

        # Filter out NaN/NaT dates
        df = df.dropna(subset=['Date'])

        # Filter out future earnings dates (must be <= today)
        now = pd.Timestamp.now()
        df = df[df['Date'] <= now]

        df['Ticker'] = ticker
        df = df.sort_values('Date').reset_index(drop=True)
        return df[['Date', 'EPS', 'Surprise_Percentage', 'Ticker']]
    
    def fetch_earnings_dates(self, ticker: str) -> pd.DataFrame:
        """
        Fetch earnings announcement dates for a stock with retries
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with earnings dates and information
        """
        try:
            logger.info(f"Fetching earnings dates for {ticker}...")
            
            def _get_earnings():
                stock = yf.Ticker(ticker)
                return stock.earnings_dates

            earnings_dates = retry_call(
                _get_earnings,
                max_retries=self.max_retries,
                description=f"earnings_dates({ticker})"
            )

            cleaned_df = self._clean_earnings_dataframe(earnings_dates, ticker)
            if not cleaned_df.empty:
                logger.info(f"Found {len(cleaned_df)} historical earnings dates for {ticker}")
                return cleaned_df
            else:
                logger.warning(f"No valid historical earnings dates found for {ticker}")
                return pd.DataFrame(columns=['Date', 'EPS', 'Surprise_Percentage', 'Ticker'])
                
        except Exception as e:
            logger.error(f"Error fetching earnings for {ticker}: {str(e)}")
            return pd.DataFrame(columns=['Date', 'EPS', 'Surprise_Percentage', 'Ticker'])
    
    def fetch_multiple_stocks(
        self,
        tickers: List[str],
        max_workers: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch price and earnings data for multiple stocks concurrently.
        
        Args:
            tickers: List of stock ticker symbols
            max_workers: Optional worker thread count override (defaults to self.max_workers)
            
        Returns:
            Tuple of (price_data_df, earnings_data_df)
        """
        if not tickers:
            return (
                pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker']),
                pd.DataFrame(columns=['Date', 'EPS', 'Surprise_Percentage', 'Ticker'])
            )

        workers = max_workers if max_workers is not None else self.max_workers
        workers = max(1, min(workers, len(tickers)))

        def _fetch_single(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
            price_data = self.fetch_stock_price_data(ticker)
            earnings_data = self.fetch_earnings_dates(ticker)
            return price_data, earnings_data

        if workers == 1 or len(tickers) == 1:
            results = [_fetch_single(t) for t in tickers]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(_fetch_single, tickers))

        all_price_data = [p for p, _ in results if not p.empty]
        all_earnings_data = [e for _, e in results if not e.empty]

        price_df = pd.concat(all_price_data, ignore_index=True) if all_price_data else pd.DataFrame(
            columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker']
        )
        earnings_df = pd.concat(all_earnings_data, ignore_index=True) if all_earnings_data else pd.DataFrame(
            columns=['Date', 'EPS', 'Surprise_Percentage', 'Ticker']
        )

        return price_df, earnings_df
    
    @staticmethod
    def validate_collected_data(
        price_df: pd.DataFrame,
        earnings_df: pd.DataFrame,
        min_price_records: int = 10,
        min_earnings_records: int = 1
    ) -> None:
        """
        Validate data shapes and required columns immediately after collection.

        Args:
            price_df: Historical stock prices DataFrame
            earnings_df: Earnings dates and surprise data DataFrame
            min_price_records: Minimum price records expected
            min_earnings_records: Minimum earnings records expected

        Raises:
            ValueError: If either DataFrame is empty, below threshold, or missing required columns.
        """
        if price_df is None or price_df.empty:
            raise ValueError("Price data collection failed: DataFrame is empty or None.")

        if len(price_df) < min_price_records:
            raise ValueError(
                f"Insufficient price data collected: got {len(price_df)} rows, "
                f"expected at least {min_price_records}."
            )

        required_price_cols = {'Date', 'Ticker', 'Adj Close', 'Volume'}
        missing_price = required_price_cols - set(price_df.columns)
        if missing_price:
            raise ValueError(f"Price DataFrame is missing required columns: {missing_price}")

        if earnings_df is None or earnings_df.empty:
            raise ValueError("Earnings data collection failed: DataFrame is empty or None.")

        if len(earnings_df) < min_earnings_records:
            raise ValueError(
                f"Insufficient earnings data collected: got {len(earnings_df)} rows, "
                f"expected at least {min_earnings_records}."
            )

        required_earnings_cols = {'Date', 'Ticker'}
        missing_earnings = required_earnings_cols - set(earnings_df.columns)
        if missing_earnings:
            raise ValueError(f"Earnings DataFrame is missing required columns: {missing_earnings}")

        # Check for ticker overlap
        price_tickers = set(price_df['Ticker'].unique())
        earnings_tickers = set(earnings_df['Ticker'].unique())
        overlap = price_tickers & earnings_tickers
        if not overlap:
            raise ValueError(
                f"No ticker overlap between price data ({price_tickers}) and "
                f"earnings data ({earnings_tickers})."
            )

        logger.info(
            f"Data validation passed: {len(price_df)} price rows, "
            f"{len(earnings_df)} earnings records across {len(overlap)} matching tickers."
        )

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


validate_collected_data = DataCollector.validate_collected_data


