"""
Sector Comparison Module
Analyzes and compares earnings impact across different market sectors
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SectorAnalyzer:
    """Analyzes earnings impact patterns across different sectors"""
    
    # Standard sector classification for common tickers
    SECTOR_MAPPING = {
        # Technology
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'GOOGL': 'Technology',
        'META': 'Technology',
        'NVDA': 'Technology',
        'ORCL': 'Technology',
        'CRM': 'Technology',
        'AMD': 'Technology',
        'INTC': 'Technology',
        'IBM': 'Technology',
        
        # Consumer Discretionary
        'AMZN': 'Consumer Discretionary',
        'TSLA': 'Consumer Discretionary',
        'HD': 'Consumer Discretionary',
        'NKE': 'Consumer Discretionary',
        'MCD': 'Consumer Discretionary',
        'SBUX': 'Consumer Discretionary',
        
        # Healthcare
        'JNJ': 'Healthcare',
        'UNH': 'Healthcare',
        'PFE': 'Healthcare',
        'MRK': 'Healthcare',
        'ABBV': 'Healthcare',
        'LLY': 'Healthcare',
        'TMO': 'Healthcare',
        'ABT': 'Healthcare',
        
        # Financials
        'JPM': 'Financials',
        'BAC': 'Financials',
        'WFC': 'Financials',
        'GS': 'Financials',
        'MS': 'Financials',
        'C': 'Financials',
        'BLK': 'Financials',
        'AXP': 'Financials',
        
        # Communication Services
        'DIS': 'Communication Services',
        'NFLX': 'Communication Services',
        'CMCSA': 'Communication Services',
        'VZ': 'Communication Services',
        'T': 'Communication Services',
        
        # Industrials
        'CAT': 'Industrials',
        'BA': 'Industrials',
        'UPS': 'Industrials',
        'HON': 'Industrials',
        'GE': 'Industrials',
        'MMM': 'Industrials',
        
        # Consumer Staples
        'PG': 'Consumer Staples',
        'KO': 'Consumer Staples',
        'PEP': 'Consumer Staples',
        'WMT': 'Consumer Staples',
        'COST': 'Consumer Staples',
        'PM': 'Consumer Staples',
        
        # Energy
        'XOM': 'Energy',
        'CVX': 'Energy',
        'COP': 'Energy',
        'SLB': 'Energy',
        'EOG': 'Energy',
        
        # Utilities
        'NEE': 'Utilities',
        'DUK': 'Utilities',
        'SO': 'Utilities',
        'D': 'Utilities',
        'AEP': 'Utilities',
        
        # Real Estate
        'AMT': 'Real Estate',
        'PLD': 'Real Estate',
        'CCI': 'Real Estate',
        'EQIX': 'Real Estate',
        'SPG': 'Real Estate',
        
        # Materials
        'LIN': 'Materials',
        'APD': 'Materials',
        'SHW': 'Materials',
        'FCX': 'Materials',
        'NEM': 'Materials',
    }
    
    def __init__(self):
        self.sector_data = None
        self.sector_metrics = None
    
    def assign_sector(self, ticker: str) -> str:
        """
        Assign sector to a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Sector name
        """
        return self.SECTOR_MAPPING.get(ticker.upper(), 'Other')
    
    def analyze_by_sector(self, event_analysis: pd.DataFrame) -> Dict:
        """
        Analyze earnings impact metrics by sector
        
        Args:
            event_analysis: DataFrame with event analysis results
            
        Returns:
            Dictionary with sector analysis results
        """
        logger.info("Performing sector comparison analysis...")
        
        # Add sector column
        data = event_analysis.copy()
        data['Sector'] = data['Ticker'].apply(self.assign_sector)
        
        self.sector_data = data
        
        # Calculate sector-level metrics
        sector_summary = data.groupby('Sector').agg({
            'PostEventReturn': ['mean', 'std', 'count'],
            'PreEventReturn': 'mean',
            'AbnormalReturnCAR': 'mean',
            'SurprisePercentage': ['mean', 'std'],
            'VolumeSpikeRatio': 'mean',
            'PreEventVolatility': 'mean',
            'PostEventVolatility': 'mean'
        }).round(4)
        
        # Flatten multi-level columns
        sector_summary.columns = [
            'AvgReturn', 'StdReturn', 'EventCount',
            'AvgPreEventReturn', 'AvgCAR',
            'AvgSurprise', 'StdSurprise',
            'AvgVolumeSpike', 'AvgPreVol', 'AvgPostVol'
        ]
        sector_summary = sector_summary.reset_index()
        
        # Calculate additional metrics
        sector_summary['VolatilityChange'] = (
            sector_summary['AvgPostVol'] - sector_summary['AvgPreVol']
        )
        sector_summary['WinRate'] = data.groupby('Sector', observed=False).apply(
            lambda x: (x['PostEventReturn'] > 0).sum() / len(x)
        ).values
        sector_summary['AvgSurpriseMagnitude'] = (
            data.assign(AbsSurprise=data['SurprisePercentage'].abs())
            .groupby('Sector', observed=False)['AbsSurprise']
            .mean()
            .values
        )
        
        self.sector_metrics = sector_summary
        
        # Prepare visualization data
        sector_returns = sector_summary[['Sector', 'AvgReturn']].sort_values('AvgReturn', ascending=False)
        sector_volatility = sector_summary[['Sector', 'AvgPostVol']].sort_values('AvgPostVol', ascending=False)
        
        results = {
            'sector_summary': sector_summary,
            'sector_returns': sector_returns,
            'sector_volatility': sector_volatility,
            'raw_data': data
        }
        
        logger.info(f"Analyzed {len(sector_summary)} sectors")
        logger.info(f"Sectors found: {', '.join(sector_summary['Sector'])}")
        
        return results
    
    def compare_sectors(self, sector1: str, sector2: str) -> Dict:
        """
        Compare two specific sectors
        
        Args:
            sector1: First sector name
            sector2: Second sector name
            
        Returns:
            Dictionary with comparison metrics
        """
        if self.sector_data is None:
            logger.error("Run analyze_by_sector first")
            return {}
        
        data = self.sector_data
        sector1_data = data[data['Sector'] == sector1]
        sector2_data = data[data['Sector'] == sector2]
        
        if len(sector1_data) == 0 or len(sector2_data) == 0:
            logger.error(f"One or both sectors not found: {sector1}, {sector2}")
            return {}
        
        comparison = {
            'sector1': {
                'name': sector1,
                'avg_return': sector1_data['PostEventReturn'].mean(),
                'std_return': sector1_data['PostEventReturn'].std(),
                'avg_surprise': sector1_data['SurprisePercentage'].mean(),
                'avg_volume_spike': sector1_data['VolumeSpikeRatio'].mean(),
                'win_rate': (sector1_data['PostEventReturn'] > 0).sum() / len(sector1_data),
                'event_count': len(sector1_data)
            },
            'sector2': {
                'name': sector2,
                'avg_return': sector2_data['PostEventReturn'].mean(),
                'std_return': sector2_data['PostEventReturn'].std(),
                'avg_surprise': sector2_data['SurprisePercentage'].mean(),
                'avg_volume_spike': sector2_data['VolumeSpikeRatio'].mean(),
                'win_rate': (sector2_data['PostEventReturn'] > 0).sum() / len(sector2_data),
                'event_count': len(sector2_data)
            }
        }
        
        # Statistical tests
        from scipy import stats
        
        # T-test for returns
        t_stat, p_value = stats.ttest_ind(
            sector1_data['PostEventReturn'],
            sector2_data['PostEventReturn'],
            equal_var=False
        )
        comparison['statistical_test'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_at_5pct': p_value < 0.05
        }
        
        return comparison
    
    def get_sector_rankings(self, metric: str = 'AvgReturn') -> pd.DataFrame:
        """
        Get sector rankings by a specific metric
        
        Args:
            metric: Metric to rank by (e.g., 'AvgReturn', 'WinRate', 'AvgSurprise')
            
        Returns:
            DataFrame with ranked sectors
        """
        if self.sector_metrics is None:
            logger.error("Run analyze_by_sector first")
            return pd.DataFrame()
        
        rankings = self.sector_metrics.sort_values(metric, ascending=False)
        rankings['Rank'] = range(1, len(rankings) + 1)
        
        return rankings[['Rank', 'Sector', metric]]
    
    def plot_sector_heatmap(self, save_path: Optional[str] = None):
        """
        Create heatmap of sector metrics
        
        Args:
            save_path: Optional path to save the figure
        """
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            
            if self.sector_metrics is None:
                logger.error("Run analyze_by_sector first")
                return
            
            # Select key metrics for heatmap
            metrics_to_show = [
                'AvgReturn', 'WinRate', 'AvgSurpriseMagnitude',
                'AvgVolumeSpike', 'VolatilityChange'
            ]
            
            heatmap_data = self.sector_metrics.set_index('Sector')[metrics_to_show]
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0)
            plt.title('Sector Performance Heatmap')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Heatmap saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")


def get_ticker_sector(ticker: str) -> str:
    """
    Utility function to get sector for a single ticker
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Sector name
    """
    analyzer = SectorAnalyzer()
    return analyzer.assign_sector(ticker)


if __name__ == "__main__":
    # Example usage
    print("Sector Analyzer Module")
    print("=" * 50)
    
    # Test sector assignment
    test_tickers = ['AAPL', 'JPM', 'XOM', 'UNKNOWN']
    for ticker in test_tickers:
        sector = get_ticker_sector(ticker)
        print(f"{ticker}: {sector}")
