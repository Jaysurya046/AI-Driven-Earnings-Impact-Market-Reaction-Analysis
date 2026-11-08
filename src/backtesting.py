"""
Backtesting Framework
Tests trading strategies based on earnings signals
"""

import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """Backtests trading strategies based on earnings announcements with realistic execution modeling"""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_bps: float = 5.0,
        next_day_execution: bool = True
    ):
        """
        Initialize backtester with transaction costs and execution delay modeling.
        
        Args:
            initial_capital: Starting capital for backtesting
            commission_rate: Broker commission fraction per side (default: 0.1% = 0.001)
            slippage_bps: Market impact / slippage in basis points per side (default: 5 bps)
            next_day_execution: Whether trades execute next business day after announcement
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.next_day_execution = next_day_execution
        
        self.capital = initial_capital
        self.positions: Dict[str, Any] = {}
        self.trades: List[Dict[str, Any]] = []
        self.portfolio_values: List[Dict[str, Any]] = []
        self.results: Optional[Dict[str, Any]] = None
    
    def reset(self):
        """Reset backtester state"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.results = None
    
    def _run_signal_strategy(
        self,
        event_analysis: pd.DataFrame,
        signal_fn: Callable[[float], int],
        strategy_name: str,
        holding_period: int = 5,
        position_size: float = 0.1,
        invert_return: bool = False
    ) -> Dict:
        """
        Shared execution engine for signal-based earnings trading strategies.

        Args:
            event_analysis: DataFrame with earnings events
            signal_fn: Callable mapping surprise percentage to trade signal (-1, 0, 1)
            strategy_name: Descriptive name of the strategy for reporting
            holding_period: Days to hold position
            position_size: Fraction of capital per trade
            invert_return: Whether to invert return relative to signal (used in mean-reversion)

        Returns:
            Dictionary with complete backtest results and performance metrics
        """
        logger.info(f"Running {strategy_name.lower()} strategy (holding={holding_period} days, size={position_size})")

        self.reset()
        data = event_analysis.copy()
        data = data.sort_values(['Ticker', 'EarningsDate'])

        portfolio_value = [self.initial_capital]
        dates = []
        trades = []

        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].copy()

            for idx, row in ticker_data.iterrows():
                surprise = row['SurprisePercentage']
                earnings_date = row['EarningsDate']

                signal = signal_fn(surprise)

                if signal != 0:
                    invest_amount = self.capital * position_size

                    if 'PostEventReturn' in row:
                        daily_return = row['PostEventReturn'] / 5.0
                        total_return = daily_return * min(holding_period, 5)
                    else:
                        total_return = 0.0

                    dt_earnings = pd.to_datetime(earnings_date)
                    if self.next_day_execution:
                        entry_date = dt_earnings + pd.tseries.offsets.BDay(1)
                        exit_date = entry_date + pd.tseries.offsets.BDay(holding_period)
                        timing_str = 'next_day_open'
                    else:
                        entry_date = dt_earnings
                        exit_date = dt_earnings + pd.Timedelta(days=holding_period)
                        timing_str = 'same_day_close'

                    if invert_return:
                        gross_return_pct = -total_return * np.sign(signal)
                    else:
                        gross_return_pct = total_return * signal

                    gross_profit_loss = invest_amount * (gross_return_pct / 100.0)

                    slippage_cost = invest_amount * (2.0 * self.slippage_bps / 10000.0)
                    commission_cost = invest_amount * (2.0 * self.commission_rate)
                    total_costs = slippage_cost + commission_cost

                    profit_loss = gross_profit_loss - total_costs
                    net_return_pct = (profit_loss / invest_amount) * 100.0 if invest_amount > 0 else 0.0

                    trade = {
                        'Ticker': ticker,
                        'EarningsDate': earnings_date,
                        'EntryDate': entry_date,
                        'ExitDate': exit_date,
                        'ExecutionTiming': timing_str,
                        'Signal': 'LONG' if signal > 0 else 'SHORT',
                        'Strategy': strategy_name,
                        'Surprise': surprise,
                        'Invested': invest_amount,
                        'GrossReturn': gross_return_pct,
                        'NetReturn': net_return_pct,
                        'Return': net_return_pct,
                        'GrossProfitLoss': gross_profit_loss,
                        'SlippageCost': slippage_cost,
                        'CommissionCost': commission_cost,
                        'ProfitLoss': profit_loss
                    }
                    trades.append(trade)

                    self.capital += profit_loss
                    portfolio_value.append(self.capital)
                    dates.append(entry_date)

        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df = trades_df.sort_values('EntryDate')
        else:
            trades_df = pd.DataFrame()

        portfolio_array = np.array(portfolio_value)
        if len(portfolio_array) > 1:
            returns = np.diff(portfolio_array) / portfolio_array[:-1]

            total_return = (portfolio_array[-1] / portfolio_array[0]) - 1
            n_periods = len(returns)
            if n_periods > 0:
                annualized_return = (1 + total_return) ** (252 / max(n_periods, 1)) - 1
                sharpe_ratio = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10) if returns.std() > 0 else 0
            else:
                annualized_return = 0
                sharpe_ratio = 0

            cumulative = np.maximum.accumulate(portfolio_array)
            drawdowns = (cumulative - portfolio_array) / cumulative
            max_drawdown = drawdowns.max()

            if len(trades_df) > 0:
                winning_trades = (trades_df['ProfitLoss'] > 0).sum()
                win_rate = winning_trades / len(trades_df)
            else:
                win_rate = 0
        else:
            total_return = 0
            annualized_return = 0
            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0

        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'winning_trades': int((trades_df['ProfitLoss'] > 0).sum()) if len(trades_df) > 0 else 0,
            'losing_trades': int((trades_df['ProfitLoss'] < 0).sum()) if len(trades_df) > 0 else 0,
            'avg_trade_return': trades_df['Return'].mean() if len(trades_df) > 0 else 0,
            'total_slippage_cost': float(trades_df['SlippageCost'].sum()) if len(trades_df) > 0 else 0.0,
            'total_commission_cost': float(trades_df['CommissionCost'].sum()) if len(trades_df) > 0 else 0.0,
            'gross_profit_loss': float(trades_df['GrossProfitLoss'].sum()) if len(trades_df) > 0 else 0.0,
            'net_profit_loss': float(trades_df['ProfitLoss'].sum()) if len(trades_df) > 0 else 0.0,
            'commission_rate': self.commission_rate,
            'slippage_bps': self.slippage_bps,
            'next_day_execution': self.next_day_execution,
            'portfolio_value': portfolio_value,
            'dates': dates if dates else [datetime.now()],
            'trades': trades
        }

        self.results = results
        logger.info(f"Backtest complete: {len(trades)} trades, Total Return: {total_return:.2%}")
        return results

    def momentum_strategy(
        self,
        event_analysis: pd.DataFrame,
        threshold: float = 2.0,
        holding_period: int = 5,
        position_size: float = 0.1
    ) -> Dict:
        """
        Momentum strategy: Buy on positive surprise, sell on negative
        
        Args:
            event_analysis: DataFrame with earnings events
            threshold: Minimum surprise percentage to trigger trade
            holding_period: Days to hold position
            position_size: Fraction of capital per trade
            
        Returns:
            Dictionary with backtest results
        """
        def _signal(surprise: float) -> int:
            if surprise >= threshold:
                return 1
            elif surprise <= -threshold:
                return -1
            return 0

        return self._run_signal_strategy(
            event_analysis=event_analysis,
            signal_fn=_signal,
            strategy_name="Momentum",
            holding_period=holding_period,
            position_size=position_size,
            invert_return=False
        )

    def mean_reversion_strategy(
        self,
        event_analysis: pd.DataFrame,
        threshold: float = 3.0,
        holding_period: int = 3,
        position_size: float = 0.1
    ) -> Dict:
        """
        Mean reversion strategy: Bet against extreme surprises with transaction costs and next-day timing
        
        Args:
            event_analysis: DataFrame with earnings events
            threshold: Extreme surprise threshold
            holding_period: Days to hold position
            position_size: Fraction of capital per trade
            
        Returns:
            Dictionary with backtest results
        """
        def _signal(surprise: float) -> int:
            if surprise >= threshold:
                return -1
            elif surprise <= -threshold:
                return 1
            return 0

        return self._run_signal_strategy(
            event_analysis=event_analysis,
            signal_fn=_signal,
            strategy_name="Mean Reversion",
            holding_period=holding_period,
            position_size=position_size,
            invert_return=True
        )
    
    def run_strategy(
        self,
        event_analysis: pd.DataFrame,
        strategy: str = 'momentum',
        **kwargs
    ) -> Dict:
        """
        Run specified backtesting strategy
        
        Args:
            event_analysis: DataFrame with earnings events
            strategy: Strategy name ('momentum' or 'mean_reversion')
            **kwargs: Additional strategy parameters
            
        Returns:
            Dictionary with backtest results
        """
        if strategy == 'momentum':
            return self.momentum_strategy(event_analysis, **kwargs)
        elif strategy == 'mean_reversion':
            return self.mean_reversion_strategy(event_analysis, **kwargs)
        else:
            logger.error(f"Unknown strategy: {strategy}")
            return {}
    
    def compare_strategies(
        self,
        event_analysis: pd.DataFrame,
        strategies: List[str] = ['momentum', 'mean_reversion']
    ) -> pd.DataFrame:
        """
        Compare multiple strategies
        
        Args:
            event_analysis: DataFrame with earnings events
            strategies: List of strategy names to compare
            
        Returns:
            DataFrame with strategy comparison
        """
        results_list = []
        
        for strategy in strategies:
            result = self.run_strategy(event_analysis, strategy=strategy)
            if result:
                result['Strategy'] = strategy
                results_list.append(result)
        
        if not results_list:
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame([
            {
                'Strategy': r['Strategy'],
                'Total Return': r['total_return'],
                'Annualized Return': r['annualized_return'],
                'Sharpe Ratio': r['sharpe_ratio'],
                'Max Drawdown': r['max_drawdown'],
                'Win Rate': r['win_rate'],
                'Total Trades': r['total_trades'],
                'Winning Trades': r['winning_trades'],
                'Avg Trade Return': r['avg_trade_return']
            }
            for r in results_list
        ])
        
        return comparison_df.round(4)
    
    def plot_portfolio_evolution(self, save_path: Optional[str] = None):
        """
        Plot portfolio value over time
        
        Args:
            save_path: Optional path to save the figure
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.results is None:
                logger.error("Run a strategy first")
                return
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.results['dates'], self.results['portfolio_value'], linewidth=2)
            ax.set_title('Portfolio Value Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value ($)')
            ax.grid(True, alpha=0.3)
            
            # Add horizontal line for initial capital
            ax.axhline(y=self.initial_capital, color='gray', linestyle='--', 
                      label=f'Initial Capital: ${self.initial_capital:,.0f}')
            ax.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting portfolio evolution: {str(e)}")


if __name__ == "__main__":
    print("Backtesting Framework Module")
    print("=" * 50)
    
    # Example usage would require actual event analysis data
    print("This module provides backtesting capabilities for earnings-based strategies.")
    print("Use with event_analysis DataFrame from the event_analysis module.")
