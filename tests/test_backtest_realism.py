"""Tests for realistic backtesting: slippage, commissions, and next-day execution timing."""
import pandas as pd
import pytest

from src.backtesting import Backtester


@pytest.fixture
def sample_events():
    return pd.DataFrame([
        {
            'Ticker': 'AAA',
            'EarningsDate': pd.Timestamp('2024-03-01'),  # Friday
            'SurprisePercentage': 5.0,                   # Triggers momentum LONG
            'PostEventReturn': 4.0                       # +4% over 5 days
        },
        {
            'Ticker': 'BBB',
            'EarningsDate': pd.Timestamp('2024-03-05'),  # Tuesday
            'SurprisePercentage': -6.0,                  # Triggers momentum SHORT
            'PostEventReturn': -3.0                      # -3% over 5 days
        }
    ])


def test_next_day_execution_timing(sample_events):
    backtester = Backtester(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_bps=5.0,
        next_day_execution=True
    )
    res = backtester.momentum_strategy(sample_events, threshold=2.0, holding_period=3)
    trades = res['trades']
    assert len(trades) == 2

    # Friday announcement executes on Monday (next business day)
    t0 = trades[0]
    assert t0['ExecutionTiming'] == 'next_day_open'
    assert t0['EntryDate'] > t0['EarningsDate']
    assert t0['EntryDate'] == pd.Timestamp('2024-03-04')  # Next business day after Friday


def test_slippage_and_commission_deductions(sample_events):
    # Zero costs baseline
    bt_zero = Backtester(
        initial_capital=100000.0,
        commission_rate=0.0,
        slippage_bps=0.0,
        next_day_execution=False
    )
    res_zero = bt_zero.momentum_strategy(sample_events, threshold=2.0)

    # Realistic costs
    bt_real = Backtester(
        initial_capital=100000.0,
        commission_rate=0.001,  # 10 bps
        slippage_bps=5.0,        # 5 bps
        next_day_execution=True
    )
    res_real = bt_real.momentum_strategy(sample_events, threshold=2.0)

    assert res_real['total_slippage_cost'] > 0
    assert res_real['total_commission_cost'] > 0
    # Net return with costs must be strictly lower than zero cost baseline
    assert res_real['final_capital'] < res_zero['final_capital']

    trade = res_real['trades'][0]
    assert trade['GrossProfitLoss'] > trade['ProfitLoss']
    assert trade['ProfitLoss'] == pytest.approx(
        trade['GrossProfitLoss'] - trade['SlippageCost'] - trade['CommissionCost'],
        abs=1e-6
    )


def test_mean_reversion_strategy_with_costs(sample_events):
    bt = Backtester(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_bps=5.0,
        next_day_execution=True
    )
    res = bt.mean_reversion_strategy(sample_events, threshold=3.0)
    assert res['total_slippage_cost'] > 0
    assert res['total_commission_cost'] > 0
    assert len(res['trades']) == 2
