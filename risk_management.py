"""
Risk Management Module

Features:
- Max Drawdown Tracking
- Position Sizing (Kelly Criterion & Fixed Fractional)
- Risk per Trade Calculator
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DrawdownMetrics:
    """Drawdown analysis results."""
    max_drawdown_pct: float
    max_drawdown_amount: float
    avg_drawdown_pct: float
    max_drawdown_duration_days: int
    avg_drawdown_duration_days: float
    recovery_time_days: int
    current_drawdown_pct: float
    drawdown_periods: int
    worst_drawdown_start: str
    worst_drawdown_end: str


@dataclass
class PositionSize:
    """Position sizing recommendation."""
    method: str
    shares: int
    position_value: float
    risk_amount: float
    risk_pct_of_capital: float
    stop_loss_price: float
    potential_loss: float
    potential_gain: float
    risk_reward_ratio: float
    kelly_fraction: float
    recommended_allocation_pct: float


def calculate_drawdown_series(equity_curve: List[float]) -> Tuple[List[float], List[float]]:
    """
    Calculate drawdown series from equity curve.

    Returns:
        (drawdown_pct, running_max)
    """
    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    return drawdown.tolist(), running_max.tolist()


def analyze_drawdowns(equity_curve: List[float], dates: List[str] = None) -> DrawdownMetrics:
    """
    Comprehensive drawdown analysis.

    Args:
        equity_curve: List of portfolio values over time
        dates: Optional list of date strings

    Returns:
        DrawdownMetrics with all analysis results
    """
    if not equity_curve or len(equity_curve) < 2:
        return DrawdownMetrics(
            max_drawdown_pct=0, max_drawdown_amount=0, avg_drawdown_pct=0,
            max_drawdown_duration_days=0, avg_drawdown_duration_days=0,
            recovery_time_days=0, current_drawdown_pct=0, drawdown_periods=0,
            worst_drawdown_start="N/A", worst_drawdown_end="N/A"
        )

    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown_pct = (equity - running_max) / running_max * 100
    drawdown_amount = equity - running_max

    # Max drawdown
    max_dd_idx = np.argmin(drawdown_pct)
    max_dd_pct = abs(drawdown_pct[max_dd_idx])
    max_dd_amount = abs(drawdown_amount[max_dd_idx])

    # Find drawdown periods
    in_drawdown = drawdown_pct < 0
    drawdown_periods = 0
    durations = []
    current_duration = 0

    for i, is_dd in enumerate(in_drawdown):
        if is_dd:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
                drawdown_periods += 1
            current_duration = 0

    if current_duration > 0:
        durations.append(current_duration)
        drawdown_periods += 1

    # Find worst drawdown period
    worst_start_idx = 0
    worst_end_idx = max_dd_idx

    # Look backwards from max drawdown to find the peak
    for i in range(max_dd_idx, -1, -1):
        if equity[i] == running_max[i]:
            worst_start_idx = i
            break

    # Find recovery (when equity exceeds previous peak)
    recovery_idx = len(equity) - 1
    peak_before_dd = running_max[max_dd_idx]
    for i in range(max_dd_idx, len(equity)):
        if equity[i] >= peak_before_dd:
            recovery_idx = i
            break

    recovery_time = recovery_idx - max_dd_idx
    max_dd_duration = max_dd_idx - worst_start_idx

    # Date formatting
    if dates and len(dates) > max(worst_start_idx, worst_end_idx):
        worst_start = dates[worst_start_idx]
        worst_end = dates[worst_end_idx]
    else:
        worst_start = f"Day {worst_start_idx}"
        worst_end = f"Day {worst_end_idx}"

    # Current drawdown
    current_dd = abs(drawdown_pct[-1]) if drawdown_pct[-1] < 0 else 0

    # Average drawdown (only during drawdown periods)
    dd_values = [abs(d) for d in drawdown_pct if d < 0]
    avg_dd = np.mean(dd_values) if dd_values else 0

    avg_duration = np.mean(durations) if durations else 0

    return DrawdownMetrics(
        max_drawdown_pct=round(max_dd_pct, 2),
        max_drawdown_amount=round(max_dd_amount, 2),
        avg_drawdown_pct=round(avg_dd, 2),
        max_drawdown_duration_days=max_dd_duration,
        avg_drawdown_duration_days=round(avg_duration, 1),
        recovery_time_days=recovery_time,
        current_drawdown_pct=round(current_dd, 2),
        drawdown_periods=drawdown_periods,
        worst_drawdown_start=worst_start,
        worst_drawdown_end=worst_end
    )


def calculate_kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate Kelly Criterion for optimal bet sizing.

    Kelly % = W - [(1-W) / R]
    Where:
        W = Win probability
        R = Win/Loss ratio

    Args:
        win_rate: Win probability (0-1)
        avg_win: Average win amount/percentage
        avg_loss: Average loss amount/percentage (positive number)

    Returns:
        Kelly fraction (0-1), capped at 0.25 for safety
    """
    if avg_loss <= 0 or win_rate <= 0:
        return 0

    win_loss_ratio = abs(avg_win) / abs(avg_loss)
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

    # Cap at 25% for safety (half-Kelly is common practice)
    kelly = max(0, min(kelly, 0.25))

    return round(kelly, 4)


def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_loss: float,
    target_price: float,
    risk_per_trade_pct: float = 2.0,
    win_rate: float = 0.65,
    avg_win_pct: float = 3.0,
    avg_loss_pct: float = 2.0,
    method: str = "fixed_fractional"
) -> PositionSize:
    """
    Calculate position size using various methods.

    Args:
        capital: Total trading capital
        entry_price: Stock entry price
        stop_loss: Stop loss price
        target_price: Target price
        risk_per_trade_pct: Max risk per trade (default 2%)
        win_rate: Historical win rate (for Kelly)
        avg_win_pct: Average win percentage (for Kelly)
        avg_loss_pct: Average loss percentage (for Kelly)
        method: "fixed_fractional", "kelly", or "half_kelly"

    Returns:
        PositionSize with all calculations
    """
    # Calculate risk per share
    risk_per_share = abs(entry_price - stop_loss)
    reward_per_share = abs(target_price - entry_price)

    if risk_per_share <= 0:
        risk_per_share = entry_price * 0.02  # Default 2% stop

    # Risk-Reward ratio
    rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

    # Kelly calculation
    kelly = calculate_kelly_criterion(win_rate, avg_win_pct, avg_loss_pct)

    # Calculate max risk amount (what we're willing to lose)
    risk_amount = capital * (risk_per_trade_pct / 100)

    # Calculate shares based on risk amount and risk per share
    shares = int(risk_amount / risk_per_share)

    # For Kelly methods, also check allocation constraint
    if method == "kelly":
        allocation_pct = kelly * 100
        max_position_value = capital * (allocation_pct / 100)
        max_shares_by_allocation = int(max_position_value / entry_price)
        shares = min(shares, max_shares_by_allocation)
    elif method == "half_kelly":
        allocation_pct = (kelly / 2) * 100
        max_position_value = capital * (allocation_pct / 100)
        max_shares_by_allocation = int(max_position_value / entry_price)
        shares = min(shares, max_shares_by_allocation)
    else:  # fixed_fractional (default) - no allocation limit, only risk limit
        allocation_pct = (shares * entry_price / capital) * 100

    shares = max(shares, 0)

    position_value = shares * entry_price
    potential_loss = shares * risk_per_share
    potential_gain = shares * reward_per_share

    return PositionSize(
        method=method,
        shares=shares,
        position_value=round(position_value, 2),
        risk_amount=round(potential_loss, 2),
        risk_pct_of_capital=round((potential_loss / capital) * 100, 2) if capital > 0 else 0,
        stop_loss_price=round(stop_loss, 2),
        potential_loss=round(potential_loss, 2),
        potential_gain=round(potential_gain, 2),
        risk_reward_ratio=round(rr_ratio, 2),
        kelly_fraction=round(kelly, 4),
        recommended_allocation_pct=round(allocation_pct, 2)
    )


def calculate_portfolio_risk(
    positions: List[Dict],
    capital: float,
    correlation_factor: float = 0.5
) -> Dict:
    """
    Calculate total portfolio risk.

    Args:
        positions: List of position dicts with 'risk_amount'
        capital: Total capital
        correlation_factor: Assumed correlation between positions (0-1)

    Returns:
        Portfolio risk metrics
    """
    if not positions:
        return {
            'total_risk': 0,
            'total_risk_pct': 0,
            'num_positions': 0,
            'avg_risk_per_position': 0,
            'diversified_risk': 0
        }

    risks = [p.get('risk_amount', 0) for p in positions]
    total_risk = sum(risks)
    num_positions = len(positions)

    # Diversified risk (simplified - assumes equal correlation)
    # sqrt(n) * avg_risk * correlation + (1-correlation) * total_risk / n
    avg_risk = total_risk / num_positions if num_positions > 0 else 0
    diversified_risk = np.sqrt(num_positions) * avg_risk * correlation_factor + \
                       (1 - correlation_factor) * avg_risk

    return {
        'total_risk': round(total_risk, 2),
        'total_risk_pct': round((total_risk / capital) * 100, 2) if capital > 0 else 0,
        'num_positions': num_positions,
        'avg_risk_per_position': round(avg_risk, 2),
        'diversified_risk': round(diversified_risk * num_positions, 2),
        'max_recommended_positions': int(10 / 2)  # 10% max portfolio risk / 2% per trade
    }


# Strategy-specific defaults based on walk-forward results
STRATEGY_DEFAULTS = {
    'win_rate': 0.826,  # 82.6% from walk-forward
    'avg_win_pct': 2.5,  # Approximate from backtest
    'avg_loss_pct': 1.8,  # Approximate from backtest
    'max_risk_per_trade': 2.0,  # Conservative 2%
    'max_portfolio_risk': 10.0,  # Max 10% total portfolio at risk
    'max_positions': 5,  # Max concurrent positions
}


def get_recommended_position_size(
    capital: float,
    entry_price: float,
    stop_loss: float,
    target_price: float
) -> PositionSize:
    """
    Get recommended position size using strategy defaults.

    This is the main function to use for quick position sizing.
    """
    return calculate_position_size(
        capital=capital,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        risk_per_trade_pct=STRATEGY_DEFAULTS['max_risk_per_trade'],
        win_rate=STRATEGY_DEFAULTS['win_rate'],
        avg_win_pct=STRATEGY_DEFAULTS['avg_win_pct'],
        avg_loss_pct=STRATEGY_DEFAULTS['avg_loss_pct'],
        method="fixed_fractional"
    )


if __name__ == "__main__":
    # Test drawdown analysis
    print("=" * 60)
    print("Testing Drawdown Analysis")
    print("=" * 60)

    # Simulated equity curve
    equity = [100000, 102000, 105000, 103000, 98000, 95000, 97000, 100000, 104000, 108000]
    dates = [f"2024-01-{i:02d}" for i in range(1, 11)]

    dd = analyze_drawdowns(equity, dates)
    print(f"Max Drawdown: {dd.max_drawdown_pct}%")
    print(f"Max DD Duration: {dd.max_drawdown_duration_days} days")
    print(f"Recovery Time: {dd.recovery_time_days} days")
    print(f"Worst Period: {dd.worst_drawdown_start} to {dd.worst_drawdown_end}")

    # Test position sizing
    print("\n" + "=" * 60)
    print("Testing Position Sizing")
    print("=" * 60)

    pos = get_recommended_position_size(
        capital=500000,
        entry_price=1500,
        stop_loss=1450,
        target_price=1600
    )

    print(f"Capital: ₹5,00,000")
    print(f"Entry: ₹1,500 | Stop: ₹1,450 | Target: ₹1,600")
    print(f"\nRecommendation:")
    print(f"  Shares: {pos.shares}")
    print(f"  Position Value: ₹{pos.position_value:,.2f}")
    print(f"  Risk Amount: ₹{pos.risk_amount:,.2f} ({pos.risk_pct_of_capital}%)")
    print(f"  Potential Gain: ₹{pos.potential_gain:,.2f}")
    print(f"  Risk:Reward: 1:{pos.risk_reward_ratio}")
    print(f"  Kelly Fraction: {pos.kelly_fraction}")
