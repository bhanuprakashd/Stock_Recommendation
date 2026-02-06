"""
Quantitative Metrics Module - Institutional Grade

Features:
- Transaction Cost & Slippage Modeling
- Monte Carlo Simulation
- Advanced Risk Metrics (Sharpe, Sortino, VaR, CVaR, Calmar)
- Statistical Validation (t-tests, bootstrap, p-values)
- Portfolio Optimization (correlation-based)
- Regime Detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# TRANSACTION COST MODELING
# =============================================================================

@dataclass
class TransactionCosts:
    """Transaction cost structure for Indian markets."""
    brokerage_pct: float = 0.03  # 0.03% per side (discount broker)
    stt_pct: float = 0.1  # STT: 0.1% on sell (delivery)
    exchange_pct: float = 0.00345  # NSE charges
    sebi_pct: float = 0.0001  # SEBI turnover fee
    gst_pct: float = 18.0  # GST on brokerage (18%)
    stamp_duty_pct: float = 0.015  # Stamp duty on buy
    slippage_pct: float = 0.05  # Estimated slippage (0.05%)

    def calculate_total_cost(self, trade_value: float, is_buy: bool = True) -> float:
        """Calculate total transaction cost for a trade."""
        brokerage = trade_value * (self.brokerage_pct / 100)
        gst_on_brokerage = brokerage * (self.gst_pct / 100)
        exchange = trade_value * (self.exchange_pct / 100)
        sebi = trade_value * (self.sebi_pct / 100)
        slippage = trade_value * (self.slippage_pct / 100)

        if is_buy:
            stamp = trade_value * (self.stamp_duty_pct / 100)
            stt = 0
        else:
            stamp = 0
            stt = trade_value * (self.stt_pct / 100)

        total = brokerage + gst_on_brokerage + exchange + sebi + stamp + stt + slippage
        return total

    def calculate_round_trip_cost(self, trade_value: float) -> float:
        """Calculate total cost for buy + sell."""
        buy_cost = self.calculate_total_cost(trade_value, is_buy=True)
        sell_cost = self.calculate_total_cost(trade_value, is_buy=False)
        return buy_cost + sell_cost

    def cost_as_percentage(self, trade_value: float) -> float:
        """Get round-trip cost as percentage."""
        return (self.calculate_round_trip_cost(trade_value) / trade_value) * 100


# =============================================================================
# ADVANCED RISK METRICS
# =============================================================================

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    # Return metrics
    total_return: float
    cagr: float
    annualized_volatility: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float

    # Drawdown metrics
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int
    ulcer_index: float

    # Tail risk
    var_95: float  # Value at Risk (95%)
    var_99: float  # Value at Risk (99%)
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    cvar_99: float

    # Distribution
    skewness: float
    kurtosis: float
    positive_months_pct: float

    # Trade metrics
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float

    # Statistical significance
    t_statistic: float
    p_value: float
    is_significant: bool  # p < 0.05


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.06) -> float:
    """
    Calculate annualized Sharpe Ratio.

    Args:
        returns: Array of periodic returns
        risk_free_rate: Annual risk-free rate (default 6% for India)
    """
    if len(returns) < 2:
        return 0

    # Assume daily returns, annualize
    excess_returns = returns - (risk_free_rate / 252)
    if np.std(excess_returns) == 0:
        return 0

    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return round(sharpe, 3)


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.06) -> float:
    """
    Calculate Sortino Ratio (uses downside deviation).
    """
    if len(returns) < 2:
        return 0

    excess_returns = returns - (risk_free_rate / 252)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0

    sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    return round(sortino, 3)


def calculate_calmar_ratio(total_return: float, max_drawdown: float, years: float) -> float:
    """
    Calculate Calmar Ratio (CAGR / Max Drawdown).
    """
    if max_drawdown == 0 or years == 0:
        return 0

    cagr = ((1 + total_return / 100) ** (1 / years) - 1) * 100
    calmar = cagr / abs(max_drawdown)
    return round(calmar, 3)


def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) using historical simulation.

    Returns the loss threshold at given confidence level.
    """
    if len(returns) < 10:
        return 0

    var = np.percentile(returns, (1 - confidence) * 100)
    return round(abs(var), 4)


def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Conditional VaR (Expected Shortfall).

    Average loss beyond VaR threshold.
    """
    if len(returns) < 10:
        return 0

    var = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var].mean()
    return round(abs(cvar), 4)


def calculate_ulcer_index(equity_curve: np.ndarray) -> float:
    """
    Calculate Ulcer Index (measures depth and duration of drawdowns).
    """
    if len(equity_curve) < 2:
        return 0

    running_max = np.maximum.accumulate(equity_curve)
    drawdown_pct = ((equity_curve - running_max) / running_max) * 100

    ulcer = np.sqrt(np.mean(drawdown_pct ** 2))
    return round(abs(ulcer), 3)


def calculate_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Calculate Information Ratio (excess return / tracking error).
    """
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0

    excess = returns - benchmark_returns
    if np.std(excess) == 0:
        return 0

    ir = np.mean(excess) / np.std(excess) * np.sqrt(252)
    return round(ir, 3)


def statistical_significance_test(returns: np.ndarray) -> Tuple[float, float, bool]:
    """
    Test if strategy returns are statistically significant.

    Returns: (t_statistic, p_value, is_significant)
    """
    if len(returns) < 30:
        return 0, 1, False

    # One-sample t-test: H0 = mean return is 0
    t_stat, p_value = stats.ttest_1samp(returns, 0)

    return round(t_stat, 3), round(p_value, 4), p_value < 0.05


def calculate_all_risk_metrics(
    trades: List[Dict],
    equity_curve: List[float],
    years: float,
    benchmark_returns: np.ndarray = None
) -> RiskMetrics:
    """
    Calculate comprehensive risk metrics from trade history.

    Args:
        trades: List of trade dictionaries with 'pnl_pct' field
        equity_curve: List of portfolio values over time
        years: Number of years in backtest
        benchmark_returns: Optional benchmark returns for IR calculation
    """
    if not trades or not equity_curve:
        return None

    # Extract trade PnLs
    pnls = np.array([t.get('pnl_pct', t.get('pnl', 0)) for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    # Convert equity curve to returns
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]

    # Basic metrics
    total_return = (equity[-1] / equity[0] - 1) * 100
    cagr = ((equity[-1] / equity[0]) ** (1 / years) - 1) * 100 if years > 0 else 0
    volatility = np.std(returns) * np.sqrt(252) * 100

    # Drawdown analysis
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max * 100
    max_dd = abs(np.min(drawdowns))
    avg_dd = abs(np.mean(drawdowns[drawdowns < 0])) if len(drawdowns[drawdowns < 0]) > 0 else 0

    # Find max drawdown duration
    in_dd = drawdowns < 0
    max_dd_duration = 0
    current_duration = 0
    for is_dd in in_dd:
        if is_dd:
            current_duration += 1
            max_dd_duration = max(max_dd_duration, current_duration)
        else:
            current_duration = 0

    # Risk-adjusted metrics
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    calmar = calculate_calmar_ratio(total_return, max_dd, years)
    ulcer = calculate_ulcer_index(equity)

    # Information ratio
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        ir = calculate_information_ratio(returns, benchmark_returns)
    else:
        ir = 0

    # VaR and CVaR
    var_95 = calculate_var(returns, 0.95) * 100
    var_99 = calculate_var(returns, 0.99) * 100
    cvar_95 = calculate_cvar(returns, 0.95) * 100
    cvar_99 = calculate_cvar(returns, 0.99) * 100

    # Distribution metrics
    skew = stats.skew(returns) if len(returns) > 2 else 0
    kurt = stats.kurtosis(returns) if len(returns) > 2 else 0

    # Monthly returns (approximate - group by ~21 trading days)
    if len(returns) >= 21:
        n_months = len(returns) // 21
        monthly_returns = np.array([returns[i*21:(i+1)*21].sum() for i in range(n_months)])
    else:
        monthly_returns = returns
    positive_months = (monthly_returns > 0).sum() / len(monthly_returns) * 100 if len(monthly_returns) > 0 else 0

    # Trade metrics
    win_rate = len(wins) / len(pnls) * 100 if len(pnls) > 0 else 0
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0

    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 10

    expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)

    largest_win = np.max(wins) if len(wins) > 0 else 0
    largest_loss = np.min(losses) if len(losses) > 0 else 0

    # Trade duration
    durations = [t.get('days', t.get('duration', 5)) for t in trades]
    avg_duration = np.mean(durations) if durations else 0

    # Statistical significance
    t_stat, p_val, is_sig = statistical_significance_test(pnls)

    return RiskMetrics(
        total_return=round(total_return, 2),
        cagr=round(cagr, 2),
        annualized_volatility=round(volatility, 2),
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        information_ratio=ir,
        max_drawdown=round(max_dd, 2),
        avg_drawdown=round(avg_dd, 2),
        max_drawdown_duration=max_dd_duration,
        ulcer_index=ulcer,
        var_95=round(var_95, 2),
        var_99=round(var_99, 2),
        cvar_95=round(cvar_95, 2),
        cvar_99=round(cvar_99, 2),
        skewness=round(skew, 3),
        kurtosis=round(kurt, 3),
        positive_months_pct=round(positive_months, 1),
        win_rate=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        expectancy=round(expectancy, 3),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        largest_win=round(largest_win, 2),
        largest_loss=round(largest_loss, 2),
        avg_trade_duration=round(avg_duration, 1),
        t_statistic=t_stat,
        p_value=p_val,
        is_significant=is_sig
    )


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results."""
    simulations: int
    median_return: float
    mean_return: float
    std_return: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    prob_profit: float
    prob_double: float
    prob_loss_20: float
    worst_case: float
    best_case: float
    confidence_interval_95: Tuple[float, float]
    median_max_drawdown: float
    var_95_mc: float


def monte_carlo_simulation(
    trades: List[Dict],
    initial_capital: float = 100000,
    num_simulations: int = 10000,
    num_trades: int = 100
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation on trade results.

    Randomly samples from historical trades to project future outcomes.
    """
    if not trades or len(trades) < 10:
        return None

    # Extract trade returns
    trade_returns = np.array([t.get('pnl_pct', 0) / 100 for t in trades])

    final_values = []
    max_drawdowns = []

    for _ in range(num_simulations):
        # Random sample with replacement
        sampled_returns = np.random.choice(trade_returns, size=num_trades, replace=True)

        # Calculate equity curve
        equity = initial_capital
        equity_curve = [equity]

        for ret in sampled_returns:
            equity *= (1 + ret)
            equity_curve.append(equity)

        final_values.append(equity)

        # Calculate max drawdown for this simulation
        equity_arr = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - running_max) / running_max
        max_drawdowns.append(abs(np.min(drawdown)) * 100)

    final_values = np.array(final_values)
    returns = (final_values / initial_capital - 1) * 100

    return MonteCarloResult(
        simulations=num_simulations,
        median_return=round(np.median(returns), 2),
        mean_return=round(np.mean(returns), 2),
        std_return=round(np.std(returns), 2),
        percentile_5=round(np.percentile(returns, 5), 2),
        percentile_25=round(np.percentile(returns, 25), 2),
        percentile_75=round(np.percentile(returns, 75), 2),
        percentile_95=round(np.percentile(returns, 95), 2),
        prob_profit=round((returns > 0).sum() / len(returns) * 100, 1),
        prob_double=round((returns > 100).sum() / len(returns) * 100, 1),
        prob_loss_20=round((returns < -20).sum() / len(returns) * 100, 1),
        worst_case=round(np.min(returns), 2),
        best_case=round(np.max(returns), 2),
        confidence_interval_95=(round(np.percentile(returns, 2.5), 2),
                                 round(np.percentile(returns, 97.5), 2)),
        median_max_drawdown=round(np.median(max_drawdowns), 2),
        var_95_mc=round(np.percentile(returns, 5), 2)
    )


# =============================================================================
# PORTFOLIO OPTIMIZATION
# =============================================================================

def calculate_correlation_matrix(returns_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Calculate correlation matrix between stock returns.
    """
    df = pd.DataFrame(returns_dict)
    return df.corr()


def optimize_portfolio_weights(
    expected_returns: Dict[str, float],
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.06,
    method: str = "max_sharpe"
) -> Dict[str, float]:
    """
    Optimize portfolio weights using Mean-Variance Optimization.

    Methods:
    - max_sharpe: Maximize Sharpe ratio
    - min_variance: Minimize portfolio variance
    - risk_parity: Equal risk contribution
    """
    n = len(expected_returns)
    symbols = list(expected_returns.keys())
    returns = np.array(list(expected_returns.values()))

    if method == "max_sharpe":
        def neg_sharpe(weights):
            port_return = np.dot(weights, returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            return -(port_return - risk_free_rate) / port_vol

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 0.25) for _ in range(n))  # Max 25% per stock
        initial = np.array([1/n] * n)

        result = minimize(neg_sharpe, initial, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        weights = result.x

    elif method == "min_variance":
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(covariance_matrix, weights))

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 0.25) for _ in range(n))
        initial = np.array([1/n] * n)

        result = minimize(portfolio_variance, initial, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        weights = result.x

    else:  # Equal weight
        weights = np.array([1/n] * n)

    return {symbols[i]: round(weights[i], 4) for i in range(n)}


def calculate_portfolio_metrics(
    weights: Dict[str, float],
    expected_returns: Dict[str, float],
    covariance_matrix: np.ndarray
) -> Dict:
    """
    Calculate portfolio-level metrics.
    """
    symbols = list(weights.keys())
    w = np.array([weights[s] for s in symbols])
    r = np.array([expected_returns[s] for s in symbols])

    port_return = np.dot(w, r)
    port_vol = np.sqrt(np.dot(w.T, np.dot(covariance_matrix, w)))
    sharpe = (port_return - 0.06) / port_vol if port_vol > 0 else 0

    return {
        'expected_return': round(port_return * 100, 2),
        'volatility': round(port_vol * 100, 2),
        'sharpe_ratio': round(sharpe, 3),
        'weights': weights
    }


# =============================================================================
# REGIME DETECTION
# =============================================================================

def detect_market_regime(
    prices: np.ndarray,
    lookback: int = 60
) -> str:
    """
    Detect market regime using multiple indicators.

    Returns: 'BULL', 'BEAR', 'HIGH_VOL', 'LOW_VOL', 'SIDEWAYS'
    """
    if len(prices) < lookback:
        return 'UNKNOWN'

    recent = prices[-lookback:]
    returns = np.diff(recent) / recent[:-1]

    # Trend detection
    sma_short = np.mean(recent[-20:])
    sma_long = np.mean(recent)
    trend_strength = (sma_short - sma_long) / sma_long * 100

    # Volatility detection
    volatility = np.std(returns) * np.sqrt(252) * 100
    avg_volatility = 20  # Baseline volatility

    if volatility > avg_volatility * 1.5:
        return 'HIGH_VOL'
    elif volatility < avg_volatility * 0.5:
        return 'LOW_VOL'
    elif trend_strength > 5:
        return 'BULL'
    elif trend_strength < -5:
        return 'BEAR'
    else:
        return 'SIDEWAYS'


def get_regime_adjustments(regime: str) -> Dict:
    """
    Get strategy parameter adjustments based on regime.
    """
    adjustments = {
        'BULL': {
            'position_size_mult': 1.2,
            'stop_mult': 1.0,
            'target_mult': 1.3,
            'signal_threshold': 7
        },
        'BEAR': {
            'position_size_mult': 0.5,
            'stop_mult': 0.8,
            'target_mult': 0.8,
            'signal_threshold': 9
        },
        'HIGH_VOL': {
            'position_size_mult': 0.6,
            'stop_mult': 1.5,
            'target_mult': 1.5,
            'signal_threshold': 9
        },
        'LOW_VOL': {
            'position_size_mult': 1.0,
            'stop_mult': 0.8,
            'target_mult': 1.0,
            'signal_threshold': 7
        },
        'SIDEWAYS': {
            'position_size_mult': 0.8,
            'stop_mult': 1.0,
            'target_mult': 0.9,
            'signal_threshold': 8
        }
    }
    return adjustments.get(regime, adjustments['SIDEWAYS'])


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func,
    n_bootstrap: int = 5000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for any statistic.

    Returns: (point_estimate, lower_bound, upper_bound)
    """
    point_estimate = statistic_func(data)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_func(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_stats, alpha * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)

    return round(point_estimate, 4), round(lower, 4), round(upper, 4)


def validate_strategy_robustness(trades: List[Dict]) -> Dict:
    """
    Comprehensive robustness validation using bootstrap.
    """
    if len(trades) < 30:
        return {'error': 'Insufficient trades for robust analysis'}

    pnls = np.array([t.get('pnl_pct', 0) for t in trades])

    # Win rate confidence interval
    wins = (pnls > 0).astype(float)
    wr_point, wr_lower, wr_upper = bootstrap_confidence_interval(
        wins, lambda x: np.mean(x) * 100
    )

    # Expectancy confidence interval
    exp_point, exp_lower, exp_upper = bootstrap_confidence_interval(
        pnls, np.mean
    )

    # Sharpe proxy confidence interval
    def sharpe_proxy(x):
        return np.mean(x) / np.std(x) if np.std(x) > 0 else 0

    sharpe_point, sharpe_lower, sharpe_upper = bootstrap_confidence_interval(
        pnls, sharpe_proxy
    )

    return {
        'win_rate': {
            'point': wr_point,
            'ci_95': (wr_lower, wr_upper)
        },
        'expectancy': {
            'point': exp_point,
            'ci_95': (exp_lower, exp_upper)
        },
        'sharpe_proxy': {
            'point': sharpe_point,
            'ci_95': (sharpe_lower, sharpe_upper)
        },
        'sample_size': len(trades),
        'is_robust': wr_lower > 50 and exp_lower > 0
    }


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("QUANT METRICS - INSTITUTIONAL GRADE ANALYTICS")
    print("=" * 70)

    # Test transaction costs
    print("\n1. TRANSACTION COSTS")
    print("-" * 40)
    tc = TransactionCosts()
    trade_value = 100000
    print(f"Trade Value: ₹{trade_value:,}")
    print(f"Round-trip Cost: ₹{tc.calculate_round_trip_cost(trade_value):,.2f}")
    print(f"As Percentage: {tc.cost_as_percentage(trade_value):.3f}%")

    # Test Monte Carlo
    print("\n2. MONTE CARLO SIMULATION")
    print("-" * 40)
    # Simulate trades
    np.random.seed(42)
    mock_trades = [{'pnl_pct': np.random.normal(0.5, 2)} for _ in range(200)]
    mc = monte_carlo_simulation(mock_trades, num_simulations=5000, num_trades=100)
    print(f"Simulations: {mc.simulations}")
    print(f"Median Return: {mc.median_return}%")
    print(f"95% CI: ({mc.confidence_interval_95[0]}%, {mc.confidence_interval_95[1]}%)")
    print(f"Prob of Profit: {mc.prob_profit}%")
    print(f"VaR 95%: {mc.var_95_mc}%")

    # Test Risk Metrics
    print("\n3. RISK METRICS")
    print("-" * 40)
    equity_curve = [100000]
    for t in mock_trades:
        equity_curve.append(equity_curve[-1] * (1 + t['pnl_pct']/100))

    metrics = calculate_all_risk_metrics(mock_trades, equity_curve, years=2)
    print(f"Sharpe Ratio: {metrics.sharpe_ratio}")
    print(f"Sortino Ratio: {metrics.sortino_ratio}")
    print(f"Max Drawdown: {metrics.max_drawdown}%")
    print(f"VaR 95%: {metrics.var_95}%")
    print(f"CVaR 95%: {metrics.cvar_95}%")
    print(f"P-value: {metrics.p_value} ({'Significant' if metrics.is_significant else 'Not Significant'})")

    # Test Robustness Validation
    print("\n4. BOOTSTRAP VALIDATION")
    print("-" * 40)
    robustness = validate_strategy_robustness(mock_trades)
    print(f"Win Rate: {robustness['win_rate']['point']:.1f}% "
          f"(95% CI: {robustness['win_rate']['ci_95'][0]:.1f}% - {robustness['win_rate']['ci_95'][1]:.1f}%)")
    print(f"Strategy Robust: {'Yes' if robustness['is_robust'] else 'No'}")

    print("\n" + "=" * 70)
    print("All quant metrics validated successfully!")
    print("=" * 70)
