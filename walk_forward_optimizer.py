"""
WALK-FORWARD OPTIMIZATION

True out-of-sample validation to detect overfitting.

Methodology:
1. Split 5-year data into rolling windows (e.g., 2-year train + 6-month test)
2. Optimize parameters on training window
3. Test on out-of-sample window (unseen data)
4. Roll forward and repeat
5. Aggregate out-of-sample results for realistic performance estimate

If in-sample >> out-of-sample: Strategy is OVERFITTED
If in-sample ≈ out-of-sample: Strategy is ROBUST
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from fetch_stock_data import fetch_stock_history
from fundamental_analyzer import get_fundamentals, analyze_fundamentals
import ta


@dataclass
class WFWindow:
    """Walk-forward window definition."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_num: int


@dataclass
class WFResult:
    """Result from one walk-forward window."""
    window_num: int
    # Training period results
    train_win_rate: float
    train_trades: int
    train_return: float
    train_expectancy: float
    # Test period results (out-of-sample)
    test_win_rate: float
    test_trades: int
    test_return: float
    test_expectancy: float
    # Optimized parameters for this window
    params: Dict


@dataclass
class ParameterSet:
    """Parameter combination for optimization."""
    target_mult: float
    stop_mult: float
    min_signals: int
    min_piotroski: int
    max_vol: float
    max_days: int
    trail_trigger: float


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization Engine.

    Validates strategy robustness by testing on truly unseen data.
    """

    def __init__(
        self,
        train_months: int = 24,
        test_months: int = 6,
        step_months: int = 6,
        start_year: int = 2021,
        end_year: int = 2026
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            train_months: Training window size (default: 24 months)
            test_months: Test window size (default: 6 months)
            step_months: How far to roll forward each iteration (default: 6 months)
            start_year: Start of backtest period
            end_year: End of backtest period
        """
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.start = datetime(start_year, 1, 1)
        self.end = datetime(end_year, 1, 31)

        # Parameter grid for optimization (reduced for speed)
        self.param_grid = {
            'target_mult': [1.5, 1.8, 2.2],
            'stop_mult': [1.0, 1.2, 1.5],
            'min_signals': [7, 8, 9],
            'min_piotroski': [5, 6, 7],
            'max_vol': [2.5, 2.8, 3.2],
            'max_days': [8, 10, 12],
            'trail_trigger': [0.30, 0.35, 0.40]
        }

        # Cache for stock data
        self._data_cache = {}
        self._fund_cache = {}

    def _generate_windows(self) -> List[WFWindow]:
        """Generate rolling walk-forward windows."""
        windows = []
        window_num = 1

        current = self.start
        while True:
            train_start = current
            train_end = train_start + timedelta(days=self.train_months * 30)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_months * 30)

            if test_end > self.end:
                break

            windows.append(WFWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                window_num=window_num
            ))

            window_num += 1
            current = current + timedelta(days=self.step_months * 30)

        return windows

    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        df = df.copy()

        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()

        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_sig'] = macd.macd_signal()

        # ADX
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['ADX'] = adx.adx()
        df['DI_plus'] = adx.adx_pos()
        df['DI_minus'] = adx.adx_neg()

        # ATR
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        df['ATR_pct'] = df['ATR'] / df['Close'] * 100

        # Volume
        df['Vol_SMA'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA']

        return df

    def _count_signals(self, df, idx, piotroski, fund_score) -> int:
        """Count entry confirmation signals."""
        if idx < 50:
            return 0

        row = df.iloc[idx]
        signals = 0

        # Fundamental signals
        if piotroski >= 6:
            signals += 1
        if piotroski >= 7:
            signals += 1
        if fund_score >= 55:
            signals += 1

        # Trend signals
        if pd.notna(row['SMA_20']) and pd.notna(row['SMA_50']):
            if row['Close'] > row['SMA_20']:
                signals += 1
            if row['SMA_20'] > row['SMA_50']:
                signals += 1

        # Momentum signals
        if pd.notna(row['RSI']) and 40 <= row['RSI'] <= 65:
            signals += 1
        if pd.notna(row['MACD']) and row['MACD'] > row['MACD_sig']:
            signals += 1

        # Volume
        if pd.notna(row['Vol_Ratio']) and row['Vol_Ratio'] > 0.8:
            signals += 1

        # ADX
        if pd.notna(row['ADX']) and row['ADX'] > 20:
            signals += 1
        if pd.notna(row['DI_plus']) and row['DI_plus'] > row['DI_minus']:
            signals += 1

        # Momentum
        if idx >= 5 and row['Close'] > df.iloc[idx-5]['Close']:
            signals += 1

        # Not overbought
        if pd.notna(row['RSI']) and row['RSI'] < 70:
            signals += 1

        return signals

    def _get_stock_data(self, symbol: str) -> Tuple[Optional[pd.DataFrame], int, float]:
        """Get stock data with caching."""
        if symbol in self._data_cache:
            return self._data_cache[symbol]

        hist = fetch_stock_history(symbol, days=1900)
        if hist is None or len(hist) < 500:
            self._data_cache[symbol] = (None, 5, 50)
            return None, 5, 50

        # Handle Date column
        if 'Date' in hist.columns:
            hist = hist.copy()
            hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
            hist.set_index('Date', inplace=True)
        if hist.index.tz:
            hist.index = hist.index.tz_localize(None)

        # Calculate indicators
        hist = self._indicators(hist)

        # Get fundamentals
        fund_score = 50
        piotroski = 5

        if symbol not in self._fund_cache:
            fund_data = get_fundamentals(symbol)
            if fund_data:
                result = analyze_fundamentals(symbol, fund_data)
                if result:
                    fund_score = result.total_score
                    piotroski = result.piotroski_score
            self._fund_cache[symbol] = (piotroski, fund_score)
        else:
            piotroski, fund_score = self._fund_cache[symbol]

        self._data_cache[symbol] = (hist, piotroski, fund_score)
        return hist, piotroski, fund_score

    def _backtest_period(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        params: ParameterSet
    ) -> Dict:
        """Run backtest for a specific period with given parameters."""

        all_trades = []

        for symbol in symbols:
            hist, piotroski, fund_score = self._get_stock_data(symbol)

            if hist is None:
                continue

            # Skip weak fundamentals
            if piotroski < params.min_piotroski:
                continue

            # Filter date range
            mask = (hist.index >= start) & (hist.index <= end)
            dates = list(hist.index[mask])

            if len(dates) < 30:
                continue

            trade = None
            cooldown = None

            for date in dates:
                idx = hist.index.get_loc(date)
                row = hist.iloc[idx]

                # Exit check
                if trade:
                    should_exit, exit_price, pnl = self._check_exit(
                        trade, row, date, hist, params
                    )
                    if should_exit:
                        all_trades.append({
                            'symbol': symbol,
                            'entry': trade['entry'],
                            'exit': date,
                            'pnl': pnl
                        })
                        trade = None
                        cooldown = date + timedelta(days=2)
                        continue

                # Entry check
                if not trade:
                    if cooldown and date < cooldown:
                        continue

                    # Volatility filter
                    if pd.notna(row['ATR_pct']) and row['ATR_pct'] > params.max_vol:
                        continue

                    signals = self._count_signals(hist, idx, piotroski, fund_score)

                    if signals >= params.min_signals:
                        price = row['Close']
                        atr = row['ATR'] if pd.notna(row['ATR']) else price * 0.015

                        trade = {
                            'entry': date,
                            'price': price,
                            'target': price + params.target_mult * atr,
                            'stop': price - params.stop_mult * atr,
                            'highest': row['High']
                        }

            # Close open trade
            if trade:
                exit_price = hist.iloc[-1]['Close']
                pnl = (exit_price - trade['price']) / trade['price'] * 100
                all_trades.append({
                    'symbol': symbol,
                    'entry': trade['entry'],
                    'exit': dates[-1],
                    'pnl': pnl
                })

        # Calculate metrics
        if not all_trades:
            return {
                'win_rate': 0,
                'trades': 0,
                'return': 0,
                'expectancy': 0
            }

        wins = [t for t in all_trades if t['pnl'] > 0]
        losses = [t for t in all_trades if t['pnl'] <= 0]

        win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
        total_return = sum(t['pnl'] for t in all_trades)
        expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)

        return {
            'win_rate': win_rate,
            'trades': len(all_trades),
            'return': total_return,
            'expectancy': expectancy
        }

    def _check_exit(
        self,
        trade: Dict,
        row,
        date: datetime,
        hist: pd.DataFrame,
        params: ParameterSet
    ) -> Tuple[bool, float, float]:
        """Check exit conditions."""
        price = row['Close']
        high = row['High']
        low = row['Low']

        if high > trade['highest']:
            trade['highest'] = high

        gain_pct = (price - trade['price']) / trade['price'] * 100
        max_gain = (trade['highest'] - trade['price']) / trade['price'] * 100
        target_gain = (trade['target'] - trade['price']) / trade['price'] * 100

        # Stop loss
        if low <= trade['stop']:
            pnl = (trade['stop'] - trade['price']) / trade['price'] * 100
            return True, trade['stop'], pnl

        # Target
        if high >= trade['target']:
            pnl = (trade['target'] - trade['price']) / trade['price'] * 100
            return True, trade['target'], pnl

        # Trailing stop
        if max_gain >= target_gain * params.trail_trigger:
            trail = trade['price'] * (1 + max_gain * 0.5 / 100)
            if low <= trail and trail > trade['price']:
                pnl = (max(trail, trade['price'] * 1.002) - trade['price']) / trade['price'] * 100
                return True, trail, pnl

        # Time exit
        days = (date - trade['entry']).days
        if days >= params.max_days:
            pnl = (price - trade['price']) / trade['price'] * 100
            return True, price, pnl

        # RSI overbought
        if pd.notna(row['RSI']) and row['RSI'] > 75 and gain_pct > 0.5:
            pnl = (price - trade['price']) / trade['price'] * 100
            return True, price, pnl

        return False, 0, 0

    def _optimize_parameters(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime
    ) -> Tuple[ParameterSet, Dict]:
        """Find best parameters for training period."""

        best_score = -float('inf')
        best_params = None
        best_result = None

        # Generate parameter combinations (simplified grid for speed)
        combinations = list(product(
            self.param_grid['target_mult'],
            self.param_grid['stop_mult'],
            self.param_grid['min_signals'],
            self.param_grid['min_piotroski'],
            self.param_grid['max_vol'],
            self.param_grid['max_days'],
            self.param_grid['trail_trigger']
        ))

        # Sample combinations for speed (full grid is 3^7 = 2187 combinations)
        np.random.seed(42)
        sample_size = min(50, len(combinations))
        sampled = [combinations[i] for i in np.random.choice(len(combinations), sample_size, replace=False)]

        # Always include the baseline parameters
        baseline = (1.8, 1.2, 8, 6, 2.8, 10, 0.35)
        if baseline not in sampled:
            sampled.append(baseline)

        for combo in sampled:
            params = ParameterSet(
                target_mult=combo[0],
                stop_mult=combo[1],
                min_signals=combo[2],
                min_piotroski=combo[3],
                max_vol=combo[4],
                max_days=combo[5],
                trail_trigger=combo[6]
            )

            result = self._backtest_period(symbols, start, end, params)

            # Score: prioritize expectancy + win rate
            score = result['expectancy'] * 10 + result['win_rate'] * 0.5

            if score > best_score and result['trades'] >= 20:
                best_score = score
                best_params = params
                best_result = result

        # Fallback to baseline if nothing found
        if best_params is None:
            best_params = ParameterSet(
                target_mult=1.8,
                stop_mult=1.2,
                min_signals=8,
                min_piotroski=6,
                max_vol=2.8,
                max_days=10,
                trail_trigger=0.35
            )
            best_result = self._backtest_period(symbols, start, end, best_params)

        return best_params, best_result

    def run_walk_forward(self, symbols: List[str]) -> Dict:
        """
        Run complete walk-forward optimization.

        Returns aggregated out-of-sample performance.
        """

        print("\n" + "="*80)
        print("WALK-FORWARD OPTIMIZATION")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Training Window:  {self.train_months} months")
        print(f"  Test Window:      {self.test_months} months")
        print(f"  Step Size:        {self.step_months} months")
        print(f"  Period:           {self.start.date()} to {self.end.date()}")
        print(f"  Stocks:           {len(symbols)}")

        windows = self._generate_windows()
        print(f"  Walk-Forward Windows: {len(windows)}")

        print("\n" + "-"*80)
        print("LOADING DATA...")
        print("-"*80)

        # Pre-load all stock data
        loaded = 0
        for symbol in symbols:
            hist, _, _ = self._get_stock_data(symbol)
            if hist is not None:
                loaded += 1
        print(f"Loaded {loaded}/{len(symbols)} stocks with sufficient data")

        results = []
        all_oos_trades = []

        print("\n" + "-"*80)
        print("RUNNING WALK-FORWARD WINDOWS")
        print("-"*80)

        for window in windows:
            print(f"\n[Window {window.window_num}/{len(windows)}]")
            print(f"  Train: {window.train_start.date()} to {window.train_end.date()}")
            print(f"  Test:  {window.test_start.date()} to {window.test_end.date()}")

            # Optimize on training period
            best_params, train_result = self._optimize_parameters(
                symbols, window.train_start, window.train_end
            )

            print(f"  Optimized params: target={best_params.target_mult}, "
                  f"stop={best_params.stop_mult}, signals={best_params.min_signals}")
            print(f"  Train: {train_result['win_rate']:.1f}% WR, "
                  f"{train_result['trades']} trades, "
                  f"{train_result['return']:+.1f}% return")

            # Test on out-of-sample period (using optimized params)
            test_result = self._backtest_period(
                symbols, window.test_start, window.test_end, best_params
            )

            print(f"  TEST:  {test_result['win_rate']:.1f}% WR, "
                  f"{test_result['trades']} trades, "
                  f"{test_result['return']:+.1f}% return")

            # Calculate degradation
            wr_degradation = train_result['win_rate'] - test_result['win_rate']
            print(f"  Degradation: {wr_degradation:+.1f}% WR")

            results.append(WFResult(
                window_num=window.window_num,
                train_win_rate=train_result['win_rate'],
                train_trades=train_result['trades'],
                train_return=train_result['return'],
                train_expectancy=train_result['expectancy'],
                test_win_rate=test_result['win_rate'],
                test_trades=test_result['trades'],
                test_return=test_result['return'],
                test_expectancy=test_result['expectancy'],
                params={
                    'target_mult': best_params.target_mult,
                    'stop_mult': best_params.stop_mult,
                    'min_signals': best_params.min_signals,
                    'min_piotroski': best_params.min_piotroski,
                    'max_vol': best_params.max_vol,
                    'max_days': best_params.max_days,
                    'trail_trigger': best_params.trail_trigger
                }
            ))

        # Aggregate results
        return self._analyze_walk_forward_results(results)

    def _analyze_walk_forward_results(self, results: List[WFResult]) -> Dict:
        """Analyze walk-forward results and detect overfitting."""

        if not results:
            print("\nNo results to analyze")
            return {}

        print("\n" + "="*80)
        print("WALK-FORWARD ANALYSIS")
        print("="*80)

        # In-sample (training) aggregates
        train_wr = np.mean([r.train_win_rate for r in results])
        train_trades = sum(r.train_trades for r in results)
        train_return = sum(r.train_return for r in results)
        train_exp = np.mean([r.train_expectancy for r in results])

        # Out-of-sample (test) aggregates
        test_wr = np.mean([r.test_win_rate for r in results])
        test_trades = sum(r.test_trades for r in results)
        test_return = sum(r.test_return for r in results)
        test_exp = np.mean([r.test_expectancy for r in results])

        print(f"\n{'Metric':<25} {'In-Sample':>15} {'Out-of-Sample':>15} {'Degradation':>15}")
        print("-"*70)
        print(f"{'Win Rate':<25} {train_wr:>14.1f}% {test_wr:>14.1f}% {train_wr - test_wr:>+14.1f}%")
        print(f"{'Total Trades':<25} {train_trades:>15} {test_trades:>15} {'':>15}")
        print(f"{'Total Return':<25} {train_return:>+14.1f}% {test_return:>+14.1f}% {train_return - test_return:>+14.1f}%")
        print(f"{'Expectancy':<25} {train_exp:>+14.3f}% {test_exp:>+14.3f}% {train_exp - test_exp:>+14.3f}%")

        # Robustness analysis
        wr_degradation = train_wr - test_wr
        robustness_ratio = test_wr / train_wr if train_wr > 0 else 0

        print("\n" + "-"*80)
        print("ROBUSTNESS ANALYSIS")
        print("-"*80)

        print(f"\nWin Rate Degradation:  {wr_degradation:+.1f}%")
        print(f"Robustness Ratio:      {robustness_ratio:.2f} (test/train)")

        # Window-by-window consistency
        print("\n" + "-"*80)
        print("WINDOW-BY-WINDOW RESULTS")
        print("-"*80)

        print(f"\n{'Window':<8} {'Train WR':>10} {'Test WR':>10} {'Degrad':>10} {'Test Ret':>12}")
        print("-"*55)

        for r in results:
            deg = r.train_win_rate - r.test_win_rate
            print(f"{r.window_num:<8} {r.train_win_rate:>9.1f}% {r.test_win_rate:>9.1f}% "
                  f"{deg:>+9.1f}% {r.test_return:>+11.1f}%")

        # Consistency metrics
        test_wrs = [r.test_win_rate for r in results]
        consistency_std = np.std(test_wrs)
        min_test_wr = min(test_wrs)
        max_test_wr = max(test_wrs)

        print(f"\nOut-of-Sample Win Rate:")
        print(f"  Mean:      {test_wr:.1f}%")
        print(f"  Std Dev:   {consistency_std:.1f}%")
        print(f"  Range:     {min_test_wr:.1f}% - {max_test_wr:.1f}%")

        # Verdict
        print("\n" + "="*80)
        print("VERDICT")
        print("="*80)

        if wr_degradation <= 5 and test_exp > 0:
            verdict = "ROBUST"
            symbol = "✅"
            explanation = "Strategy performs consistently in unseen data"
        elif wr_degradation <= 10 and test_exp > 0:
            verdict = "ACCEPTABLE"
            symbol = "🔶"
            explanation = "Minor degradation, strategy is usable"
        elif wr_degradation <= 15:
            verdict = "MODERATE OVERFIT"
            symbol = "⚠️"
            explanation = "Noticeable degradation, reduce parameter complexity"
        else:
            verdict = "OVERFITTED"
            symbol = "❌"
            explanation = "Significant overfitting detected, reconsider strategy"

        print(f"\n{symbol} {verdict}")
        print(f"   {explanation}")

        print(f"\n   In-Sample Win Rate:     {train_wr:.1f}%")
        print(f"   Out-of-Sample Win Rate: {test_wr:.1f}%")
        print(f"   Degradation:            {wr_degradation:+.1f}%")
        print(f"   Robustness Ratio:       {robustness_ratio:.2f}")

        # Expected real-world performance
        print(f"\n   EXPECTED REAL-WORLD PERFORMANCE:")
        print(f"   ─────────────────────────────────")
        print(f"   Win Rate:    {test_wr:.1f}%")
        print(f"   Expectancy:  {test_exp:+.3f}% per trade")

        if test_exp > 0:
            print(f"\n   ✅ Strategy has POSITIVE out-of-sample expectancy")
        else:
            print(f"\n   ⚠️ Strategy has NEGATIVE out-of-sample expectancy")

        return {
            'in_sample': {
                'win_rate': train_wr,
                'trades': train_trades,
                'return': train_return,
                'expectancy': train_exp
            },
            'out_of_sample': {
                'win_rate': test_wr,
                'trades': test_trades,
                'return': test_return,
                'expectancy': test_exp
            },
            'degradation': {
                'win_rate': wr_degradation,
                'return': train_return - test_return,
                'expectancy': train_exp - test_exp
            },
            'robustness_ratio': robustness_ratio,
            'verdict': verdict,
            'windows': len(results),
            'consistency_std': consistency_std
        }


def run_walk_forward_validation(symbols: List[str] = None) -> Dict:
    """
    Run walk-forward validation on the strategy.

    This is the main entry point for validating strategy robustness.
    """

    if symbols is None:
        symbols = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
            "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "BAJFINANCE",
            "HCLTECH", "SUNPHARMA", "TITAN", "WIPRO", "ULTRACEMCO",
            "NESTLEIND", "POWERGRID", "NTPC", "TECHM", "JSWSTEEL",
            "TATASTEEL", "INDUSINDBK", "GRASIM", "ADANIPORTS", "ONGC",
            "BAJAJFINSV", "DRREDDY", "CIPLA", "EICHERMOT", "HEROMOTOCO",
            "COALINDIA", "BRITANNIA", "DIVISLAB", "BPCL", "HINDALCO"
        ]

    optimizer = WalkForwardOptimizer(
        train_months=24,  # 2 years training
        test_months=6,    # 6 months out-of-sample test
        step_months=6     # Roll forward every 6 months
    )

    return optimizer.run_walk_forward(symbols)


if __name__ == "__main__":
    results = run_walk_forward_validation()
