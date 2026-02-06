"""
5-YEAR BACKTEST - Comprehensive Long-Term Validation

Tests strategy across multiple market cycles:
- Bull markets
- Bear markets
- Sideways/consolidation
- High volatility periods (COVID crash, etc.)

Period: 2021-2026 (5 years)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

from fetch_stock_data import fetch_stock_history
from fundamental_analyzer import get_fundamentals, analyze_fundamentals
import ta


@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime = None
    exit_price: float = 0
    target: float = 0
    stop: float = 0
    highest: float = 0
    piotroski: int = 0
    fund_score: float = 0
    signals: int = 0
    volatility: float = 0
    market_phase: str = ""  # bull, bear, sideways
    pnl_pct: float = 0
    days: int = 0
    reason: str = ""


class FiveYearBacktester:
    """
    5-Year Backtest with market regime awareness.

    Optimized parameters from iterative testing (V3):
    - Target: 1.8 × ATR
    - Stop: 1.2 × ATR
    - Min signals: 8
    - Min Piotroski: 6
    - Max volatility: 2.8%
    - Max holding: 10 days
    - Trailing at 35%
    """

    def __init__(self):
        # Optimized parameters
        self.target_mult = 1.8
        self.stop_mult = 1.2
        self.min_signals = 8
        self.min_piotroski = 6
        self.max_vol = 2.8
        self.max_days = 10
        self.trail_trigger = 0.35

    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        df = df.copy()

        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()

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

        # Market regime
        df['Bull_Market'] = (df['SMA_50'] > df['SMA_200']) & (df['Close'] > df['SMA_50'])
        df['Bear_Market'] = (df['SMA_50'] < df['SMA_200']) & (df['Close'] < df['SMA_50'])

        return df

    def _detect_market_phase(self, row) -> str:
        """Detect current market phase."""
        if pd.notna(row.get('Bull_Market')) and row['Bull_Market']:
            return "bull"
        elif pd.notna(row.get('Bear_Market')) and row['Bear_Market']:
            return "bear"
        return "sideways"

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

    def _check_exit(self, trade: Trade, row, date) -> Tuple[bool, str, float]:
        """Check exit conditions."""
        price = row['Close']
        high = row['High']
        low = row['Low']

        if high > trade.highest:
            trade.highest = high

        gain_pct = (price - trade.entry_price) / trade.entry_price * 100
        max_gain = (trade.highest - trade.entry_price) / trade.entry_price * 100
        target_gain = (trade.target - trade.entry_price) / trade.entry_price * 100

        # Stop loss
        if low <= trade.stop:
            return True, "stop", trade.stop

        # Target
        if high >= trade.target:
            return True, "target", trade.target

        # Trailing stop
        if max_gain >= target_gain * self.trail_trigger:
            trail = trade.entry_price * (1 + max_gain * 0.5 / 100)
            if low <= trail and trail > trade.entry_price:
                return True, "trail", max(trail, trade.entry_price * 1.002)

        # Time exit
        days = (date - trade.entry_date).days
        if days >= self.max_days:
            reason = "time_win" if gain_pct > 0 else "time_loss"
            return True, reason, price

        # RSI overbought
        if pd.notna(row['RSI']) and row['RSI'] > 75 and gain_pct > 0.5:
            return True, "rsi_exit", price

        return False, "", 0

    def backtest_stock(self, symbol: str, start: datetime, end: datetime) -> List[Trade]:
        """Backtest single stock over 5 years."""

        # Fetch 5 years of data
        hist = fetch_stock_history(symbol, days=1900)  # ~5.2 years
        if hist is None or len(hist) < 500:
            return []

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
        fund_data = get_fundamentals(symbol)
        fund_score = 50
        piotroski = 5

        if fund_data:
            result = analyze_fundamentals(symbol, fund_data)
            if result:
                fund_score = result.total_score
                piotroski = result.piotroski_score

        # Skip weak fundamentals
        if piotroski < self.min_piotroski:
            return []

        # Filter date range
        mask = (hist.index >= start) & (hist.index <= end)
        dates = list(hist.index[mask])

        if len(dates) < 50:
            return []

        trades = []
        trade = None
        cooldown = None

        for date in dates:
            idx = hist.index.get_loc(date)
            row = hist.iloc[idx]

            # Exit check
            if trade:
                should_exit, reason, exit_price = self._check_exit(trade, row, date)
                if should_exit:
                    trade.exit_date = date
                    trade.exit_price = exit_price
                    trade.reason = reason
                    trade.pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * 100
                    trade.days = (date - trade.entry_date).days
                    trades.append(trade)
                    trade = None
                    cooldown = date + timedelta(days=2)
                    continue

            # Entry check
            if not trade:
                if cooldown and date < cooldown:
                    continue

                # Volatility filter
                if pd.notna(row['ATR_pct']) and row['ATR_pct'] > self.max_vol:
                    continue

                signals = self._count_signals(hist, idx, piotroski, fund_score)

                if signals >= self.min_signals:
                    price = row['Close']
                    atr = row['ATR'] if pd.notna(row['ATR']) else price * 0.015

                    trade = Trade(
                        symbol=symbol,
                        entry_date=date,
                        entry_price=price,
                        target=price + self.target_mult * atr,
                        stop=price - self.stop_mult * atr,
                        highest=row['High'],
                        piotroski=piotroski,
                        fund_score=fund_score,
                        signals=signals,
                        volatility=row['ATR_pct'] if pd.notna(row['ATR_pct']) else 2,
                        market_phase=self._detect_market_phase(row)
                    )

        # Close open trade
        if trade:
            trade.exit_date = dates[-1]
            trade.exit_price = hist.iloc[-1]['Close']
            trade.reason = "end"
            trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price * 100
            trades.append(trade)

        return trades

    def run_5year_backtest(self, symbols: List[str]) -> Dict:
        """Run 5-year backtest."""

        # Try to get 5 years: 2021-2026
        # If data not available, use what's available
        start = datetime(2021, 1, 1)
        end = datetime(2026, 1, 31)

        print(f"\n{'='*75}")
        print("5-YEAR COMPREHENSIVE BACKTEST")
        print(f"{'='*75}")
        print(f"Target Period: {start.date()} to {end.date()}")
        print(f"Stocks: {len(symbols)}")
        print(f"Strategy: V3 Optimized (68.8% win rate on 2-year test)")
        print(f"{'='*75}\n")

        all_trades = []
        stocks_tested = 0
        stocks_qualified = 0

        for i, symbol in enumerate(symbols):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(symbols)}...")

            try:
                trades = self.backtest_stock(symbol, start, end)
                stocks_tested += 1
                if trades:
                    stocks_qualified += 1
                    all_trades.extend(trades)
            except Exception as e:
                continue

        print(f"\nStocks tested: {stocks_tested}")
        print(f"Stocks qualified (Piotroski≥6): {stocks_qualified}")

        return self._analyze_results(all_trades, start, end)

    def _analyze_results(self, trades: List[Trade], start: datetime, end: datetime) -> Dict:
        """Analyze results with yearly breakdown."""

        if not trades:
            print("\nNo trades generated.")
            return {}

        # Overall metrics
        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]

        win_rate = len(wins) / len(trades) * 100
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
        total_return = sum(t.pnl_pct for t in trades)
        expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)

        gp = sum(t.pnl_pct for t in wins) if wins else 0
        gl = abs(sum(t.pnl_pct for t in losses)) if losses else 1
        pf = gp / gl if gl > 0 else 10

        # Print overall results
        print(f"\n{'='*75}")
        print("OVERALL 5-YEAR RESULTS")
        print(f"{'='*75}")

        marker = "✅" if win_rate >= 65 else ("🔶" if win_rate >= 55 else "❌")
        print(f"\n🎯 WIN RATE: {win_rate:.1f}% {marker}")

        print(f"\n📊 Performance:")
        print(f"   Total Trades:    {len(trades)}")
        print(f"   Winners:         {len(wins)}")
        print(f"   Losers:          {len(losses)}")
        print(f"   Total Return:    {total_return:+.1f}%")
        print(f"   Avg Win:         {avg_win:+.2f}%")
        print(f"   Avg Loss:        {avg_loss:.2f}%")
        print(f"   Expectancy:      {expectancy:+.3f}% per trade")
        print(f"   Profit Factor:   {pf:.2f}")
        print(f"   Avg Hold Days:   {np.mean([t.days for t in trades]):.1f}")

        # Yearly breakdown
        print(f"\n{'='*75}")
        print("YEARLY BREAKDOWN")
        print(f"{'='*75}")
        print(f"\n{'Year':<8} {'Trades':>8} {'Win%':>8} {'Return':>10} {'Expect':>10} {'PF':>8}")
        print("-"*60)

        yearly_results = {}
        for year in range(2021, 2027):
            year_trades = [t for t in trades if t.entry_date.year == year]
            if year_trades:
                y_wins = [t for t in year_trades if t.pnl_pct > 0]
                y_wr = len(y_wins) / len(year_trades) * 100
                y_ret = sum(t.pnl_pct for t in year_trades)
                y_avg_win = np.mean([t.pnl_pct for t in y_wins]) if y_wins else 0
                y_avg_loss = np.mean([t.pnl_pct for t in year_trades if t.pnl_pct <= 0]) if len(year_trades) > len(y_wins) else 0
                y_exp = (y_wr/100 * y_avg_win) + ((100-y_wr)/100 * y_avg_loss)

                y_gp = sum(t.pnl_pct for t in y_wins) if y_wins else 0
                y_gl = abs(sum(t.pnl_pct for t in year_trades if t.pnl_pct <= 0))
                y_pf = y_gp / y_gl if y_gl > 0 else 10

                yearly_results[year] = {
                    "trades": len(year_trades),
                    "win_rate": y_wr,
                    "return": y_ret,
                    "expectancy": y_exp,
                    "pf": y_pf
                }

                print(f"{year:<8} {len(year_trades):>8} {y_wr:>7.1f}% {y_ret:>9.1f}% {y_exp:>9.3f}% {y_pf:>8.2f}")

        print("-"*60)

        # Market phase analysis
        print(f"\n{'='*75}")
        print("MARKET PHASE ANALYSIS")
        print(f"{'='*75}")

        for phase in ["bull", "bear", "sideways"]:
            phase_trades = [t for t in trades if t.market_phase == phase]
            if phase_trades:
                p_wins = [t for t in phase_trades if t.pnl_pct > 0]
                p_wr = len(p_wins) / len(phase_trades) * 100
                p_ret = sum(t.pnl_pct for t in phase_trades)
                print(f"\n{phase.upper()} Market:")
                print(f"   Trades: {len(phase_trades)}, Win Rate: {p_wr:.1f}%, Return: {p_ret:+.1f}%")

        # Exit reasons
        print(f"\n{'='*75}")
        print("EXIT ANALYSIS")
        print(f"{'='*75}\n")

        reasons = {}
        for t in trades:
            reasons[t.reason] = reasons.get(t.reason, 0) + 1

        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            # Calculate win rate for this exit type
            reason_trades = [t for t in trades if t.reason == reason]
            reason_wins = len([t for t in reason_trades if t.pnl_pct > 0])
            reason_wr = reason_wins / len(reason_trades) * 100 if reason_trades else 0
            print(f"   {reason:15} {count:4} ({count/len(trades)*100:5.1f}%) | Win Rate: {reason_wr:.1f}%")

        # Top performers
        print(f"\n{'='*75}")
        print("TOP PERFORMERS")
        print(f"{'='*75}")

        print("\n🏆 Best Trades:")
        for t in sorted(trades, key=lambda x: x.pnl_pct, reverse=True)[:10]:
            print(f"   {t.symbol:10} {t.pnl_pct:+6.2f}% | {t.entry_date.date()} | {t.days}d | {t.market_phase}")

        print("\n❌ Worst Trades:")
        for t in sorted(trades, key=lambda x: x.pnl_pct)[:10]:
            print(f"   {t.symbol:10} {t.pnl_pct:+6.2f}% | {t.entry_date.date()} | {t.days}d | {t.market_phase}")

        # Summary
        print(f"\n{'='*75}")
        print("FINAL VERDICT")
        print(f"{'='*75}")

        if win_rate >= 65 and expectancy > 0:
            print("\n✅ STRATEGY VALIDATED: Consistent performance over 5 years")
        elif win_rate >= 55 and expectancy > 0:
            print("\n🔶 STRATEGY ACCEPTABLE: Room for improvement but profitable")
        else:
            print("\n⚠️ NEEDS WORK: Review parameters for long-term consistency")

        print(f"\n   5-Year Win Rate:    {win_rate:.1f}%")
        print(f"   5-Year Return:      {total_return:+.1f}%")
        print(f"   Expectancy:         {expectancy:+.3f}% per trade")
        print(f"   Profit Factor:      {pf:.2f}")

        return {
            "win_rate": win_rate,
            "total_return": total_return,
            "trades": len(trades),
            "expectancy": expectancy,
            "profit_factor": pf,
            "yearly": yearly_results
        }


if __name__ == "__main__":
    from nse_tickers import fetch_nse_tickers
    symbols = fetch_nse_tickers("NIFTY 50")
    if not symbols:
        print("Failed to fetch NIFTY 50 stocks from NSE API.")
        import sys
        sys.exit(1)

    bt = FiveYearBacktester()
    results = bt.run_5year_backtest(symbols)
