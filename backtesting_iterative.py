"""
ITERATIVE BACKTEST - Learn and Improve Strategy

Process:
1. Run backtest → Analyze results → Identify weaknesses → Improve
2. Repeat for multiple iterations
3. Test on full available data (2+ years)
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
    pnl_pct: float = 0
    days: int = 0
    reason: str = ""


class IterativeBacktester:
    """
    Iteratively improved backtester.

    Iteration 1: Base strategy
    Iteration 2: Add volatility filter
    Iteration 3: Improve exit timing
    Iteration 4: Add market regime filter
    """

    def __init__(self, version: int = 4):
        self.version = version

        # Parameters tuned by iteration
        self.params = {
            1: {  # Base
                "target_mult": 2.0,
                "stop_mult": 1.5,
                "min_signals": 7,
                "min_piotroski": 5,
                "max_vol": 5.0,
                "max_days": 15,
                "trail_trigger": 0.5,
            },
            2: {  # + Volatility filter
                "target_mult": 2.0,
                "stop_mult": 1.5,
                "min_signals": 7,
                "min_piotroski": 6,
                "max_vol": 3.0,  # Tighter
                "max_days": 12,
                "trail_trigger": 0.4,
            },
            3: {  # + Better exits
                "target_mult": 1.8,  # Smaller target
                "stop_mult": 1.2,    # Tighter stop
                "min_signals": 8,
                "min_piotroski": 6,
                "max_vol": 2.8,
                "max_days": 10,
                "trail_trigger": 0.35,
            },
            4: {  # + Market regime
                "target_mult": 1.6,
                "stop_mult": 1.0,
                "min_signals": 8,
                "min_piotroski": 6,
                "max_vol": 2.5,
                "max_days": 8,
                "trail_trigger": 0.3,
                "use_market_filter": True,
            }
        }[version]

    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()

        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_sig'] = macd.macd_signal()

        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['ADX'] = adx.adx()
        df['DI_plus'] = adx.adx_pos()
        df['DI_minus'] = adx.adx_neg()

        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        df['ATR_pct'] = df['ATR'] / df['Close'] * 100

        df['Vol_SMA'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA']

        # Market regime (for v4)
        df['Market_Bullish'] = df['SMA_50'] > df['SMA_200']

        return df

    def _count_signals(self, df, idx, piotroski, fund_score) -> int:
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

        # Price momentum
        if idx >= 5 and row['Close'] > df.iloc[idx-5]['Close']:
            signals += 1

        # Not overbought
        if pd.notna(row['RSI']) and row['RSI'] < 70:
            signals += 1

        return signals

    def _check_exit(self, trade: Trade, row, date) -> Tuple[bool, str, float]:
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
        if max_gain >= target_gain * self.params["trail_trigger"]:
            trail = trade.entry_price * (1 + max_gain * 0.5 / 100)
            if low <= trail and trail > trade.entry_price:
                return True, "trail", max(trail, trade.entry_price * 1.002)

        # Time exit
        days = (date - trade.entry_date).days
        if days >= self.params["max_days"]:
            reason = "time_win" if gain_pct > 0 else "time_loss"
            return True, reason, price

        # RSI overbought
        if pd.notna(row['RSI']) and row['RSI'] > 75 and gain_pct > 0.5:
            return True, "rsi_exit", price

        return False, "", 0

    def backtest_stock(self, symbol, start, end, fund_data=None) -> List[Trade]:
        hist = fetch_stock_history(symbol, days=800)
        if hist is None or len(hist) < 250:
            return []

        if 'Date' in hist.columns:
            hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
            hist.set_index('Date', inplace=True)
        if hist.index.tz:
            hist.index = hist.index.tz_localize(None)

        hist = self._indicators(hist)

        # Get fundamentals
        if fund_data is None:
            fund_data = get_fundamentals(symbol)

        fund_score = 50
        piotroski = 5
        if fund_data:
            result = analyze_fundamentals(symbol, fund_data)
            if result:
                fund_score = result.total_score
                piotroski = result.piotroski_score

        # Skip weak fundamentals
        if piotroski < self.params["min_piotroski"]:
            return []

        mask = (hist.index >= start) & (hist.index <= end)
        dates = list(hist.index[mask])

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
                if pd.notna(row['ATR_pct']) and row['ATR_pct'] > self.params["max_vol"]:
                    continue

                # Market regime filter (v4)
                if self.params.get("use_market_filter"):
                    if pd.notna(row['Market_Bullish']) and not row['Market_Bullish']:
                        continue

                signals = self._count_signals(hist, idx, piotroski, fund_score)

                if signals >= self.params["min_signals"]:
                    price = row['Close']
                    atr = row['ATR'] if pd.notna(row['ATR']) else price * 0.015

                    trade = Trade(
                        symbol=symbol,
                        entry_date=date,
                        entry_price=price,
                        target=price + self.params["target_mult"] * atr,
                        stop=price - self.params["stop_mult"] * atr,
                        highest=row['High'],
                        piotroski=piotroski,
                        fund_score=fund_score,
                        signals=signals,
                        volatility=row['ATR_pct'] if pd.notna(row['ATR_pct']) else 2
                    )

        if trade:
            last = hist.iloc[-1]
            trade.exit_date = hist.index[-1]
            trade.exit_price = last['Close']
            trade.reason = "end"
            trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price * 100
            trades.append(trade)

        return trades

    def backtest(self, symbols: List[str], start: datetime, end: datetime) -> Dict:
        all_trades = []

        for symbol in symbols:
            try:
                trades = self.backtest_stock(symbol, start, end)
                all_trades.extend(trades)
            except:
                continue

        return self._results(all_trades)

    def _results(self, trades: List[Trade]) -> Dict:
        if not trades:
            return {"win_rate": 0, "trades": 0, "return": 0, "expectancy": 0, "pf": 0}

        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]

        win_rate = len(wins) / len(trades) * 100
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
        total = sum(t.pnl_pct for t in trades)
        expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)

        gp = sum(t.pnl_pct for t in wins) if wins else 0
        gl = abs(sum(t.pnl_pct for t in losses)) if losses else 1
        pf = gp / gl if gl > 0 else 10

        reasons = {}
        for t in trades:
            reasons[t.reason] = reasons.get(t.reason, 0) + 1

        return {
            "win_rate": win_rate,
            "trades": len(trades),
            "return": total,
            "expectancy": expectancy,
            "pf": pf,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_days": np.mean([t.days for t in trades]),
            "reasons": reasons
        }


def run_iterative_improvement():
    """Run all iterations and show improvement."""

    print("\n" + "="*80)
    print("ITERATIVE BACKTEST IMPROVEMENT")
    print("="*80)

    symbols = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "BAJFINANCE",
        "HCLTECH", "SUNPHARMA", "TITAN", "WIPRO", "ULTRACEMCO",
        "NESTLEIND", "POWERGRID", "NTPC", "TECHM", "JSWSTEEL"
    ]

    # Full available period (2024-02 to 2026-01)
    start = datetime(2024, 3, 1)
    end = datetime(2026, 1, 31)

    print(f"\nTesting Period: {start.date()} to {end.date()} (~2 years)")
    print(f"Stocks: {len(symbols)}\n")

    results = []

    for v in range(1, 5):
        print(f"Running Version {v}...")
        bt = IterativeBacktester(version=v)
        r = bt.backtest(symbols, start, end)
        results.append((v, r))

    # Display results
    print("\n" + "-"*90)
    print(f"{'Version':<12} {'Win%':>8} {'Trades':>8} {'Return':>10} {'Expect':>10} {'PF':>8} {'AvgDays':>8}")
    print("-"*90)

    for v, r in results:
        desc = {
            1: "Base",
            2: "+VolFilter",
            3: "+BetterExit",
            4: "+MktRegime"
        }[v]
        print(f"V{v} {desc:<9} {r['win_rate']:>7.1f}% {r['trades']:>8} {r['return']:>9.1f}% {r['expectancy']:>9.3f}% {r['pf']:>8.2f} {r['avg_days']:>8.1f}")

    print("-"*90)

    # Best version
    best = max(results, key=lambda x: (x[1]['expectancy'], x[1]['win_rate']))
    print(f"\n🏆 Best Version: V{best[0]}")
    print(f"   Win Rate:    {best[1]['win_rate']:.1f}%")
    print(f"   Return:      {best[1]['return']:.1f}%")
    print(f"   Expectancy:  {best[1]['expectancy']:.3f}%")
    print(f"   Profit Factor: {best[1]['pf']:.2f}")

    # Show exit reasons for best
    print(f"\n   Exit Breakdown:")
    for reason, count in sorted(best[1]['reasons'].items(), key=lambda x: -x[1]):
        print(f"      {reason:15} {count:4} ({count/best[1]['trades']*100:.1f}%)")

    # Improvement summary
    print("\n" + "="*80)
    print("ITERATION IMPROVEMENTS")
    print("="*80)

    base = results[0][1]
    final = results[-1][1]

    print(f"\n{'Metric':<20} {'V1 (Base)':>12} {'V4 (Final)':>12} {'Change':>12}")
    print("-"*60)
    print(f"{'Win Rate':<20} {base['win_rate']:>11.1f}% {final['win_rate']:>11.1f}% {final['win_rate']-base['win_rate']:>+11.1f}%")
    print(f"{'Total Return':<20} {base['return']:>11.1f}% {final['return']:>11.1f}% {final['return']-base['return']:>+11.1f}%")
    print(f"{'Expectancy':<20} {base['expectancy']:>11.3f}% {final['expectancy']:>11.3f}% {final['expectancy']-base['expectancy']:>+11.3f}%")
    print(f"{'Profit Factor':<20} {base['pf']:>12.2f} {final['pf']:>12.2f} {final['pf']-base['pf']:>+12.2f}")
    print(f"{'Trades':<20} {base['trades']:>12} {final['trades']:>12} {final['trades']-base['trades']:>+12}")

    return results


if __name__ == "__main__":
    run_iterative_improvement()
