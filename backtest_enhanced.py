"""
ENHANCED INSTITUTIONAL BACKTESTING ENGINE
==========================================

Uses ALL AVAILABLE technical and fundamental signals:

TECHNICAL SIGNALS (16 total):
1. Price > SMA20
2. SMA20 > SMA50
3. SMA50 > SMA200 (Golden cross setup)
4. RSI in range (40-65)
5. RSI not overbought (<70)
6. MACD > Signal line
7. MACD histogram rising
8. ADX > 20 (trending)
9. DI+ > DI- (bullish directional)
10. Volume > 0.8x average
11. ROC_5 > 0 (5-day momentum)
12. ROC_10 > 0 (10-day momentum)
13. SuperTrend bullish
14. Stochastic not overbought (K < 80)
15. Price > lower Bollinger Band
16. EMA12 > SMA20 (short-term trend)

FUNDAMENTAL SIGNALS (8 total):
1. Piotroski >= 6 (basic)
2. Piotroski >= 7 (strong)
3. Piotroski >= 8 (very strong)
4. Fundamental score >= 55
5. Fundamental score >= 65 (high quality)
6. ROE > 15% (profitable)
7. Debt/Equity < 100% (low leverage)
8. Profit margin > 10%

TOTAL: 24 signals for entry confirmation
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
from quant_metrics import (
    TransactionCosts, calculate_all_risk_metrics, monte_carlo_simulation,
    validate_strategy_robustness, detect_market_regime, get_regime_adjustments,
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_var, calculate_cvar
)
import ta


@dataclass
class EnhancedTrade:
    """Trade with full signal tracking."""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime = None
    exit_price: float = 0
    shares: int = 100
    target: float = 0
    stop: float = 0
    highest: float = 0
    lowest: float = float('inf')
    piotroski: int = 0
    fund_score: float = 0
    tech_signals: int = 0
    fund_signals: int = 0
    total_signals: int = 0
    volatility: float = 0
    regime: str = ""
    pnl_gross: float = 0
    pnl_net: float = 0
    transaction_cost: float = 0
    slippage_cost: float = 0
    days: int = 0
    reason: str = ""
    mae: float = 0
    mfe: float = 0


@dataclass
class EnhancedConfig:
    """
    Enhanced configuration with expanded signal filtering.
    """
    # Entry - ATR multiples
    target_mult: float = 6.0
    stop_mult: float = 3.0

    # Signal thresholds (out of 24 total signals)
    min_tech_signals: int = 12      # Minimum technical signals (out of 16)
    min_fund_signals: int = 5       # Minimum fundamental signals (out of 8)
    min_total_signals: int = 17     # Minimum total signals (out of 24)

    # Fundamental requirements
    min_piotroski: int = 7
    min_fund_score: float = 55
    min_roe: float = 0.15           # 15% ROE minimum
    max_debt_equity: float = 100    # Maximum D/E ratio
    min_profit_margin: float = 0.10 # 10% profit margin

    # Technical requirements
    max_vol: float = 2.0            # Max ATR%
    require_supertrend: bool = True # Require SuperTrend bullish

    # Exit parameters
    max_days: int = 90              # Maximum hold (3 months)
    min_hold_days: int = 20         # Minimum hold (1 month)
    trail_trigger: float = 0.75
    trail_pct: float = 0.65

    # Risk parameters
    risk_per_trade: float = 0.02
    max_positions: int = 3

    # Drawdown control
    max_drawdown_pct: float = 25.0
    reduce_size_at_dd: float = 15.0
    size_reduction_factor: float = 0.7

    # Costs
    include_costs: bool = True
    slippage_pct: float = 0.03
    use_regime: bool = True


class EnhancedBacktester:
    """
    Enhanced backtester using ALL available signals.
    """

    def __init__(self, config: EnhancedConfig = None):
        self.config = config or EnhancedConfig()
        self.costs = TransactionCosts()
        self._data_cache = {}
        self._fund_cache = {}

    def _get_stock_data(self, symbol: str) -> Tuple[Optional[pd.DataFrame], Dict]:
        """Get stock data with fundamentals."""
        if symbol in self._data_cache:
            return self._data_cache[symbol]

        hist = fetch_stock_history(symbol, days=3700)
        if hist is None or len(hist) < 500:
            self._data_cache[symbol] = (None, {})
            return None, {}

        # Handle Date column
        if 'Date' in hist.columns:
            hist = hist.copy()
            hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
            hist.set_index('Date', inplace=True)
        if hist.index.tz:
            hist.index = hist.index.tz_localize(None)

        # Calculate ALL indicators
        hist = self._calculate_all_indicators(hist)

        # Get fundamentals
        fund_data = {}
        if symbol not in self._fund_cache:
            raw_fund = get_fundamentals(symbol)
            if raw_fund:
                result = analyze_fundamentals(symbol, raw_fund)
                if result:
                    fund_data = {
                        'piotroski': result.piotroski_score,
                        'fund_score': result.total_score,
                        'roe': raw_fund.get('roe', 0) or 0,
                        'debt_equity': raw_fund.get('debt_to_equity', 100) or 100,
                        'profit_margin': raw_fund.get('profit_margin', 0) or 0,
                        'revenue_growth': raw_fund.get('revenue_growth', 0) or 0,
                        'earnings_growth': raw_fund.get('earnings_growth', 0) or 0,
                    }
            self._fund_cache[symbol] = fund_data
        else:
            fund_data = self._fund_cache[symbol]

        self._data_cache[symbol] = (hist, fund_data)
        return hist, fund_data

    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ALL technical indicators."""
        df = df.copy()

        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()

        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_sig'] = macd.macd_signal()
        df['MACD_hist'] = macd.macd_diff()

        # ADX
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['ADX'] = adx.adx()
        df['DI_plus'] = adx.adx_pos()
        df['DI_minus'] = adx.adx_neg()

        # ATR
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        df['ATR_pct'] = df['ATR'] / df['Close'] * 100

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_mid'] = bb.bollinger_mavg()
        df['BB_pband'] = bb.bollinger_pband()  # %B indicator

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()

        # Volume
        df['Vol_SMA'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA']

        # Momentum
        df['ROC_5'] = df['Close'].pct_change(5) * 100
        df['ROC_10'] = df['Close'].pct_change(10) * 100
        df['ROC_20'] = df['Close'].pct_change(20) * 100

        # SuperTrend
        df = self._calculate_supertrend(df)

        # Market regime
        df['Bull_Market'] = (df['SMA_50'] > df['SMA_200']) & (df['Close'] > df['SMA_50'])
        df['Bear_Market'] = (df['SMA_50'] < df['SMA_200']) & (df['Close'] < df['SMA_50'])

        return df

    def _calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """Calculate SuperTrend indicator."""
        hl2 = (df['High'] + df['Low']) / 2
        atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=period)

        upper_basic = hl2 + (multiplier * atr)
        lower_basic = hl2 - (multiplier * atr)

        upper_band = upper_basic.copy()
        lower_band = lower_basic.copy()
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)

        for i in range(period, len(df)):
            # Upper band
            if upper_basic.iloc[i] < upper_band.iloc[i-1] or df['Close'].iloc[i-1] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_basic.iloc[i]
            else:
                upper_band.iloc[i] = upper_band.iloc[i-1]

            # Lower band
            if lower_basic.iloc[i] > lower_band.iloc[i-1] or df['Close'].iloc[i-1] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_basic.iloc[i]
            else:
                lower_band.iloc[i] = lower_band.iloc[i-1]

            # Direction
            if i < period + 1:
                direction.iloc[i] = 1
            elif supertrend.iloc[i-1] == upper_band.iloc[i-1]:
                direction.iloc[i] = -1 if df['Close'].iloc[i] > upper_band.iloc[i] else 1
            else:
                direction.iloc[i] = 1 if df['Close'].iloc[i] < lower_band.iloc[i] else -1

            # SuperTrend value
            supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

        df['SuperTrend'] = supertrend
        df['ST_Direction'] = direction  # 1 = bullish, -1 = bearish

        return df

    def _count_tech_signals(self, df: pd.DataFrame, idx: int) -> Tuple[int, List[str]]:
        """Count technical signals (16 possible)."""
        if idx < 50:
            return 0, []

        row = df.iloc[idx]
        prev_row = df.iloc[idx-1] if idx > 0 else row
        signals = 0
        signal_list = []

        # 1. Price > SMA20
        if pd.notna(row['SMA_20']) and row['Close'] > row['SMA_20']:
            signals += 1
            signal_list.append("Price > SMA20")

        # 2. SMA20 > SMA50
        if pd.notna(row['SMA_20']) and pd.notna(row['SMA_50']) and row['SMA_20'] > row['SMA_50']:
            signals += 1
            signal_list.append("SMA20 > SMA50")

        # 3. SMA50 > SMA200 (Golden cross setup)
        if pd.notna(row['SMA_50']) and pd.notna(row['SMA_200']) and row['SMA_50'] > row['SMA_200']:
            signals += 1
            signal_list.append("SMA50 > SMA200")

        # 4. RSI in range (40-65)
        if pd.notna(row['RSI']) and 40 <= row['RSI'] <= 65:
            signals += 1
            signal_list.append(f"RSI optimal ({row['RSI']:.0f})")

        # 5. RSI not overbought (<70)
        if pd.notna(row['RSI']) and row['RSI'] < 70:
            signals += 1
            signal_list.append("Not overbought")

        # 6. MACD > Signal
        if pd.notna(row['MACD']) and pd.notna(row['MACD_sig']) and row['MACD'] > row['MACD_sig']:
            signals += 1
            signal_list.append("MACD bullish")

        # 7. MACD histogram rising
        if pd.notna(row['MACD_hist']) and pd.notna(prev_row['MACD_hist']) and row['MACD_hist'] > prev_row['MACD_hist']:
            signals += 1
            signal_list.append("MACD momentum")

        # 8. ADX > 20 (trending)
        if pd.notna(row['ADX']) and row['ADX'] > 20:
            signals += 1
            signal_list.append(f"ADX trending ({row['ADX']:.0f})")

        # 9. DI+ > DI-
        if pd.notna(row['DI_plus']) and pd.notna(row['DI_minus']) and row['DI_plus'] > row['DI_minus']:
            signals += 1
            signal_list.append("DI+ > DI-")

        # 10. Volume > 0.8x average
        if pd.notna(row['Vol_Ratio']) and row['Vol_Ratio'] > 0.8:
            signals += 1
            signal_list.append(f"Volume ({row['Vol_Ratio']:.1f}x)")

        # 11. ROC_5 > 0 (5-day momentum)
        if pd.notna(row['ROC_5']) and row['ROC_5'] > 0:
            signals += 1
            signal_list.append("5D momentum +")

        # 12. ROC_10 > 0 (10-day momentum)
        if pd.notna(row['ROC_10']) and row['ROC_10'] > 0:
            signals += 1
            signal_list.append("10D momentum +")

        # 13. SuperTrend bullish
        if pd.notna(row.get('ST_Direction')) and row['ST_Direction'] == 1:
            signals += 1
            signal_list.append("SuperTrend bullish")

        # 14. Stochastic not overbought (K < 80)
        if pd.notna(row['Stoch_K']) and row['Stoch_K'] < 80:
            signals += 1
            signal_list.append("Stoch not OB")

        # 15. Price > lower Bollinger Band
        if pd.notna(row['BB_lower']) and row['Close'] > row['BB_lower']:
            signals += 1
            signal_list.append("Above BB lower")

        # 16. EMA12 > SMA20
        if pd.notna(row['EMA_12']) and pd.notna(row['SMA_20']) and row['EMA_12'] > row['SMA_20']:
            signals += 1
            signal_list.append("EMA12 > SMA20")

        return signals, signal_list

    def _count_fund_signals(self, fund_data: Dict) -> Tuple[int, List[str]]:
        """Count fundamental signals (8 possible)."""
        signals = 0
        signal_list = []

        piotroski = fund_data.get('piotroski', 0)
        fund_score = fund_data.get('fund_score', 0)
        roe = fund_data.get('roe', 0)
        debt_equity = fund_data.get('debt_equity', 100)
        profit_margin = fund_data.get('profit_margin', 0)

        # 1. Piotroski >= 6
        if piotroski >= 6:
            signals += 1
            signal_list.append(f"Piotroski >= 6 ({piotroski})")

        # 2. Piotroski >= 7
        if piotroski >= 7:
            signals += 1
            signal_list.append(f"Piotroski >= 7 ({piotroski})")

        # 3. Piotroski >= 8
        if piotroski >= 8:
            signals += 1
            signal_list.append(f"Piotroski >= 8 ({piotroski})")

        # 4. Fundamental score >= 55
        if fund_score >= 55:
            signals += 1
            signal_list.append(f"Fund score >= 55 ({fund_score:.0f})")

        # 5. Fundamental score >= 65
        if fund_score >= 65:
            signals += 1
            signal_list.append(f"Fund score >= 65 ({fund_score:.0f})")

        # 6. ROE > 15%
        if roe and roe > 0.15:
            signals += 1
            signal_list.append(f"ROE > 15% ({roe*100:.1f}%)")

        # 7. Debt/Equity < 100%
        if debt_equity and debt_equity < 100:
            signals += 1
            signal_list.append(f"Low D/E ({debt_equity:.0f}%)")

        # 8. Profit margin > 10%
        if profit_margin and profit_margin > 0.10:
            signals += 1
            signal_list.append(f"Margin > 10% ({profit_margin*100:.1f}%)")

        return signals, signal_list

    def _detect_regime(self, row) -> str:
        """Detect market regime."""
        if pd.notna(row.get('Bull_Market')) and row['Bull_Market']:
            if pd.notna(row.get('ATR_pct')) and row['ATR_pct'] > 3:
                return 'HIGH_VOL'
            return 'BULL'
        elif pd.notna(row.get('Bear_Market')) and row['Bear_Market']:
            return 'BEAR'
        elif pd.notna(row.get('ATR_pct')) and row['ATR_pct'] < 1.5:
            return 'LOW_VOL'
        return 'SIDEWAYS'

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply realistic slippage."""
        slippage = price * (self.config.slippage_pct / 100)
        return price + slippage if is_buy else price - slippage

    def _check_exit(self, trade: EnhancedTrade, row, date, regime_adj: Dict) -> Tuple[bool, str, float]:
        """Check exit conditions."""
        price = row['Close']
        high = row['High']
        low = row['Low']
        days_held = (date - trade.entry_date).days

        # Update MAE/MFE
        if high > trade.highest:
            trade.highest = high
            trade.mfe = (high - trade.entry_price) / trade.entry_price * 100
        if low < trade.lowest:
            trade.lowest = low
            trade.mae = (trade.entry_price - low) / trade.entry_price * 100

        max_gain = (trade.highest - trade.entry_price) / trade.entry_price * 100
        target_gain = (trade.target - trade.entry_price) / trade.entry_price * 100

        # 1. Stop loss
        adj_stop = trade.entry_price - (trade.entry_price - trade.stop) * regime_adj.get('stop_mult', 1.0)
        if low <= adj_stop:
            return True, "stop", self._apply_slippage(adj_stop, is_buy=False)

        # 2. Target hit
        if high >= trade.target:
            return True, "target", self._apply_slippage(trade.target, is_buy=False)

        # 3. Minimum hold period
        if days_held < self.config.min_hold_days:
            return False, "", 0

        # 4. Trailing stop
        if max_gain >= target_gain * self.config.trail_trigger:
            trail_level = trade.entry_price * (1 + max_gain * self.config.trail_pct / 100)
            min_profit = trade.entry_price * 1.03
            if low <= trail_level and trail_level >= min_profit:
                return True, "trail", self._apply_slippage(trail_level, is_buy=False)

        # 5. Max hold
        if days_held >= self.config.max_days:
            reason = "max_hold_profit" if price > trade.entry_price else "max_hold_loss"
            return True, reason, self._apply_slippage(price, is_buy=False)

        return False, "", 0

    def backtest_stock(self, symbol: str, start: datetime, end: datetime, capital_per_trade: float = 100000) -> List[EnhancedTrade]:
        """Backtest single stock."""
        hist, fund_data = self._get_stock_data(symbol)

        if hist is None or not fund_data:
            return []

        # Check fundamental requirements
        piotroski = fund_data.get('piotroski', 0)
        fund_score = fund_data.get('fund_score', 0)
        roe = fund_data.get('roe', 0)
        debt_equity = fund_data.get('debt_equity', 100)
        profit_margin = fund_data.get('profit_margin', 0)

        if piotroski < self.config.min_piotroski:
            return []
        if fund_score < self.config.min_fund_score:
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

            regime = self._detect_regime(row)
            regime_adj = get_regime_adjustments(regime) if self.config.use_regime else {}

            # Exit check
            if trade:
                should_exit, reason, exit_price = self._check_exit(trade, row, date, regime_adj)
                if should_exit:
                    trade.exit_date = date
                    trade.exit_price = exit_price
                    trade.reason = reason
                    trade.days = (date - trade.entry_date).days

                    trade.pnl_gross = (exit_price - trade.entry_price) / trade.entry_price * 100

                    if self.config.include_costs:
                        trade_value = trade.shares * trade.entry_price
                        trade.transaction_cost = self.costs.calculate_round_trip_cost(trade_value)
                        trade.slippage_cost = trade_value * (self.config.slippage_pct / 100) * 2
                        total_cost_pct = (trade.transaction_cost + trade.slippage_cost) / trade_value * 100
                        trade.pnl_net = trade.pnl_gross - total_cost_pct
                    else:
                        trade.pnl_net = trade.pnl_gross

                    trades.append(trade)
                    trade = None
                    cooldown = date + timedelta(days=2)
                    continue

            # Entry check
            if not trade:
                if cooldown and date < cooldown:
                    continue

                # Volatility filter
                atr_pct = row.get('ATR_pct', 3)
                max_vol_adj = self.config.max_vol * regime_adj.get('vol_mult', 1.0)
                if pd.notna(atr_pct) and atr_pct > max_vol_adj:
                    continue

                # Count signals
                tech_signals, tech_list = self._count_tech_signals(hist, idx)
                fund_signals, fund_list = self._count_fund_signals(fund_data)
                total_signals = tech_signals + fund_signals

                # Check signal thresholds
                if tech_signals < self.config.min_tech_signals:
                    continue
                if fund_signals < self.config.min_fund_signals:
                    continue
                if total_signals < self.config.min_total_signals:
                    continue

                # SuperTrend requirement
                if self.config.require_supertrend:
                    if not (pd.notna(row.get('ST_Direction')) and row['ST_Direction'] == 1):
                        continue

                # Entry!
                entry_price = self._apply_slippage(row['Close'], is_buy=True)
                atr = row.get('ATR', entry_price * 0.02)

                target_adj = regime_adj.get('target_mult', 1.0)
                stop_adj = regime_adj.get('stop_mult', 1.0)

                target = entry_price + (self.config.target_mult * atr * target_adj)
                stop = entry_price - (self.config.stop_mult * atr * stop_adj)

                trade = EnhancedTrade(
                    symbol=symbol,
                    entry_date=date,
                    entry_price=entry_price,
                    target=target,
                    stop=stop,
                    highest=row['High'],
                    lowest=row['Low'],
                    piotroski=piotroski,
                    fund_score=fund_score,
                    tech_signals=tech_signals,
                    fund_signals=fund_signals,
                    total_signals=total_signals,
                    volatility=atr_pct,
                    regime=regime
                )

        return trades

    def run_backtest(self, symbols: List[str], start: datetime, end: datetime) -> Dict:
        """Run backtest on multiple symbols."""
        print(f"\n{'='*80}")
        print("ENHANCED INSTITUTIONAL BACKTEST (24 SIGNALS)")
        print("="*80)
        print(f"Period: {start.date()} to {end.date()}")
        print(f"Stocks: {len(symbols)}")
        print(f"Signal Requirements: Tech>={self.config.min_tech_signals}/16, Fund>={self.config.min_fund_signals}/8, Total>={self.config.min_total_signals}/24")
        print("="*80)

        all_trades = []

        for i, symbol in enumerate(symbols):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(symbols)}...")
            trades = self.backtest_stock(symbol, start, end)
            all_trades.extend(trades)

        if not all_trades:
            print("No trades generated.")
            return {}

        # Apply drawdown control
        all_trades = self._apply_drawdown_control(all_trades)

        return self._analyze_results(all_trades, start, end)

    def _apply_drawdown_control(self, trades: List[EnhancedTrade]) -> List[EnhancedTrade]:
        """Apply drawdown control."""
        if not trades:
            return trades

        sorted_trades = sorted(trades, key=lambda x: x.exit_date)
        equity = 100000
        peak_equity = 100000

        for trade in sorted_trades:
            current_dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0

            if current_dd >= self.config.max_drawdown_pct:
                size_mult = 0.5
            elif current_dd >= self.config.reduce_size_at_dd:
                size_mult = self.config.size_reduction_factor
            else:
                size_mult = 1.0

            trade.pnl_net *= size_mult
            trade.pnl_gross *= size_mult
            equity = equity * (1 + trade.pnl_net / 100)
            peak_equity = max(peak_equity, equity)

        # Recalculate max DD
        equity = 100000
        peak = 100000
        max_dd_after = 0
        for trade in sorted_trades:
            equity = equity * (1 + trade.pnl_net / 100)
            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100
            max_dd_after = max(max_dd_after, dd)

        print(f"Drawdown Control: Max DD = {max_dd_after:.1f}%")
        return sorted_trades

    def _analyze_results(self, trades: List[EnhancedTrade], start: datetime, end: datetime) -> Dict:
        """Analyze backtest results."""
        if not trades:
            return {}

        years = (end - start).days / 365.25

        pnls = np.array([t.pnl_net for t in trades])
        pnls_gross = np.array([t.pnl_gross for t in trades])

        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        win_rate = len(wins) / len(pnls) * 100
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        gross_profit = np.sum(wins) if len(wins) > 0 else 0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 10

        expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)

        total_return_net = np.sum(pnls)
        total_return_gross = np.sum(pnls_gross)
        total_costs = total_return_gross - total_return_net

        # Equity curve
        equity = [100000]
        for t in sorted(trades, key=lambda x: x.exit_date):
            equity.append(equity[-1] * (1 + t.pnl_net/100))
        equity = np.array(equity)

        # Drawdowns
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max * 100
        max_dd = abs(np.min(drawdowns))

        # Risk metrics
        returns = np.diff(equity) / equity[:-1]
        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)
        var_95 = calculate_var(returns) * 100
        cvar_95 = calculate_cvar(returns) * 100

        cagr = ((equity[-1] / equity[0]) ** (1/years) - 1) * 100 if years > 0 else 0
        calmar = cagr / max_dd if max_dd > 0 else 0

        # MAE/MFE
        maes = [t.mae for t in trades if t.mae > 0]
        mfes = [t.mfe for t in trades if t.mfe > 0]
        avg_mae = np.mean(maes) if maes else 0
        avg_mfe = np.mean(mfes) if mfes else 0

        # Statistical significance
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(pnls, 0)

        # Monte Carlo
        trade_dicts = [{'pnl_pct': t.pnl_net} for t in trades]
        mc = monte_carlo_simulation(trade_dicts, num_simulations=5000, num_trades=100)

        # Bootstrap validation
        robustness = validate_strategy_robustness(trade_dicts)

        # Average signals
        avg_tech = np.mean([t.tech_signals for t in trades])
        avg_fund = np.mean([t.fund_signals for t in trades])
        avg_total = np.mean([t.total_signals for t in trades])

        # Print results
        print(f"\n{'='*80}")
        print("ENHANCED BACKTEST RESULTS (24 SIGNALS)")
        print(f"{'='*80}")

        print(f"\n📊 PERFORMANCE METRICS (NET OF COSTS)")
        print(f"-" * 50)
        print(f"Win Rate:           {win_rate:.1f}%")
        print(f"Total Trades:       {len(trades)}")
        print(f"Profit Factor:      {profit_factor:.2f}")
        print(f"Expectancy:         {expectancy:+.3f}% per trade")
        print(f"Total Return (Net): {total_return_net:+.1f}%")
        print(f"Total Return (Gross): {total_return_gross:+.1f}%")
        print(f"Transaction Costs:  {total_costs:.1f}%")
        print(f"CAGR:               {cagr:+.1f}%")

        print(f"\n📉 RISK METRICS")
        print(f"-" * 50)
        print(f"Sharpe Ratio:       {sharpe:.2f}")
        print(f"Sortino Ratio:      {sortino:.2f}")
        print(f"Calmar Ratio:       {calmar:.2f}")
        print(f"Max Drawdown:       {max_dd:.1f}%")
        print(f"VaR 95%:            {var_95:.2f}%")
        print(f"CVaR 95%:           {cvar_95:.2f}%")

        print(f"\n📐 SIGNAL QUALITY")
        print(f"-" * 50)
        print(f"Avg Tech Signals:   {avg_tech:.1f}/16")
        print(f"Avg Fund Signals:   {avg_fund:.1f}/8")
        print(f"Avg Total Signals:  {avg_total:.1f}/24")

        print(f"\n📐 STATISTICAL VALIDATION")
        print(f"-" * 50)
        print(f"T-Statistic:        {t_stat:.2f}")
        print(f"P-Value:            {p_value:.4f}")
        print(f"Significant:        {'Yes ✓' if p_value < 0.05 else 'No'}")

        print(f"\n🎲 MONTE CARLO (100 trades, 5000 sims)")
        print(f"-" * 50)
        if mc:
            print(f"Median Return:      {mc.median_return:+.1f}%")
            print(f"95% CI:             ({mc.confidence_interval_95[0]:+.1f}%, {mc.confidence_interval_95[1]:+.1f}%)")
            print(f"Prob of Profit:     {mc.prob_profit:.1f}%")
            print(f"VaR 95% (MC):       {mc.var_95_mc:+.1f}%")

        print(f"\n🎯 TRADE QUALITY")
        print(f"-" * 50)
        print(f"Avg Win:            {avg_win:+.2f}%")
        print(f"Avg Loss:           {avg_loss:.2f}%")
        print(f"Largest Win:        {np.max(pnls):+.2f}%")
        print(f"Largest Loss:       {np.min(pnls):.2f}%")
        print(f"Avg MAE:            {avg_mae:.2f}%")
        print(f"Avg MFE:            {avg_mfe:.2f}%")
        print(f"Avg Hold Days:      {np.mean([t.days for t in trades]):.1f}")

        # Calculate score
        mc_prob = mc.prob_profit if mc else 50.0
        score = self._calculate_score(
            win_rate, sharpe, sortino, calmar, max_dd,
            profit_factor, expectancy, p_value, mc_prob
        )

        print(f"\n{'='*80}")
        print(f"SYSTEM SCORE: {score}/100")
        print(f"{'='*80}")

        if score >= 95:
            print("🏆 TOP HEDGE FUND LEVEL")
        elif score >= 85:
            print("⭐ INSTITUTIONAL/QUANT LEVEL")
        elif score >= 75:
            print("📈 SEMI-PROFESSIONAL LEVEL")
        else:
            print("📊 ADVANCED RETAIL LEVEL")

        return {
            'trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'total_return_net': total_return_net,
            'cagr': cagr,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            't_stat': t_stat,
            'p_value': p_value,
            'mc_prob_profit': mc.prob_profit if mc else 50.0,
            'score': score,
            'avg_tech_signals': avg_tech,
            'avg_fund_signals': avg_fund,
            'avg_total_signals': avg_total,
        }

    def _calculate_score(self, win_rate, sharpe, sortino, calmar, max_dd,
                        profit_factor, expectancy, p_value, mc_prob) -> int:
        """Calculate system score."""
        score = 0

        # Win Rate (max 15)
        if win_rate >= 80: score += 15
        elif win_rate >= 70: score += 12
        elif win_rate >= 60: score += 9
        elif win_rate >= 50: score += 6

        # Sharpe (max 15)
        if sharpe >= 2.5: score += 15
        elif sharpe >= 2.0: score += 12
        elif sharpe >= 1.5: score += 9
        elif sharpe >= 1.0: score += 6

        # Sortino (max 10)
        if sortino >= 3.5: score += 10
        elif sortino >= 2.5: score += 8
        elif sortino >= 1.5: score += 5

        # Calmar (max 10)
        if calmar >= 2.0: score += 10
        elif calmar >= 1.5: score += 8
        elif calmar >= 1.0: score += 5

        # Max DD (max 10)
        if max_dd <= 10: score += 10
        elif max_dd <= 15: score += 8
        elif max_dd <= 20: score += 5

        # Profit Factor (max 10)
        if profit_factor >= 2.5: score += 10
        elif profit_factor >= 2.0: score += 8
        elif profit_factor >= 1.5: score += 5

        # Expectancy (max 10)
        if expectancy >= 0.8: score += 10
        elif expectancy >= 0.5: score += 8
        elif expectancy >= 0.3: score += 5

        # P-Value (max 10)
        if p_value < 0.001: score += 10
        elif p_value < 0.01: score += 8
        elif p_value < 0.05: score += 5

        # MC Prob (max 10)
        if mc_prob >= 98: score += 10
        elif mc_prob >= 95: score += 8
        elif mc_prob >= 90: score += 5

        return min(score, 100)


def run_enhanced_optimization(symbols: List[str], years: int = 10):
    """Run optimization with ALL signals."""
    print("\n" + "="*80)
    print("ENHANCED OPTIMIZATION - 24 SIGNAL SYSTEM")
    print("="*80)

    end_date = datetime(2026, 1, 31)
    start_date = datetime(2026 - years, 1, 1)

    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Stocks: {len(symbols)}")

    best_score = 0
    best_config = None
    best_results = None
    all_results = []

    # Enhanced parameter space
    param_grid = {
        'target_mult': [5.0, 6.0, 7.0, 8.0],
        'stop_mult': [2.0, 2.5, 3.0],
        'min_tech_signals': [11, 12, 13, 14],      # Out of 16
        'min_fund_signals': [4, 5, 6],              # Out of 8
        'min_total_signals': [16, 17, 18, 19],      # Out of 24
        'min_piotroski': [6, 7, 8],
        'max_vol': [1.8, 2.0, 2.5],
        'require_supertrend': [True, False],
    }

    # Generate combinations
    all_combos = list(product(
        param_grid['target_mult'],
        param_grid['stop_mult'],
        param_grid['min_tech_signals'],
        param_grid['min_fund_signals'],
        param_grid['min_total_signals'],
        param_grid['min_piotroski'],
        param_grid['max_vol'],
        param_grid['require_supertrend'],
    ))

    print(f"Total combinations: {len(all_combos)}")

    # Sample for faster testing
    import random
    if len(all_combos) > 200:
        sampled = random.sample(all_combos, 200)
        # Add known good configs
        known_good = [
            (6.0, 3.0, 13, 5, 18, 7, 2.0, True),
            (7.0, 3.0, 12, 5, 17, 7, 2.0, True),
            (6.0, 2.5, 14, 6, 19, 7, 1.8, True),
            (5.0, 2.5, 12, 5, 17, 6, 2.5, False),
        ]
        for kg in known_good:
            if kg not in sampled:
                sampled.append(kg)
    else:
        sampled = all_combos

    print(f"Testing {len(sampled)} configurations...")
    print("-"*80)

    for i, combo in enumerate(sampled):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(sampled)} (Best: {best_score})")

        config = EnhancedConfig(
            target_mult=combo[0],
            stop_mult=combo[1],
            min_tech_signals=combo[2],
            min_fund_signals=combo[3],
            min_total_signals=combo[4],
            min_piotroski=combo[5],
            max_vol=combo[6],
            require_supertrend=combo[7],
            include_costs=True,
            use_regime=True,
        )

        bt = EnhancedBacktester(config)

        # Test on subset first
        test_symbols = symbols[:30]
        results = bt.run_backtest(test_symbols, start_date, end_date)

        if results:
            all_results.append({
                'config': combo,
                'score': results.get('score', 0),
                'win_rate': results.get('win_rate', 0),
                'sharpe': results.get('sharpe', 0),
                'expectancy': results.get('expectancy', 0),
                'trades': results.get('trades', 0),
                'avg_signals': results.get('avg_total_signals', 0),
            })

            if results.get('score', 0) > best_score:
                best_score = results['score']
                best_config = config
                best_results = results
                print(f"\n✨ NEW BEST: Score={best_score}")
                print(f"   Tech>={combo[2]}/16, Fund>={combo[3]}/8, Total>={combo[4]}/24")
                print(f"   Target={combo[0]}x, Stop={combo[1]}x, Piotroski>={combo[5]}")

    # Final results
    print(f"\n{'='*80}")
    print("TOP 10 CONFIGURATIONS")
    print("="*80)

    sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:10]
    for i, r in enumerate(sorted_results, 1):
        print(f"{i}. Score: {r['score']} | WR: {r['win_rate']:.1f}% | Sharpe: {r['sharpe']:.2f} | "
              f"Trades: {r['trades']} | Avg Signals: {r['avg_signals']:.1f}/24")
        print(f"   Config: Target={r['config'][0]}x, Stop={r['config'][1]}x, "
              f"Tech>={r['config'][2]}, Fund>={r['config'][3]}, Total>={r['config'][4]}")

    return best_config, best_results


if __name__ == "__main__":
    import sys

    from nse_tickers import fetch_nse_tickers

    # Determine index from command-line arguments
    if "--nifty500" in sys.argv:
        index_name = "NIFTY 500"
    elif "--nifty200" in sys.argv:
        index_name = "NIFTY 200"
    elif "--nifty100" in sys.argv:
        index_name = "NIFTY 100"
    else:
        index_name = "NIFTY 50"

    print(f"Fetching {index_name} stocks...")
    symbols = fetch_nse_tickers(index_name)
    symbols = [s for s in symbols if not s.startswith("NIFTY")]
    if not symbols:
        print(f"Failed to fetch {index_name} stocks from NSE API.")
        sys.exit(1)
    print(f"Loaded {len(symbols)} stocks from {index_name}")

    print("\n" + "="*80)
    print("ENHANCED 24-SIGNAL BACKTESTING SYSTEM")
    print("="*80)

    run_enhanced_optimization(symbols, years=10)
