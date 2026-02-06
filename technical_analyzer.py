"""
Technical Analysis for Swing Trading

Includes:
- Basic indicators (RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX)
- SuperTrend indicator (popular trend-following indicator)
- Candlestick pattern detection (Hammer, Engulfing, Doji, Morning Star)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class TechnicalScore:
    """Technical analysis score breakdown."""
    symbol: str
    total_score: float  # 0-100
    trend_score: float
    momentum_score: float
    volatility_score: float
    volume_score: float
    signals: list
    indicators: Dict
    swing_signal: str  # BUY, SELL, HOLD
    candlestick_patterns: List[str] = field(default_factory=list)
    supertrend_signal: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD, Signal line, and Histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std: int = 2):
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std_dev = prices.rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, sma, lower


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                         k_period: int = 14, d_period: int = 3):
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k, d


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)

    tr = calculate_atr(high, low, close, 1) * period  # Simplified TR
    atr = calculate_atr(high, low, close, period)

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return adx


def calculate_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate SuperTrend indicator.

    SuperTrend is a popular trend-following indicator that uses ATR
    to determine trend direction and potential reversal points.

    Args:
        high, low, close: Price series
        period: ATR period (default 10)
        multiplier: ATR multiplier (default 3.0)

    Returns:
        (supertrend_line, direction) where direction is 1 (bullish) or -1 (bearish)
    """
    hl2 = (high + low) / 2

    # Calculate ATR
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # Basic bands
    upper_basic = hl2 + (multiplier * atr)
    lower_basic = hl2 - (multiplier * atr)

    # Initialize
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=float)

    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()

    for i in range(period, len(close)):
        # Upper band
        if upper_basic.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
            upper_band.iloc[i] = upper_basic.iloc[i]
        else:
            upper_band.iloc[i] = upper_band.iloc[i-1]

        # Lower band
        if lower_basic.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
            lower_band.iloc[i] = lower_basic.iloc[i]
        else:
            lower_band.iloc[i] = lower_band.iloc[i-1]

        # Supertrend direction
        if i == period:
            direction.iloc[i] = 1
        elif supertrend.iloc[i-1] == upper_band.iloc[i-1]:
            direction.iloc[i] = -1 if close.iloc[i] > upper_band.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if close.iloc[i] < lower_band.iloc[i] else -1

        supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

    return supertrend, direction


def detect_candlestick_patterns(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> Tuple[List[str], int]:
    """
    Detect common candlestick patterns on recent candles.

    Detects:
    - Doji (indecision)
    - Hammer (bullish reversal)
    - Inverted Hammer (bullish reversal)
    - Bullish Engulfing (strong bullish)
    - Bearish Engulfing (strong bearish)
    - Morning Star (bullish reversal)
    - Evening Star (bearish reversal)

    Returns:
        (list of pattern names, score adjustment)
    """
    patterns = []
    score_adj = 0

    if len(close) < 3:
        return patterns, score_adj

    # Current and previous candles
    o, h, l, c = open_.iloc[-1], high.iloc[-1], low.iloc[-1], close.iloc[-1]
    o1, h1, l1, c1 = open_.iloc[-2], high.iloc[-2], low.iloc[-2], close.iloc[-2]
    o2, h2, l2, c2 = open_.iloc[-3], high.iloc[-3], low.iloc[-3], close.iloc[-3]

    body = abs(c - o)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    total_range = h - l if h != l else 0.001

    body1 = abs(c1 - o1)

    # === SINGLE CANDLE PATTERNS ===

    # Doji (very small body)
    if body < total_range * 0.1:
        patterns.append("Doji (indecision)")

    # Hammer (bullish reversal after downtrend)
    # Small body at top, long lower shadow
    if (lower_shadow > body * 2 and
        upper_shadow < body * 0.5 and
        c1 < o1):  # Previous candle was bearish
        patterns.append("Hammer (bullish reversal)")
        score_adj += 5

    # Inverted Hammer (bullish reversal)
    # Small body at bottom, long upper shadow
    if (upper_shadow > body * 2 and
        lower_shadow < body * 0.5 and
        c1 < o1):
        patterns.append("Inverted Hammer (potential bullish)")
        score_adj += 3

    # === TWO CANDLE PATTERNS ===

    # Bullish Engulfing
    if (c > o and  # Current is bullish
        c1 < o1 and  # Previous is bearish
        o < c1 and c > o1 and  # Current body engulfs previous
        body > body1 * 1.2):
        patterns.append("Bullish Engulfing (strong bullish)")
        score_adj += 8

    # Bearish Engulfing
    if (c < o and  # Current is bearish
        c1 > o1 and  # Previous is bullish
        o > c1 and c < o1 and  # Current body engulfs previous
        body > body1 * 1.2):
        patterns.append("Bearish Engulfing (strong bearish)")
        score_adj -= 8

    # === THREE CANDLE PATTERNS ===

    # Morning Star (bullish reversal)
    body2 = abs(c2 - o2)
    if (c2 < o2 and  # First: bearish
        body1 < body2 * 0.3 and  # Second: small body (star)
        c > o and  # Third: bullish
        c > (o2 + c2) / 2):  # Third closes above midpoint of first
        patterns.append("Morning Star (bullish reversal)")
        score_adj += 10

    # Evening Star (bearish reversal)
    if (c2 > o2 and  # First: bullish
        body1 < body2 * 0.3 and  # Second: small body (star)
        c < o and  # Third: bearish
        c < (o2 + c2) / 2):  # Third closes below midpoint of first
        patterns.append("Evening Star (bearish reversal)")
        score_adj -= 10

    return patterns, score_adj


def analyze_technicals(symbol: str, df: pd.DataFrame) -> Optional[TechnicalScore]:
    """
    Perform technical analysis for swing trading.

    Args:
        symbol: Stock symbol
        df: DataFrame with Date, Open, High, Low, Close, Volume

    Returns:
        TechnicalScore object
    """
    if df is None or len(df) < 30:
        return None

    signals = []
    indicators = {}

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    current_price = close.iloc[-1]

    # === TREND ANALYSIS (30 points) ===
    trend_score = 0

    # Moving Averages
    sma_20 = close.rolling(window=20).mean()
    sma_50 = close.rolling(window=50).mean() if len(close) >= 50 else sma_20
    sma_200 = close.rolling(window=200).mean() if len(close) >= 200 else None

    indicators['SMA_20'] = sma_20.iloc[-1]
    indicators['SMA_50'] = sma_50.iloc[-1]
    indicators['SMA_200'] = sma_200.iloc[-1] if sma_200 is not None else None
    indicators['price'] = current_price
    indicators['data_days'] = len(close)

    # Price above SMA20 (10 points)
    if current_price > sma_20.iloc[-1]:
        trend_score += 10
        signals.append("Price above SMA20 (bullish)")

    # Price above SMA50 (10 points)
    if len(close) >= 50 and current_price > sma_50.iloc[-1]:
        trend_score += 10
        signals.append("Price above SMA50 (uptrend)")

    # SMA20 > SMA50 (Golden cross potential) (10 points)
    if len(close) >= 50 and sma_20.iloc[-1] > sma_50.iloc[-1]:
        trend_score += 10
        signals.append("SMA20 > SMA50 (bullish alignment)")

    # BONUS: SMA200 analysis (if available - 1 year data)
    if sma_200 is not None and not pd.isna(sma_200.iloc[-1]):
        if current_price > sma_200.iloc[-1]:
            signals.append("Price above SMA200 (long-term bullish)")
            indicators['long_term_trend'] = 'BULLISH'
        else:
            signals.append("Price below SMA200 (long-term bearish)")
            indicators['long_term_trend'] = 'BEARISH'

    # SuperTrend Analysis
    supertrend_signal = "NEUTRAL"
    if len(close) >= 15:
        try:
            st_line, st_dir = calculate_supertrend(high, low, close)
            indicators['SuperTrend'] = st_line.iloc[-1]
            indicators['SuperTrend_Dir'] = st_dir.iloc[-1]

            if st_dir.iloc[-1] == 1:
                supertrend_signal = "BULLISH"
                signals.append("SuperTrend bullish (price above ST line)")
                trend_score = min(30, trend_score + 3)  # Bonus points
            elif st_dir.iloc[-1] == -1:
                supertrend_signal = "BEARISH"
                signals.append("SuperTrend bearish (price below ST line)")
        except Exception:
            pass  # Skip if calculation fails

    # === MOMENTUM ANALYSIS (30 points) ===
    momentum_score = 0

    # RSI
    rsi = calculate_rsi(close)
    current_rsi = rsi.iloc[-1]
    indicators['RSI'] = current_rsi

    if 40 <= current_rsi <= 60:
        momentum_score += 10
        signals.append(f"RSI neutral zone: {current_rsi:.1f}")
    elif 30 <= current_rsi < 40:
        momentum_score += 15
        signals.append(f"RSI oversold bounce: {current_rsi:.1f} (BUY signal)")
    elif 60 < current_rsi <= 70:
        momentum_score += 8
        signals.append(f"RSI strong momentum: {current_rsi:.1f}")
    elif current_rsi < 30:
        momentum_score += 5
        signals.append(f"RSI oversold: {current_rsi:.1f} (wait for reversal)")
    elif current_rsi > 70:
        signals.append(f"RSI overbought: {current_rsi:.1f} (caution)")

    # MACD
    macd, signal_line, histogram = calculate_macd(close)
    indicators['MACD'] = macd.iloc[-1]
    indicators['MACD_Signal'] = signal_line.iloc[-1]
    indicators['MACD_Hist'] = histogram.iloc[-1]

    if macd.iloc[-1] > signal_line.iloc[-1]:
        momentum_score += 10
        signals.append("MACD bullish crossover")
    if histogram.iloc[-1] > 0 and histogram.iloc[-2] < 0:
        momentum_score += 10
        signals.append("MACD histogram turned positive (momentum shift)")

    # === VOLATILITY ANALYSIS (20 points) ===
    volatility_score = 0

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
    indicators['BB_Upper'] = bb_upper.iloc[-1]
    indicators['BB_Middle'] = bb_middle.iloc[-1]
    indicators['BB_Lower'] = bb_lower.iloc[-1]

    bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
    indicators['BB_Width'] = bb_width

    # Price near lower band (potential swing entry)
    bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
    indicators['BB_Position'] = bb_position

    if bb_position < 0.2:
        volatility_score += 15
        signals.append("Price near lower Bollinger Band (potential entry)")
    elif bb_position < 0.4:
        volatility_score += 10
        signals.append("Price in lower Bollinger zone")
    elif 0.4 <= bb_position <= 0.6:
        volatility_score += 5

    # ATR for stop-loss calculation
    atr = calculate_atr(high, low, close)
    indicators['ATR'] = atr.iloc[-1]
    indicators['ATR_Pct'] = (atr.iloc[-1] / current_price) * 100

    if indicators['ATR_Pct'] < 3:
        volatility_score += 5
        signals.append(f"Low volatility: ATR {indicators['ATR_Pct']:.1f}%")

    # === VOLUME ANALYSIS (20 points) ===
    volume_score = 0

    avg_volume = volume.rolling(window=20).mean()
    current_volume = volume.iloc[-1]
    volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1

    indicators['Volume'] = current_volume
    indicators['Avg_Volume'] = avg_volume.iloc[-1]
    indicators['Volume_Ratio'] = volume_ratio

    if volume_ratio > 1.5:
        volume_score += 15
        signals.append(f"High volume: {volume_ratio:.1f}x average (strong interest)")
    elif volume_ratio > 1:
        volume_score += 10
        signals.append("Above average volume")
    elif volume_ratio > 0.7:
        volume_score += 5

    # Volume trend
    vol_sma_5 = volume.rolling(window=5).mean()
    vol_sma_20 = volume.rolling(window=20).mean()
    if vol_sma_5.iloc[-1] > vol_sma_20.iloc[-1]:
        volume_score += 5
        signals.append("Rising volume trend")

    # === CANDLESTICK PATTERN DETECTION ===
    candlestick_patterns = []
    candle_score_adj = 0
    if 'Open' in df.columns:
        try:
            candlestick_patterns, candle_score_adj = detect_candlestick_patterns(
                df['Open'], high, low, close
            )
            for pattern in candlestick_patterns:
                signals.append(f"Candlestick: {pattern}")
        except Exception:
            pass

    # Calculate total score (with candlestick adjustment)
    total_score = trend_score + momentum_score + volatility_score + volume_score
    total_score = max(0, min(100, total_score + candle_score_adj))

    # Determine swing signal
    if total_score >= 70 and current_rsi < 65:
        swing_signal = "BUY"
    elif total_score >= 55:
        swing_signal = "HOLD"
    else:
        swing_signal = "WAIT"

    # Override for overbought
    if current_rsi > 75:
        swing_signal = "AVOID"

    return TechnicalScore(
        symbol=symbol,
        total_score=total_score,
        trend_score=trend_score,
        momentum_score=momentum_score,
        volatility_score=volatility_score,
        volume_score=volume_score,
        signals=signals,
        indicators=indicators,
        swing_signal=swing_signal,
        candlestick_patterns=candlestick_patterns,
        supertrend_signal=supertrend_signal
    )


def calculate_swing_targets(df: pd.DataFrame, tech_score: TechnicalScore) -> Dict:
    """
    Calculate entry, target, and stop-loss for multi-week swing trade.

    OPTIMIZED PARAMETERS (10-year backtest, Score=74/100):
    - Stop: 3.0x ATR (wide to avoid noise)
    - Target: 6.0x ATR (2:1 R:R for meaningful profits)
    - Expected hold: 2-3 months

    Returns:
        Dict with entry, target, stop_loss, risk_reward
    """
    close = df['Close']
    high = df['High']
    low = df['Low']

    current_price = close.iloc[-1]
    atr = tech_score.indicators.get('ATR', current_price * 0.02)

    # Entry: Current price
    entry = current_price

    # Stop Loss: 3x ATR below entry (wide stop for multi-week holding)
    recent_low = low.tail(20).min()
    stop_loss = max(entry - (3.0 * atr), recent_low * 0.95)

    # Target: 6x ATR above entry (for ~10-15% upside)
    target = entry + (6.0 * atr)

    # Alternative target: Recent resistance if higher
    recent_high = high.tail(30).max()
    if recent_high > target:
        target = recent_high * 0.98

    risk_reward = (target - entry) / (entry - stop_loss) if entry > stop_loss else 0

    return {
        'entry': round(entry, 2),
        'target': round(target, 2),
        'stop_loss': round(stop_loss, 2),
        'risk': round(entry - stop_loss, 2),
        'reward': round(target - entry, 2),
        'risk_reward': round(risk_reward, 2),
        'upside_pct': round(((target - entry) / entry) * 100, 2)
    }


if __name__ == "__main__":
    from fetch_stock_data import fetch_stock_history

    print("Testing technical analysis (with SuperTrend + Candlestick patterns)...")

    test_symbols = ["RELIANCE", "TCS", "INFY"]

    for symbol in test_symbols:
        df = fetch_stock_history(symbol)  # Uses default 365 days
        if df is not None:
            score = analyze_technicals(symbol, df)
            if score:
                print(f"\n{'='*55}")
                print(f"{symbol}: {score.total_score:.1f}/100 - {score.swing_signal}")
                print(f"{'='*55}")
                print(f"  Trend:       {score.trend_score}/30")
                print(f"  Momentum:    {score.momentum_score}/30")
                print(f"  Volatility:  {score.volatility_score}/20")
                print(f"  Volume:      {score.volume_score}/20")
                print(f"  SuperTrend:  {score.supertrend_signal}")
                if score.candlestick_patterns:
                    print(f"  Patterns:    {', '.join(score.candlestick_patterns)}")
                print(f"\nSignals:")
                for sig in score.signals[:6]:
                    print(f"  • {sig}")

                targets = calculate_swing_targets(df, score)
                print(f"\nSwing Trade Setup:")
                print(f"  Entry: ₹{targets['entry']}")
                print(f"  Target: ₹{targets['target']} (+{targets['upside_pct']}%)")
                print(f"  Stop Loss: ₹{targets['stop_loss']}")
                print(f"  Risk:Reward = 1:{targets['risk_reward']}")
