"""
Multibagger Stock Screener for NSE

Identifies stocks with 2x-10x potential over 1-5 years using:
- Fundamental Analysis (70% weight): Quality, Growth, Value, Financial Strength
- Technical Analysis (30% weight): Trend, Momentum, Accumulation

Scoring: 0-100 composite (70 fundamental + 30 technical)
Categories:
  - Strong Multibagger (>=75)
  - Potential Multibagger (>=60)
  - Watchlist (>=45)
  - Avoid (<45)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import time

from nse_tickers import fetch_nse_tickers
from fetch_stock_data import fetch_stock_history, fetch_multiple_stocks
from fundamental_analyzer import get_fundamentals, calculate_piotroski_score
from technical_analyzer import analyze_technicals, TechnicalScore


# =============================================================================
# DATACLASS
# =============================================================================

@dataclass
class MultibaggerScore:
    """Multibagger screening score breakdown."""
    symbol: str
    company_name: str
    sector: str
    industry: str
    current_price: float
    market_cap_cr: float  # Market cap in Crores
    market_cap_category: str  # "Small Cap", "Mid Cap", "Large Cap"

    # Composite
    total_score: float  # 0-100
    category: str  # "Strong Multibagger", "Potential Multibagger", "Watchlist", "Avoid"

    # Fundamental breakdown (0-70)
    fundamental_total: float
    quality_score: float  # 0-25
    growth_score: float  # 0-20
    value_score: float  # 0-15
    financial_strength_score: float  # 0-10

    # Technical breakdown (0-30)
    technical_total: float
    trend_score: float  # 0-15
    momentum_score: float  # 0-10
    accumulation_score: float  # 0-5

    # Key metrics
    piotroski_score: int
    piotroski_rating: str
    pe_ratio: Optional[float]
    peg_ratio: Optional[float]
    roe_pct: Optional[float]
    debt_to_equity: Optional[float]
    revenue_growth_pct: Optional[float]
    earnings_growth_pct: Optional[float]
    rsi: float
    above_200dma: bool
    above_50dma: bool

    # Time-horizon scores (computed by classify_by_time_horizon)
    short_mid_term_score: float = 0.0
    long_term_score: float = 0.0

    # Signals
    fundamental_signals: List[str] = field(default_factory=list)
    technical_signals: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    why_multibagger: List[str] = field(default_factory=list)

    # Raw data
    fundamentals: Dict = field(default_factory=dict)
    indicators: Dict = field(default_factory=dict)


@dataclass
class TimeHorizonRecommendation:
    """Stocks classified by investment time horizon."""
    short_mid_term: List[MultibaggerScore] = field(default_factory=list)  # 1-6 months
    long_term: List[MultibaggerScore] = field(default_factory=list)       # 1-5 years
    short_mid_reasons: Dict = field(default_factory=dict)  # symbol -> List[str]
    long_term_reasons: Dict = field(default_factory=dict)  # symbol -> List[str]


# =============================================================================
# FUNDAMENTAL SCORING (0-70 POINTS)
# =============================================================================

def calculate_multibagger_fundamental_score(
    fundamentals: Dict,
) -> Tuple[float, float, float, float, float, List[str], List[str], List[str]]:
    """
    Calculate multibagger fundamental score (0-70 points).

    Returns:
        (total, quality, growth, value, strength, signals, risk_factors, why_reasons)
    """
    signals = []
    risk_factors = []
    why_reasons = []

    # === QUALITY SCORE (0-25 points) ===
    quality = 0

    # ROE (10 points)
    roe = fundamentals.get('roe')
    if roe is not None:
        roe_pct = roe * 100
        if roe_pct > 20:
            quality += 10
            signals.append(f"Excellent ROE: {roe_pct:.1f}% (>20%)")
            why_reasons.append(f"High return on equity ({roe_pct:.1f}%) - efficient capital allocation")
        elif roe_pct > 15:
            quality += 7
            signals.append(f"Strong ROE: {roe_pct:.1f}% (>15%)")
            why_reasons.append(f"Strong ROE of {roe_pct:.1f}% indicates quality business")
        elif roe_pct > 10:
            quality += 4
            signals.append(f"Decent ROE: {roe_pct:.1f}%")
        elif roe_pct > 5:
            quality += 2
        else:
            risk_factors.append(f"Low ROE: {roe_pct:.1f}% - weak profitability")
    else:
        risk_factors.append("ROE data unavailable")

    # ROA (7 points)
    roa = fundamentals.get('roa')
    if roa is not None:
        roa_pct = roa * 100
        if roa_pct > 10:
            quality += 7
            signals.append(f"Excellent ROA: {roa_pct:.1f}%")
        elif roa_pct > 5:
            quality += 5
            signals.append(f"Good ROA: {roa_pct:.1f}%")
        elif roa_pct > 3:
            quality += 3
        elif roa_pct > 0:
            quality += 1

    # Operating Margin (8 points)
    op_margin = fundamentals.get('operating_margin')
    if op_margin is not None:
        op_pct = op_margin * 100
        if op_pct > 20:
            quality += 8
            signals.append(f"High operating margin: {op_pct:.1f}%")
            why_reasons.append(f"Wide operating margin ({op_pct:.1f}%) - strong pricing power")
        elif op_pct > 15:
            quality += 6
            signals.append(f"Good operating margin: {op_pct:.1f}%")
        elif op_pct > 10:
            quality += 4
        elif op_pct > 5:
            quality += 2
        else:
            risk_factors.append(f"Thin operating margin: {op_pct:.1f}%")

    # === GROWTH SCORE (0-20 points) ===
    growth = 0

    # Revenue Growth (8 points)
    rev_growth = fundamentals.get('revenue_growth')
    if rev_growth is not None:
        rev_pct = rev_growth * 100
        if rev_pct > 25:
            growth += 8
            signals.append(f"Explosive revenue growth: {rev_pct:.1f}%")
            why_reasons.append(f"Revenue growing at {rev_pct:.1f}% - rapid business expansion")
        elif rev_pct > 15:
            growth += 6
            signals.append(f"Strong revenue growth: {rev_pct:.1f}%")
            why_reasons.append(f"Healthy revenue growth of {rev_pct:.1f}%")
        elif rev_pct > 10:
            growth += 4
            signals.append(f"Moderate revenue growth: {rev_pct:.1f}%")
        elif rev_pct > 5:
            growth += 2
        elif rev_pct < 0:
            risk_factors.append(f"Revenue declining: {rev_pct:.1f}%")
    else:
        risk_factors.append("Revenue growth data unavailable")

    # Earnings Growth (12 points)
    earn_growth = fundamentals.get('earnings_growth')
    if earn_growth is not None:
        earn_pct = earn_growth * 100
        if earn_pct > 30:
            growth += 12
            signals.append(f"Exceptional earnings growth: {earn_pct:.1f}%")
            why_reasons.append(f"Earnings surging {earn_pct:.1f}% - strong compounding potential")
        elif earn_pct > 20:
            growth += 9
            signals.append(f"Strong earnings growth: {earn_pct:.1f}%")
            why_reasons.append(f"Earnings growing {earn_pct:.1f}% - compounding machine")
        elif earn_pct > 15:
            growth += 6
            signals.append(f"Good earnings growth: {earn_pct:.1f}%")
        elif earn_pct > 10:
            growth += 3
        elif earn_pct < 0:
            risk_factors.append(f"Earnings declining: {earn_pct:.1f}%")
    else:
        risk_factors.append("Earnings growth data unavailable")

    # === VALUE SCORE (0-15 points) ===
    value = 0

    # PEG Ratio (7 points)
    peg = fundamentals.get('peg_ratio')
    if peg is not None and peg > 0:
        if peg < 1:
            value += 7
            signals.append(f"Excellent PEG: {peg:.2f} - growth at bargain price")
            why_reasons.append(f"PEG ratio of {peg:.2f} - undervalued relative to growth")
        elif peg < 1.5:
            value += 5
            signals.append(f"Good PEG: {peg:.2f} - reasonable valuation")
        elif peg < 2:
            value += 3
            signals.append(f"Fair PEG: {peg:.2f}")
        elif peg > 3:
            risk_factors.append(f"Expensive PEG: {peg:.2f} - priced for perfection")
    else:
        signals.append("PEG ratio unavailable - limited valuation insight")

    # Forward PE improvement (4 points)
    pe = fundamentals.get('pe_ratio')
    forward_pe = fundamentals.get('forward_pe')
    if pe and forward_pe and forward_pe > 0:
        if forward_pe < pe:
            value += 4
            signals.append(f"Forward PE ({forward_pe:.1f}) < Trailing PE ({pe:.1f}) - earnings improving")
        elif forward_pe < 20:
            value += 2
            signals.append(f"Forward PE reasonable: {forward_pe:.1f}")
    elif pe:
        if pe < 15:
            value += 3
            signals.append(f"Attractive PE: {pe:.1f}")
        elif pe < 25:
            value += 1
        elif pe > 50:
            risk_factors.append(f"Very high PE: {pe:.1f}")

    # EV/EBITDA (4 points)
    ev_ebitda = fundamentals.get('ev_ebitda')
    if ev_ebitda is not None and ev_ebitda > 0:
        if ev_ebitda < 10:
            value += 4
            signals.append(f"Low EV/EBITDA: {ev_ebitda:.1f}")
        elif ev_ebitda < 15:
            value += 3
        elif ev_ebitda < 20:
            value += 1
        elif ev_ebitda > 30:
            risk_factors.append(f"High EV/EBITDA: {ev_ebitda:.1f}")

    # === FINANCIAL STRENGTH SCORE (0-10 points) ===
    strength = 0

    # Debt/Equity (5 points)
    de = fundamentals.get('debt_to_equity')
    if de is not None:
        if de < 30:
            strength += 5
            signals.append(f"Very low debt: D/E {de:.1f}%")
            why_reasons.append(f"Low debt (D/E {de:.1f}%) - financially strong")
        elif de < 50:
            strength += 3
            signals.append(f"Low debt: D/E {de:.1f}%")
        elif de < 100:
            strength += 1
        else:
            risk_factors.append(f"High debt: D/E {de:.1f}%")

    # Cash > Debt (3 points)
    cash = fundamentals.get('total_cash')
    debt = fundamentals.get('total_debt')
    if cash and debt:
        if cash > debt:
            strength += 3
            signals.append("Net cash positive - no debt concerns")
            why_reasons.append("Cash exceeds debt - strong balance sheet")
        elif cash > debt * 0.5:
            strength += 1
    elif cash and cash > 0 and (debt is None or debt == 0):
        strength += 3
        signals.append("Debt-free company with cash reserves")
        why_reasons.append("Zero debt - pristine balance sheet")

    # Current Ratio (2 points)
    cr = fundamentals.get('current_ratio')
    if cr is not None:
        if cr > 2:
            strength += 2
            signals.append(f"Strong liquidity: CR {cr:.2f}")
        elif cr > 1.5:
            strength += 1
        elif cr < 1:
            risk_factors.append(f"Weak liquidity: CR {cr:.2f}")

    # === BONUSES (within 70-point cap) ===
    bonus = 0

    # Piotroski F-Score bonus
    piotroski, piotroski_rating, _ = calculate_piotroski_score(fundamentals)
    if piotroski >= 7:
        bonus += 3
        signals.append(f"Strong Piotroski F-Score: {piotroski}/9")
        why_reasons.append(f"Piotroski F-Score {piotroski}/9 - academically validated quality")
    elif piotroski >= 5:
        bonus += 1
        signals.append(f"Moderate Piotroski F-Score: {piotroski}/9")

    # Market cap bonus (small/mid caps have more multibagger potential)
    market_cap = fundamentals.get('market_cap')
    if market_cap:
        market_cap_cr = market_cap / 1e7
        if market_cap_cr < 10000:
            bonus += 3
            signals.append(f"Small cap ({market_cap_cr:,.0f} Cr) - higher multibagger potential")
            why_reasons.append(f"Small cap at {market_cap_cr:,.0f} Cr - room for significant re-rating")
        elif market_cap_cr < 50000:
            bonus += 1
            signals.append(f"Mid cap ({market_cap_cr:,.0f} Cr) - growth runway available")

    total = min(70, quality + growth + value + strength + bonus)

    return total, quality, growth, value, strength, signals, risk_factors, why_reasons


# =============================================================================
# TECHNICAL SCORING (0-30 POINTS)
# =============================================================================

def calculate_multibagger_technical_score(
    tech_score: TechnicalScore,
    df: pd.DataFrame,
) -> Tuple[float, float, float, float, List[str], List[str]]:
    """
    Calculate multibagger technical score (0-30 points).

    Args:
        tech_score: Existing TechnicalScore from technical_analyzer
        df: Raw OHLCV DataFrame

    Returns:
        (total, trend, momentum, accumulation, signals, risk_factors)
    """
    signals = []
    risk_factors = []
    indicators = tech_score.indicators

    close = df['Close']
    volume = df['Volume']

    # === TREND SCORE (0-15 points) ===
    trend = 0

    # Price above 200 DMA (6 points) - most important for multibaggers
    sma_200 = indicators.get('SMA_200')
    current_price = indicators.get('price', close.iloc[-1])
    above_200dma = False

    if sma_200 is not None and not pd.isna(sma_200):
        if current_price > sma_200:
            trend += 6
            above_200dma = True
            signals.append("Price above 200 DMA - long-term uptrend confirmed")
        else:
            risk_factors.append("Price below 200 DMA - long-term trend is bearish")
    elif len(close) >= 200:
        sma_200_calc = close.rolling(window=200).mean().iloc[-1]
        if current_price > sma_200_calc:
            trend += 6
            above_200dma = True
            signals.append("Price above 200 DMA - long-term uptrend confirmed")
        else:
            risk_factors.append("Price below 200 DMA - long-term trend is bearish")
    else:
        signals.append("Insufficient data for 200 DMA (need 1 year history)")

    # Price above 50 DMA (4 points)
    sma_50 = indicators.get('SMA_50')
    above_50dma = False
    if sma_50 is not None and not pd.isna(sma_50):
        if current_price > sma_50:
            trend += 4
            above_50dma = True
            signals.append("Price above 50 DMA - medium-term uptrend")
        else:
            risk_factors.append("Price below 50 DMA - medium-term weakness")

    # SuperTrend bullish (3 points)
    if tech_score.supertrend_signal == "BULLISH":
        trend += 3
        signals.append("SuperTrend bullish - trend-following confirmation")
    elif tech_score.supertrend_signal == "BEARISH":
        risk_factors.append("SuperTrend bearish")

    # Near 52-week high (2 points) - relative strength
    if len(close) >= 200:
        high_52w = close.tail(252).max() if len(close) >= 252 else close.max()
        pct_from_high = (current_price / high_52w) * 100
        if pct_from_high >= 80:
            trend += 2
            signals.append(f"Near 52-week high ({pct_from_high:.0f}%) - strong relative strength")
        elif pct_from_high < 60:
            risk_factors.append(f"Far from 52-week high ({pct_from_high:.0f}%)")

    # === MOMENTUM SCORE (0-10 points) ===
    momentum = 0

    # RSI zone (5 points)
    rsi = indicators.get('RSI', 50)
    if 50 <= rsi <= 65:
        momentum += 5
        signals.append(f"RSI in ideal zone: {rsi:.1f} (50-65)")
    elif 40 <= rsi < 50:
        momentum += 3
        signals.append(f"RSI building momentum: {rsi:.1f}")
    elif 65 < rsi <= 70:
        momentum += 2
        signals.append(f"RSI strong but watch for overbought: {rsi:.1f}")
    elif rsi > 70:
        risk_factors.append(f"RSI overbought: {rsi:.1f} - may correct short-term")
    elif rsi < 30:
        risk_factors.append(f"RSI oversold: {rsi:.1f} - wait for reversal confirmation")

    # MACD bullish (5 points)
    macd = indicators.get('MACD')
    macd_signal = indicators.get('MACD_Signal')
    macd_hist = indicators.get('MACD_Hist')

    if macd is not None and macd_signal is not None:
        if macd > macd_signal and macd_hist is not None and macd_hist > 0:
            momentum += 5
            signals.append("MACD bullish with positive histogram")
        elif macd > macd_signal:
            momentum += 3
            signals.append("MACD bullish crossover")
        elif macd < macd_signal:
            risk_factors.append("MACD bearish")

    # === ACCUMULATION SCORE (0-5 points) ===
    accumulation = 0

    # Volume trend (3 points)
    if len(volume) >= 20:
        vol_sma_5 = volume.tail(5).mean()
        vol_sma_20 = volume.tail(20).mean()
        if vol_sma_20 > 0 and vol_sma_5 > vol_sma_20:
            accumulation += 3
            vol_ratio = vol_sma_5 / vol_sma_20
            signals.append(f"Rising volume trend ({vol_ratio:.1f}x vs 20-day avg)")
        elif vol_sma_20 > 0 and vol_sma_5 > vol_sma_20 * 0.8:
            accumulation += 1

    # Up-volume vs down-volume (2 points)
    if len(close) >= 20 and len(volume) >= 20:
        recent_close = close.tail(20)
        recent_volume = volume.tail(20)
        recent_open = df['Open'].tail(20) if 'Open' in df.columns else recent_close.shift(1)

        up_days = recent_close > recent_open
        down_days = recent_close <= recent_open

        up_volume = recent_volume[up_days].mean() if up_days.sum() > 0 else 0
        down_volume = recent_volume[down_days].mean() if down_days.sum() > 0 else 1

        if down_volume > 0 and up_volume > down_volume * 1.2:
            accumulation += 2
            signals.append("Smart money accumulation: higher volume on up days")

    total = min(30, trend + momentum + accumulation)

    return total, trend, momentum, accumulation, signals, risk_factors


# =============================================================================
# PER-STOCK ANALYSIS
# =============================================================================

def screen_stock_for_multibagger(
    symbol: str,
    hist_data: pd.DataFrame = None,
) -> Optional[MultibaggerScore]:
    """
    Analyze a single stock for multibagger potential.

    Args:
        symbol: NSE stock symbol
        hist_data: Pre-fetched OHLCV data (optional)

    Returns:
        MultibaggerScore or None if analysis fails
    """
    # Step 1: Fetch fundamentals
    fundamentals = get_fundamentals(symbol)
    if fundamentals is None:
        return None

    # Step 2: Market cap categorization
    market_cap = fundamentals.get('market_cap')
    if market_cap is None:
        return None

    market_cap_cr = market_cap / 1e7
    if market_cap_cr < 10000:
        cap_category = "Small Cap"
    elif market_cap_cr < 50000:
        cap_category = "Mid Cap"
    else:
        cap_category = "Large Cap"

    # Step 3: Fundamental scoring
    (fund_total, quality, growth, value, strength,
     fund_signals, fund_risks, why_reasons) = calculate_multibagger_fundamental_score(fundamentals)

    # Step 4: Fetch price data if not provided
    if hist_data is None:
        hist_data = fetch_stock_history(symbol, days=365)

    if hist_data is None or len(hist_data) < 30:
        return None

    # Step 5: Technical analysis
    tech_score = analyze_technicals(symbol, hist_data)
    if tech_score is None:
        return None

    # Step 6: Technical scoring
    (tech_total, trend, momentum, accumulation,
     tech_signals, tech_risks) = calculate_multibagger_technical_score(tech_score, hist_data)

    # Step 7: Composite score
    total_score = fund_total + tech_total

    # Step 8: Category
    if total_score >= 75:
        category = "Strong Multibagger"
    elif total_score >= 60:
        category = "Potential Multibagger"
    elif total_score >= 45:
        category = "Watchlist"
    else:
        category = "Avoid"

    # Step 9: Combine risk factors
    all_risks = fund_risks + tech_risks

    # Step 10: Key metrics extraction
    roe = fundamentals.get('roe')
    roe_pct = roe * 100 if roe else None
    rev_g = fundamentals.get('revenue_growth')
    rev_g_pct = rev_g * 100 if rev_g else None
    earn_g = fundamentals.get('earnings_growth')
    earn_g_pct = earn_g * 100 if earn_g else None

    piotroski, piotroski_rating, _ = calculate_piotroski_score(fundamentals)

    rsi = tech_score.indicators.get('RSI', 50)
    sma_200 = tech_score.indicators.get('SMA_200')
    sma_50 = tech_score.indicators.get('SMA_50')
    current_price = tech_score.indicators.get('price', hist_data['Close'].iloc[-1])

    above_200 = False
    if sma_200 is not None and not pd.isna(sma_200):
        above_200 = current_price > sma_200
    elif len(hist_data['Close']) >= 200:
        sma_200_calc = hist_data['Close'].rolling(200).mean().iloc[-1]
        above_200 = current_price > sma_200_calc

    above_50 = False
    if sma_50 is not None and not pd.isna(sma_50):
        above_50 = current_price > sma_50

    return MultibaggerScore(
        symbol=symbol,
        company_name=fundamentals.get('company_name', symbol),
        sector=fundamentals.get('sector', 'Unknown'),
        industry=fundamentals.get('industry', 'Unknown'),
        current_price=round(current_price, 2),
        market_cap_cr=round(market_cap_cr, 0),
        market_cap_category=cap_category,
        total_score=round(total_score, 1),
        category=category,
        fundamental_total=round(fund_total, 1),
        quality_score=round(quality, 1),
        growth_score=round(growth, 1),
        value_score=round(value, 1),
        financial_strength_score=round(strength, 1),
        technical_total=round(tech_total, 1),
        trend_score=round(trend, 1),
        momentum_score=round(momentum, 1),
        accumulation_score=round(accumulation, 1),
        piotroski_score=piotroski,
        piotroski_rating=piotroski_rating,
        pe_ratio=fundamentals.get('pe_ratio'),
        peg_ratio=fundamentals.get('peg_ratio'),
        roe_pct=round(roe_pct, 1) if roe_pct else None,
        debt_to_equity=fundamentals.get('debt_to_equity'),
        revenue_growth_pct=round(rev_g_pct, 1) if rev_g_pct else None,
        earnings_growth_pct=round(earn_g_pct, 1) if earn_g_pct else None,
        rsi=round(rsi, 1),
        above_200dma=above_200,
        above_50dma=above_50,
        fundamental_signals=fund_signals,
        technical_signals=tech_signals,
        risk_factors=all_risks,
        why_multibagger=why_reasons[:5],
        fundamentals=fundamentals,
        indicators=tech_score.indicators,
    )


# =============================================================================
# BATCH SCREENER
# =============================================================================

def run_multibagger_screener(
    symbols: List[str] = None,
    index: str = "NIFTY 500",
    top_n: int = 20,
    min_score: float = 45.0,
    min_market_cap_cr: float = 500,
    max_market_cap_cr: float = 1000000,
    sector_filter: str = None,
    max_workers: int = 5,
    progress_callback: Callable = None,
) -> List[MultibaggerScore]:
    """
    Run multibagger screener on a list of stocks.

    Args:
        symbols: List of NSE symbols (fetched from index if None)
        index: NSE index to scan (default NIFTY 500)
        top_n: Return top N results
        min_score: Minimum composite score to include
        min_market_cap_cr: Min market cap in Crores (filter micro-caps)
        max_market_cap_cr: Max market cap in Crores
        sector_filter: Only include this sector (None for all)
        max_workers: Parallel workers for data fetch
        progress_callback: fn(current, total, symbol) for progress updates

    Returns:
        List of MultibaggerScore sorted by total_score descending
    """
    # Get symbols
    if symbols is None:
        print(f"Fetching tickers from {index}...")
        symbols = fetch_nse_tickers(index)
        symbols = [s for s in symbols if s != index.replace(" ", "")]

    if not symbols:
        print("No symbols found.")
        return []

    print(f"Fetching price data for {len(symbols)} stocks...")
    stock_data = fetch_multiple_stocks(symbols, days=365, max_workers=max_workers)

    results = []
    analyzed = 0
    failed = 0
    total = len(symbols)

    print(f"\nScreening {total} stocks for multibagger potential...")

    for i, symbol in enumerate(symbols):
        if progress_callback:
            progress_callback(i + 1, total, symbol)

        try:
            hist = stock_data.get(symbol)
            score = screen_stock_for_multibagger(symbol, hist)

            if score is None:
                failed += 1
                continue

            analyzed += 1

            # Apply filters
            if score.total_score < min_score:
                continue
            if score.market_cap_cr < min_market_cap_cr:
                continue
            if score.market_cap_cr > max_market_cap_cr:
                continue
            if sector_filter and sector_filter.lower() not in score.sector.lower():
                continue

            results.append(score)

        except Exception as e:
            failed += 1
            continue

        if (i + 1) % 20 == 0:
            print(f"Progress: {i+1}/{total} | Passed: {len(results)} | Failed: {failed}")

    # Sort by score descending
    results.sort(key=lambda x: x.total_score, reverse=True)

    print(f"\nScreening complete:")
    print(f"  Analyzed: {analyzed}/{total}")
    print(f"  Failed: {failed}")
    print(f"  Passed filters: {len(results)}")

    strong = sum(1 for r in results if r.category == "Strong Multibagger")
    potential = sum(1 for r in results if r.category == "Potential Multibagger")
    watchlist = sum(1 for r in results if r.category == "Watchlist")
    print(f"  Strong: {strong} | Potential: {potential} | Watchlist: {watchlist}")

    return results[:top_n]


# =============================================================================
# TIME-HORIZON CLASSIFICATION
# =============================================================================

def classify_by_time_horizon(
    results: List[MultibaggerScore],
) -> TimeHorizonRecommendation:
    """
    Classify screener results into short-mid term and long term recommendations.

    Short-Mid Term (1-6 months):
      - 55% technical weight, 45% fundamental weight
      - Must have strong technical setup for near-term entry
      - Requires: above 50 DMA, RSI 40-70, MACD bullish

    Long Term (1-5 years):
      - 85% fundamental weight, 15% technical weight
      - Must have exceptional fundamentals for compounding
      - Requires: Piotroski >= 6, ROE > 12%

    Args:
        results: List of MultibaggerScore from screener

    Returns:
        TimeHorizonRecommendation with classified lists
    """
    short_mid = []
    long_term = []
    short_mid_reasons = {}
    long_term_reasons = {}

    for r in results:
        # ===== SHORT-MID TERM SCORING =====
        # Reweight: 55% technical + 45% fundamental
        smt_score = (r.technical_total / 30) * 55 + (r.fundamental_total / 70) * 45
        r.short_mid_term_score = round(smt_score, 1)

        smt_reasons = []
        smt_qualifies = True

        # Must-have checks
        if not r.above_50dma:
            smt_qualifies = False
        else:
            smt_reasons.append("Price above 50 DMA - medium-term uptrend intact")

        if r.rsi < 40 or r.rsi > 70:
            smt_qualifies = False
        else:
            smt_reasons.append(f"RSI at {r.rsi:.0f} - in momentum sweet zone (40-70)")

        # MACD check
        macd = r.indicators.get('MACD')
        macd_signal = r.indicators.get('MACD_Signal')
        macd_bullish = macd is not None and macd_signal is not None and macd > macd_signal
        if not macd_bullish:
            smt_qualifies = False
        else:
            smt_reasons.append("MACD bullish crossover - positive momentum")

        # Bonus reasons (enhance score but not required)
        if r.above_200dma:
            smt_score += 5
            smt_reasons.append("Above 200 DMA - long-term trend supports entry")

        supertrend = r.indicators.get('SuperTrend_Dir')
        if supertrend == 1:
            smt_score += 3
            smt_reasons.append("SuperTrend bullish - trend-following confirmation")

        if r.momentum_score >= 7:
            smt_reasons.append(f"Strong momentum score ({r.momentum_score:.0f}/10)")

        if r.accumulation_score >= 3:
            smt_reasons.append("Volume accumulation detected - smart money buying")

        # Fundamental floor for short-mid term
        if r.fundamental_total >= 35:
            smt_reasons.append(f"Solid fundamentals ({r.fundamental_total:.0f}/70) backing the technical setup")

        r.short_mid_term_score = round(min(100, smt_score), 1)

        if smt_qualifies and r.short_mid_term_score >= 50:
            short_mid.append(r)
            short_mid_reasons[r.symbol] = smt_reasons

        # ===== LONG TERM SCORING =====
        # Reweight: 85% fundamental + 15% technical
        lt_score = (r.fundamental_total / 70) * 85 + (r.technical_total / 30) * 15
        r.long_term_score = round(lt_score, 1)

        lt_reasons = []
        lt_qualifies = True

        # Must-have checks
        if r.piotroski_score < 6:
            lt_qualifies = False
        else:
            lt_reasons.append(f"Piotroski F-Score {r.piotroski_score}/9 - academically validated quality")

        roe = r.roe_pct or 0
        if roe < 12:
            lt_qualifies = False
        else:
            lt_reasons.append(f"ROE of {roe:.1f}% - strong return on equity for compounding")

        de = r.debt_to_equity
        if de is not None and de > 100:
            lt_qualifies = False
        elif de is not None and de < 50:
            lt_reasons.append(f"Low debt (D/E: {de:.0f}%) - financial strength for long haul")

        # Bonus reasons
        if r.market_cap_category in ("Small Cap", "Mid Cap"):
            lt_score += 5
            lt_reasons.append(f"{r.market_cap_category} ({r.market_cap_cr:,.0f} Cr) - significant growth runway")

        rev_g = r.revenue_growth_pct
        if rev_g is not None and rev_g > 15:
            lt_score += 3
            lt_reasons.append(f"Revenue growing {rev_g:+.1f}% - strong business expansion")

        earn_g = r.earnings_growth_pct
        if earn_g is not None and earn_g > 20:
            lt_score += 3
            lt_reasons.append(f"Earnings growing {earn_g:+.1f}% - compounding machine")

        peg = r.peg_ratio
        if peg is not None and 0 < peg < 1.5:
            lt_score += 2
            lt_reasons.append(f"PEG ratio {peg:.2f} - growth at reasonable price")

        if r.quality_score >= 18:
            lt_reasons.append(f"High quality score ({r.quality_score:.0f}/25) - durable business moat")

        if r.financial_strength_score >= 7:
            lt_reasons.append(f"Strong balance sheet ({r.financial_strength_score:.0f}/10)")

        # Cash rich
        cash = r.fundamentals.get('total_cash')
        debt = r.fundamentals.get('total_debt')
        if cash and debt and cash > debt:
            lt_reasons.append("Net cash positive - self-funded growth")

        r.long_term_score = round(min(100, lt_score), 1)

        if lt_qualifies and r.long_term_score >= 50:
            long_term.append(r)
            long_term_reasons[r.symbol] = lt_reasons

    # Sort each list by respective score
    short_mid.sort(key=lambda x: x.short_mid_term_score, reverse=True)
    long_term.sort(key=lambda x: x.long_term_score, reverse=True)

    return TimeHorizonRecommendation(
        short_mid_term=short_mid,
        long_term=long_term,
        short_mid_reasons=short_mid_reasons,
        long_term_reasons=long_term_reasons,
    )


def run_multibagger_screener_with_recommendations(
    symbols: List[str] = None,
    index: str = "NIFTY 500",
    top_n: int = 30,
    min_score: float = 45.0,
    min_market_cap_cr: float = 500,
    max_market_cap_cr: float = 1000000,
    sector_filter: str = None,
    max_workers: int = 5,
    progress_callback: Callable = None,
) -> Tuple[List[MultibaggerScore], TimeHorizonRecommendation]:
    """
    Run multibagger screener and classify results by time horizon.

    Returns:
        (all_results, time_horizon_recommendations)
    """
    results = run_multibagger_screener(
        symbols=symbols,
        index=index,
        top_n=top_n,
        min_score=min_score,
        min_market_cap_cr=min_market_cap_cr,
        max_market_cap_cr=max_market_cap_cr,
        sector_filter=sector_filter,
        max_workers=max_workers,
        progress_callback=progress_callback,
    )

    recommendations = classify_by_time_horizon(results)

    print(f"\nTime-Horizon Classification:")
    print(f"  Short-Mid Term picks: {len(recommendations.short_mid_term)}")
    print(f"  Long Term picks: {len(recommendations.long_term)}")

    return results, recommendations


# =============================================================================
# HTML REPORT CARD
# =============================================================================

def generate_multibagger_report_card(score: MultibaggerScore) -> str:
    """Generate an HTML report card for a multibagger candidate."""

    # Category badge
    badge_map = {
        "Strong Multibagger": ("mb-badge-strong", "#10b981"),
        "Potential Multibagger": ("mb-badge-potential", "#3b82f6"),
        "Watchlist": ("mb-badge-watchlist", "#f59e0b"),
        "Avoid": ("mb-badge-watchlist", "#ef4444"),
    }
    badge_class, badge_color = badge_map.get(score.category, ("mb-badge-watchlist", "#64748b"))

    # Score color
    if score.total_score >= 75:
        score_color = "#10b981"
    elif score.total_score >= 60:
        score_color = "#3b82f6"
    elif score.total_score >= 45:
        score_color = "#f59e0b"
    else:
        score_color = "#ef4444"

    # Fundamental bar width (out of 70%)
    fund_width = (score.fundamental_total / 70) * 100
    tech_width = (score.technical_total / 30) * 100

    # Key metrics
    pe_str = f"{score.pe_ratio:.1f}" if score.pe_ratio else "N/A"
    peg_str = f"{score.peg_ratio:.2f}" if score.peg_ratio else "N/A"
    roe_str = f"{score.roe_pct:.1f}%" if score.roe_pct else "N/A"
    de_str = f"{score.debt_to_equity:.1f}" if score.debt_to_equity is not None else "N/A"
    rev_str = f"{score.revenue_growth_pct:+.1f}%" if score.revenue_growth_pct is not None else "N/A"
    earn_str = f"{score.earnings_growth_pct:+.1f}%" if score.earnings_growth_pct is not None else "N/A"
    dma_200 = "Yes" if score.above_200dma else "No"
    dma_50 = "Yes" if score.above_50dma else "No"

    # Why multibagger reasons
    reasons_html = ""
    for i, reason in enumerate(score.why_multibagger[:5], 1):
        reasons_html += f'<div class="mb-reason-card">{i}. {reason}</div>'

    # Risk factors
    risks_html = ""
    for risk in score.risk_factors[:4]:
        risks_html += f'<div style="background:#fef2f2;border-left:3px solid #ef4444;border-radius:8px;padding:0.6rem 1rem;margin:0.3rem 0;font-size:0.85rem;color:#991b1b;">{risk}</div>'

    # Fundamental signals
    fund_signals_html = ""
    for sig in score.fundamental_signals[:6]:
        fund_signals_html += f"<div style='font-size:0.82rem;padding:0.2rem 0;color:#1e293b;'>&#8226; {sig}</div>"

    # Technical signals
    tech_signals_html = ""
    for sig in score.technical_signals[:6]:
        tech_signals_html += f"<div style='font-size:0.82rem;padding:0.2rem 0;color:#1e293b;'>&#8226; {sig}</div>"

    html = f"""
    <div style="background:#ffffff;border-radius:16px;padding:1.5rem;margin-bottom:1.5rem;border:1px solid #e2e8f0;box-shadow:0 4px 20px rgba(0,0,0,0.06);">
        <!-- Header -->
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
            <div>
                <div style="font-size:1.3rem;font-weight:700;color:#1e293b;">{score.symbol}</div>
                <div style="font-size:0.85rem;color:#64748b;">{score.company_name} | {score.sector}</div>
                <div style="font-size:0.8rem;color:#94a3b8;">{score.market_cap_category} | {score.market_cap_cr:,.0f} Cr</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:2.5rem;font-weight:800;color:{score_color};">{score.total_score:.0f}</div>
                <span class="{badge_class}">{score.category}</span>
            </div>
        </div>

        <!-- Score Bars -->
        <div style="margin:1rem 0;">
            <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#64748b;margin-bottom:0.3rem;">
                <span>Fundamental: {score.fundamental_total:.0f}/70</span>
                <span>Technical: {score.technical_total:.0f}/30</span>
            </div>
            <div style="display:flex;height:20px;border-radius:10px;overflow:hidden;background:#f1f5f9;">
                <div style="width:{fund_width * 0.7:.0f}%;background:linear-gradient(90deg,#3b82f6,#8b5cf6);"></div>
                <div style="width:{tech_width * 0.3:.0f}%;background:linear-gradient(90deg,#10b981,#06b6d4);"></div>
            </div>
        </div>

        <!-- Key Metrics Grid -->
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;margin:1rem 0;">
            <div style="background:#f8fafc;border-radius:8px;padding:0.6rem;text-align:center;">
                <div style="font-size:1.1rem;font-weight:700;color:#1e293b;">{roe_str}</div>
                <div style="font-size:0.7rem;color:#64748b;">ROE</div>
            </div>
            <div style="background:#f8fafc;border-radius:8px;padding:0.6rem;text-align:center;">
                <div style="font-size:1.1rem;font-weight:700;color:#1e293b;">{de_str}</div>
                <div style="font-size:0.7rem;color:#64748b;">Debt/Equity</div>
            </div>
            <div style="background:#f8fafc;border-radius:8px;padding:0.6rem;text-align:center;">
                <div style="font-size:1.1rem;font-weight:700;color:#1e293b;">{rev_str}</div>
                <div style="font-size:0.7rem;color:#64748b;">Rev Growth</div>
            </div>
            <div style="background:#f8fafc;border-radius:8px;padding:0.6rem;text-align:center;">
                <div style="font-size:1.1rem;font-weight:700;color:#1e293b;">{earn_str}</div>
                <div style="font-size:0.7rem;color:#64748b;">Earn Growth</div>
            </div>
            <div style="background:#f8fafc;border-radius:8px;padding:0.6rem;text-align:center;">
                <div style="font-size:1.1rem;font-weight:700;color:#1e293b;">{peg_str}</div>
                <div style="font-size:0.7rem;color:#64748b;">PEG Ratio</div>
            </div>
            <div style="background:#f8fafc;border-radius:8px;padding:0.6rem;text-align:center;">
                <div style="font-size:1.1rem;font-weight:700;color:#1e293b;">{score.piotroski_score}/9</div>
                <div style="font-size:0.7rem;color:#64748b;">Piotroski</div>
            </div>
            <div style="background:#f8fafc;border-radius:8px;padding:0.6rem;text-align:center;">
                <div style="font-size:1.1rem;font-weight:700;color:#1e293b;">{score.rsi:.0f}</div>
                <div style="font-size:0.7rem;color:#64748b;">RSI</div>
            </div>
            <div style="background:#f8fafc;border-radius:8px;padding:0.6rem;text-align:center;">
                <div style="font-size:1.1rem;font-weight:700;color:{'#10b981' if score.above_200dma else '#ef4444'};">{dma_200}</div>
                <div style="font-size:0.7rem;color:#64748b;">Above 200 DMA</div>
            </div>
        </div>

        <!-- Why Multibagger -->
        {"<div style='margin-top:1rem;'><div style=\"font-size:0.9rem;font-weight:600;color:#1e293b;margin-bottom:0.5rem;\">Why Multibagger Potential</div>" + reasons_html + "</div>" if reasons_html else ""}

        <!-- Risk Factors -->
        {"<div style='margin-top:1rem;'><div style=\"font-size:0.9rem;font-weight:600;color:#991b1b;margin-bottom:0.5rem;\">Risk Factors</div>" + risks_html + "</div>" if risks_html else ""}
    </div>
    """

    return html


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multibagger Stock Screener")
    parser.add_argument("--quick", action="store_true", help="Quick scan (15 popular stocks)")
    parser.add_argument("--nifty500", action="store_true", help="Full NIFTY 500 scan")
    parser.add_argument("--nifty100", action="store_true", help="NIFTY 100 scan")
    parser.add_argument("--sector", type=str, help="Filter by sector (e.g., Technology, Healthcare)")
    parser.add_argument("--smallcap", action="store_true", help="Small caps only (<10,000 Cr)")
    parser.add_argument("--midcap", action="store_true", help="Mid caps only (<50,000 Cr)")
    parser.add_argument("--min-score", type=float, default=45, help="Minimum score (default 45)")
    parser.add_argument("--top", type=int, default=20, help="Top N results (default 20)")

    args = parser.parse_args()

    # Determine stock universe
    if args.quick:
        symbols = [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
            "HINDUNILVR", "SBIN", "ITC", "BHARTIARTL", "LT",
            "SUNPHARMA", "TITAN", "BAJFINANCE", "MARUTI", "WIPRO"
        ]
        index = "QUICK"
    elif args.nifty500:
        symbols = None
        index = "NIFTY 500"
    elif args.nifty100:
        symbols = None
        index = "NIFTY 100"
    else:
        symbols = None
        index = "NIFTY 50"

    # Market cap filter
    max_cap = 1000000
    if args.smallcap:
        max_cap = 10000
    elif args.midcap:
        max_cap = 50000

    print("=" * 70)
    print("MULTIBAGGER STOCK SCREENER")
    print("=" * 70)
    print(f"Universe: {index}")
    print(f"Min Score: {args.min_score}")
    print(f"Sector: {args.sector or 'All'}")
    print(f"Max Market Cap: {max_cap:,} Cr")
    print("=" * 70)

    results = run_multibagger_screener(
        symbols=symbols,
        index=index,
        top_n=args.top,
        min_score=args.min_score,
        max_market_cap_cr=max_cap,
        sector_filter=args.sector,
    )

    if results:
        print("\n" + "=" * 70)
        print(f"TOP {len(results)} MULTIBAGGER CANDIDATES")
        print("=" * 70)

        for i, r in enumerate(results, 1):
            print(f"\n{'─' * 60}")
            print(f"#{i} {r.symbol} ({r.company_name})")
            print(f"   Score: {r.total_score:.0f}/100 | {r.category}")
            print(f"   Sector: {r.sector} | {r.market_cap_category} ({r.market_cap_cr:,.0f} Cr)")
            print(f"   Fundamental: {r.fundamental_total:.0f}/70 | Technical: {r.technical_total:.0f}/30")
            print(f"   ROE: {r.roe_pct or 'N/A'}% | D/E: {r.debt_to_equity or 'N/A'} | Piotroski: {r.piotroski_score}/9")
            print(f"   Rev Growth: {r.revenue_growth_pct or 'N/A'}% | Earn Growth: {r.earnings_growth_pct or 'N/A'}%")
            print(f"   200 DMA: {'Above' if r.above_200dma else 'Below'} | RSI: {r.rsi:.0f}")
            if r.why_multibagger:
                print(f"   Why: {r.why_multibagger[0]}")
    else:
        print("\nNo stocks matched the criteria.")
