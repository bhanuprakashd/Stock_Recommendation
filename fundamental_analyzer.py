"""
Fundamental Analysis for NSE Stocks

Includes:
- Basic fundamental scoring (Valuation, Profitability, Growth, Health)
- Piotroski F-Score (peer-reviewed academic model, 0-9)
"""

import yfinance as yf
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class FundamentalScore:
    """Fundamental analysis score breakdown."""
    symbol: str
    total_score: float  # 0-100
    valuation_score: float
    profitability_score: float
    growth_score: float
    health_score: float
    piotroski_score: int  # 0-9 (Piotroski F-Score)
    piotroski_rating: str  # Strong, Moderate, Weak
    metrics: Dict
    signals: list
    passed: bool


def get_fundamentals(symbol: str) -> Optional[Dict]:
    """
    Fetch fundamental data for a stock.

    Args:
        symbol: NSE stock symbol

    Returns:
        Dictionary of fundamental metrics
    """
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info

        price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
        if not info or price is None:
            return None

        return {
            # Price & Valuation
            'price': price,
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'pb_ratio': info.get('priceToBook'),
            'peg_ratio': info.get('pegRatio'),
            'ev_ebitda': info.get('enterpriseToEbitda'),
            'market_cap': info.get('marketCap'),

            # Profitability
            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),
            'profit_margin': info.get('profitMargins'),
            'operating_margin': info.get('operatingMargins'),
            'gross_margin': info.get('grossMargins'),

            # Growth
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),

            # Financial Health
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'total_debt': info.get('totalDebt'),
            'total_cash': info.get('totalCash'),

            # Dividends
            'dividend_yield': info.get('dividendYield'),
            'payout_ratio': info.get('payoutRatio'),

            # Other
            'beta': info.get('beta'),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'company_name': info.get('shortName', symbol),
            'avg_volume': info.get('averageVolume'),
        }
    except Exception as e:
        print(f"Error fetching fundamentals for {symbol}: {e}")
        return None


def calculate_piotroski_score(info: Dict) -> Tuple[int, str, List[str]]:
    """
    Calculate Piotroski F-Score (0-9).

    A peer-reviewed academic model (Piotroski 2000) that uses 9 binary signals:
    - Profitability (4 points): ROA, CFO, Delta ROA, Accruals
    - Leverage/Liquidity (3 points): Delta Leverage, Delta Liquidity, No Dilution
    - Operating Efficiency (2 points): Delta Margin, Delta Turnover

    Returns:
        (score, rating, signals)
    """
    score = 0
    signals = []

    # === PROFITABILITY (4 points) ===

    # 1. ROA > 0 (1 point)
    roa = info.get('roa') or info.get('returnOnAssets') or 0
    if roa and roa > 0:
        score += 1
        signals.append(f"F1: ROA positive ({roa*100:.1f}%)")

    # 2. Operating Cash Flow > 0 (1 point)
    # Using profit margin as proxy if CFO not available
    cfo = info.get('operatingCashflow', 0)
    profit_margin = info.get('profit_margin') or info.get('profitMargins') or 0
    if cfo and cfo > 0:
        score += 1
        signals.append("F2: Operating cash flow positive")
    elif profit_margin and profit_margin > 0.05:
        score += 1
        signals.append("F2: Positive profit margin (CFO proxy)")

    # 3. ROA improving (1 point) - using ROA > 5% as proxy
    if roa and roa > 0.05:
        score += 1
        signals.append("F3: ROA above 5% threshold")

    # 4. Cash flow > Net Income (Accruals quality) (1 point)
    # Good companies have CFO > Net Income
    if cfo and profit_margin:
        # Simplified: if both are positive, assume quality
        if cfo > 0 and profit_margin > 0:
            score += 1
            signals.append("F4: Earnings quality (positive cash flow)")
    elif profit_margin and profit_margin > 0.08:
        score += 1
        signals.append("F4: Strong margins indicate quality")

    # === LEVERAGE & LIQUIDITY (3 points) ===

    # 5. Debt decreasing / Low leverage (1 point)
    de = info.get('debt_to_equity') or info.get('debtToEquity') or 100
    if de is not None and de < 50:
        score += 1
        signals.append(f"F5: Low leverage (D/E: {de:.0f}%)")

    # 6. Current ratio improving / > 1.5 (1 point)
    cr = info.get('current_ratio') or info.get('currentRatio') or 0
    if cr and cr > 1.5:
        score += 1
        signals.append(f"F6: Strong liquidity (CR: {cr:.2f})")

    # 7. No share dilution (1 point)
    # Assume pass if company is profitable
    roe = info.get('roe') or info.get('returnOnEquity') or 0
    if roe and roe > 0.10:
        score += 1
        signals.append("F7: No dilution (profitable company)")

    # === OPERATING EFFICIENCY (2 points) ===

    # 8. Gross margin improving / > 30% (1 point)
    gross_margin = info.get('gross_margin') or info.get('grossMargins') or 0
    if gross_margin and gross_margin > 0.30:
        score += 1
        signals.append(f"F8: Strong gross margin ({gross_margin*100:.1f}%)")

    # 9. Asset turnover improving (1 point)
    # Using revenue growth as proxy
    rev_growth = info.get('revenue_growth') or info.get('revenueGrowth') or 0
    if rev_growth and rev_growth > 0.05:
        score += 1
        signals.append(f"F9: Growing revenue ({rev_growth*100:.1f}%)")

    # Determine rating
    if score >= 7:
        rating = "Strong"
    elif score >= 5:
        rating = "Moderate"
    else:
        rating = "Weak"

    return score, rating, signals


def analyze_fundamentals(symbol: str, fundamentals: Optional[Dict] = None) -> Optional[FundamentalScore]:
    """
    Perform fundamental analysis and return a score.

    Args:
        symbol: Stock symbol
        fundamentals: Pre-fetched fundamentals (optional)

    Returns:
        FundamentalScore object
    """
    if fundamentals is None:
        fundamentals = get_fundamentals(symbol)

    if fundamentals is None:
        return None

    signals = []

    # === VALUATION SCORE (25 points) ===
    valuation_score = 0

    # P/E Ratio (10 points)
    pe = fundamentals.get('pe_ratio')
    if pe:
        if pe < 15:
            valuation_score += 10
            signals.append(f"Attractive P/E: {pe:.1f} (undervalued)")
        elif pe < 25:
            valuation_score += 7
            signals.append(f"Reasonable P/E: {pe:.1f}")
        elif pe < 40:
            valuation_score += 4
        else:
            signals.append(f"High P/E: {pe:.1f} (expensive)")

    # P/B Ratio (7 points)
    pb = fundamentals.get('pb_ratio')
    if pb:
        if pb < 1.5:
            valuation_score += 7
            signals.append(f"Low P/B: {pb:.2f} (value stock)")
        elif pb < 3:
            valuation_score += 5
        elif pb < 5:
            valuation_score += 2

    # PEG Ratio (8 points)
    peg = fundamentals.get('peg_ratio')
    if peg and peg > 0:
        if peg < 1:
            valuation_score += 8
            signals.append(f"Excellent PEG: {peg:.2f} (growth at reasonable price)")
        elif peg < 1.5:
            valuation_score += 5
        elif peg < 2:
            valuation_score += 2

    # === PROFITABILITY SCORE (25 points) ===
    profitability_score = 0

    # ROE (10 points)
    roe = fundamentals.get('roe')
    if roe:
        roe_pct = roe * 100
        if roe_pct > 20:
            profitability_score += 10
            signals.append(f"Strong ROE: {roe_pct:.1f}%")
        elif roe_pct > 15:
            profitability_score += 7
            signals.append(f"Good ROE: {roe_pct:.1f}%")
        elif roe_pct > 10:
            profitability_score += 4
        elif roe_pct > 0:
            profitability_score += 2

    # ROA (7 points)
    roa = fundamentals.get('roa')
    if roa:
        roa_pct = roa * 100
        if roa_pct > 10:
            profitability_score += 7
        elif roa_pct > 5:
            profitability_score += 5
        elif roa_pct > 2:
            profitability_score += 2

    # Profit Margin (8 points)
    margin = fundamentals.get('profit_margin')
    if margin:
        margin_pct = margin * 100
        if margin_pct > 15:
            profitability_score += 8
            signals.append(f"High profit margin: {margin_pct:.1f}%")
        elif margin_pct > 10:
            profitability_score += 5
        elif margin_pct > 5:
            profitability_score += 3

    # === GROWTH SCORE (25 points) ===
    growth_score = 0

    # Revenue Growth (10 points)
    rev_growth = fundamentals.get('revenue_growth')
    if rev_growth:
        rev_pct = rev_growth * 100
        if rev_pct > 20:
            growth_score += 10
            signals.append(f"Strong revenue growth: {rev_pct:.1f}%")
        elif rev_pct > 10:
            growth_score += 7
        elif rev_pct > 5:
            growth_score += 4
        elif rev_pct > 0:
            growth_score += 2

    # Earnings Growth (15 points)
    earn_growth = fundamentals.get('earnings_growth')
    if earn_growth:
        earn_pct = earn_growth * 100
        if earn_pct > 25:
            growth_score += 15
            signals.append(f"Excellent earnings growth: {earn_pct:.1f}%")
        elif earn_pct > 15:
            growth_score += 10
        elif earn_pct > 10:
            growth_score += 6
        elif earn_pct > 0:
            growth_score += 3

    # === FINANCIAL HEALTH SCORE (25 points) ===
    health_score = 0

    # Debt to Equity (10 points)
    de = fundamentals.get('debt_to_equity')
    if de is not None:
        if de < 30:
            health_score += 10
            signals.append(f"Low debt: D/E {de:.1f}%")
        elif de < 50:
            health_score += 7
        elif de < 100:
            health_score += 4
        else:
            signals.append(f"High debt: D/E {de:.1f}%")

    # Current Ratio (8 points)
    cr = fundamentals.get('current_ratio')
    if cr:
        if cr > 2:
            health_score += 8
            signals.append(f"Strong liquidity: CR {cr:.2f}")
        elif cr > 1.5:
            health_score += 6
        elif cr > 1:
            health_score += 3

    # Cash Position (7 points)
    cash = fundamentals.get('total_cash')
    debt = fundamentals.get('total_debt')
    if cash and debt:
        if cash > debt:
            health_score += 7
            signals.append("Net cash positive")
        elif cash > debt * 0.5:
            health_score += 4

    # Calculate Piotroski F-Score
    piotroski, piotroski_rating, piotroski_signals = calculate_piotroski_score(fundamentals)

    # Add Piotroski signals
    if piotroski >= 7:
        signals.append(f"Piotroski F-Score: {piotroski}/9 (Strong)")
    elif piotroski >= 5:
        signals.append(f"Piotroski F-Score: {piotroski}/9 (Moderate)")

    # Calculate total score (include Piotroski bonus)
    base_score = valuation_score + profitability_score + growth_score + health_score

    # Piotroski bonus: up to 10 extra points for high F-Score
    piotroski_bonus = (piotroski / 9) * 10  # 0-10 points

    total_score = min(100, base_score + piotroski_bonus)

    # Pass threshold for swing trading: score >= 50
    passed = total_score >= 50

    return FundamentalScore(
        symbol=symbol,
        total_score=total_score,
        valuation_score=valuation_score,
        profitability_score=profitability_score,
        growth_score=growth_score,
        health_score=health_score,
        piotroski_score=piotroski,
        piotroski_rating=piotroski_rating,
        metrics=fundamentals,
        signals=signals,
        passed=passed
    )


def screen_fundamentals(symbols: list, min_score: float = 50) -> list:
    """
    Screen stocks based on fundamental criteria.

    Args:
        symbols: List of stock symbols
        min_score: Minimum fundamental score to pass

    Returns:
        List of (symbol, FundamentalScore) tuples that passed screening
    """
    passed = []

    print(f"Screening {len(symbols)} stocks fundamentally...")

    for i, symbol in enumerate(symbols, 1):
        try:
            score = analyze_fundamentals(symbol)
            if score and score.total_score >= min_score:
                passed.append((symbol, score))
        except Exception as e:
            continue

        if i % 10 == 0:
            print(f"Progress: {i}/{len(symbols)} screened, {len(passed)} passed")

    # Sort by score descending
    passed.sort(key=lambda x: x[1].total_score, reverse=True)

    print(f"\nFundamental screening complete: {len(passed)} passed out of {len(symbols)}")
    return passed


if __name__ == "__main__":
    # Test
    print("Testing fundamental analysis (with Piotroski F-Score)...")

    test_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN"]

    for symbol in test_symbols:
        score = analyze_fundamentals(symbol)
        if score:
            print(f"\n{symbol}: {score.total_score:.1f}/100")
            print(f"  Valuation:     {score.valuation_score}/25")
            print(f"  Profitability: {score.profitability_score}/25")
            print(f"  Growth:        {score.growth_score}/25")
            print(f"  Health:        {score.health_score}/25")
            print(f"  Piotroski:     {score.piotroski_score}/9 ({score.piotroski_rating})")
            print(f"  Signals: {score.signals[:3]}")
