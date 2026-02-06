"""
Swing Trading Report Generator

Provides detailed reasoning and analysis for swing trade recommendations.
Includes confidence scoring, selection criteria explanation, and time estimates.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

from fetch_stock_data import fetch_stock_history
from fundamental_analyzer import analyze_fundamentals, FundamentalScore
from technical_analyzer import analyze_technicals, calculate_swing_targets, TechnicalScore

# News integration (optional)
try:
    from news_fetcher import fetch_stock_news
    from news_sentiment import analyze_news_sentiment, calculate_confidence_adjustment
    from news_trigger import apply_news_trigger
    NEWS_AVAILABLE = True
except ImportError:
    NEWS_AVAILABLE = False


@dataclass
class SwingReport:
    """Comprehensive swing trade report."""
    symbol: str
    company_name: str
    sector: str
    report_date: str

    # Prices
    current_price: float
    entry_price: float
    target_price: float
    stop_loss: float
    upside_pct: float
    downside_risk_pct: float
    risk_reward: float

    # Scores
    technical_score: float
    fundamental_score: float
    composite_score: float
    confidence_score: float

    # Signal
    signal: str  # BUY, HOLD, WAIT
    signal_strength: str  # STRONG, MODERATE, WEAK

    # Time Estimates
    estimated_days_to_target: int
    time_horizon: str  # "3-5 days", "1-2 weeks", etc.

    # Detailed Reasoning
    why_selected: List[str] = field(default_factory=list)
    how_selected: Dict = field(default_factory=dict)
    confidence_breakdown: Dict = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)

    # Technical Details
    technical_signals: List[str] = field(default_factory=list)
    fundamental_signals: List[str] = field(default_factory=list)
    key_levels: Dict = field(default_factory=dict)

    # News Analysis (optional)
    news_score: Optional[float] = None
    news_sentiment: Optional[str] = None
    news_signals: List[str] = field(default_factory=list)
    news_warnings: List[str] = field(default_factory=list)
    news_triggered: bool = False
    news_trigger_mode: Optional[str] = None
    news_adjustment: float = 0.0


def calculate_confidence_score(
    tech_score: TechnicalScore,
    fund_score: FundamentalScore,
    df: pd.DataFrame
) -> tuple:
    """
    Calculate confidence score with detailed breakdown.

    Returns:
        (confidence_score, breakdown_dict)
    """
    breakdown = {}
    total = 0

    # 1. Technical-Fundamental Alignment (25 points)
    tech_fund_diff = abs(tech_score.total_score - fund_score.total_score)
    if tech_fund_diff < 10:
        alignment_score = 25
        breakdown['alignment'] = {'score': 25, 'reason': 'Technical & fundamental strongly aligned'}
    elif tech_fund_diff < 20:
        alignment_score = 18
        breakdown['alignment'] = {'score': 18, 'reason': 'Technical & fundamental moderately aligned'}
    elif tech_fund_diff < 30:
        alignment_score = 10
        breakdown['alignment'] = {'score': 10, 'reason': 'Some divergence between technical & fundamental'}
    else:
        alignment_score = 5
        breakdown['alignment'] = {'score': 5, 'reason': 'Significant divergence - higher uncertainty'}
    total += alignment_score

    # 2. Trend Strength (25 points)
    rsi = tech_score.indicators.get('RSI', 50)
    if 40 <= rsi <= 60:
        trend_score = 20
        breakdown['trend'] = {'score': 20, 'reason': f'RSI {rsi:.1f} in neutral zone - balanced momentum'}
    elif 30 <= rsi < 40 or 60 < rsi <= 70:
        trend_score = 25
        breakdown['trend'] = {'score': 25, 'reason': f'RSI {rsi:.1f} showing directional momentum'}
    elif rsi < 30:
        trend_score = 15
        breakdown['trend'] = {'score': 15, 'reason': f'RSI {rsi:.1f} oversold - reversal potential but risky'}
    else:
        trend_score = 10
        breakdown['trend'] = {'score': 10, 'reason': f'RSI {rsi:.1f} overbought - limited upside'}
    total += trend_score

    # 3. Volume Confirmation (20 points)
    vol_ratio = tech_score.indicators.get('Volume_Ratio', 1)
    if vol_ratio > 1.5:
        vol_score = 20
        breakdown['volume'] = {'score': 20, 'reason': f'Volume {vol_ratio:.1f}x average - strong conviction'}
    elif vol_ratio > 1.0:
        vol_score = 15
        breakdown['volume'] = {'score': 15, 'reason': f'Volume {vol_ratio:.1f}x average - moderate interest'}
    elif vol_ratio > 0.7:
        vol_score = 10
        breakdown['volume'] = {'score': 10, 'reason': f'Volume {vol_ratio:.1f}x average - low participation'}
    else:
        vol_score = 5
        breakdown['volume'] = {'score': 5, 'reason': f'Volume {vol_ratio:.1f}x average - weak interest'}
    total += vol_score

    # 4. Data Quality (15 points)
    data_points = len(df)
    if data_points >= 60:
        data_score = 15
        breakdown['data_quality'] = {'score': 15, 'reason': f'{data_points} days of data - reliable analysis'}
    elif data_points >= 40:
        data_score = 10
        breakdown['data_quality'] = {'score': 10, 'reason': f'{data_points} days of data - adequate analysis'}
    else:
        data_score = 5
        breakdown['data_quality'] = {'score': 5, 'reason': f'{data_points} days of data - limited history'}
    total += data_score

    # 5. Risk-Reward Quality (15 points)
    # This will be calculated later with actual R:R ratio
    breakdown['risk_reward'] = {'score': 0, 'reason': 'Pending calculation'}

    return total, breakdown


def estimate_time_to_target(
    df: pd.DataFrame,
    current_price: float,
    target_price: float,
    tech_score: TechnicalScore
) -> tuple:
    """
    Estimate time to reach target based on historical volatility and momentum.

    Returns:
        (estimated_days, time_horizon_string)
    """
    # Calculate average daily move
    daily_returns = df['Close'].pct_change().dropna()
    avg_daily_move = daily_returns.abs().mean()

    # Calculate required move
    required_move = (target_price - current_price) / current_price

    # ATR-based estimate
    atr = tech_score.indicators.get('ATR', current_price * 0.02)
    atr_pct = atr / current_price

    # Base estimate: required move / average daily move
    if avg_daily_move > 0:
        base_days = required_move / avg_daily_move
    else:
        base_days = required_move / 0.01  # Assume 1% daily move

    # Adjust based on momentum
    rsi = tech_score.indicators.get('RSI', 50)
    if 30 <= rsi <= 40:  # Oversold bounce potential
        momentum_factor = 0.7  # Faster
    elif 60 <= rsi <= 70:  # Strong momentum
        momentum_factor = 0.8
    elif rsi > 70:  # Overbought - slower
        momentum_factor = 1.3
    else:
        momentum_factor = 1.0

    # Adjust based on trend
    if tech_score.trend_score >= 25:  # Strong trend
        trend_factor = 0.8
    elif tech_score.trend_score >= 15:
        trend_factor = 1.0
    else:
        trend_factor = 1.3  # Weak trend - slower

    estimated_days = int(base_days * momentum_factor * trend_factor)

    # Cap between 3 and 30 days for swing trading
    estimated_days = max(3, min(30, estimated_days))

    # Convert to readable horizon
    if estimated_days <= 5:
        horizon = "3-5 days"
    elif estimated_days <= 10:
        horizon = "1-2 weeks"
    elif estimated_days <= 15:
        horizon = "2-3 weeks"
    elif estimated_days <= 21:
        horizon = "3-4 weeks"
    else:
        horizon = "4+ weeks"

    return estimated_days, horizon


def generate_why_selected(
    tech_score: TechnicalScore,
    fund_score: FundamentalScore,
    composite_score: float,
    upside_pct: float,
    risk_reward: float
) -> List[str]:
    """Generate reasons why this stock was selected."""
    reasons = []

    # Composite score reason
    if composite_score >= 60:
        reasons.append(f"Strong composite score of {composite_score:.1f}/100 indicates high-quality opportunity")
    elif composite_score >= 50:
        reasons.append(f"Solid composite score of {composite_score:.1f}/100 meets selection criteria")
    else:
        reasons.append(f"Composite score of {composite_score:.1f}/100 passes minimum threshold")

    # Technical reasons
    if tech_score.trend_score >= 20:
        reasons.append("Price trading above key moving averages showing bullish trend")
    if tech_score.momentum_score >= 20:
        reasons.append("Momentum indicators (RSI, MACD) showing bullish signals")
    if tech_score.volume_score >= 15:
        reasons.append("Volume confirms price action with above-average participation")

    # Fundamental reasons
    if fund_score.valuation_score >= 15:
        reasons.append("Attractive valuation metrics (P/E, P/B) suggest undervaluation")
    if fund_score.profitability_score >= 15:
        reasons.append("Strong profitability (ROE, margins) indicates quality business")
    if fund_score.growth_score >= 15:
        reasons.append("Healthy growth trajectory in revenue and earnings")

    # Risk-reward reason
    if risk_reward >= 2.5:
        reasons.append(f"Excellent risk-reward ratio of 1:{risk_reward:.1f} favors the trade")
    elif risk_reward >= 2.0:
        reasons.append(f"Good risk-reward ratio of 1:{risk_reward:.1f}")
    else:
        reasons.append(f"Acceptable risk-reward ratio of 1:{risk_reward:.1f}")

    # Upside reason
    reasons.append(f"Target offers {upside_pct:.1f}% upside potential")

    return reasons


def generate_risk_factors(
    tech_score: TechnicalScore,
    fund_score: FundamentalScore,
    df: pd.DataFrame
) -> List[str]:
    """Generate risk factors to watch."""
    risks = []

    # Technical risks
    rsi = tech_score.indicators.get('RSI', 50)
    if rsi > 65:
        risks.append(f"RSI at {rsi:.1f} approaching overbought - watch for pullback")
    if rsi < 35:
        risks.append(f"RSI at {rsi:.1f} in oversold zone - could fall further before reversal")

    if tech_score.trend_score < 15:
        risks.append("Weak trend structure - price may not sustain upward move")

    if tech_score.volume_score < 10:
        risks.append("Low volume participation - breakouts may fail without volume")

    # Volatility risk
    atr_pct = tech_score.indicators.get('ATR_Pct', 2)
    if atr_pct > 4:
        risks.append(f"High volatility (ATR {atr_pct:.1f}%) - larger stop loss required")

    # Fundamental risks
    metrics = fund_score.metrics
    de = metrics.get('debt_to_equity')
    if de and de > 100:
        risks.append(f"High debt level (D/E: {de:.0f}%) - financial stress risk")

    pe = metrics.get('pe_ratio')
    if pe and pe > 40:
        risks.append(f"High valuation (P/E: {pe:.1f}) - limited upside, high downside risk")

    # Market risk
    risks.append("Broader market correction could impact stock regardless of fundamentals")
    risks.append("Sector-specific news or events could cause sudden moves")

    return risks[:6]  # Limit to top 6 risks


def generate_swing_report(
    symbol: str,
    df: pd.DataFrame = None,
    include_news: bool = False
) -> Optional[SwingReport]:
    """
    Generate comprehensive swing trade report for a stock.

    Args:
        symbol: Stock symbol
        df: Pre-fetched historical data (optional)
        include_news: Whether to include news sentiment analysis

    Returns:
        SwingReport object or None if stock doesn't qualify
    """
    # Fetch data if not provided (365 days - gold standard)
    if df is None:
        df = fetch_stock_history(symbol)  # Uses default 365 days

    if df is None or len(df) < 30:
        return None

    # Get analyses
    tech_score = analyze_technicals(symbol, df)
    fund_score = analyze_fundamentals(symbol)

    if tech_score is None or fund_score is None:
        return None

    # Calculate composite score
    composite_score = (fund_score.total_score * 0.4) + (tech_score.total_score * 0.6)

    # Calculate trade setup
    targets = calculate_swing_targets(df, tech_score)

    current_price = df['Close'].iloc[-1]
    entry_price = targets['entry']
    target_price = targets['target']
    stop_loss = targets['stop_loss']
    upside_pct = targets['upside_pct']
    downside_risk_pct = ((entry_price - stop_loss) / entry_price) * 100
    risk_reward = targets['risk_reward']

    # Calculate confidence score
    confidence, confidence_breakdown = calculate_confidence_score(tech_score, fund_score, df)

    # Add risk-reward to confidence
    if risk_reward >= 2.5:
        rr_score = 15
        rr_reason = f'Excellent R:R of 1:{risk_reward:.1f}'
    elif risk_reward >= 2.0:
        rr_score = 12
        rr_reason = f'Good R:R of 1:{risk_reward:.1f}'
    elif risk_reward >= 1.5:
        rr_score = 8
        rr_reason = f'Acceptable R:R of 1:{risk_reward:.1f}'
    else:
        rr_score = 4
        rr_reason = f'Poor R:R of 1:{risk_reward:.1f}'

    confidence_breakdown['risk_reward'] = {'score': rr_score, 'reason': rr_reason}
    confidence += rr_score

    # Determine signal and strength
    if tech_score.swing_signal == "BUY" and composite_score >= 55:
        signal = "BUY"
        signal_strength = "STRONG" if composite_score >= 65 else "MODERATE"
    elif tech_score.swing_signal in ["BUY", "HOLD"] and composite_score >= 45:
        signal = "HOLD"
        signal_strength = "MODERATE" if composite_score >= 55 else "WEAK"
    else:
        signal = "WAIT"
        signal_strength = "WEAK"

    # Estimate time to target
    estimated_days, time_horizon = estimate_time_to_target(
        df, current_price, target_price, tech_score
    )

    # Generate reasoning
    why_selected = generate_why_selected(
        tech_score, fund_score, composite_score, upside_pct, risk_reward
    )

    how_selected = {
        'technical_analysis': {
            'score': tech_score.total_score,
            'weight': '60%',
            'components': {
                'trend': f'{tech_score.trend_score}/30',
                'momentum': f'{tech_score.momentum_score}/30',
                'volatility': f'{tech_score.volatility_score}/20',
                'volume': f'{tech_score.volume_score}/20'
            }
        },
        'fundamental_analysis': {
            'score': fund_score.total_score,
            'weight': '40%',
            'components': {
                'valuation': f'{fund_score.valuation_score}/25',
                'profitability': f'{fund_score.profitability_score}/25',
                'growth': f'{fund_score.growth_score}/25',
                'health': f'{fund_score.health_score}/25'
            }
        },
        'composite_formula': 'Composite = (Technical × 0.6) + (Fundamental × 0.4)',
        'filters_applied': [
            f'Composite Score >= 45 (Got: {composite_score:.1f})',
            f'Upside >= 5% (Got: {upside_pct:.1f}%)',
            f'Risk:Reward >= 1.5 (Got: 1:{risk_reward:.1f})',
            f'Signal: {tech_score.swing_signal}'
        ]
    }

    risk_factors = generate_risk_factors(tech_score, fund_score, df)

    # Key levels
    key_levels = {
        'support_1': round(stop_loss, 2),
        'support_2': round(df['Low'].tail(20).min(), 2),
        'resistance_1': round(target_price, 2),
        'resistance_2': round(df['High'].tail(20).max(), 2),
        'sma_20': round(tech_score.indicators.get('SMA_20', 0), 2),
        'sma_50': round(tech_score.indicators.get('SMA_50', 0), 2),
        'bollinger_upper': round(tech_score.indicators.get('BB_Upper', 0), 2),
        'bollinger_lower': round(tech_score.indicators.get('BB_Lower', 0), 2)
    }

    # News Analysis (if enabled and available)
    news_score = None
    news_sentiment_label = None
    news_signals = []
    news_warnings = []
    news_triggered = False
    news_trigger_mode = None
    news_adjustment = 0.0

    if include_news and NEWS_AVAILABLE:
        try:
            articles = fetch_stock_news(symbol, lookback_days=3)
            if articles:
                sentiment = analyze_news_sentiment(articles)
                news_score = sentiment.overall_score
                news_sentiment_label = sentiment.sentiment_label
                news_signals = sentiment.top_signals[:5]

                # Calculate confidence adjustment
                news_adjustment = calculate_confidence_adjustment(sentiment)
                confidence += news_adjustment  # Add to total confidence

                # Apply news triggers
                trigger_result = apply_news_trigger(
                    symbol=symbol,
                    signal=signal,
                    confidence=confidence,
                    composite_score=composite_score,
                    technical_score=tech_score.total_score,
                    fundamental_score=fund_score.total_score,
                    articles=articles,
                    sentiment=sentiment
                )

                news_triggered = trigger_result.triggered
                news_trigger_mode = trigger_result.mode
                news_warnings = trigger_result.warnings

                # Update signal if triggered
                if trigger_result.triggered:
                    signal = trigger_result.updated_signal
                    if signal == "BUY":
                        signal_strength = "STRONG" if composite_score >= 60 else "MODERATE"

        except Exception as e:
            print(f"News analysis error for {symbol}: {e}")

    return SwingReport(
        symbol=symbol,
        company_name=fund_score.metrics.get('company_name', symbol),
        sector=fund_score.metrics.get('sector', 'Unknown'),
        report_date=datetime.now().strftime('%Y-%m-%d %H:%M'),
        current_price=round(current_price, 2),
        entry_price=entry_price,
        target_price=target_price,
        stop_loss=stop_loss,
        upside_pct=upside_pct,
        downside_risk_pct=round(downside_risk_pct, 2),
        risk_reward=risk_reward,
        technical_score=tech_score.total_score,
        fundamental_score=fund_score.total_score,
        composite_score=round(composite_score, 1),
        confidence_score=confidence,
        signal=signal,
        signal_strength=signal_strength,
        estimated_days_to_target=estimated_days,
        time_horizon=time_horizon,
        why_selected=why_selected,
        how_selected=how_selected,
        confidence_breakdown=confidence_breakdown,
        risk_factors=risk_factors,
        technical_signals=tech_score.signals[:6],
        fundamental_signals=fund_score.signals[:6],
        key_levels=key_levels,
        news_score=news_score,
        news_sentiment=news_sentiment_label,
        news_signals=news_signals,
        news_warnings=news_warnings,
        news_triggered=news_triggered,
        news_trigger_mode=news_trigger_mode,
        news_adjustment=news_adjustment
    )


def print_swing_report(report: SwingReport):
    """Print detailed swing trade report."""

    print("\n" + "=" * 80)
    print(f"SWING TRADE REPORT: {report.symbol}")
    print("=" * 80)
    print(f"Company: {report.company_name}")
    print(f"Sector: {report.sector}")
    print(f"Report Date: {report.report_date}")

    # Signal Box
    print("\n" + "─" * 80)
    signal_emoji = "🟢" if report.signal == "BUY" else "🟡" if report.signal == "HOLD" else "🔴"
    print(f"SIGNAL: {signal_emoji} {report.signal} ({report.signal_strength})")
    print(f"CONFIDENCE: {report.confidence_score}/100")
    print(f"TIME HORIZON: {report.time_horizon} (Est. {report.estimated_days_to_target} trading days)")
    print("─" * 80)

    # Trade Setup
    print("\n📊 TRADE SETUP")
    print("─" * 40)
    print(f"  Current Price:  ₹{report.current_price:.2f}")
    print(f"  Entry Price:    ₹{report.entry_price:.2f}")
    print(f"  Target Price:   ₹{report.target_price:.2f} (+{report.upside_pct:.1f}%)")
    print(f"  Stop Loss:      ₹{report.stop_loss:.2f} (-{report.downside_risk_pct:.1f}%)")
    print(f"  Risk:Reward:    1:{report.risk_reward:.1f}")

    # Scores
    print("\n📈 SCORES")
    print("─" * 40)
    print(f"  Composite Score:   {report.composite_score}/100")
    print(f"  ├─ Technical:      {report.technical_score}/100 (weight: 60%)")
    print(f"  └─ Fundamental:    {report.fundamental_score}/100 (weight: 40%)")

    # Why Selected
    print("\n✅ WHY THIS STOCK WAS SELECTED")
    print("─" * 40)
    for i, reason in enumerate(report.why_selected, 1):
        print(f"  {i}. {reason}")

    # How Selected
    print("\n🔍 HOW IT WAS SELECTED")
    print("─" * 40)
    how = report.how_selected

    print("  Technical Analysis (60% weight):")
    tech_comp = how['technical_analysis']['components']
    print(f"    • Trend:      {tech_comp['trend']}")
    print(f"    • Momentum:   {tech_comp['momentum']}")
    print(f"    • Volatility: {tech_comp['volatility']}")
    print(f"    • Volume:     {tech_comp['volume']}")

    print("\n  Fundamental Analysis (40% weight):")
    fund_comp = how['fundamental_analysis']['components']
    print(f"    • Valuation:     {fund_comp['valuation']}")
    print(f"    • Profitability: {fund_comp['profitability']}")
    print(f"    • Growth:        {fund_comp['growth']}")
    print(f"    • Health:        {fund_comp['health']}")

    print(f"\n  Formula: {how['composite_formula']}")

    print("\n  Filters Applied:")
    for f in how['filters_applied']:
        print(f"    ✓ {f}")

    # Confidence Breakdown
    print("\n🎯 CONFIDENCE SCORE BREAKDOWN")
    print("─" * 40)
    for key, item in report.confidence_breakdown.items():
        print(f"  {key.title():15} : {item['score']:2}/{'25' if key != 'data_quality' and key != 'risk_reward' else '15'} - {item['reason']}")
    print(f"  {'─' * 35}")
    print(f"  {'TOTAL':15} : {report.confidence_score}/100")

    # Technical Signals
    print("\n📉 TECHNICAL SIGNALS")
    print("─" * 40)
    for signal in report.technical_signals:
        print(f"  • {signal}")

    # Fundamental Signals
    print("\n💰 FUNDAMENTAL SIGNALS")
    print("─" * 40)
    for signal in report.fundamental_signals:
        print(f"  • {signal}")

    # News Analysis (if available)
    if report.news_score is not None:
        print("\n📰 NEWS ANALYSIS")
        print("─" * 40)
        sentiment_indicator = "🟢" if report.news_score >= 65 else "🔴" if report.news_score <= 35 else "🟡"
        print(f"  Sentiment Score: {sentiment_indicator} {report.news_score:.0f}/100 ({report.news_sentiment})")
        print(f"  Confidence Adj:  {report.news_adjustment:+.1f} points")

        if report.news_triggered:
            print(f"  Trigger Mode:    {report.news_trigger_mode}")

        if report.news_signals:
            print("  Key Headlines:")
            for signal in report.news_signals[:3]:
                print(f"    {signal}")

        if report.news_warnings:
            print("  Warnings:")
            for warning in report.news_warnings[:2]:
                print(f"    ⚠ {warning}")

    # Key Levels
    print("\n📍 KEY PRICE LEVELS")
    print("─" * 40)
    levels = report.key_levels
    print(f"  Resistance 2:    ₹{levels['resistance_2']:.2f}")
    print(f"  Resistance 1:    ₹{levels['resistance_1']:.2f} (Target)")
    print(f"  Bollinger Upper: ₹{levels['bollinger_upper']:.2f}")
    print(f"  SMA 20:          ₹{levels['sma_20']:.2f}")
    print(f"  Current:         ₹{report.current_price:.2f} ◄")
    print(f"  SMA 50:          ₹{levels['sma_50']:.2f}")
    print(f"  Bollinger Lower: ₹{levels['bollinger_lower']:.2f}")
    print(f"  Support 1:       ₹{levels['support_1']:.2f} (Stop Loss)")
    print(f"  Support 2:       ₹{levels['support_2']:.2f}")

    # Risk Factors
    print("\n⚠️  RISK FACTORS")
    print("─" * 40)
    for i, risk in enumerate(report.risk_factors, 1):
        print(f"  {i}. {risk}")

    # Time Estimate Explanation
    print("\n⏱️  TIME ESTIMATE REASONING")
    print("─" * 40)
    print(f"  Estimated Days: {report.estimated_days_to_target} trading days")
    print(f"  Time Horizon:   {report.time_horizon}")
    print("  Factors considered:")
    print("    • Historical daily volatility")
    print("    • Current momentum (RSI-based)")
    print("    • Trend strength")
    print("    • Distance to target")

    print("\n" + "=" * 80)
    print("DISCLAIMER: This is not financial advice. Do your own research.")
    print("=" * 80 + "\n")


def generate_reports_for_sector(
    sector: str,
    fetch_all: bool = False,
    top_n: int = 5
) -> List[SwingReport]:
    """
    Generate swing reports for all qualifying stocks in a sector.
    """
    from sectorial_tickers import get_predefined_sector_tickers, get_all_sector_tickers
    from fetch_stock_data import fetch_multiple_stocks

    if fetch_all:
        symbols = get_all_sector_tickers(sector, index="NIFTY 500")
    else:
        symbols = get_predefined_sector_tickers(sector)

    if not symbols:
        print(f"No stocks found for sector: {sector}")
        return []

    print(f"\nGenerating reports for {len(symbols)} stocks in {sector} sector...")

    # Fetch data (365 days - gold standard)
    stock_data = fetch_multiple_stocks(symbols)  # Uses default 365 days

    # Generate reports
    reports = []
    for symbol, df in stock_data.items():
        report = generate_swing_report(symbol, df)
        if report and report.signal in ["BUY", "HOLD"]:
            reports.append(report)

    # Sort by composite score
    reports.sort(key=lambda x: x.composite_score, reverse=True)

    return reports[:top_n]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        print(f"Generating swing report for {symbol}...")
        report = generate_swing_report(symbol)
        if report:
            print_swing_report(report)
        else:
            print(f"Could not generate report for {symbol}")
    else:
        # Demo with SBIN
        print("Usage: python swing_report.py <SYMBOL>")
        print("Example: python swing_report.py SBIN")
        print("\nRunning demo with SBIN...")
        report = generate_swing_report("SBIN")
        if report:
            print_swing_report(report)
