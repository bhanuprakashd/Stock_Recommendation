"""
Swing Trading Stock Recommender

Combines fundamental screening with technical analysis to provide
actionable swing trading recommendations.
"""

import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from nse_tickers import fetch_nse_tickers, fetch_all_nse_tickers
from fetch_stock_data import fetch_stock_history, fetch_multiple_stocks
from fundamental_analyzer import analyze_fundamentals, get_fundamentals, FundamentalScore
from technical_analyzer import analyze_technicals, calculate_swing_targets, TechnicalScore
from sectorial_tickers import (
    get_predefined_sector_tickers,
    get_all_sector_tickers,
    resolve_sector_name,
    PREDEFINED_SECTORS
)
from swing_report import generate_swing_report, print_swing_report, generate_reports_for_sector

# News integration (optional - graceful fallback if not available)
try:
    from news_fetcher import fetch_stock_news, NewsArticle
    from news_sentiment import analyze_news_sentiment, NewsSentiment
    from news_trigger import apply_news_trigger, enhance_recommendation, TriggerResult
    NEWS_AVAILABLE = True
except ImportError:
    NEWS_AVAILABLE = False
    print("Note: News modules not available. Running without news triggers.")


@dataclass
class SwingRecommendation:
    """Complete swing trade recommendation."""
    rank: int
    symbol: str
    company_name: str
    sector: str
    current_price: float

    # Scores
    fundamental_score: float
    technical_score: float
    composite_score: float

    # Trade Setup
    entry: float
    target: float
    stop_loss: float
    upside_pct: float
    risk_reward: float
    swing_signal: str

    # Analysis
    fundamental_signals: List[str]
    technical_signals: List[str]

    # Key Metrics
    pe_ratio: Optional[float]
    roe: Optional[float]
    rsi: float

    # News Analysis (optional)
    news_score: Optional[float] = None
    news_sentiment: Optional[str] = None
    news_triggered: bool = False
    news_trigger_mode: Optional[str] = None
    news_signals: List[str] = None
    news_warnings: List[str] = None
    confidence: float = 50.0  # Overall confidence including news adjustment


def analyze_stock_complete(
    symbol: str,
    hist_data: pd.DataFrame = None,
    include_news: bool = False
) -> Optional[SwingRecommendation]:
    """
    Complete analysis of a single stock for swing trading.

    Args:
        symbol: Stock symbol
        hist_data: Pre-fetched historical data (optional)
        include_news: Whether to include news sentiment analysis

    Returns:
        SwingRecommendation if stock passes criteria, None otherwise
    """
    try:
        # Fetch data if not provided
        if hist_data is None:
            hist_data = fetch_stock_history(symbol)  # Uses default 365 days

        if hist_data is None or len(hist_data) < 30:
            return None

        # Fundamental Analysis
        fund_score = analyze_fundamentals(symbol)
        if fund_score is None or fund_score.total_score < 40:
            return None

        # Technical Analysis
        tech_score = analyze_technicals(symbol, hist_data)
        if tech_score is None or tech_score.total_score < 45:
            return None

        # Combined Score (50% fundamental, 50% technical for swing)
        composite_score = (fund_score.total_score * 0.4) + (tech_score.total_score * 0.6)

        # Must have BUY or HOLD signal
        if tech_score.swing_signal not in ["BUY", "HOLD"]:
            return None

        # Calculate trade setup
        targets = calculate_swing_targets(hist_data, tech_score)

        # Minimum upside requirement for swing trade
        if targets['upside_pct'] < 5:
            return None

        # Minimum risk:reward ratio
        if targets['risk_reward'] < 1.5:
            return None

        # Get metrics
        metrics = fund_score.metrics

        # Initialize news variables
        news_score = None
        news_sentiment_label = None
        news_triggered = False
        news_trigger_mode = None
        news_signals = []
        news_warnings = []
        confidence = 50.0  # Base confidence

        # News Analysis (if enabled and available)
        if include_news and NEWS_AVAILABLE:
            try:
                # Fetch and analyze news
                articles = fetch_stock_news(symbol, lookback_days=3)
                if articles:
                    sentiment = analyze_news_sentiment(articles)
                    news_score = sentiment.overall_score
                    news_sentiment_label = sentiment.sentiment_label
                    news_signals = sentiment.top_signals[:5]

                    # Apply news triggers
                    trigger_result = apply_news_trigger(
                        symbol=symbol,
                        signal=tech_score.swing_signal,
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
                    confidence = trigger_result.updated_confidence

                    # Update signal if triggered
                    if trigger_result.triggered:
                        tech_score.swing_signal = trigger_result.updated_signal

            except Exception as e:
                print(f"News analysis error for {symbol}: {e}")

        return SwingRecommendation(
            rank=0,  # Will be set later
            symbol=symbol,
            company_name=metrics.get('company_name', symbol),
            sector=metrics.get('sector', 'Unknown'),
            current_price=hist_data['Close'].iloc[-1],
            fundamental_score=fund_score.total_score,
            technical_score=tech_score.total_score,
            composite_score=composite_score,
            entry=targets['entry'],
            target=targets['target'],
            stop_loss=targets['stop_loss'],
            upside_pct=targets['upside_pct'],
            risk_reward=targets['risk_reward'],
            swing_signal=tech_score.swing_signal,
            fundamental_signals=fund_score.signals[:5],
            technical_signals=tech_score.signals[:5],
            pe_ratio=metrics.get('pe_ratio'),
            roe=metrics.get('roe', 0) * 100 if metrics.get('roe') else None,
            rsi=tech_score.indicators.get('RSI', 0),
            news_score=news_score,
            news_sentiment=news_sentiment_label,
            news_triggered=news_triggered,
            news_trigger_mode=news_trigger_mode,
            news_signals=news_signals or [],
            news_warnings=news_warnings or [],
            confidence=confidence
        )

    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None


def get_swing_recommendations(
    symbols: List[str] = None,
    index: str = "NIFTY 200",
    top_n: int = 10,
    min_composite_score: float = 55,
    max_workers: int = 5,
    lookback_days: int = 365,
    include_news: bool = False
) -> List[SwingRecommendation]:
    """
    Get top swing trading recommendations.

    Args:
        symbols: List of symbols to analyze (optional)
        index: NSE index to use if symbols not provided
        top_n: Number of top recommendations to return
        min_composite_score: Minimum composite score threshold
        max_workers: Parallel workers for data fetching
        lookback_days: Historical data period (default: 365 - gold standard)
                       - 90: Fast, basic indicators
                       - 180: Better context, 6 months
                       - 365: Gold standard (default), includes SMA200, full cycles
        include_news: Include news sentiment analysis and triggers (default: False)

    Returns:
        List of SwingRecommendation sorted by composite score
    """
    # Get symbols
    if symbols is None:
        print(f"Fetching {index} constituents...")
        symbols = fetch_nse_tickers(index)
        symbols = [s for s in symbols if s != index.replace(" ", "")]

    lookback_label = {90: "3 months", 180: "6 months", 365: "1 year"}.get(lookback_days, f"{lookback_days} days")

    news_status = "Enabled" if include_news and NEWS_AVAILABLE else "Disabled"

    print(f"\n{'='*60}")
    print(f"SWING TRADING SCREENER")
    print(f"{'='*60}")
    print(f"Analyzing {len(symbols)} stocks...")
    print(f"Lookback period: {lookback_label}")
    print(f"Minimum composite score: {min_composite_score}")
    print(f"News triggers: {news_status}")
    print(f"{'='*60}\n")

    # Fetch historical data for all symbols
    print("Step 1: Fetching historical data...")
    stock_data = fetch_multiple_stocks(symbols, days=lookback_days, max_workers=max_workers)

    # Analyze each stock
    analysis_type = "fundamental + technical + news" if include_news else "fundamental + technical"
    print(f"\nStep 2: Running {analysis_type} analysis...")
    recommendations = []
    analyzed = 0

    for i, (symbol, hist_data) in enumerate(stock_data.items(), 1):
        try:
            rec = analyze_stock_complete(symbol, hist_data, include_news=include_news)
            if rec and rec.composite_score >= min_composite_score:
                recommendations.append(rec)
            analyzed += 1

            if i % 20 == 0:
                print(f"Progress: {i}/{len(stock_data)} analyzed, {len(recommendations)} passed")

        except Exception as e:
            continue

    print(f"\nAnalysis complete: {analyzed} analyzed, {len(recommendations)} recommendations")

    # Sort by composite score
    recommendations.sort(key=lambda x: x.composite_score, reverse=True)

    # Assign ranks
    for i, rec in enumerate(recommendations[:top_n], 1):
        rec.rank = i

    return recommendations[:top_n]


def print_recommendations(recommendations: List[SwingRecommendation]):
    """Print recommendations in a formatted report."""

    print("\n" + "=" * 80)
    print(f"SWING TRADING RECOMMENDATIONS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    if not recommendations:
        print("\nNo stocks meet the swing trading criteria at this time.")
        print("Try adjusting the parameters or wait for better market conditions.")
        return

    print(f"\nTop {len(recommendations)} Swing Trade Opportunities:\n")

    for rec in recommendations:
        print(f"\n{'='*80}")
        print(f"#{rec.rank} {rec.symbol} - {rec.company_name}")
        print(f"{'='*80}")
        print(f"Sector: {rec.sector}")
        print(f"Signal: {rec.swing_signal}")
        print()

        print("SCORES:")
        print(f"  Composite Score:   {rec.composite_score:.1f}/100")
        print(f"  Technical Score:   {rec.technical_score:.1f}/100")
        print(f"  Fundamental Score: {rec.fundamental_score:.1f}/100")
        print()

        print("TRADE SETUP:")
        print(f"  Current Price: ₹{rec.current_price:.2f}")
        print(f"  Entry:         ₹{rec.entry:.2f}")
        print(f"  Target:        ₹{rec.target:.2f} (+{rec.upside_pct:.1f}%)")
        print(f"  Stop Loss:     ₹{rec.stop_loss:.2f}")
        print(f"  Risk:Reward:   1:{rec.risk_reward:.1f}")
        print()

        print("KEY METRICS:")
        print(f"  P/E Ratio: {rec.pe_ratio:.1f}" if rec.pe_ratio else "  P/E Ratio: N/A")
        print(f"  ROE: {rec.roe:.1f}%" if rec.roe else "  ROE: N/A")
        print(f"  RSI: {rec.rsi:.1f}")
        print()

        print("TECHNICAL SIGNALS:")
        for sig in rec.technical_signals[:4]:
            print(f"  • {sig}")
        print()

        print("FUNDAMENTAL SIGNALS:")
        for sig in rec.fundamental_signals[:4]:
            print(f"  • {sig}")

        # News Analysis (if available)
        if rec.news_score is not None:
            print()
            print("NEWS ANALYSIS:")
            print(f"  Sentiment Score: {rec.news_score:.0f}/100 ({rec.news_sentiment})")
            print(f"  Confidence:      {rec.confidence:.0f}%")
            if rec.news_triggered:
                print(f"  Trigger Mode:    {rec.news_trigger_mode}")
            if rec.news_signals:
                print("  Key News:")
                for sig in rec.news_signals[:3]:
                    print(f"    {sig}")
            if rec.news_warnings:
                print("  Warnings:")
                for warning in rec.news_warnings[:2]:
                    print(f"    ⚠ {warning}")

    print("\n" + "=" * 80)
    print("DISCLAIMER")
    print("=" * 80)
    print("""
This analysis is for educational purposes only and should not be considered
as financial advice. Swing trading involves significant risk. Past performance
does not guarantee future results. Always do your own research and consider
consulting with a qualified financial advisor before making investment decisions.
    """)
    print("=" * 80)


def quick_scan(symbols: List[str] = None) -> List[SwingRecommendation]:
    """
    Quick scan of a small set of stocks for testing.

    Args:
        symbols: List of symbols (default: popular large caps)

    Returns:
        List of recommendations
    """
    if symbols is None:
        symbols = [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
            "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
            "AXISBANK", "MARUTI", "TITAN", "BAJFINANCE", "WIPRO"
        ]

    return get_swing_recommendations(
        symbols=symbols,
        top_n=5,
        min_composite_score=50,
        max_workers=5
    )


def sector_scan(sector: str, top_n: int = 5, fetch_all: bool = False) -> List[SwingRecommendation]:
    """
    Scan a specific sector for swing trading opportunities.

    Args:
        sector: Sector name or alias (e.g., "banking", "IT", "pharma")
        top_n: Number of top recommendations
        fetch_all: If True, fetch ALL stocks in sector from NIFTY 500

    Returns:
        List of recommendations from that sector
    """
    resolved = resolve_sector_name(sector)

    if fetch_all:
        # Get ALL stocks in sector from NIFTY 500
        symbols = get_all_sector_tickers(sector, index="NIFTY 500")
    else:
        # Get predefined major stocks only
        symbols = get_predefined_sector_tickers(sector)

    if not symbols:
        print(f"No stocks found for sector: {sector}")
        print(f"Available sectors: {', '.join(PREDEFINED_SECTORS.keys())}")
        return []

    mode = "ALL" if fetch_all else "major"
    print(f"\nScanning {resolved} sector ({len(symbols)} {mode} stocks)...")

    return get_swing_recommendations(
        symbols=symbols,
        top_n=top_n,
        min_composite_score=45,  # Lower threshold for sector scans
        max_workers=5
    )


if __name__ == "__main__":
    import sys

    # Parse --days option
    lookback_days = 365  # Default: Gold standard (1 year)
    if "--days" in sys.argv:
        idx = sys.argv.index("--days")
        if idx + 1 < len(sys.argv):
            try:
                lookback_days = int(sys.argv[idx + 1])
                sys.argv.pop(idx)  # Remove --days
                sys.argv.pop(idx)  # Remove the value
            except ValueError:
                print("Invalid --days value. Use 90, 180, or 365")
                sys.exit(1)

    # Parse --with-news option
    include_news = "--with-news" in sys.argv
    if include_news:
        sys.argv.remove("--with-news")
        if not NEWS_AVAILABLE:
            print("Warning: News modules not available. Running without news triggers.")
            include_news = False

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Full scan of NIFTY 200
        print("Running full NIFTY 200 scan...")
        recs = get_swing_recommendations(
            index="NIFTY 200",
            top_n=10,
            min_composite_score=55,
            lookback_days=lookback_days,
            include_news=include_news
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "--nifty50":
        # Scan NIFTY 50
        print("Running NIFTY 50 scan...")
        recs = get_swing_recommendations(
            index="NIFTY 50",
            top_n=5,
            min_composite_score=50,
            lookback_days=lookback_days,
            include_news=include_news
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "--sector":
        # Sector-based scan (predefined major stocks)
        if len(sys.argv) > 2:
            sector = " ".join(sys.argv[2:])
            recs = sector_scan(sector, top_n=5, fetch_all=False)
        else:
            print("Usage: python swing_recommender.py --sector <sector_name>")
            print("\nAvailable sectors:")
            for sec in PREDEFINED_SECTORS.keys():
                print(f"  - {sec}")
            print("\nAliases: banking, IT, pharma, auto, fmcg, metal, power, infra, realty")
            sys.exit(0)

    elif len(sys.argv) > 1 and sys.argv[1] == "--sector-all":
        # Sector-based scan (ALL stocks in sector from NIFTY 500)
        if len(sys.argv) > 2:
            sector = " ".join(sys.argv[2:])
            recs = sector_scan(sector, top_n=10, fetch_all=True)
        else:
            print("Usage: python swing_recommender.py --sector-all <sector_name>")
            print("This scans ALL stocks in the sector from NIFTY 500")
            print("\nExample: python swing_recommender.py --sector-all banking")
            sys.exit(0)

    elif len(sys.argv) > 1 and sys.argv[1] == "--report":
        # Generate detailed report for sector
        if len(sys.argv) > 2:
            sector = " ".join(sys.argv[2:])
            print(f"\nGenerating detailed swing reports for {sector} sector...")
            reports = generate_reports_for_sector(sector, fetch_all=True, top_n=5)
            if reports:
                for report in reports:
                    print_swing_report(report)
            else:
                print("No stocks in this sector meet the swing trading criteria.")
        else:
            print("Usage: python swing_recommender.py --report <sector_name>")
            print("Generates detailed reports with time estimates")
            print("\nExample: python swing_recommender.py --report pharma")
            sys.exit(0)
        sys.exit(0)
    else:
        # Quick scan
        print("Running quick scan (15 large caps)...")
        print("\nOptions:")
        print("  --nifty50              Scan NIFTY 50 (index-based, 50 stocks)")
        print("  --full                 Scan NIFTY 200 (index-based, 200 stocks)")
        print("  --sector <name>        Scan sector - major stocks only (e.g., 17 banks)")
        print("  --sector-all <name>    Scan sector - ALL stocks (e.g., 92 banks)")
        print("  --report <sector>      Detailed reports with time estimates")
        print()
        print("  --days <90|180|365>    Lookback period (default: 365)")
        print("                         90  = 3 months (fast, basic)")
        print("                         180 = 6 months (better context)")
        print("                         365 = 1 year (gold standard, SMA200)")
        print()
        print("  --with-news            Include news sentiment analysis")
        print("                         Requires: transformers, torch")
        print()
        recs = quick_scan()

    print_recommendations(recs)
