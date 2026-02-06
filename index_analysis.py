"""
Index-Based Stock Analysis

Perform state-of-the-art swing trading analysis on index constituents.
Supports NIFTY 50, NIFTY 100, NIFTY 200, NIFTY 500, and sector indices.
"""

import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from nse_tickers import fetch_nse_tickers
from fetch_stock_data import fetch_multiple_stocks
from state_of_art_analyzer import (
    analyze_stock,
    score_technical,
    score_fundamental,
    StateOfArtResult,
    print_state_of_art_report
)


# Supported indices
SUPPORTED_INDICES = {
    "NIFTY 50": "Top 50 large-cap stocks",
    "NIFTY NEXT 50": "Next 50 large-cap stocks (51-100)",
    "NIFTY 100": "Top 100 stocks",
    "NIFTY 200": "Top 200 stocks",
    "NIFTY 500": "Top 500 stocks",
    "NIFTY MIDCAP 100": "Top 100 mid-cap stocks",
    "NIFTY SMALLCAP 100": "Top 100 small-cap stocks",
    "NIFTY BANK": "Banking sector",
    "NIFTY IT": "IT sector",
    "NIFTY PHARMA": "Pharma sector",
    "NIFTY AUTO": "Auto sector",
    "NIFTY FMCG": "FMCG sector",
    "NIFTY METAL": "Metal sector",
    "NIFTY REALTY": "Real Estate sector",
    "NIFTY ENERGY": "Energy sector",
    "NIFTY INFRA": "Infrastructure sector",
    "NIFTY PSE": "Public Sector Enterprises",
    "NIFTY PRIVATE BANK": "Private Banks",
    "NIFTY PSU BANK": "PSU Banks",
    "NIFTY FIN SERVICE": "Financial Services",
}


@dataclass
class IndexAnalysisResult:
    """Result of index analysis."""
    index_name: str
    analysis_date: str
    total_stocks: int
    analyzed_stocks: int
    qualifying_stocks: int
    top_recommendations: List[StateOfArtResult]
    summary_stats: Dict


def get_index_stocks(index_name: str) -> List[str]:
    """
    Get all stock symbols for a given index.

    Args:
        index_name: Name of the index (e.g., "NIFTY 50", "NIFTY BANK")

    Returns:
        List of stock symbols
    """
    try:
        symbols = fetch_nse_tickers(index_name)
        # Remove index name itself if present
        symbols = [s for s in symbols if s.upper() not in [
            index_name.upper(),
            index_name.replace(" ", "").upper(),
            "NIFTY50", "NIFTY100", "NIFTY200", "NIFTY500"
        ]]
        return symbols
    except Exception as e:
        print(f"Error fetching {index_name} constituents: {e}")
        return []


def analyze_index(
    index_name: str = "NIFTY 50",
    top_n: int = 10,
    min_confidence: float = 60,
    min_composite_score: float = 50,
    lookback_days: int = 365,
    max_workers: int = 5
) -> IndexAnalysisResult:
    """
    Analyze all stocks in an index and return top recommendations.

    Args:
        index_name: Name of the index to analyze
        top_n: Number of top recommendations to return
        min_confidence: Minimum confidence score (0-100)
        min_composite_score: Minimum composite score (0-100)
        lookback_days: Historical data period (90, 180, 365)
        max_workers: Parallel workers for data fetching

    Returns:
        IndexAnalysisResult with top recommendations
    """
    print("\n" + "█" * 80)
    print(f"█ INDEX ANALYSIS: {index_name}")
    print("█" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Lookback Period: {lookback_days} days")
    print(f"Min Confidence: {min_confidence}%")
    print(f"Min Composite Score: {min_composite_score}")
    print("█" * 80 + "\n")

    # Get index constituents
    print("Step 1: Fetching index constituents...")
    symbols = get_index_stocks(index_name)

    if not symbols:
        print(f"Error: Could not fetch stocks for {index_name}")
        return None

    print(f"Found {len(symbols)} stocks in {index_name}\n")

    # Fetch historical data
    print("Step 2: Fetching historical data...")
    stock_data = fetch_multiple_stocks(symbols, days=lookback_days, max_workers=max_workers)
    print(f"Successfully fetched data for {len(stock_data)} stocks\n")

    # Analyze each stock
    print("Step 3: Running state-of-the-art analysis...")
    recommendations = []
    analyzed = 0
    errors = 0

    for i, (symbol, df) in enumerate(stock_data.items(), 1):
        try:
            result = analyze_stock(symbol, lookback_days=lookback_days)

            if result:
                analyzed += 1
                # Apply filters
                if result.confidence >= min_confidence and result.composite_score >= min_composite_score:
                    recommendations.append(result)
            else:
                errors += 1

        except Exception as e:
            errors += 1
            continue

        if i % 10 == 0:
            print(f"Progress: {i}/{len(stock_data)} analyzed, {len(recommendations)} qualifying")

    print(f"\nAnalysis complete:")
    print(f"  Total in index: {len(symbols)}")
    print(f"  Successfully analyzed: {analyzed}")
    print(f"  Meeting criteria: {len(recommendations)}")
    print(f"  Errors: {errors}")

    # Sort by composite score and confidence
    recommendations.sort(key=lambda x: (x.composite_score, x.confidence), reverse=True)

    # Take top N
    top_recs = recommendations[:top_n]

    # Calculate summary stats
    if recommendations:
        avg_composite = sum(r.composite_score for r in recommendations) / len(recommendations)
        avg_confidence = sum(r.confidence for r in recommendations) / len(recommendations)
        avg_upside = sum(r.upside_pct for r in recommendations) / len(recommendations)

        # Sector distribution
        sectors = {}
        for r in recommendations:
            sectors[r.sector] = sectors.get(r.sector, 0) + 1

        summary_stats = {
            'avg_composite_score': round(avg_composite, 1),
            'avg_confidence': round(avg_confidence, 1),
            'avg_upside': round(avg_upside, 1),
            'sector_distribution': sectors,
            'signal_distribution': {
                'STRONG BUY': sum(1 for r in recommendations if r.signal == "STRONG BUY"),
                'BUY': sum(1 for r in recommendations if r.signal == "BUY"),
                'HOLD': sum(1 for r in recommendations if r.signal == "HOLD"),
            }
        }
    else:
        summary_stats = {}

    return IndexAnalysisResult(
        index_name=index_name,
        analysis_date=datetime.now().strftime('%Y-%m-%d %H:%M'),
        total_stocks=len(symbols),
        analyzed_stocks=analyzed,
        qualifying_stocks=len(recommendations),
        top_recommendations=top_recs,
        summary_stats=summary_stats
    )


def print_index_analysis_report(result: IndexAnalysisResult):
    """Print comprehensive index analysis report."""

    print("\n" + "█" * 80)
    print(f"█ {result.index_name} ANALYSIS REPORT")
    print("█" * 80)
    print(f"Report Date: {result.analysis_date}")

    # Summary
    print(f"\n{'━' * 80}")
    print("SUMMARY")
    print(f"{'━' * 80}")
    print(f"  Total Stocks in Index:    {result.total_stocks}")
    print(f"  Successfully Analyzed:    {result.analyzed_stocks}")
    print(f"  Meeting Criteria:         {result.qualifying_stocks}")
    print(f"  Top Recommendations:      {len(result.top_recommendations)}")

    if result.summary_stats:
        print(f"\n  Average Composite Score:  {result.summary_stats['avg_composite_score']}/100")
        print(f"  Average Confidence:       {result.summary_stats['avg_confidence']}%")
        print(f"  Average Upside:           {result.summary_stats['avg_upside']}%")

        # Signal distribution
        print(f"\n  Signal Distribution:")
        for signal, count in result.summary_stats['signal_distribution'].items():
            print(f"    {signal}: {count}")

        # Sector distribution
        if result.summary_stats.get('sector_distribution'):
            print(f"\n  Sector Distribution:")
            for sector, count in sorted(result.summary_stats['sector_distribution'].items(),
                                        key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {sector}: {count}")

    # Top Recommendations Table
    if result.top_recommendations:
        print(f"\n{'━' * 80}")
        print(f"TOP {len(result.top_recommendations)} RECOMMENDATIONS")
        print(f"{'━' * 80}")

        # Header
        print(f"\n{'Rank':<5} {'Symbol':<12} {'Signal':<12} {'Confidence':<12} {'Composite':<10} {'Upside':<10} {'R:R':<8} {'Time':<12}")
        print("─" * 85)

        for i, rec in enumerate(result.top_recommendations, 1):
            print(f"{i:<5} {rec.symbol:<12} {rec.signal:<12} {rec.confidence:<12.0f}% {rec.composite_score:<10.1f} {rec.upside_pct:<10.1f}% {rec.risk_reward:<8.1f} {rec.time_horizon:<12}")

        print("─" * 85)

        # Detailed reports for top 3
        print(f"\n{'━' * 80}")
        print("DETAILED ANALYSIS - TOP 3")
        print(f"{'━' * 80}")

        for rec in result.top_recommendations[:3]:
            print_state_of_art_report(rec)

    else:
        print(f"\n{'━' * 80}")
        print("NO RECOMMENDATIONS")
        print(f"{'━' * 80}")
        print("\nNo stocks in this index meet the criteria:")
        print(f"  - Minimum Confidence: 60%")
        print(f"  - Minimum Composite Score: 50")
        print(f"  - Minimum Upside: 5%")
        print(f"  - Minimum Risk:Reward: 1.5")
        print("\nTry adjusting the parameters or wait for better market conditions.")

    # Footer
    print("\n" + "█" * 80)
    print("DISCLAIMER: This analysis is for educational purposes only.")
    print("Past performance does not guarantee future results.")
    print("█" * 80 + "\n")


def compare_indices(
    indices: List[str] = None,
    top_n: int = 5,
    lookback_days: int = 365
) -> Dict[str, IndexAnalysisResult]:
    """
    Compare multiple indices and their top recommendations.

    Args:
        indices: List of index names to compare
        top_n: Top recommendations per index
        lookback_days: Historical data period

    Returns:
        Dict mapping index name to its analysis result
    """
    if indices is None:
        indices = ["NIFTY 50", "NIFTY BANK", "NIFTY IT", "NIFTY PHARMA"]

    results = {}

    for index in indices:
        print(f"\n{'='*80}")
        print(f"Analyzing {index}...")
        print(f"{'='*80}")

        result = analyze_index(
            index_name=index,
            top_n=top_n,
            lookback_days=lookback_days
        )

        if result:
            results[index] = result

    # Print comparison
    print("\n" + "█" * 80)
    print("INDEX COMPARISON SUMMARY")
    print("█" * 80)

    print(f"\n{'Index':<20} {'Analyzed':<12} {'Qualifying':<12} {'Avg Score':<12} {'Avg Confidence':<15}")
    print("─" * 75)

    for index, result in results.items():
        avg_score = result.summary_stats.get('avg_composite_score', 0)
        avg_conf = result.summary_stats.get('avg_confidence', 0)
        print(f"{index:<20} {result.analyzed_stocks:<12} {result.qualifying_stocks:<12} {avg_score:<12.1f} {avg_conf:<15.1f}%")

    print("─" * 75)

    # Best picks across all indices
    all_recs = []
    for result in results.values():
        all_recs.extend(result.top_recommendations)

    all_recs.sort(key=lambda x: (x.composite_score, x.confidence), reverse=True)

    print(f"\n{'━' * 80}")
    print("BEST PICKS ACROSS ALL INDICES")
    print(f"{'━' * 80}")

    for i, rec in enumerate(all_recs[:10], 1):
        print(f"{i}. {rec.symbol:<12} | {rec.sector:<20} | Score: {rec.composite_score:.1f} | Confidence: {rec.confidence:.0f}%")

    return results


def quick_index_scan(index_name: str = "NIFTY 50") -> List[StateOfArtResult]:
    """
    Quick scan of an index with default parameters.

    Args:
        index_name: Name of the index

    Returns:
        List of top recommendations
    """
    result = analyze_index(
        index_name=index_name,
        top_n=10,
        min_confidence=50,
        min_composite_score=45,
        lookback_days=365
    )

    if result:
        print_index_analysis_report(result)
        return result.top_recommendations

    return []


def list_supported_indices():
    """Print all supported indices."""
    print("\n" + "=" * 60)
    print("SUPPORTED INDICES")
    print("=" * 60)

    print("\nBroad Market Indices:")
    for idx in ["NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200", "NIFTY 500"]:
        print(f"  • {idx}: {SUPPORTED_INDICES.get(idx, '')}")

    print("\nMarket Cap Indices:")
    for idx in ["NIFTY MIDCAP 100", "NIFTY SMALLCAP 100"]:
        print(f"  • {idx}: {SUPPORTED_INDICES.get(idx, '')}")

    print("\nSector Indices:")
    for idx in ["NIFTY BANK", "NIFTY IT", "NIFTY PHARMA", "NIFTY AUTO",
                "NIFTY FMCG", "NIFTY METAL", "NIFTY REALTY", "NIFTY ENERGY"]:
        print(f"  • {idx}: {SUPPORTED_INDICES.get(idx, '')}")

    print("\nBanking Indices:")
    for idx in ["NIFTY PRIVATE BANK", "NIFTY PSU BANK", "NIFTY FIN SERVICE"]:
        print(f"  • {idx}: {SUPPORTED_INDICES.get(idx, '')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_supported_indices()
        elif sys.argv[1] == "--compare":
            if len(sys.argv) > 2:
                indices = sys.argv[2:]
            else:
                indices = ["NIFTY 50", "NIFTY BANK", "NIFTY IT"]
            compare_indices(indices)
        else:
            index_name = " ".join(sys.argv[1:]).upper()
            if index_name in SUPPORTED_INDICES:
                quick_index_scan(index_name)
            else:
                print(f"Unknown index: {index_name}")
                print("Use --list to see supported indices")
    else:
        print("Index Analysis Tool")
        print("=" * 40)
        print("\nUsage:")
        print("  python index_analysis.py <INDEX_NAME>")
        print("  python index_analysis.py --list")
        print("  python index_analysis.py --compare INDEX1 INDEX2 ...")
        print("\nExamples:")
        print("  python index_analysis.py NIFTY 50")
        print("  python index_analysis.py NIFTY BANK")
        print("  python index_analysis.py --compare NIFTY 50 NIFTY IT")
        print("\nDefault: Running NIFTY 50 analysis...")
        print()

        quick_index_scan("NIFTY 50")
