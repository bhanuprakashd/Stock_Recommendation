"""
Fetch historical stock data for NSE tickers.

Default: 365 days (1 year) - Gold standard for swing trading analysis.
Includes SMA200, full market cycles, and seasonal patterns.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from nse_tickers import fetch_nse_tickers, fetch_all_nse_tickers


def fetch_stock_history(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data for a single stock.

    Args:
        symbol: NSE stock symbol (e.g., 'RELIANCE', 'TCS')
        days: Number of days of history to fetch (default: 365 - gold standard)
              - 90 days: Basic swing trading (SMA20, SMA50, RSI, MACD)
              - 180 days: Better context (6 months of data)
              - 365 days: Gold standard (includes SMA200, full cycles, seasonal patterns)

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    try:
        # Add .NS suffix for NSE stocks
        ticker = f"{symbol}.NS"

        # Calculate period string
        if days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        elif days <= 180:
            period = "6mo"
        elif days <= 365:
            period = "1y"
        elif days <= 730:
            period = "2y"
        elif days <= 1825:
            period = "5y"
        elif days <= 3650:
            period = "10y"
        else:
            period = "max"

        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df is None or df.empty:
            return None

        # Reset index to get Date as column
        df = df.reset_index()

        # Standardize column names
        df = df.rename(columns={
            "Datetime": "Date",
            "Adj Close": "Adj_Close"
        })

        # Select columns
        cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in cols if c in df.columns]]

        # Convert date and sort
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        return df

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def fetch_multiple_stocks(
    symbols: List[str],
    days: int = 365,
    max_workers: int = 5,
    delay: float = 0.2
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple stocks in parallel.

    Args:
        symbols: List of NSE stock symbols
        days: Number of days of history
        max_workers: Number of parallel threads
        delay: Delay between requests to avoid rate limiting

    Returns:
        Dictionary mapping symbol -> DataFrame
    """
    results = {}
    failed = []

    def fetch_with_delay(symbol: str) -> Tuple[str, Optional[pd.DataFrame]]:
        time.sleep(delay)
        df = fetch_stock_history(symbol, days)
        return symbol, df

    print(f"Fetching {len(symbols)} stocks ({days} days history)...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_with_delay, s): s for s in symbols}

        for i, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]
            try:
                sym, df = future.result()
                if df is not None and not df.empty:
                    results[sym] = df
                else:
                    failed.append(sym)
            except Exception as e:
                failed.append(symbol)

            if i % 10 == 0:
                print(f"Progress: {i}/{len(symbols)} stocks fetched")

    print(f"Progress: {len(symbols)}/{len(symbols)} stocks fetched")
    print(f"\nCompleted: {len(results)} success, {len(failed)} failed")
    if failed:
        print(f"Failed symbols: {failed[:10]}{'...' if len(failed) > 10 else ''}")

    return results


def fetch_nifty50_data(days: int = 365) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for all NIFTY 50 stocks.
    Default: 365 days (1 year) - Gold standard.
    """
    tickers = fetch_nse_tickers("NIFTY 50")
    tickers = [t for t in tickers if t != "NIFTY 50"]
    return fetch_multiple_stocks(tickers, days=days)


def fetch_nifty100_data(days: int = 365) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for all NIFTY 100 stocks.
    Default: 365 days (1 year) - Gold standard.
    """
    tickers = fetch_nse_tickers("NIFTY 100")
    tickers = [t for t in tickers if t != "NIFTY100"]
    return fetch_multiple_stocks(tickers, days=days)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing single stock fetch (365 days - gold standard)...")
    print("=" * 60)

    df = fetch_stock_history("RELIANCE")  # Uses default 365 days
    if df is not None and not df.empty:
        print(f"\nRELIANCE - {len(df)} trading days:")
        print(df.head())
        print(f"\nLatest price: ₹{df['Close'].iloc[-1]:.2f}")
    else:
        print("Failed to fetch RELIANCE data")

    print("\n" + "=" * 60)
    print("Testing batch fetch (5 stocks, 365 days)...")
    print("=" * 60)

    test_symbols = ["TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN"]
    data = fetch_multiple_stocks(test_symbols)  # Uses default 365 days

    print("\nSummary:")
    for symbol, df in data.items():
        latest = df['Close'].iloc[-1] if 'Close' in df.columns else 0
        print(f"  {symbol}: {len(df)} days, Latest: ₹{latest:.2f}")
