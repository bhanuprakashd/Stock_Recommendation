"""
Download and cache stock data locally.

Standalone script that pre-fetches price history and fundamentals
for NSE stocks and saves to live_data/ folder. The app reads from
these cached files instead of hitting yfinance every time.

Usage:
    python download_data.py                          # NIFTY 50 (default)
    python download_data.py --index "NIFTY 200"      # NIFTY 200
    python download_data.py --index "NIFTY 500" --days 730  # 2 years
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from nse_tickers import fetch_nse_tickers

# Base directory for cached data
LIVE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_data")
PRICES_DIR = os.path.join(LIVE_DATA_DIR, "prices")
FUNDAMENTALS_PATH = os.path.join(LIVE_DATA_DIR, "fundamentals.csv")
METADATA_PATH = os.path.join(LIVE_DATA_DIR, "metadata.json")


def _ensure_dirs():
    """Create live_data/ and live_data/prices/ if they don't exist."""
    os.makedirs(PRICES_DIR, exist_ok=True)


def _download_price_history(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """Download OHLCV data for a single stock."""
    ticker = f"{symbol}.NS"

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

    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, progress=False, timeout=15)

            if df is None or df.empty:
                stock = yf.Ticker(ticker)
                df = stock.history(period=period)

            if df is None or df.empty:
                if attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                return None

            df = df.reset_index()

            # Handle MultiIndex columns from yf.download
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if col[1] == '' or col[1] == ticker else col[0]
                              for col in df.columns]

            df = df.rename(columns={"Datetime": "Date", "Adj Close": "Adj_Close"})

            cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in cols if c in df.columns]]

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)

            return df

        except Exception as e:
            if attempt < 2:
                time.sleep(1 + attempt)
                continue
            return None


def _download_fundamentals(symbol: str) -> Optional[Dict]:
    """Download fundamental data for a single stock."""
    for attempt in range(3):
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.info

            if not info or len(info) < 5:
                if attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                return None

            price = (info.get('regularMarketPrice')
                     or info.get('currentPrice')
                     or info.get('previousClose')
                     or info.get('open'))
            if price is None:
                return None

            return {
                'symbol': symbol,
                'price': price,
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'peg_ratio': info.get('pegRatio'),
                'ev_ebitda': info.get('enterpriseToEbitda'),
                'market_cap': info.get('marketCap'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'gross_margin': info.get('grossMargins'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'total_debt': info.get('totalDebt'),
                'total_cash': info.get('totalCash'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                'beta': info.get('beta'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'company_name': info.get('shortName', symbol),
                'avg_volume': info.get('averageVolume'),
                'operatingCashflow': info.get('operatingCashflow'),
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(1 + attempt)
                continue
            return None


def download_all_prices(symbols: List[str], days: int = 365, max_workers: int = 5) -> int:
    """
    Download price history for all symbols and save as CSV files.

    Returns:
        Number of successfully downloaded stocks.
    """
    _ensure_dirs()
    success = 0
    failed = []

    def fetch_one(symbol):
        time.sleep(0.2)
        return symbol, _download_price_history(symbol, days)

    print(f"\n[Prices] Downloading {len(symbols)} stocks ({days} days)...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, s): s for s in symbols}

        for i, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]
            try:
                sym, df = future.result()
                if df is not None and not df.empty:
                    csv_path = os.path.join(PRICES_DIR, f"{sym}.csv")
                    df.to_csv(csv_path, index=False)
                    success += 1
                else:
                    failed.append(sym)
            except Exception:
                failed.append(symbol)

            if i % 10 == 0 or i == len(symbols):
                print(f"  Progress: {i}/{len(symbols)} ({success} OK, {len(failed)} failed)")

    if failed:
        print(f"  Failed: {failed[:15]}{'...' if len(failed) > 15 else ''}")

    return success


def download_all_fundamentals(symbols: List[str], max_workers: int = 3) -> int:
    """
    Download fundamentals for all symbols and save as single CSV.

    Returns:
        Number of successfully downloaded stocks.
    """
    _ensure_dirs()
    rows = []
    failed = []

    def fetch_one(symbol):
        time.sleep(0.3)
        return symbol, _download_fundamentals(symbol)

    print(f"\n[Fundamentals] Downloading {len(symbols)} stocks...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, s): s for s in symbols}

        for i, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]
            try:
                sym, data = future.result()
                if data is not None:
                    rows.append(data)
                else:
                    failed.append(sym)
            except Exception:
                failed.append(symbol)

            if i % 10 == 0 or i == len(symbols):
                print(f"  Progress: {i}/{len(symbols)} ({len(rows)} OK, {len(failed)} failed)")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(FUNDAMENTALS_PATH, index=False)

    if failed:
        print(f"  Failed: {failed[:15]}{'...' if len(failed) > 15 else ''}")

    return len(rows)


def save_metadata(index: str, stock_count: int, price_count: int, fund_count: int, days: int):
    """Save metadata about the download."""
    _ensure_dirs()
    meta = {
        "timestamp": datetime.now().isoformat(),
        "index": index,
        "total_symbols": stock_count,
        "prices_downloaded": price_count,
        "fundamentals_downloaded": fund_count,
        "days_of_history": days,
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(meta, f, indent=2)


def get_data_freshness() -> Optional[Dict]:
    """
    Check how fresh the cached data is.

    Returns:
        Dict with 'timestamp', 'age_hours', 'index', etc. or None if no cached data.
    """
    if not os.path.exists(METADATA_PATH):
        return None

    try:
        with open(METADATA_PATH, 'r') as f:
            meta = json.load(f)

        ts = datetime.fromisoformat(meta['timestamp'])
        age = datetime.now() - ts
        meta['age_hours'] = age.total_seconds() / 3600
        meta['age_str'] = _format_age(age.total_seconds())
        return meta
    except Exception:
        return None


def _format_age(seconds: float) -> str:
    """Format age in human-readable form."""
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    elif seconds < 86400:
        return f"{int(seconds / 3600)}h ago"
    else:
        return f"{int(seconds / 86400)}d ago"


def is_data_stale(max_age_hours: float = 24.0) -> bool:
    """Check if cached data is older than max_age_hours."""
    freshness = get_data_freshness()
    if freshness is None:
        return True
    return freshness['age_hours'] > max_age_hours


def load_cached_price(symbol: str) -> Optional[pd.DataFrame]:
    """Load cached price data for a symbol."""
    csv_path = os.path.join(PRICES_DIR, f"{symbol}.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception:
        return None


def load_cached_fundamentals(symbol: str) -> Optional[Dict]:
    """Load cached fundamentals for a symbol."""
    if not os.path.exists(FUNDAMENTALS_PATH):
        return None
    try:
        df = pd.read_csv(FUNDAMENTALS_PATH)
        row = df[df['symbol'] == symbol]
        if row.empty:
            return None
        record = row.iloc[0].to_dict()
        # Convert NaN to None
        return {k: (None if pd.isna(v) else v) for k, v in record.items()}
    except Exception:
        return None


def refresh_all_data(index: str = "NIFTY 50", days: int = 365):
    """
    Full refresh: download all prices and fundamentals.
    Can be called from app.py or standalone.
    """
    print("=" * 60)
    print(f"DOWNLOADING DATA: {index} ({days} days)")
    print("=" * 60)

    # Fetch ticker list
    symbols = fetch_nse_tickers(index)
    symbols = [s for s in symbols if not s.startswith("NIFTY")]

    if not symbols:
        print("ERROR: Could not fetch ticker list from any source.")
        return False

    print(f"Found {len(symbols)} stocks")

    # Download prices
    price_count = download_all_prices(symbols, days=days)

    # Download fundamentals
    fund_count = download_all_fundamentals(symbols)

    # Save metadata
    save_metadata(index, len(symbols), price_count, fund_count, days)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Prices:       {price_count}/{len(symbols)}")
    print(f"  Fundamentals: {fund_count}/{len(symbols)}")
    print(f"  Saved to:     {LIVE_DATA_DIR}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NSE stock data")
    parser.add_argument("--index", default="NIFTY 50", help="NSE index (default: NIFTY 50)")
    parser.add_argument("--days", type=int, default=365, help="Days of price history (default: 365)")
    args = parser.parse_args()

    refresh_all_data(index=args.index, days=args.days)
