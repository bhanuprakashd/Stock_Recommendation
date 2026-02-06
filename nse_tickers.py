"""
Fetch NSE Ticker Labels/Symbols

Multiple fallback sources:
1. NSE India API (works from Indian IPs)
2. NSE Archives CSV (works globally)
3. Wikipedia scraping (works globally)
"""

import requests
import pandas as pd
from typing import List, Optional


# NSE Archives CSV URLs for index constituents
_NSE_CSV_URLS = {
    "NIFTY 50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY 100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "NIFTY 200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "NIFTY 500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "NIFTY NEXT 50": "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv",
}


def _fetch_from_nse_api(index: str) -> List[str]:
    """Fetch tickers from NSE India API (may fail outside India)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }

    index_encoded = index.replace(" ", "%20")
    url = f"https://www.nseindia.com/api/equity-stockIndices?index={index_encoded}"

    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        tickers = [stock["symbol"] for stock in data.get("data", [])]
        tickers = [t for t in tickers if t != index.replace(" ", "")]
        return tickers

    except Exception:
        return []


def _fetch_from_nse_csv(index: str) -> List[str]:
    """Fetch tickers from NSE Archives CSV (works globally)."""
    url = _NSE_CSV_URLS.get(index)
    if not url:
        return []

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        df = pd.read_csv(pd.io.common.StringIO(response.text))
        # Column name is usually "Symbol"
        symbol_col = None
        for col in df.columns:
            if col.strip().lower() == "symbol":
                symbol_col = col
                break

        if symbol_col:
            return df[symbol_col].str.strip().tolist()
        return []

    except Exception:
        return []


def _fetch_from_wikipedia(index: str) -> List[str]:
    """Fetch NIFTY 50 tickers from Wikipedia as last resort."""
    if "50" not in index:
        return []

    try:
        url = "https://en.wikipedia.org/wiki/NIFTY_50"
        tables = pd.read_html(url)
        for table in tables:
            for col in table.columns:
                if "symbol" in col.lower():
                    symbols = table[col].dropna().str.strip().tolist()
                    if len(symbols) >= 40:
                        return symbols
        return []
    except Exception:
        return []


def fetch_nse_tickers(index: str = "NIFTY 500") -> List[str]:
    """
    Fetch NSE stock ticker symbols from a given index.
    Tries multiple sources with fallback.

    Args:
        index: NSE index name (e.g., "NIFTY 50", "NIFTY 100", "NIFTY 500")

    Returns:
        List of ticker symbols
    """
    # Source 1: NSE API (fastest, but blocked outside India)
    tickers = _fetch_from_nse_api(index)
    if tickers:
        print(f"Fetched {len(tickers)} tickers from NSE API")
        return tickers

    # Source 2: NSE Archives CSV (works globally)
    tickers = _fetch_from_nse_csv(index)
    if tickers:
        print(f"Fetched {len(tickers)} tickers from NSE CSV")
        return tickers

    # Source 3: Wikipedia (NIFTY 50 only, last resort)
    tickers = _fetch_from_wikipedia(index)
    if tickers:
        print(f"Fetched {len(tickers)} tickers from Wikipedia")
        return tickers

    print(f"Failed to fetch tickers for {index} from all sources")
    return []


def fetch_all_nse_tickers() -> List[str]:
    """
    Fetch all NSE equity tickers.

    Returns:
        List of all NSE ticker symbols
    """
    try:
        from nsepython import nse_eq_symbols
        return nse_eq_symbols()
    except ImportError:
        return fetch_nse_tickers("NIFTY 500")
    except Exception:
        return []


if __name__ == "__main__":
    print("Fetching NIFTY 50 tickers...")
    nifty50 = fetch_nse_tickers("NIFTY 50")
    print(f"Found {len(nifty50)} tickers:")
    print(nifty50)

    print("\nFetching NIFTY 100 tickers...")
    nifty100 = fetch_nse_tickers("NIFTY 100")
    print(f"Found {len(nifty100)} tickers")
