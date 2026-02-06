"""
Fetch NSE Ticker Labels/Symbols
"""

import requests
from typing import List, Optional


def fetch_nse_tickers(index: str = "NIFTY 500") -> List[str]:
    """
    Fetch NSE stock ticker symbols from a given index.

    Args:
        index: NSE index name (e.g., "NIFTY 50", "NIFTY 100", "NIFTY 500")

    Returns:
        List of ticker symbols
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # NSE API endpoint for index constituents
    index_encoded = index.replace(" ", "%20")
    url = f"https://www.nseindia.com/api/equity-stockIndices?index={index_encoded}"

    try:
        # First get cookies from main page
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        # Then fetch index data
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        tickers = [stock["symbol"] for stock in data.get("data", [])]

        # Remove index name itself if present
        tickers = [t for t in tickers if t != index.replace(" ", "")]

        return tickers

    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return []


def fetch_all_nse_tickers() -> List[str]:
    """
    Fetch all NSE equity tickers using nsepython library.

    Returns:
        List of all NSE ticker symbols
    """
    try:
        from nsepython import nse_eq_symbols
        return nse_eq_symbols()
    except ImportError:
        print("nsepython not installed. Install with: pip install nsepython")
        # Fallback to NIFTY 500
        return fetch_nse_tickers("NIFTY 500")
    except Exception as e:
        print(f"Error: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    print("Fetching NIFTY 50 tickers...")
    nifty50 = fetch_nse_tickers("NIFTY 50")
    print(f"Found {len(nifty50)} tickers:")
    print(nifty50)

    print("\nFetching all NSE tickers...")
    all_tickers = fetch_all_nse_tickers()
    print(f"Found {len(all_tickers)} tickers")
    print(f"First 20: {all_tickers[:20]}")
