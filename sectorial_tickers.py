"""
Sectorial Tickers - Get NSE stocks filtered by sector
"""

import yfinance as yf
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json

from nse_tickers import fetch_nse_tickers, fetch_all_nse_tickers


# Predefined sector mappings for common NSE stocks
SECTOR_CACHE = {}


def get_stock_sector(symbol: str) -> Optional[str]:
    """
    Get sector for a single stock.

    Args:
        symbol: NSE stock symbol

    Returns:
        Sector name or None
    """
    if symbol in SECTOR_CACHE:
        return SECTOR_CACHE[symbol]

    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info
        sector = info.get('sector')
        if sector:
            SECTOR_CACHE[symbol] = sector
        return sector
    except Exception:
        return None


def get_sectors_for_tickers(
    symbols: List[str],
    max_workers: int = 10,
    delay: float = 0.1
) -> Dict[str, str]:
    """
    Get sectors for multiple tickers.

    Args:
        symbols: List of stock symbols
        max_workers: Parallel workers
        delay: Delay between requests

    Returns:
        Dict mapping symbol -> sector
    """
    results = {}

    def fetch_sector(symbol: str) -> Tuple[str, Optional[str]]:
        time.sleep(delay)
        return symbol, get_stock_sector(symbol)

    print(f"Fetching sectors for {len(symbols)} stocks...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_sector, s): s for s in symbols}

        for i, future in enumerate(as_completed(futures), 1):
            symbol, sector = future.result()
            if sector:
                results[symbol] = sector

            if i % 20 == 0:
                print(f"Progress: {i}/{len(symbols)}")

    print(f"Found sectors for {len(results)} stocks")
    return results


def get_available_sectors(index: str = "NIFTY 200") -> List[str]:
    """
    Get list of all available sectors in an index.

    Args:
        index: NSE index name

    Returns:
        List of unique sector names
    """
    symbols = fetch_nse_tickers(index)
    symbols = [s for s in symbols if s != index.replace(" ", "")]

    sector_map = get_sectors_for_tickers(symbols[:100])  # Sample first 100
    sectors = sorted(set(sector_map.values()))

    return sectors


def get_tickers_by_sector(
    sector: str,
    index: str = "NIFTY 200",
    symbols: List[str] = None
) -> List[str]:
    """
    Get all tickers belonging to a specific sector.

    Args:
        sector: Sector name (e.g., "Technology", "Financial Services")
        index: NSE index to search in
        symbols: Pre-defined list of symbols (optional)

    Returns:
        List of stock symbols in that sector
    """
    if symbols is None:
        symbols = fetch_nse_tickers(index)
        symbols = [s for s in symbols if s != index.replace(" ", "")]

    sector_map = get_sectors_for_tickers(symbols)

    # Case-insensitive matching
    sector_lower = sector.lower()
    matching = [
        sym for sym, sec in sector_map.items()
        if sector_lower in sec.lower()
    ]

    return sorted(matching)


def get_tickers_grouped_by_sector(
    index: str = "NIFTY 100",
    symbols: List[str] = None
) -> Dict[str, List[str]]:
    """
    Get all tickers grouped by their sectors.

    Args:
        index: NSE index to use
        symbols: Pre-defined list of symbols (optional)

    Returns:
        Dict mapping sector -> list of symbols
    """
    if symbols is None:
        symbols = fetch_nse_tickers(index)
        symbols = [s for s in symbols if s != index.replace(" ", "")]

    sector_map = get_sectors_for_tickers(symbols)

    # Group by sector
    grouped = {}
    for symbol, sector in sector_map.items():
        if sector not in grouped:
            grouped[sector] = []
        grouped[sector].append(symbol)

    # Sort symbols within each sector
    for sector in grouped:
        grouped[sector].sort()

    return dict(sorted(grouped.items()))


# Common sector shortcuts
SECTOR_ALIASES = {
    "it": "Technology",
    "tech": "Technology",
    "technology": "Technology",
    "banking": "Financial Services",
    "bank": "Financial Services",
    "finance": "Financial Services",
    "financial": "Financial Services",
    "pharma": "Healthcare",
    "healthcare": "Healthcare",
    "health": "Healthcare",
    "auto": "Consumer Cyclical",
    "automobile": "Consumer Cyclical",
    "automotive": "Consumer Cyclical",
    "fmcg": "Consumer Defensive",
    "consumer": "Consumer Defensive",
    "energy": "Energy",
    "oil": "Energy",
    "power": "Utilities",
    "utilities": "Utilities",
    "metal": "Basic Materials",
    "metals": "Basic Materials",
    "materials": "Basic Materials",
    "infra": "Industrials",
    "infrastructure": "Industrials",
    "industrial": "Industrials",
    "realty": "Real Estate",
    "real estate": "Real Estate",
    "telecom": "Communication Services",
    "communication": "Communication Services",
}


def resolve_sector_name(sector_input: str) -> str:
    """
    Resolve sector alias to full sector name.

    Args:
        sector_input: Sector name or alias

    Returns:
        Full sector name
    """
    return SECTOR_ALIASES.get(sector_input.lower(), sector_input)


def get_sector_tickers(sector: str, index: str = "NIFTY 200") -> List[str]:
    """
    Convenience function to get tickers by sector with alias support.

    Args:
        sector: Sector name or alias (e.g., "IT", "banking", "pharma")
        index: NSE index to search

    Returns:
        List of stock symbols
    """
    resolved_sector = resolve_sector_name(sector)
    return get_tickers_by_sector(resolved_sector, index)


def print_sector_summary(grouped: Dict[str, List[str]]):
    """Print a formatted summary of sectors and their stocks."""
    print("\n" + "=" * 60)
    print("SECTOR-WISE STOCK DISTRIBUTION")
    print("=" * 60)

    for sector, stocks in grouped.items():
        print(f"\n{sector} ({len(stocks)} stocks)")
        print("-" * 40)
        # Print in rows of 5
        for i in range(0, len(stocks), 5):
            row = stocks[i:i+5]
            print("  " + ", ".join(row))


# Pre-defined sector stock lists (for quick access without API calls)
PREDEFINED_SECTORS = {
    "Technology": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS",
        "COFORGE", "PERSISTENT", "LTTS", "TATAELXSI", "MINDTREE"
    ],
    "Financial Services": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "BANKBARODA",
        "PNB", "INDUSINDBK", "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE",
        "ICICIPRULI", "ICICIGI", "CHOLAFIN", "SHRIRAMFIN", "M&MFIN"
    ],
    "Healthcare": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP", "FORTIS",
        "BIOCON", "LUPIN", "AUROPHARMA", "TORNTPHARM", "ALKEM", "GLAND"
    ],
    "Consumer Cyclical": [
        "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT",
        "TITAN", "TRENT", "PAGEIND", "RELAXO", "BATAINDIA"
    ],
    "Consumer Defensive": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO",
        "GODREJCP", "COLPAL", "TATACONSUM", "VBL", "UBL"
    ],
    "Energy": [
        "RELIANCE", "ONGC", "IOC", "BPCL", "HINDPETRO", "GAIL", "PETRONET",
        "OIL", "MRPL", "CASTROLIND"
    ],
    "Industrials": [
        "LT", "SIEMENS", "ABB", "HAVELLS", "BHARATFORG", "CUMMINSIND",
        "THERMAX", "GRINDWELL", "HONAUT", "BEL", "HAL", "BHEL"
    ],
    "Basic Materials": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA", "NMDC",
        "JINDALSTEL", "SAIL", "NATIONALUM", "HINDZINC"
    ],
    "Utilities": [
        "NTPC", "POWERGRID", "ADANIGREEN", "TATAPOWER", "TORNTPOWER",
        "NHPC", "SJVN", "JSW ENERGY", "CESC"
    ],
    "Real Estate": [
        "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "BRIGADE",
        "SOBHA", "LODHA", "PHOENIXLTD"
    ],
    "Communication Services": [
        "BHARTIARTL", "INDIAMART", "JUSTDIAL", "ROUTE", "NAZARA"
    ]
}


def get_predefined_sector_tickers(sector: str) -> List[str]:
    """
    Get tickers from predefined sector lists (no API calls needed).

    Args:
        sector: Sector name or alias

    Returns:
        List of stock symbols
    """
    resolved = resolve_sector_name(sector)

    # Find matching sector (case-insensitive partial match)
    for sec_name, stocks in PREDEFINED_SECTORS.items():
        if resolved.lower() in sec_name.lower():
            return stocks

    return []


def get_all_sector_tickers(
    sector: str,
    index: str = "NIFTY 500",
    use_cache: bool = True
) -> List[str]:
    """
    Get ALL stocks in a sector from NSE (not just predefined).

    Args:
        sector: Sector name or alias
        index: Index to search (larger index = more stocks)
        use_cache: Use cached sector data if available

    Returns:
        List of all stock symbols in that sector
    """
    resolved = resolve_sector_name(sector)

    # Fetch all symbols from index
    print(f"Fetching all {resolved} stocks from {index}...")
    symbols = fetch_nse_tickers(index)
    symbols = [s for s in symbols if s not in [index.replace(" ", ""), index]]

    # Get sectors for all symbols
    sector_map = get_sectors_for_tickers(symbols)

    # Filter by sector (case-insensitive partial match)
    sector_lower = resolved.lower()
    matching = [
        sym for sym, sec in sector_map.items()
        if sector_lower in sec.lower() or sec.lower() in sector_lower
    ]

    print(f"Found {len(matching)} stocks in {resolved} sector")
    return sorted(matching)


def get_complete_sector_breakdown(index: str = "NIFTY 500") -> Dict[str, List[str]]:
    """
    Get complete sector breakdown for an entire index.

    Args:
        index: NSE index name

    Returns:
        Dict mapping sector -> list of ALL symbols in that sector
    """
    print(f"Fetching complete sector breakdown for {index}...")
    symbols = fetch_nse_tickers(index)
    symbols = [s for s in symbols if s not in [index.replace(" ", ""), index]]

    return get_tickers_grouped_by_sector(symbols=symbols)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # Get ALL stocks in a sector
        if len(sys.argv) > 2:
            sector = " ".join(sys.argv[2:])
            tickers = get_all_sector_tickers(sector, index="NIFTY 500")
            if tickers:
                print(f"\nAll {len(tickers)} stocks in sector:")
                for i, t in enumerate(tickers, 1):
                    print(f"  {i:3}. {t}")
        else:
            print("Usage: python sectorial_tickers.py --all <sector>")
            print("Example: python sectorial_tickers.py --all banking")

    elif len(sys.argv) > 1 and sys.argv[1] == "--breakdown":
        # Full sector breakdown
        grouped = get_complete_sector_breakdown("NIFTY 200")
        print_sector_summary(grouped)

    elif len(sys.argv) > 1:
        # Get tickers for specified sector
        sector = " ".join(sys.argv[1:])
        print(f"\nFetching tickers for sector: {sector}")

        # Try predefined first (fast)
        tickers = get_predefined_sector_tickers(sector)
        if tickers:
            print(f"\nFrom predefined list ({len(tickers)} major stocks):")
            print(", ".join(tickers))
            print(f"\nUse --all {sector} to get ALL stocks in this sector")
        else:
            # Fetch from API
            tickers = get_sector_tickers(sector, index="NIFTY 100")
            if tickers:
                print(f"\nFound {len(tickers)} stocks:")
                print(", ".join(tickers))
            else:
                print(f"\nNo stocks found for sector: {sector}")
                print("\nAvailable sectors:")
                for sec in PREDEFINED_SECTORS.keys():
                    print(f"  - {sec}")

    else:
        # Show all predefined sectors
        print("\n" + "=" * 60)
        print("PREDEFINED SECTOR STOCK LISTS")
        print("=" * 60)
        print("\nUsage: python sectorial_tickers.py <sector>")
        print("Example: python sectorial_tickers.py banking")
        print("         python sectorial_tickers.py IT")
        print("         python sectorial_tickers.py pharma")

        print("\n" + "-" * 60)
        print("Available Sectors & Aliases:")
        print("-" * 60)

        for sector, stocks in PREDEFINED_SECTORS.items():
            aliases = [k for k, v in SECTOR_ALIASES.items() if v == sector]
            alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
            print(f"\n{sector}{alias_str}")
            print(f"  Stocks: {', '.join(stocks[:8])}{'...' if len(stocks) > 8 else ''}")
            print(f"  Count: {len(stocks)}")
