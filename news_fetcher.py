"""
News Fetcher for NSE Stocks

Fetches news from multiple sources:
1. NewsAPI (primary, requires API key)
2. Google News RSS (fallback, no API key needed)
"""

import os
import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests
from diskcache import Cache

from config.news_settings import (
    NEWSAPI_KEY,
    NEWS_SOURCES,
    CACHE_CONFIG,
    get_company_name,
    get_enabled_sources
)


# Initialize cache
CACHE_DIR = os.path.join(os.path.dirname(__file__), CACHE_CONFIG.get('cache_dir', '.news_cache'))
news_cache = Cache(CACHE_DIR, size_limit=CACHE_CONFIG.get('max_size_mb', 100) * 1024 * 1024)


@dataclass
class NewsArticle:
    """Represents a single news article."""
    title: str
    source: str
    published_at: datetime
    url: str
    description: str = ""
    relevance_score: float = 1.0
    symbol: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'source': self.source,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'url': self.url,
            'description': self.description,
            'relevance_score': self.relevance_score,
            'symbol': self.symbol
        }

    @property
    def age_hours(self) -> float:
        """Get age of article in hours."""
        if not self.published_at:
            return 999
        delta = datetime.now() - self.published_at
        return delta.total_seconds() / 3600

    @property
    def full_text(self) -> str:
        """Get combined title and description for sentiment analysis."""
        return f"{self.title}. {self.description}" if self.description else self.title


def _get_cache_key(symbol: str, lookback_days: int) -> str:
    """Generate cache key for news query."""
    key_str = f"news_{symbol}_{lookback_days}_{datetime.now().strftime('%Y%m%d_%H')}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _parse_datetime(date_str: str) -> Optional[datetime]:
    """Parse datetime from various formats."""
    if not date_str:
        return None

    formats = [
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%d %H:%M:%S',
        '%a, %d %b %Y %H:%M:%S %z',
        '%a, %d %b %Y %H:%M:%S GMT',
        '%Y-%m-%d'
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.replace('+0000', '').strip(), fmt.replace(' %z', ''))
        except ValueError:
            continue

    return None


def fetch_newsapi(
    symbol: str,
    company_name: str = None,
    lookback_days: int = 3
) -> List[NewsArticle]:
    """
    Fetch news from NewsAPI.

    Args:
        symbol: Stock symbol (e.g., 'RELIANCE')
        company_name: Full company name for better search
        lookback_days: Days of history to fetch

    Returns:
        List of NewsArticle objects
    """
    if not NEWSAPI_KEY:
        return []

    if not company_name:
        company_name = get_company_name(symbol)

    # Build search query
    query = f'"{company_name}" OR "{symbol}" stock India'

    # Calculate date range
    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')

    url = "https://newsapi.org/v2/everything"
    params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 20,
        'apiKey': NEWSAPI_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = []
        for item in data.get('articles', []):
            article = NewsArticle(
                title=item.get('title', ''),
                source=item.get('source', {}).get('name', 'NewsAPI'),
                published_at=_parse_datetime(item.get('publishedAt', '')),
                url=item.get('url', ''),
                description=item.get('description', '') or '',
                symbol=symbol
            )
            # Calculate relevance based on symbol/company name presence
            text = f"{article.title} {article.description}".lower()
            if symbol.lower() in text or company_name.lower() in text:
                article.relevance_score = 1.0
            else:
                article.relevance_score = 0.7

            articles.append(article)

        return articles

    except requests.RequestException as e:
        print(f"NewsAPI error for {symbol}: {e}")
        return []


def fetch_google_news_rss(
    symbol: str,
    company_name: str = None,
    lookback_days: int = 3
) -> List[NewsArticle]:
    """
    Fetch news from Google News RSS feed.

    Args:
        symbol: Stock symbol
        company_name: Full company name
        lookback_days: Days of history to fetch

    Returns:
        List of NewsArticle objects
    """
    if not company_name:
        company_name = get_company_name(symbol)

    # Build search query
    query = f"{company_name} stock NSE"
    encoded_query = quote_plus(query)

    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()

        # Parse RSS XML
        root = ET.fromstring(response.content)
        articles = []

        cutoff_date = datetime.now() - timedelta(days=lookback_days)

        for item in root.findall('.//item'):
            title = item.find('title')
            link = item.find('link')
            pub_date = item.find('pubDate')
            source = item.find('source')

            if title is None or link is None:
                continue

            published_at = _parse_datetime(pub_date.text if pub_date is not None else '')

            # Skip old articles
            if published_at and published_at < cutoff_date:
                continue

            article = NewsArticle(
                title=title.text or '',
                source=source.text if source is not None else 'Google News',
                published_at=published_at,
                url=link.text or '',
                description='',  # RSS doesn't include description
                symbol=symbol
            )

            # Calculate relevance
            text = article.title.lower()
            if symbol.lower() in text or company_name.lower() in text:
                article.relevance_score = 1.0
            else:
                article.relevance_score = 0.8

            articles.append(article)

        return articles[:15]  # Limit to 15 articles

    except Exception as e:
        print(f"Google News RSS error for {symbol}: {e}")
        return []


def fetch_stock_news(
    symbol: str,
    lookback_days: int = 3,
    use_cache: bool = True
) -> List[NewsArticle]:
    """
    Fetch news for a stock from all enabled sources.

    Args:
        symbol: Stock symbol (e.g., 'RELIANCE')
        lookback_days: Days of history to fetch (default: 3)
        use_cache: Whether to use cached results

    Returns:
        List of NewsArticle objects, sorted by recency
    """
    # Check cache
    cache_key = _get_cache_key(symbol, lookback_days)
    if use_cache and cache_key in news_cache:
        cached = news_cache.get(cache_key)
        if cached:
            return cached

    company_name = get_company_name(symbol)
    all_articles = []
    seen_titles = set()

    # Fetch from NewsAPI (if enabled and has API key)
    if NEWS_SOURCES.get('newsapi', {}).get('enabled', False) and NEWSAPI_KEY:
        newsapi_articles = fetch_newsapi(symbol, company_name, lookback_days)
        for article in newsapi_articles:
            title_hash = hashlib.md5(article.title.lower().encode()).hexdigest()
            if title_hash not in seen_titles:
                seen_titles.add(title_hash)
                all_articles.append(article)

    # Fetch from Google News RSS (fallback)
    if NEWS_SOURCES.get('google_news', {}).get('enabled', False):
        google_articles = fetch_google_news_rss(symbol, company_name, lookback_days)
        for article in google_articles:
            title_hash = hashlib.md5(article.title.lower().encode()).hexdigest()
            if title_hash not in seen_titles:
                seen_titles.add(title_hash)
                all_articles.append(article)

    # Sort by recency (newest first)
    all_articles.sort(key=lambda x: x.published_at or datetime.min, reverse=True)

    # Cache results
    if use_cache and all_articles:
        expire_seconds = CACHE_CONFIG.get('expire_hours', 2) * 3600
        news_cache.set(cache_key, all_articles, expire=expire_seconds)

    return all_articles


def fetch_multiple_stock_news(
    symbols: List[str],
    lookback_days: int = 3,
    max_workers: int = 5
) -> Dict[str, List[NewsArticle]]:
    """
    Fetch news for multiple stocks.

    Args:
        symbols: List of stock symbols
        lookback_days: Days of history
        max_workers: Parallel workers

    Returns:
        Dict mapping symbol to list of articles
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}

    def fetch_one(symbol: str) -> tuple:
        articles = fetch_stock_news(symbol, lookback_days)
        return symbol, articles

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, s): s for s in symbols}

        for future in as_completed(futures):
            try:
                symbol, articles = future.result()
                results[symbol] = articles
            except Exception as e:
                symbol = futures[future]
                print(f"Error fetching news for {symbol}: {e}")
                results[symbol] = []

    return results


def clear_news_cache():
    """Clear the news cache."""
    news_cache.clear()
    print("News cache cleared.")


def get_cache_stats() -> Dict:
    """Get news cache statistics."""
    return {
        'size_bytes': news_cache.volume(),
        'size_mb': round(news_cache.volume() / (1024 * 1024), 2),
        'items': len(news_cache)
    }


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        symbol = "RELIANCE"

    print(f"\n{'='*60}")
    print(f"Fetching news for {symbol}")
    print(f"{'='*60}\n")

    articles = fetch_stock_news(symbol, lookback_days=3)

    if not articles:
        print("No news articles found.")
    else:
        print(f"Found {len(articles)} articles:\n")
        for i, article in enumerate(articles[:10], 1):
            age = f"{article.age_hours:.1f}h ago" if article.age_hours < 999 else "Unknown"
            print(f"{i}. [{article.source}] ({age})")
            print(f"   {article.title}")
            print(f"   URL: {article.url[:60]}...")
            print()

    print(f"\nCache stats: {get_cache_stats()}")
