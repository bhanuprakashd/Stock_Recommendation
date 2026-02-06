"""
News Integration Configuration

Settings for news fetching, sentiment analysis, and trigger modes.
"""

import os
from typing import Dict, List


# =============================================================================
# API CONFIGURATION
# =============================================================================

# NewsAPI settings (https://newsapi.org)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
NEWSAPI_ENABLED = bool(NEWSAPI_KEY)

# Data source priorities
NEWS_SOURCES = {
    'newsapi': {
        'enabled': NEWSAPI_ENABLED,
        'base_url': 'https://newsapi.org/v2/everything',
        'rate_limit_per_day': 100,  # Free tier
        'priority': 1
    },
    'moneycontrol': {
        'enabled': True,
        'priority': 2
    },
    'google_news': {
        'enabled': True,
        'priority': 3  # Fallback via RSS
    }
}


# =============================================================================
# SENTIMENT ANALYSIS CONFIGURATION
# =============================================================================

SENTIMENT_CONFIG = {
    # Model selection: 'finbert' (accurate) or 'keyword' (fast)
    'model': 'finbert',

    # FinBERT settings
    'finbert': {
        'model_name': 'ProsusAI/finbert',
        'use_gpu': True,  # Will fallback to CPU if unavailable
        'batch_size': 16,
        'max_length': 512
    },

    # Sentiment thresholds (FinBERT output is 0-1)
    'threshold_positive': 0.65,
    'threshold_negative': 0.35,

    # Score mapping: FinBERT label → numeric score (0-100)
    'score_mapping': {
        'positive': 80,   # Base score for positive
        'neutral': 50,    # Base score for neutral
        'negative': 20    # Base score for negative
    }
}

# Financial keyword dictionaries for fallback
FINANCIAL_KEYWORDS = {
    'positive': [
        'beat', 'beats', 'exceeds', 'exceeding', 'exceeded',
        'upgrade', 'upgraded', 'upgrades',
        'gains', 'gain', 'gained', 'rally', 'rallies', 'rallied',
        'bullish', 'bull', 'surge', 'surges', 'surged',
        'approval', 'approved', 'approves',
        'acquisition', 'acquires', 'acquired', 'merger',
        'profit', 'profitable', 'profits',
        'growth', 'growing', 'grew', 'expands', 'expansion',
        'record', 'breakthrough', 'success', 'successful',
        'outperform', 'outperforms', 'outperformed',
        'strong', 'stronger', 'strength',
        'positive', 'optimistic', 'confidence', 'confident'
    ],
    'negative': [
        'miss', 'misses', 'missed', 'falls short',
        'downgrade', 'downgraded', 'downgrades',
        'loss', 'losses', 'losing', 'lost',
        'bearish', 'bear', 'decline', 'declines', 'declined',
        'delay', 'delayed', 'delays', 'postpone', 'postponed',
        'regulatory', 'investigation', 'probe', 'scrutiny',
        'lawsuit', 'legal', 'litigation', 'penalty', 'fine',
        'bankruptcy', 'default', 'restructuring',
        'layoff', 'layoffs', 'job cuts', 'workforce reduction',
        'weak', 'weaker', 'weakness',
        'concern', 'concerns', 'worried', 'warning',
        'underperform', 'underperforms', 'underperformed',
        'negative', 'pessimistic', 'uncertainty'
    ],
    'event_triggers': {
        'earnings': ['earnings', 'EPS', 'Q1', 'Q2', 'Q3', 'Q4', 'quarterly', 'results', 'guidance'],
        'M&A': ['merger', 'acquisition', 'acquires', 'acquired', 'deal', 'buyout', 'takeover'],
        'regulatory': ['regulatory', 'compliance', 'approval', 'FDA', 'SEBI', 'RBI', 'ban', 'investigation'],
        'dividend': ['dividend', 'payout', 'distribution', 'buyback', 'bonus'],
        'analyst': ['upgrade', 'downgrade', 'rating', 'target price', 'analyst', 'coverage']
    }
}


# =============================================================================
# TRIGGER MODE CONFIGURATION
# =============================================================================

TRIGGER_CONFIG = {
    # Mode 1: Entry Signal (Aggressive)
    # Upgrades borderline stocks on positive news
    'entry_signal': {
        'enabled': True,
        'news_score_threshold': 75,      # Min news score to trigger
        'composite_score_threshold': 50,  # Min composite to consider
        'max_recency_hours': 24,         # News must be < 24 hours old
        'confidence_boost': 20,          # Points added to confidence
        'signal_upgrade': {
            'WAIT': 'HOLD',
            'HOLD': 'BUY'
        }
    },

    # Mode 2: Confirmation (Conservative)
    # Boosts already-good picks when all pillars align
    'confirmation': {
        'enabled': True,
        'news_score_threshold': 70,
        'composite_score_threshold': 60,
        'technical_score_threshold': 55,
        'fundamental_score_threshold': 55,
        'confidence_boost': 15,
        'signal_upgrade': {
            'HOLD': 'BUY',
            'BUY': 'STRONG BUY'
        }
    },

    # Mode 3: Risk Filter (Protective)
    # Downgrades stocks with negative news
    'risk_filter': {
        'enabled': True,
        'negative_news_threshold': 70,   # Score below 30 = negative
        'max_recency_hours': 48,         # Check news up to 48 hours old
        'confidence_penalty': -15,       # Points removed from confidence
        'signal_downgrade': {
            'STRONG BUY': 'BUY',
            'BUY': 'HOLD',
            'HOLD': 'WAIT'
        },
        'add_warning': True
    },

    # Mode 4: Dynamic Recalculation
    # Triggers full rescan on major events
    'dynamic_recalc': {
        'enabled': True,
        'major_events': ['earnings', 'M&A', 'regulatory'],
        'score_change_threshold': 15,    # Rescan if score would change > 15 pts
        'cooldown_hours': 4              # Don't rescan same stock within 4 hours
    }
}


# =============================================================================
# CONFIDENCE ADJUSTMENT SETTINGS
# =============================================================================

CONFIDENCE_ADJUSTMENT = {
    # Base adjustment range
    'min_adjustment': -15,
    'max_adjustment': 20,

    # Calculation formula: (news_score - 50) / divisor
    'divisor': 2.5,  # Results in -20 to +20 base range

    # Bonuses
    'recency_bonus': {
        'threshold_hours': 24,
        'bonus_points': 5
    },
    'event_bonus': {
        'qualifying_events': ['earnings', 'M&A'],
        'bonus_points': 5
    },
    'multi_source_bonus': {
        'min_sources': 2,
        'bonus_points': 3
    }
}


# =============================================================================
# CACHING CONFIGURATION
# =============================================================================

CACHE_CONFIG = {
    'enabled': True,
    'provider': 'diskcache',
    'cache_dir': '.news_cache',
    'expire_hours': 2,        # News cache expires after 2 hours
    'max_size_mb': 100,
    'sentiment_cache_hours': 4  # Sentiment results cached longer
}


# =============================================================================
# RISK SAFEGUARDS
# =============================================================================

NEWS_SAFEGUARDS = {
    # Prevent over-reliance on news
    'max_confidence_from_news': 20,      # Cap news contribution to confidence

    # Data quality requirements
    'min_articles_for_signal': 1,        # Need at least 1 article
    'max_news_age_hours': 72,            # Ignore news older than 3 days
    'require_multi_source': False,       # Optional: require 2+ sources

    # Trading safeguards
    'max_signal_upgrade_per_day': 5,     # Max stocks upgraded by news per day
    'avoid_earnings_day_trades': False,  # Optional: skip same-day earnings

    # Sector limits
    'max_sector_concentration': 3,       # Max stocks in same sector from news triggers
}


# =============================================================================
# COMPANY NAME MAPPINGS
# =============================================================================

# Map stock symbols to company names for better news search
COMPANY_NAMES = {
    'RELIANCE': 'Reliance Industries',
    'TCS': 'Tata Consultancy Services',
    'INFY': 'Infosys',
    'HDFCBANK': 'HDFC Bank',
    'ICICIBANK': 'ICICI Bank',
    'SBIN': 'State Bank of India',
    'BHARTIARTL': 'Bharti Airtel',
    'ITC': 'ITC Limited',
    'KOTAKBANK': 'Kotak Mahindra Bank',
    'LT': 'Larsen & Toubro',
    'AXISBANK': 'Axis Bank',
    'MARUTI': 'Maruti Suzuki',
    'TITAN': 'Titan Company',
    'BAJFINANCE': 'Bajaj Finance',
    'WIPRO': 'Wipro',
    'HCLTECH': 'HCL Technologies',
    'SUNPHARMA': 'Sun Pharma',
    'ASIANPAINT': 'Asian Paints',
    'TATAMOTORS': 'Tata Motors',
    'TATASTEEL': 'Tata Steel',
    'POWERGRID': 'Power Grid Corporation',
    'NTPC': 'NTPC Limited',
    'ONGC': 'Oil and Natural Gas Corporation',
    'COALINDIA': 'Coal India',
    'ADANIPORTS': 'Adani Ports',
    'ADANIENT': 'Adani Enterprises',
    'TECHM': 'Tech Mahindra',
    'ULTRACEMCO': 'UltraTech Cement',
    'HINDALCO': 'Hindalco Industries',
    'JSWSTEEL': 'JSW Steel',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_company_name(symbol: str) -> str:
    """Get company name for a symbol, or return symbol if not found."""
    return COMPANY_NAMES.get(symbol.upper(), symbol)


def get_enabled_sources() -> List[str]:
    """Get list of enabled news sources."""
    return [name for name, config in NEWS_SOURCES.items() if config.get('enabled', False)]


def is_trigger_enabled(mode: str) -> bool:
    """Check if a trigger mode is enabled."""
    return TRIGGER_CONFIG.get(mode, {}).get('enabled', False)
