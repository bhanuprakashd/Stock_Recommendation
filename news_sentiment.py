"""
News Sentiment Analysis

Uses FinBERT for financial sentiment analysis with keyword-based fallback.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Tuple
import warnings

from news_fetcher import NewsArticle
from config.news_settings import (
    SENTIMENT_CONFIG,
    FINANCIAL_KEYWORDS,
    CONFIDENCE_ADJUSTMENT
)

# Suppress transformers warnings
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class NewsSentiment:
    """Result of news sentiment analysis."""
    overall_score: float           # 0-100 (weighted average)
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    event_type: Optional[str] = None  # 'earnings', 'M&A', 'regulatory', etc.
    recency_hours: float = 999     # Hours since most recent news
    articles_analyzed: int = 0
    top_signals: List[str] = field(default_factory=list)
    article_sentiments: List[Dict] = field(default_factory=list)
    model_used: str = "keyword"    # 'finbert' or 'keyword'

    @property
    def sentiment_label(self) -> str:
        """Get overall sentiment label."""
        if self.overall_score >= 65:
            return "Positive"
        elif self.overall_score <= 35:
            return "Negative"
        else:
            return "Neutral"

    @property
    def is_positive(self) -> bool:
        """Check if overall sentiment is positive."""
        return self.overall_score >= SENTIMENT_CONFIG.get('threshold_positive', 0.65) * 100

    @property
    def is_negative(self) -> bool:
        """Check if overall sentiment is negative."""
        return self.overall_score <= SENTIMENT_CONFIG.get('threshold_negative', 0.35) * 100

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'overall_score': round(self.overall_score, 1),
            'sentiment_label': self.sentiment_label,
            'positive_count': self.positive_count,
            'negative_count': self.negative_count,
            'neutral_count': self.neutral_count,
            'event_type': self.event_type,
            'recency_hours': round(self.recency_hours, 1),
            'articles_analyzed': self.articles_analyzed,
            'top_signals': self.top_signals[:5],
            'model_used': self.model_used
        }


class FinBERTAnalyzer:
    """
    FinBERT-based sentiment analyzer.

    Uses ProsusAI/finbert model for financial sentiment classification.
    Falls back to keyword-based analysis if model unavailable.
    """

    _instance = None
    _model = None
    _tokenizer = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not FinBERTAnalyzer._initialized:
            self._load_model()
            FinBERTAnalyzer._initialized = True

    def _load_model(self):
        """Load FinBERT model (lazy initialization)."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            import torch

            model_name = SENTIMENT_CONFIG['finbert']['model_name']

            # Check for GPU
            device = 0 if torch.cuda.is_available() and SENTIMENT_CONFIG['finbert']['use_gpu'] else -1

            # Load model
            self._classifier = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=device,
                max_length=SENTIMENT_CONFIG['finbert']['max_length'],
                truncation=True
            )

            FinBERTAnalyzer._model = self._classifier
            print(f"FinBERT loaded successfully (device: {'GPU' if device >= 0 else 'CPU'})")

        except ImportError as e:
            print(f"FinBERT not available (missing dependencies): {e}")
            print("Falling back to keyword-based sentiment analysis.")
            FinBERTAnalyzer._model = None

        except Exception as e:
            print(f"Error loading FinBERT: {e}")
            print("Falling back to keyword-based sentiment analysis.")
            FinBERTAnalyzer._model = None

    @property
    def is_available(self) -> bool:
        """Check if FinBERT model is loaded."""
        return FinBERTAnalyzer._model is not None

    def analyze_text(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (label, score) where label is 'positive'/'negative'/'neutral'
        """
        if not self.is_available:
            return self._keyword_analyze(text)

        try:
            # Truncate to max length
            max_len = SENTIMENT_CONFIG['finbert']['max_length']
            text = text[:max_len * 4]  # Rough char estimate

            result = FinBERTAnalyzer._model(text)[0]
            label = result['label'].lower()
            score = result['score']

            return label, score

        except Exception as e:
            print(f"FinBERT analysis error: {e}")
            return self._keyword_analyze(text)

    def _keyword_analyze(self, text: str) -> Tuple[str, float]:
        """Fallback keyword-based sentiment analysis."""
        text_lower = text.lower()

        pos_count = sum(1 for word in FINANCIAL_KEYWORDS['positive'] if word in text_lower)
        neg_count = sum(1 for word in FINANCIAL_KEYWORDS['negative'] if word in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return 'neutral', 0.5

        if pos_count > neg_count:
            score = min(0.5 + (pos_count - neg_count) * 0.1, 0.95)
            return 'positive', score
        elif neg_count > pos_count:
            score = max(0.5 - (neg_count - pos_count) * 0.1, 0.05)
            return 'negative', 1 - score
        else:
            return 'neutral', 0.5

    def analyze_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Analyze multiple texts."""
        if not self.is_available:
            return [self._keyword_analyze(t) for t in texts]

        try:
            batch_size = SENTIMENT_CONFIG['finbert']['batch_size']
            results = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # Truncate each text
                batch = [t[:SENTIMENT_CONFIG['finbert']['max_length'] * 4] for t in batch]

                batch_results = FinBERTAnalyzer._model(batch)
                for r in batch_results:
                    results.append((r['label'].lower(), r['score']))

            return results

        except Exception as e:
            print(f"FinBERT batch analysis error: {e}")
            return [self._keyword_analyze(t) for t in texts]


def detect_event_type(articles: List[NewsArticle]) -> Optional[str]:
    """
    Detect if articles mention specific event types.

    Args:
        articles: List of news articles

    Returns:
        Event type ('earnings', 'M&A', 'regulatory', 'dividend', 'analyst') or None
    """
    event_keywords = FINANCIAL_KEYWORDS['event_triggers']
    event_counts = {event: 0 for event in event_keywords}

    for article in articles:
        text = f"{article.title} {article.description}".lower()
        for event, keywords in event_keywords.items():
            if any(kw.lower() in text for kw in keywords):
                event_counts[event] += 1

    # Return most common event type if significant
    if event_counts:
        max_event = max(event_counts.items(), key=lambda x: x[1])
        if max_event[1] >= 1:  # At least 1 article mentions it
            return max_event[0]

    return None


def calculate_recency_weight(age_hours: float) -> float:
    """
    Calculate recency weight for an article.

    Recent news is weighted higher.

    Args:
        age_hours: Age of article in hours

    Returns:
        Weight between 0.5 and 1.0
    """
    if age_hours <= 24:
        return 1.0
    elif age_hours <= 72:  # 3 days
        return 0.9
    elif age_hours <= 168:  # 7 days
        return 0.7
    else:
        return 0.5


def extract_top_signals(articles: List[NewsArticle], sentiments: List[Tuple[str, float]]) -> List[str]:
    """
    Extract top signals from analyzed articles.

    Args:
        articles: List of articles
        sentiments: List of (label, score) tuples

    Returns:
        List of signal strings
    """
    signals = []

    for article, (label, score) in zip(articles, sentiments):
        if label == 'positive' and score > 0.7:
            # Extract key phrases from title
            title = article.title
            if len(title) > 80:
                title = title[:77] + "..."
            signals.append(f"[+] {title}")
        elif label == 'negative' and score > 0.7:
            title = article.title
            if len(title) > 80:
                title = title[:77] + "..."
            signals.append(f"[-] {title}")

    return signals[:5]


def analyze_news_sentiment(
    articles: List[NewsArticle],
    use_finbert: bool = True
) -> NewsSentiment:
    """
    Analyze sentiment of news articles.

    Args:
        articles: List of NewsArticle objects
        use_finbert: Whether to use FinBERT (default True)

    Returns:
        NewsSentiment object with analysis results
    """
    if not articles:
        return NewsSentiment(
            overall_score=50.0,
            articles_analyzed=0,
            top_signals=["No recent news found"]
        )

    # Initialize analyzer
    analyzer = FinBERTAnalyzer() if use_finbert else None

    # Prepare texts for analysis
    texts = [article.full_text for article in articles]

    # Analyze sentiment
    if use_finbert and analyzer and analyzer.is_available:
        sentiments = analyzer.analyze_batch(texts)
        model_used = "finbert"
    else:
        # Keyword fallback
        sentiments = [KeywordSentimentAnalyzer.analyze(t) for t in texts]
        model_used = "keyword"

    # Calculate weighted scores
    score_mapping = SENTIMENT_CONFIG['score_mapping']
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    weighted_scores = []
    article_sentiments = []

    for article, (label, confidence) in zip(articles, sentiments):
        # Get recency weight
        recency_weight = calculate_recency_weight(article.age_hours)

        # Get base score for label
        base_score = score_mapping.get(label, 50)

        # Adjust score based on confidence
        if label == 'positive':
            score = base_score + (confidence - 0.5) * 40  # 60-100 range
            positive_count += 1
        elif label == 'negative':
            score = base_score - (confidence - 0.5) * 40  # 0-40 range
            negative_count += 1
        else:
            score = base_score
            neutral_count += 1

        # Apply relevance and recency weights
        weighted_score = score * article.relevance_score * recency_weight
        weighted_scores.append((weighted_score, recency_weight * article.relevance_score))

        # Store individual article sentiment
        article_sentiments.append({
            'title': article.title[:100],
            'source': article.source,
            'label': label,
            'confidence': round(confidence, 2),
            'score': round(score, 1),
            'age_hours': round(article.age_hours, 1)
        })

    # Calculate weighted average
    total_weight = sum(w for _, w in weighted_scores)
    if total_weight > 0:
        overall_score = sum(s * w for s, w in weighted_scores) / total_weight
    else:
        overall_score = 50.0

    # Clamp to 0-100
    overall_score = max(0, min(100, overall_score))

    # Get recency of most recent article
    recency_hours = min(a.age_hours for a in articles) if articles else 999

    # Detect event type
    event_type = detect_event_type(articles)

    # Extract top signals
    top_signals = extract_top_signals(articles, sentiments)

    return NewsSentiment(
        overall_score=overall_score,
        positive_count=positive_count,
        negative_count=negative_count,
        neutral_count=neutral_count,
        event_type=event_type,
        recency_hours=recency_hours,
        articles_analyzed=len(articles),
        top_signals=top_signals,
        article_sentiments=article_sentiments,
        model_used=model_used
    )


class KeywordSentimentAnalyzer:
    """Simple keyword-based sentiment analyzer (fallback)."""

    @staticmethod
    def analyze(text: str) -> Tuple[str, float]:
        """
        Analyze sentiment using keyword matching.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (label, confidence)
        """
        text_lower = text.lower()

        pos_keywords = FINANCIAL_KEYWORDS['positive']
        neg_keywords = FINANCIAL_KEYWORDS['negative']

        pos_count = sum(1 for word in pos_keywords if word in text_lower)
        neg_count = sum(1 for word in neg_keywords if word in text_lower)

        total = pos_count + neg_count

        if total == 0:
            return 'neutral', 0.5

        pos_ratio = pos_count / total

        if pos_ratio > 0.6:
            confidence = 0.5 + (pos_ratio - 0.5) * 0.8
            return 'positive', min(confidence, 0.95)
        elif pos_ratio < 0.4:
            confidence = 0.5 + (0.5 - pos_ratio) * 0.8
            return 'negative', min(confidence, 0.95)
        else:
            return 'neutral', 0.5


def calculate_confidence_adjustment(sentiment: NewsSentiment) -> float:
    """
    Calculate confidence adjustment based on news sentiment.

    Args:
        sentiment: NewsSentiment analysis result

    Returns:
        Confidence adjustment (-15 to +20 points)
    """
    config = CONFIDENCE_ADJUSTMENT

    # Base adjustment: (score - 50) / divisor
    base_adj = (sentiment.overall_score - 50) / config['divisor']

    # Recency bonus
    recency_bonus = 0
    if sentiment.recency_hours < config['recency_bonus']['threshold_hours']:
        recency_bonus = config['recency_bonus']['bonus_points']

    # Event bonus
    event_bonus = 0
    if sentiment.event_type in config['event_bonus']['qualifying_events']:
        event_bonus = config['event_bonus']['bonus_points']

    # Multi-source bonus (if multiple articles)
    multi_bonus = 0
    if sentiment.articles_analyzed >= config['multi_source_bonus']['min_sources']:
        multi_bonus = config['multi_source_bonus']['bonus_points']

    # Calculate total adjustment
    total_adj = base_adj + recency_bonus + event_bonus + multi_bonus

    # Clamp to configured range
    return max(config['min_adjustment'], min(config['max_adjustment'], total_adj))


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    from news_fetcher import fetch_stock_news

    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        symbol = "RELIANCE"

    print(f"\n{'='*60}")
    print(f"Analyzing news sentiment for {symbol}")
    print(f"{'='*60}\n")

    # Fetch news
    articles = fetch_stock_news(symbol, lookback_days=3)
    print(f"Fetched {len(articles)} articles\n")

    if not articles:
        print("No articles found.")
        sys.exit(0)

    # Analyze sentiment
    sentiment = analyze_news_sentiment(articles, use_finbert=True)

    print(f"SENTIMENT ANALYSIS RESULTS")
    print(f"{'─'*40}")
    print(f"Overall Score:     {sentiment.overall_score:.1f}/100 ({sentiment.sentiment_label})")
    print(f"Model Used:        {sentiment.model_used}")
    print(f"Articles Analyzed: {sentiment.articles_analyzed}")
    print(f"Most Recent:       {sentiment.recency_hours:.1f} hours ago")
    print(f"Event Detected:    {sentiment.event_type or 'None'}")
    print(f"\nBreakdown:")
    print(f"  Positive: {sentiment.positive_count}")
    print(f"  Negative: {sentiment.negative_count}")
    print(f"  Neutral:  {sentiment.neutral_count}")

    if sentiment.top_signals:
        print(f"\nTop Signals:")
        for signal in sentiment.top_signals:
            print(f"  {signal}")

    # Calculate confidence adjustment
    conf_adj = calculate_confidence_adjustment(sentiment)
    print(f"\nConfidence Adjustment: {conf_adj:+.1f} points")

    print(f"\n{'='*60}")
