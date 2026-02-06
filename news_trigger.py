"""
News Trigger Engine

Implements 4-mode news-based trigger system for swing trading:
1. Entry Signal - Upgrade borderline stocks on positive news
2. Confirmation - Boost high-conviction picks when all pillars align
3. Risk Filter - Downgrade stocks with negative news
4. Dynamic Recalc - Trigger rescan on major events
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from copy import deepcopy

from news_fetcher import NewsArticle, fetch_stock_news
from news_sentiment import (
    NewsSentiment,
    analyze_news_sentiment,
    calculate_confidence_adjustment
)
from config.news_settings import (
    TRIGGER_CONFIG,
    NEWS_SAFEGUARDS,
    is_trigger_enabled
)


@dataclass
class TriggerResult:
    """Result of applying news triggers."""
    triggered: bool = False
    mode: Optional[str] = None           # Which mode triggered
    original_signal: str = ""
    updated_signal: str = ""
    original_confidence: float = 0
    updated_confidence: float = 0
    confidence_adjustment: float = 0
    warnings: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    news_sentiment: Optional[NewsSentiment] = None
    should_rescan: bool = False          # For dynamic recalc mode

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'triggered': self.triggered,
            'mode': self.mode,
            'original_signal': self.original_signal,
            'updated_signal': self.updated_signal,
            'original_confidence': round(self.original_confidence, 1),
            'updated_confidence': round(self.updated_confidence, 1),
            'confidence_adjustment': round(self.confidence_adjustment, 1),
            'warnings': self.warnings,
            'reasons': self.reasons,
            'should_rescan': self.should_rescan,
            'news_score': round(self.news_sentiment.overall_score, 1) if self.news_sentiment else None,
            'event_type': self.news_sentiment.event_type if self.news_sentiment else None
        }


@dataclass
class StockContext:
    """Context for trigger evaluation."""
    symbol: str
    signal: str                  # Current signal (WAIT, HOLD, BUY, STRONG BUY)
    confidence: float            # Current confidence (0-100)
    composite_score: float       # Current composite score (0-100)
    technical_score: float = 0   # Technical score (0-100)
    fundamental_score: float = 0 # Fundamental score (0-100)


class NewsTriggerEngine:
    """
    Engine for applying news-based triggers to stock recommendations.

    Modes:
    1. entry_signal - Upgrades borderline stocks on positive news
    2. confirmation - Boosts already-good picks when all pillars align
    3. risk_filter - Downgrades stocks with negative news
    4. dynamic_recalc - Triggers full rescan on major events
    """

    SIGNAL_ORDER = ['WAIT', 'HOLD', 'BUY', 'STRONG BUY']

    def __init__(self, modes: List[str] = None):
        """
        Initialize trigger engine.

        Args:
            modes: List of modes to enable (default: all enabled modes from config)
        """
        if modes is None:
            self.enabled_modes = [m for m in TRIGGER_CONFIG if is_trigger_enabled(m)]
        else:
            self.enabled_modes = modes

    def apply_triggers(
        self,
        context: StockContext,
        articles: List[NewsArticle] = None,
        sentiment: NewsSentiment = None
    ) -> TriggerResult:
        """
        Apply all enabled news triggers to a stock.

        Args:
            context: Stock context with current scores/signal
            articles: Pre-fetched news articles (optional)
            sentiment: Pre-analyzed sentiment (optional)

        Returns:
            TriggerResult with updated signal/confidence
        """
        # Fetch news if not provided
        if articles is None:
            articles = fetch_stock_news(context.symbol, lookback_days=3)

        # Analyze sentiment if not provided
        if sentiment is None:
            sentiment = analyze_news_sentiment(articles)

        # Initialize result
        result = TriggerResult(
            original_signal=context.signal,
            updated_signal=context.signal,
            original_confidence=context.confidence,
            updated_confidence=context.confidence,
            news_sentiment=sentiment
        )

        # Apply base confidence adjustment from news
        base_adj = calculate_confidence_adjustment(sentiment)
        result.confidence_adjustment = base_adj

        # Apply each enabled mode in priority order
        for mode in ['risk_filter', 'entry_signal', 'confirmation', 'dynamic_recalc']:
            if mode not in self.enabled_modes:
                continue

            if mode == 'risk_filter':
                self._apply_risk_filter(context, sentiment, result)
            elif mode == 'entry_signal':
                self._apply_entry_signal(context, sentiment, result)
            elif mode == 'confirmation':
                self._apply_confirmation(context, sentiment, result)
            elif mode == 'dynamic_recalc':
                self._apply_dynamic_recalc(context, sentiment, result)

            # Stop if a trigger fired (except risk_filter which is additive)
            if result.triggered and mode != 'risk_filter':
                break

        # Apply final confidence adjustment (capped)
        max_adj = NEWS_SAFEGUARDS['max_confidence_from_news']
        result.confidence_adjustment = max(
            -max_adj,
            min(max_adj, result.confidence_adjustment)
        )
        result.updated_confidence = max(0, min(100,
            context.confidence + result.confidence_adjustment
        ))

        return result

    def _apply_entry_signal(
        self,
        context: StockContext,
        sentiment: NewsSentiment,
        result: TriggerResult
    ):
        """
        Mode 1: Entry Signal

        Upgrades borderline stocks on positive news.
        Triggers when:
        - News score > 75
        - Composite score > 50 (borderline)
        - News is recent (< 24 hours)
        """
        config = TRIGGER_CONFIG['entry_signal']

        # Check conditions
        if sentiment.overall_score < config['news_score_threshold']:
            return

        if context.composite_score < config['composite_score_threshold']:
            return

        if sentiment.recency_hours > config['max_recency_hours']:
            return

        # Already at max signal
        if context.signal == 'STRONG BUY':
            return

        # Apply upgrade
        upgrade_map = config['signal_upgrade']
        if context.signal in upgrade_map:
            new_signal = upgrade_map[context.signal]
            result.triggered = True
            result.mode = 'entry_signal'
            result.updated_signal = new_signal
            result.confidence_adjustment += config['confidence_boost']
            result.reasons.append(
                f"Entry Signal: Positive news ({sentiment.overall_score:.0f}/100) "
                f"upgrades {context.signal} → {new_signal}"
            )

    def _apply_confirmation(
        self,
        context: StockContext,
        sentiment: NewsSentiment,
        result: TriggerResult
    ):
        """
        Mode 2: Confirmation

        Boosts high-conviction picks when all pillars align.
        Triggers when:
        - News score > 70
        - Composite score > 60
        - Technical score > 55
        - Fundamental score > 55
        """
        config = TRIGGER_CONFIG['confirmation']

        # Check all conditions
        if sentiment.overall_score < config['news_score_threshold']:
            return

        if context.composite_score < config['composite_score_threshold']:
            return

        if context.technical_score < config['technical_score_threshold']:
            return

        if context.fundamental_score < config['fundamental_score_threshold']:
            return

        # Already at max signal
        if context.signal == 'STRONG BUY':
            return

        # Apply confirmation
        upgrade_map = config['signal_upgrade']
        if context.signal in upgrade_map:
            new_signal = upgrade_map[context.signal]
            result.triggered = True
            result.mode = 'confirmation'
            result.updated_signal = new_signal
            result.confidence_adjustment += config['confidence_boost']
            result.reasons.append(
                f"Confirmation: All pillars aligned (Tech={context.technical_score:.0f}, "
                f"Fund={context.fundamental_score:.0f}, News={sentiment.overall_score:.0f}) "
                f"- {context.signal} → {new_signal}"
            )

    def _apply_risk_filter(
        self,
        context: StockContext,
        sentiment: NewsSentiment,
        result: TriggerResult
    ):
        """
        Mode 3: Risk Filter

        Downgrades stocks with negative news.
        Triggers when:
        - News score < 30 (negative)
        - News is recent (< 48 hours)
        """
        config = TRIGGER_CONFIG['risk_filter']

        # Check for negative news (score below 100 - threshold)
        negative_threshold = 100 - config['negative_news_threshold']

        if sentiment.overall_score > negative_threshold:
            return

        if sentiment.recency_hours > config['max_recency_hours']:
            return

        # Apply downgrade
        downgrade_map = config['signal_downgrade']
        if context.signal in downgrade_map:
            new_signal = downgrade_map[context.signal]
            result.triggered = True
            result.mode = 'risk_filter'
            result.updated_signal = new_signal
            result.confidence_adjustment += config['confidence_penalty']

            result.warnings.append(
                f"RISK ALERT: Negative news ({sentiment.overall_score:.0f}/100) detected"
            )
            result.reasons.append(
                f"Risk Filter: Negative news downgrades {context.signal} → {new_signal}"
            )

            # Add specific warning if configured
            if config.get('add_warning') and sentiment.top_signals:
                for signal in sentiment.top_signals[:2]:
                    if signal.startswith('[-]'):
                        result.warnings.append(signal)

    def _apply_dynamic_recalc(
        self,
        context: StockContext,
        sentiment: NewsSentiment,
        result: TriggerResult
    ):
        """
        Mode 4: Dynamic Recalculation

        Flags stocks for rescan on major events.
        Triggers when:
        - Major event detected (earnings, M&A, regulatory)
        """
        config = TRIGGER_CONFIG['dynamic_recalc']

        if sentiment.event_type not in config['major_events']:
            return

        # Flag for rescan
        result.should_rescan = True
        result.reasons.append(
            f"Dynamic Recalc: Major event detected ({sentiment.event_type}) "
            f"- recommend full rescan"
        )


def apply_news_trigger(
    symbol: str,
    signal: str,
    confidence: float,
    composite_score: float,
    technical_score: float = 0,
    fundamental_score: float = 0,
    articles: List[NewsArticle] = None,
    sentiment: NewsSentiment = None,
    modes: List[str] = None
) -> TriggerResult:
    """
    Convenience function to apply news triggers.

    Args:
        symbol: Stock symbol
        signal: Current signal (WAIT, HOLD, BUY, STRONG BUY)
        confidence: Current confidence (0-100)
        composite_score: Current composite score (0-100)
        technical_score: Technical score (0-100)
        fundamental_score: Fundamental score (0-100)
        articles: Pre-fetched news articles (optional)
        sentiment: Pre-analyzed sentiment (optional)
        modes: Specific modes to apply (optional)

    Returns:
        TriggerResult with updated values
    """
    context = StockContext(
        symbol=symbol,
        signal=signal,
        confidence=confidence,
        composite_score=composite_score,
        technical_score=technical_score,
        fundamental_score=fundamental_score
    )

    engine = NewsTriggerEngine(modes=modes)
    return engine.apply_triggers(context, articles, sentiment)


def enhance_recommendation(
    recommendation: Any,
    articles: List[NewsArticle] = None,
    sentiment: NewsSentiment = None
) -> Any:
    """
    Enhance a recommendation with news triggers.

    Args:
        recommendation: SwingRecommendation or similar object
        articles: Pre-fetched news articles (optional)
        sentiment: Pre-analyzed sentiment (optional)

    Returns:
        Enhanced recommendation with news adjustments
    """
    # Create context from recommendation
    context = StockContext(
        symbol=getattr(recommendation, 'symbol', ''),
        signal=getattr(recommendation, 'swing_signal', 'HOLD'),
        confidence=getattr(recommendation, 'confidence', 50),
        composite_score=getattr(recommendation, 'composite_score', 50),
        technical_score=getattr(recommendation, 'technical_score', 0),
        fundamental_score=getattr(recommendation, 'fundamental_score', 0)
    )

    # Apply triggers
    engine = NewsTriggerEngine()
    result = engine.apply_triggers(context, articles, sentiment)

    # Update recommendation
    if result.triggered:
        if hasattr(recommendation, 'swing_signal'):
            recommendation.swing_signal = result.updated_signal
        if hasattr(recommendation, 'signal'):
            recommendation.signal = result.updated_signal
        if hasattr(recommendation, 'confidence'):
            recommendation.confidence = result.updated_confidence

    # Add news attributes
    recommendation.news_triggered = result.triggered
    recommendation.news_trigger_mode = result.mode
    recommendation.news_sentiment = result.news_sentiment
    recommendation.news_warnings = result.warnings
    recommendation.news_reasons = result.reasons

    return recommendation


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
    print(f"Testing News Triggers for {symbol}")
    print(f"{'='*60}\n")

    # Simulate different scenarios
    test_cases = [
        {
            'name': 'Borderline stock with positive news',
            'signal': 'HOLD',
            'confidence': 55,
            'composite_score': 52,
            'technical_score': 55,
            'fundamental_score': 50
        },
        {
            'name': 'Good stock needing confirmation',
            'signal': 'HOLD',
            'confidence': 65,
            'composite_score': 62,
            'technical_score': 60,
            'fundamental_score': 58
        },
        {
            'name': 'Stock at risk',
            'signal': 'BUY',
            'confidence': 70,
            'composite_score': 65,
            'technical_score': 60,
            'fundamental_score': 55
        }
    ]

    # Fetch real news
    print(f"Fetching news for {symbol}...")
    articles = fetch_stock_news(symbol, lookback_days=3)
    print(f"Found {len(articles)} articles\n")

    # Analyze sentiment
    sentiment = analyze_news_sentiment(articles)
    print(f"News Sentiment: {sentiment.overall_score:.1f}/100 ({sentiment.sentiment_label})")
    print(f"Event Detected: {sentiment.event_type or 'None'}")
    print()

    # Test each scenario
    for tc in test_cases:
        print(f"{'─'*60}")
        print(f"Scenario: {tc['name']}")
        print(f"  Initial: {tc['signal']} (Conf: {tc['confidence']}, Comp: {tc['composite_score']})")

        result = apply_news_trigger(
            symbol=symbol,
            signal=tc['signal'],
            confidence=tc['confidence'],
            composite_score=tc['composite_score'],
            technical_score=tc['technical_score'],
            fundamental_score=tc['fundamental_score'],
            articles=articles,
            sentiment=sentiment
        )

        print(f"  Triggered: {result.triggered} (Mode: {result.mode or 'None'})")
        print(f"  Updated:  {result.updated_signal} (Conf: {result.updated_confidence:.1f})")
        print(f"  Adjustment: {result.confidence_adjustment:+.1f} points")

        if result.reasons:
            print(f"  Reasons:")
            for reason in result.reasons:
                print(f"    - {reason}")

        if result.warnings:
            print(f"  Warnings:")
            for warning in result.warnings:
                print(f"    - {warning}")

        print()

    print(f"{'='*60}")
