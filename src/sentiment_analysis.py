"""
News Sentiment Analysis Module
Integrates sentiment analysis from news articles around earnings announcements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzes news sentiment around earnings announcements"""
    
    def __init__(self, method: str = 'textblob'):
        """
        Initialize sentiment analyzer
        
        Args:
            method: Sentiment analysis method ('textblob' or 'vader')
        """
        self.method = method
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentiment analysis model"""
        if self.method == 'textblob':
            try:
                from textblob import TextBlob
                self.model = TextBlob
                logger.info("TextBlob sentiment analyzer initialized")
            except ImportError:
                logger.error("TextBlob not installed. Run: pip install textblob")
                raise
        elif self.method == 'vader':
            try:
                from nltk.sentiment import SentimentIntensityAnalyzer
                import nltk
                # Download VADER lexicon if not already downloaded
                try:
                    nltk.data.find('sentiment/vader_lexicon.zip')
                except LookupError:
                    nltk.download('vader_lexicon', quiet=True)
                self.model = SentimentIntensityAnalyzer()
                logger.info("VADER sentiment analyzer initialized")
            except ImportError:
                logger.error("NLTK not installed. Run: pip install nltk")
                raise
        else:
            logger.error(f"Unknown method: {self.method}")
            raise ValueError(f"Method must be 'textblob' or 'vader', got '{self.method}'")
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text or not isinstance(text, str):
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'compound': 0.0,
                'positive': 0.0,
                'neutral': 1.0,
                'negative': 0.0
            }
        
        if self.method == 'textblob':
            blob = self.model(text)
            polarity = blob.sentiment.polarity  # Range: -1 to 1
            subjectivity = blob.sentiment.subjectivity  # Range: 0 to 1
            
            # Convert to VADER-like format for consistency
            if polarity > 0:
                positive = abs(polarity)
                negative = 0.0
            else:
                positive = 0.0
                negative = abs(polarity)
            neutral = 1.0 - positive - negative
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'compound': polarity,  # Use polarity as compound for TextBlob
                'positive': max(0, positive),
                'neutral': max(0, neutral),
                'negative': max(0, negative)
            }
        
        elif self.method == 'vader':
            scores = self.model.polarity_scores(text)
            return {
                'polarity': scores['compound'],
                'subjectivity': 0.5,  # VADER doesn't provide subjectivity
                'compound': scores['compound'],
                'positive': scores['pos'],
                'neutral': scores['neu'],
                'negative': scores['neg']
            }
    
    def analyze_multiple_texts(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment of multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with sentiment scores for each text
        """
        results = []
        for i, text in enumerate(texts):
            sentiment = self.analyze_text(text)
            sentiment['text_id'] = i
            sentiment['text_length'] = len(text) if text else 0
            results.append(sentiment)
        
        return pd.DataFrame(results)
    
    def classify_sentiment(self, polarity: float, thresholds: Tuple[float, float] = (-0.1, 0.1)) -> str:
        """
        Classify sentiment based on polarity score
        
        Args:
            polarity: Polarity score (-1 to 1)
            thresholds: (negative_threshold, positive_threshold)
            
        Returns:
            Sentiment label ('Positive', 'Negative', or 'Neutral')
        """
        neg_thresh, pos_thresh = thresholds
        
        if polarity >= pos_thresh:
            return 'Positive'
        elif polarity <= neg_thresh:
            return 'Negative'
        else:
            return 'Neutral'
    
    def aggregate_sentiment(
        self,
        sentiment_df: pd.DataFrame,
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            group_by: Column to group by (e.g., 'Ticker', 'Date')
            
        Returns:
            DataFrame with aggregated sentiment metrics
        """
        if group_by:
            agg_df = sentiment_df.groupby(group_by).agg({
                'polarity': ['mean', 'std', 'count'],
                'positive': 'mean',
                'neutral': 'mean',
                'negative': 'mean',
                'subjectivity': 'mean'
            }).round(4)
            
            # Flatten column names
            agg_df.columns = [
                'AvgPolarity', 'StdPolarity', 'ArticleCount',
                'AvgPositive', 'AvgNeutral', 'AvgNegative',
                'AvgSubjectivity'
            ]
            agg_df = agg_df.reset_index()
            
            # Add sentiment distribution
            agg_df['BullishPercent'] = agg_df['AvgPositive'] / (
                agg_df['AvgPositive'] + agg_df['AvgNegative'] + 1e-10
            ) * 100
            
            return agg_df
        else:
            return sentiment_df.describe()


class EarningsSentimentIntegration:
    """Integrates sentiment analysis with earnings event data"""
    
    def __init__(self, sentiment_method: str = 'textblob'):
        """
        Initialize earnings sentiment integrator
        
        Args:
            sentiment_method: Method for sentiment analysis
        """
        self.sentiment_analyzer = SentimentAnalyzer(method=sentiment_method)
    
    def generate_mock_news(self, ticker: str, earnings_date: datetime, 
                          surprise: float) -> List[Dict]:
        """
        Generate mock news articles for testing
        
        Args:
            ticker: Stock ticker
            earnings_date: Earnings announcement date
            surprise: Earnings surprise percentage
            
        Returns:
            List of mock news articles
        """
        np.random.seed(int(ticker.__hash__()) % (2**32))
        
        # Generate headlines based on surprise
        if surprise > 5:
            positive_templates = [
                f"{ticker} Crushes Earnings Expectations",
                f"{ticker} Reports Stellar Quarterly Results",
                f"{ticker} Beats Estimates by Wide Margin",
                f"Strong Earnings Drive {ticker} Higher",
                f"{ticker} Exceeds Wall Street Forecasts"
            ]
            templates = positive_templates
        elif surprise < -5:
            negative_templates = [
                f"{ticker} Misses Earnings Estimates",
                f"{ticker} Disappoints Investors with Weak Results",
                f"{ticker} Falls Short of Expectations",
                f"Weak Earnings Pressure {ticker}",
                f"{ticker} Warns on Future Guidance"
            ]
            templates = negative_templates
        else:
            neutral_templates = [
                f"{ticker} Reports Mixed Quarterly Results",
                f"{ticker} Meets Earnings Expectations",
                f"{ticker} Posts In-Line Earnings",
                f"Analysts React to {ticker} Earnings",
                f"{ticker} Announces Quarterly Results"
            ]
            templates = neutral_templates
        
        # Generate articles
        articles = []
        for i in range(np.random.randint(3, 8)):
            days_before = np.random.randint(-5, 1)
            article_date = earnings_date + timedelta(days=days_before)
            
            headline = np.random.choice(templates)
            
            # Generate body text
            body_templates = [
                f"Analysts are watching {ticker} closely as the company reports its quarterly earnings. ",
                f"Market participants expect significant movement in {ticker} following the announcement. ",
                f"Investors are evaluating {ticker}'s performance against industry peers. ",
                f"The latest earnings report from {ticker} provides insights into market conditions. "
            ]
            body = np.random.choice(body_templates) * 2
            
            articles.append({
                'Ticker': ticker,
                'Date': article_date,
                'Headline': headline,
                'Body': body,
                'Source': f'News Source {i+1}',
                'EarningsDate': earnings_date,
                'Surprise': surprise
            })
        
        return articles
    
    def analyze_earnings_sentiment(
        self,
        event_analysis: pd.DataFrame,
        use_real_news: bool = False
    ) -> pd.DataFrame:
        """
        Analyze sentiment around earnings announcements
        
        Args:
            event_analysis: DataFrame with earnings events
            use_real_news: Whether to fetch real news (requires API)
            
        Returns:
            DataFrame with sentiment metrics per earnings event
        """
        logger.info("Analyzing sentiment around earnings announcements...")
        
        all_articles = []
        
        # Generate or fetch news articles
        for idx, row in event_analysis.iterrows():
            ticker = row['Ticker']
            earnings_date = row['EarningsDate']
            surprise = row.get('SurprisePercentage', 0)
            
            if use_real_news:
                # TODO: Implement real news fetching (requires news API)
                logger.warning("Real news fetching not yet implemented. Using mock data.")
                articles = self.generate_mock_news(ticker, earnings_date, surprise)
            else:
                articles = self.generate_mock_news(ticker, earnings_date, surprise)
            
            all_articles.extend(articles)
        
        if not all_articles:
            logger.error("No articles generated")
            return pd.DataFrame()
        
        articles_df = pd.DataFrame(all_articles)
        
        # Analyze sentiment for each article
        logger.info(f"Analyzing sentiment for {len(articles_df)} articles...")
        
        # Combine headline and body for analysis
        articles_df['FullText'] = articles_df['Headline'] + ' ' + articles_df['Body']
        
        sentiment_results = []
        for idx, row in articles_df.iterrows():
            sentiment = self.sentiment_analyzer.analyze_text(row['FullText'])
            sentiment['Ticker'] = row['Ticker']
            sentiment['Date'] = row['Date']
            sentiment['EarningsDate'] = row['EarningsDate']
            sentiment['Headline'] = row['Headline']
            sentiment_results.append(sentiment)
        
        sentiment_df = pd.DataFrame(sentiment_results)
        
        # Aggregate by earnings event
        aggregated = sentiment_df.groupby(['Ticker', 'EarningsDate']).agg({
            'polarity': ['mean', 'std', 'count'],
            'positive': 'mean',
            'neutral': 'mean',
            'negative': 'mean',
            'compound': 'mean'
        }).round(4)
        
        # Flatten columns
        aggregated.columns = [
            'AvgSentiment', 'SentimentStd', 'ArticleCount',
            'AvgPositive', 'AvgNeutral', 'AvgNegative',
            'AvgCompound'
        ]
        aggregated = aggregated.reset_index()
        
        # Add sentiment classification
        aggregated['SentimentLabel'] = aggregated['AvgSentiment'].apply(
            self.sentiment_analyzer.classify_sentiment
        )
        
        # Calculate bullishness
        aggregated['BullishPercent'] = aggregated['AvgPositive'] / (
            aggregated['AvgPositive'] + aggregated['AvgNegative'] + 1e-10
        ) * 100
        
        logger.info(f"Sentiment analysis complete for {len(aggregated)} earnings events")
        
        return aggregated
    
    def merge_with_event_analysis(
        self,
        event_analysis: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge sentiment data with event analysis
        
        Args:
            event_analysis: Original event analysis DataFrame
            sentiment_df: Sentiment analysis results
            
        Returns:
            Merged DataFrame
        """
        merged = event_analysis.merge(
            sentiment_df,
            on=['Ticker', 'EarningsDate'],
            how='left'
        )
        
        logger.info(f"Merged dataset has {len(merged)} records with sentiment data")
        
        return merged
    
    def analyze_sentiment_impact(
        self,
        merged_df: pd.DataFrame
    ) -> Dict:
        """
        Analyze how sentiment impacts market reactions
        
        Args:
            merged_df: DataFrame with both event analysis and sentiment
            
        Returns:
            Dictionary with analysis results
        """
        if 'AvgSentiment' not in merged_df.columns:
            logger.error("Sentiment data not found in DataFrame")
            return {}
        
        # Correlation analysis
        correlations = {
            'sentiment_vs_return': merged_df['AvgSentiment'].corr(merged_df['PostEventReturn']),
            'sentiment_vs_surprise': merged_df['AvgSentiment'].corr(merged_df['SurprisePercentage']),
            'bullish_vs_return': merged_df['BullishPercent'].corr(merged_df['PostEventReturn'])
        }
        
        # Group by sentiment label
        by_sentiment = merged_df.groupby('SentimentLabel').agg({
            'PostEventReturn': ['mean', 'std', 'count'],
            'SurprisePercentage': 'mean'
        }).round(4)
        
        by_sentiment.columns = ['AvgReturn', 'StdReturn', 'EventCount', 'AvgSurprise']
        by_sentiment = by_sentiment.reset_index()
        
        # Predictive power
        # High sentiment vs low sentiment performance
        high_sentiment = merged_df[merged_df['AvgSentiment'] > 0.1]['PostEventReturn'].mean()
        low_sentiment = merged_df[merged_df['AvgSentiment'] < -0.1]['PostEventReturn'].mean()
        
        results = {
            'correlations': correlations,
            'by_sentiment_label': by_sentiment,
            'high_sentiment_avg_return': high_sentiment,
            'low_sentiment_avg_return': low_sentiment,
            'sentiment_premium': high_sentiment - low_sentiment
        }
        
        logger.info("Sentiment impact analysis complete")
        logger.info(f"Correlation (Sentiment vs Return): {correlations['sentiment_vs_return']:.3f}")
        logger.info(f"Sentiment Premium: {results['sentiment_premium']:.2f}%")
        
        return results


if __name__ == "__main__":
    print("Sentiment Analysis Module")
    print("=" * 50)
    
    # Test basic sentiment analysis
    analyzer = SentimentAnalyzer(method='textblob')
    
    test_texts = [
        "Company reports excellent earnings, beating all expectations!",
        "Disappointing results as company misses revenue targets.",
        "Mixed quarter with some strengths and weaknesses."
    ]
    
    for text in test_texts:
        result = analyzer.analyze_text(text)
        label = analyzer.classify_sentiment(result['polarity'])
        print(f"\nText: {text[:50]}...")
        print(f"Polarity: {result['polarity']:.3f}, Label: {label}")
