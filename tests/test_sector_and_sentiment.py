"""Tests for sector comparison and sentiment analysis modules."""
import pandas as pd
import pytest

from src.sector_comparison import SectorAnalyzer
from src.sentiment_analysis import SentimentAnalyzer


def test_sector_analyzer_mapping():
    analyzer = SectorAnalyzer()
    assert analyzer.assign_sector("AAPL") == "Technology"
    assert analyzer.assign_sector("JPM") == "Financials"
    assert analyzer.assign_sector("XOM") == "Energy"
    assert analyzer.assign_sector("UNKNOWN_TICKER") == "Other"


def test_sector_mapping_json_file_loading(tmp_path):
    """Verify SectorAnalyzer loads mappings from JSON file and handles fallbacks"""
    import json

    custom_json = tmp_path / "custom_sectors.json"
    custom_json.write_text(json.dumps({"XYZ": "SpaceExploration", "ABC": "Biotech"}))

    # Load from custom path
    custom_analyzer = SectorAnalyzer(mapping_path=custom_json)
    assert custom_analyzer.assign_sector("XYZ") == "SpaceExploration"
    assert custom_analyzer.assign_sector("ABC") == "Biotech"

    # Fallback when file doesn't exist
    missing_analyzer = SectorAnalyzer(mapping_path=tmp_path / "nonexistent.json")
    assert missing_analyzer.assign_sector("AAPL") == "Technology"



def test_sector_analysis_grouping(event_analysis_df):
    analyzer = SectorAnalyzer()
    res = analyzer.analyze_by_sector(event_analysis_df)
    assert 'sector_summary' in res
    assert 'sector_returns' in res
    assert 'sector_volatility' in res
    assert isinstance(res['sector_summary'], pd.DataFrame)
    assert len(res['sector_summary']) > 0


def test_sentiment_analyzer():
    analyzer = SentimentAnalyzer(method='textblob')

    pos = analyzer.analyze_text("Company reports record high revenue, astonishing earnings growth and booming demand!")
    assert pos['polarity'] > 0.1
    assert analyzer.classify_sentiment(pos['polarity']) == 'Positive'

    neg = analyzer.analyze_text("Disastrous earnings miss, severe losses, catastrophic decline in quarterly revenue.")
    assert neg['polarity'] < -0.1
    assert analyzer.classify_sentiment(neg['polarity']) == 'Negative'

    neu = analyzer.analyze_text("The company will hold its annual shareholder meeting on Tuesday at 10 AM.")
    assert abs(neu['polarity']) <= 0.1
    assert analyzer.classify_sentiment(neu['polarity']) == 'Neutral'
