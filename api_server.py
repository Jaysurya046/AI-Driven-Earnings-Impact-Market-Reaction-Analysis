"""
FastAPI Server for Earnings Predictions
Provides REST API endpoints for model predictions and analysis
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Earnings Impact Analysis API",
    description="REST API for earnings impact prediction and analysis",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
analysis_state = {
    'pipeline': None,
    'event_analysis': None,
    'features': None,
    'model_trainer': None,
    'last_updated': None
}


class TickerRequest(BaseModel):
    tickers: List[str]
    lookback_years: int = 5
    event_window: int = 5


class PredictionRequest(BaseModel):
    ticker: str
    features: Dict[str, float]


class SentimentRequest(BaseModel):
    text: str


@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Earnings Impact Analysis API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "analysis_loaded": analysis_state['pipeline'] is not None,
        "last_updated": analysis_state['last_updated']
    }


@app.post("/analyze")
async def run_analysis(request: TickerRequest):
    """
    Run complete earnings analysis pipeline
    
    Args:
        request: TickerRequest with tickers and parameters
        
    Returns:
        Analysis summary
    """
    try:
        from main import EarningsAnalysisPipeline
        
        config = {
            'lookback_years': request.lookback_years,
            'event_window': request.event_window,
            'estimation_window': 60,
            'label_threshold': 0.0,
            'data_dir': 'data',
            'models_dir': 'models',
            'viz_dir': 'visualizations'
        }
        
        pipeline = EarningsAnalysisPipeline(config)
        
        # Run data collection and event analysis
        price_data, earnings_data = pipeline.step_1_collect_data(request.tickers)
        event_analysis = pipeline.step_2_event_analysis()
        features, labels = pipeline.step_3_feature_engineering()
        model_comparison = pipeline.step_4_train_models()
        
        # Store in global state
        analysis_state['pipeline'] = pipeline
        analysis_state['event_analysis'] = event_analysis
        analysis_state['features'] = features
        analysis_state['model_trainer'] = pipeline.model_trainer
        analysis_state['last_updated'] = pd.Timestamp.now()
        
        # Return summary
        return {
            "status": "success",
            "events_analyzed": len(event_analysis),
            "tickers": request.tickers,
            "date_range": {
                "start": str(event_analysis['EarningsDate'].min()),
                "end": str(event_analysis['EarningsDate'].max())
            },
            "model_comparison": model_comparison.to_dict('records'),
            "summary_stats": {
                "avg_surprise": float(event_analysis['SurprisePercentage'].mean()),
                "avg_post_event_return": float(event_analysis['PostEventReturn'].mean()),
                "correlation": float(event_analysis['SurprisePercentage'].corr(event_analysis['PostEventReturn']))
            }
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sector-analysis")
async def sector_analysis():
    """
    Get sector comparison analysis
    
    Returns:
        Sector analysis results
    """
    if analysis_state['event_analysis'] is None:
        raise HTTPException(status_code=400, detail="Run /analyze first")
    
    try:
        from src.sector_comparison import SectorAnalyzer
        
        analyzer = SectorAnalyzer()
        results = analyzer.analyze_by_sector(analysis_state['event_analysis'])
        
        return {
            "sector_summary": results['sector_summary'].to_dict('records'),
            "sector_returns": results['sector_returns'].to_dict('records'),
            "sector_volatility": results['sector_volatility'].to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest")
async def backtest(strategy: str = "momentum", threshold: float = 2.0):
    """
    Run backtesting on earnings strategies
    
    Args:
        strategy: Strategy name ('momentum' or 'mean_reversion')
        threshold: Surprise threshold for trading signals
        
    Returns:
        Backtest results
    """
    if analysis_state['event_analysis'] is None:
        raise HTTPException(status_code=400, detail="Run /analyze first")
    
    try:
        from src.backtesting import Backtester
        
        backtester = Backtester(initial_capital=100000)
        results = backtester.run_strategy(
            analysis_state['event_analysis'],
            strategy=strategy,
            threshold=threshold
        )
        
        # Convert numpy types to Python types
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.float64, np.float32)):
                serializable_results[key] = float(value)
            elif isinstance(value, (np.int64, np.int32)):
                serializable_results[key] = int(value)
            else:
                serializable_results[key] = value
        
        return serializable_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sentiment/analyze")
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment of text
    
    Args:
        request: SentimentRequest with text
        
    Returns:
        Sentiment analysis results
    """
    try:
        from src.sentiment_analysis import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(method='textblob')
        result = analyzer.analyze_text(request.text)
        result['label'] = analyzer.classify_sentiment(result['polarity'])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/{ticker}")
async def get_prediction(ticker: str):
    """
    Get latest prediction for a ticker
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Prediction results
    """
    if analysis_state['event_analysis'] is None:
        raise HTTPException(status_code=400, detail="Run /analyze first")
    
    try:
        # Get latest event for ticker
        ticker_data = analysis_state['event_analysis'][
            analysis_state['event_analysis']['Ticker'] == ticker
        ]
        
        if len(ticker_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data for ticker: {ticker}")
        
        latest = ticker_data.sort_values('EarningsDate').iloc[-1]
        
        # Get model prediction if available
        prediction = None
        if analysis_state['model_trainer'] is not None:
            # Use best model for prediction
            model_comparison = pd.read_csv('data/model_comparison.csv')
            best_model_name = model_comparison.loc[model_comparison['F1-Score'].idxmax(), 'Model']
            
            # Note: This is simplified - real implementation would need proper feature preparation
            prediction = {
                "model": best_model_name,
                "latest_surprise": float(latest['SurprisePercentage']),
                "latest_return": float(latest['PostEventReturn'])
            }
        
        return {
            "ticker": ticker,
            "latest_earnings_date": str(latest['EarningsDate']),
            "surprise_percentage": float(latest['SurprisePercentage']),
            "post_event_return": float(latest['PostEventReturn']),
            "prediction": prediction
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/events")
async def get_events(ticker: Optional[str] = None, limit: int = 100):
    """
    Get earnings events data
    
    Args:
        ticker: Optional ticker filter
        limit: Maximum number of records
        
    Returns:
        Events DataFrame
    """
    if analysis_state['event_analysis'] is None:
        raise HTTPException(status_code=400, detail="Run /analyze first")
    
    try:
        data = analysis_state['event_analysis']
        
        if ticker:
            data = data[data['Ticker'] == ticker]
        
        data = data.head(limit)
        
        # Convert to JSON-serializable format
        records = []
        for _, row in data.iterrows():
            record = {}
            for col, val in row.items():
                if isinstance(val, pd.Timestamp):
                    record[col] = str(val)
                elif isinstance(val, (np.float64, np.float32)):
                    record[col] = float(val)
                elif isinstance(val, (np.int64, np.int32)):
                    record[col] = int(val)
                else:
                    record[col] = val
            records.append(record)
        
        return {"events": records, "count": len(records)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """
    Get overall analysis metrics
    
    Returns:
        Summary metrics
    """
    if analysis_state['event_analysis'] is None:
        raise HTTPException(status_code=400, detail="Run /analyze first")
    
    try:
        data = analysis_state['event_analysis']
        
        return {
            "total_events": len(data),
            "unique_tickers": data['Ticker'].nunique(),
            "date_range": {
                "start": str(data['EarningsDate'].min()),
                "end": str(data['EarningsDate'].max())
            },
            "returns": {
                "avg_pre_event": float(data['PreEventReturn'].mean()),
                "avg_post_event": float(data['PostEventReturn'].mean()),
                "avg_car": float(data['AbnormalReturnCAR'].mean())
            },
            "surprise": {
                "avg": float(data['SurprisePercentage'].mean()),
                "std": float(data['SurprisePercentage'].std()),
                "correlation_with_return": float(data['SurprisePercentage'].corr(data['PostEventReturn']))
            },
            "volatility": {
                "avg_pre": float(data['PreEventVolatility'].mean()),
                "avg_post": float(data['PostEventVolatility'].mean())
            },
            "volume": {
                "avg_spike_ratio": float(data['VolumeSpikeRatio'].mean())
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
