"""
FastAPI Server for Earnings Predictions
Provides thread-safe REST API endpoints for model predictions, background analysis, and monitoring
"""

import asyncio
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Dict, List, Optional
import uuid

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config()

app = FastAPI(
    title="Earnings Impact Analysis API",
    description="REST API for earnings impact prediction and analysis",
    version="1.0.0"
)

# Secure CORS configuration:
# When allow_origins contains "*", allow_credentials MUST be False according to CORS specification.
cors_origins = config.get('cors_origins', ['*'])
allow_credentials = False if '*' in cors_origins else True

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global thread-safe state container
class AnalysisStateManager:
    """Thread-safe manager for global analysis state."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.pipeline = None
        self.event_analysis: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.model_trainer = None
        self.model_comparison: Optional[pd.DataFrame] = None
        self.last_updated: Optional[str] = None
        self.is_analyzing: bool = False
        self.current_task_id: Optional[str] = None
        self.task_history: Dict[str, Dict] = {}

    async def get_state_summary(self) -> Dict:
        async with self._lock:
            return {
                "status": "healthy",
                "analysis_loaded": self.pipeline is not None,
                "is_analyzing": self.is_analyzing,
                "current_task_id": self.current_task_id,
                "last_updated": self.last_updated,
                "total_events": len(self.event_analysis) if self.event_analysis is not None else 0,
            }

    async def reset(self):
        async with self._lock:
            self.pipeline = None
            self.event_analysis = None
            self.features = None
            self.model_trainer = None
            self.model_comparison = None
            self.last_updated = None
            self.is_analyzing = False
            self.current_task_id = None


state_manager = AnalysisStateManager()


class TickerRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1, description="List of stock tickers to analyze")
    lookback_years: int = Field(5, ge=1, le=20)
    event_window: int = Field(5, ge=1, le=30)
    run_in_background: bool = Field(False, description="Whether to execute analysis asynchronously")


class PredictionRequest(BaseModel):
    ticker: str
    features: Optional[Dict[str, float]] = None


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1)


@app.get("/")
async def root():
    """API root health check"""
    return {
        "message": "Earnings Impact Analysis API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check reporting lock-protected state"""
    return await state_manager.get_state_summary()


def _execute_pipeline_sync(task_id: str, tickers: List[str], lookback_years: int, event_window: int):
    """Worker function executed in worker thread to prevent blocking the async event loop."""
    from main import EarningsAnalysisPipeline

    pipeline_cfg = {
        'lookback_years': lookback_years,
        'event_window': event_window,
        'estimation_window': config.get('estimation_window', 60),
        'label_threshold': config.get('label_threshold', 0.0),
        'data_dir': config.get('data_dir', 'data'),
        'models_dir': config.get('models_dir', 'models'),
        'viz_dir': config.get('viz_dir', 'visualizations')
    }

    pipeline = EarningsAnalysisPipeline(pipeline_cfg)
    price_data, earnings_data = pipeline.step_1_collect_data(tickers)
    event_analysis = pipeline.step_2_event_analysis()
    features, labels = pipeline.step_3_feature_engineering()
    model_comparison = pipeline.step_4_train_models()

    return {
        'pipeline': pipeline,
        'event_analysis': event_analysis,
        'features': features,
        'model_trainer': pipeline.model_trainer,
        'model_comparison': model_comparison,
    }


async def _background_task_wrapper(task_id: str, tickers: List[str], lookback_years: int, event_window: int):
    try:
        results = await asyncio.to_thread(
            _execute_pipeline_sync, task_id, tickers, lookback_years, event_window
        )
        async with state_manager._lock:
            state_manager.pipeline = results['pipeline']
            state_manager.event_analysis = results['event_analysis']
            state_manager.features = results['features']
            state_manager.model_trainer = results['model_trainer']
            state_manager.model_comparison = results['model_comparison']
            state_manager.last_updated = datetime.now(timezone.utc).isoformat()
            state_manager.is_analyzing = False
            state_manager.current_task_id = None
            state_manager.task_history[task_id] = {
                'status': 'completed',
                'finished_at': datetime.now(timezone.utc).isoformat(),
                'events_count': len(results['event_analysis']),
            }
    except Exception as e:
        logger.error(f"Background analysis task {task_id} failed: {e}", exc_info=True)
        async with state_manager._lock:
            state_manager.is_analyzing = False
            state_manager.current_task_id = None
            state_manager.task_history[task_id] = {
                'status': 'failed',
                'error': str(e),
                'finished_at': datetime.now(timezone.utc).isoformat(),
            }


@app.post("/analyze")
async def run_analysis(request: TickerRequest, background_tasks: BackgroundTasks):
    """
    Execute earnings analysis pipeline with concurrency guards.
    Can be run synchronously or offloaded to background task.
    """
    async with state_manager._lock:
        if state_manager.is_analyzing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"An analysis is already running (Task ID: {state_manager.current_task_id})"
            )
        task_id = str(uuid.uuid4())
        state_manager.is_analyzing = True
        state_manager.current_task_id = task_id
        state_manager.task_history[task_id] = {
            'status': 'running',
            'started_at': datetime.now(timezone.utc).isoformat(),
            'tickers': request.tickers
        }

    if request.run_in_background:
        background_tasks.add_task(
            _background_task_wrapper,
            task_id,
            request.tickers,
            request.lookback_years,
            request.event_window
        )
        return {
            "status": "accepted",
            "message": "Analysis started in background",
            "task_id": task_id
        }

    try:
        # Offload CPU/IO work to a thread pool to avoid blocking the event loop
        results = await asyncio.to_thread(
            _execute_pipeline_sync,
            task_id,
            request.tickers,
            request.lookback_years,
            request.event_window
        )
        async with state_manager._lock:
            state_manager.pipeline = results['pipeline']
            state_manager.event_analysis = results['event_analysis']
            state_manager.features = results['features']
            state_manager.model_trainer = results['model_trainer']
            state_manager.model_comparison = results['model_comparison']
            state_manager.last_updated = datetime.now(timezone.utc).isoformat()
            state_manager.is_analyzing = False
            state_manager.current_task_id = None
            state_manager.task_history[task_id] = {
                'status': 'completed',
                'finished_at': datetime.now(timezone.utc).isoformat(),
                'events_count': len(results['event_analysis']),
            }

        ea = results['event_analysis']
        mc = results['model_comparison']

        return {
            "status": "success",
            "task_id": task_id,
            "events_analyzed": len(ea),
            "tickers": request.tickers,
            "date_range": {
                "start": str(ea['EarningsDate'].min()) if len(ea) else None,
                "end": str(ea['EarningsDate'].max()) if len(ea) else None,
            },
            "model_comparison": mc.to_dict('records') if mc is not None else [],
            "summary_stats": {
                "avg_surprise": float(ea['SurprisePercentage'].mean()) if len(ea) else 0.0,
                "avg_post_event_return": float(ea['PostEventReturn'].mean()) if len(ea) else 0.0,
                "correlation": float(ea['SurprisePercentage'].corr(ea['PostEventReturn'])) if len(ea) > 1 else 0.0,
            }
        }
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        async with state_manager._lock:
            state_manager.is_analyzing = False
            state_manager.current_task_id = None
            state_manager.task_history[task_id] = {
                'status': 'failed',
                'error': str(e),
                'finished_at': datetime.now(timezone.utc).isoformat(),
            }
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Retrieve status of background analysis task."""
    async with state_manager._lock:
        if task_id not in state_manager.task_history:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return state_manager.task_history[task_id]


@app.get("/sector-analysis")
async def sector_analysis():
    """Get sector comparison analysis"""
    async with state_manager._lock:
        event_analysis = state_manager.event_analysis

    if event_analysis is None or event_analysis.empty:
        raise HTTPException(status_code=400, detail="Run /analyze first")

    try:
        from src.sector_comparison import SectorAnalyzer
        analyzer = SectorAnalyzer()
        results = await asyncio.to_thread(analyzer.analyze_by_sector, event_analysis)

        return {
            "sector_summary": results['sector_summary'].to_dict('records'),
            "sector_returns": results['sector_returns'].to_dict('records'),
            "sector_volatility": results['sector_volatility'].to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest")
async def backtest(strategy: str = "momentum", threshold: float = 2.0):
    """Run backtesting on earnings strategies"""
    async with state_manager._lock:
        event_analysis = state_manager.event_analysis

    if event_analysis is None or event_analysis.empty:
        raise HTTPException(status_code=400, detail="Run /analyze first")

    try:
        from src.backtesting import Backtester
        backtester = Backtester(initial_capital=100000)
        results = await asyncio.to_thread(
            backtester.run_strategy,
            event_analysis,
            strategy=strategy,
            threshold=threshold
        )

        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.floating, float)):
                serializable_results[key] = float(value)
            elif isinstance(value, (np.integer, int)):
                serializable_results[key] = int(value)
            else:
                serializable_results[key] = value

        return serializable_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sentiment/analyze")
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of text"""
    try:
        from src.sentiment_analysis import SentimentAnalyzer
        analyzer = SentimentAnalyzer(method='textblob')
        result = await asyncio.to_thread(analyzer.analyze_text, request.text)
        result['label'] = analyzer.classify_sentiment(result['polarity'])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/{ticker}")
async def get_prediction(ticker: str):
    """Get latest prediction for a ticker using trained models and features"""
    async with state_manager._lock:
        event_analysis = state_manager.event_analysis
        features = state_manager.features
        trainer = state_manager.model_trainer
        model_comparison = state_manager.model_comparison

    if event_analysis is None or event_analysis.empty:
        raise HTTPException(status_code=400, detail="Run /analyze first")

    ticker_events = event_analysis[event_analysis['Ticker'] == ticker]
    if len(ticker_events) == 0:
        raise HTTPException(status_code=404, detail=f"No data for ticker: {ticker}")

    latest = ticker_events.sort_values('EarningsDate').iloc[-1]

    prediction = None
    if trainer is not None and len(trainer.models) > 0 and features is not None:
        best_model_name = list(trainer.models.keys())[0]
        if model_comparison is not None and not model_comparison.empty and 'F1-Score' in model_comparison.columns:
            best_model_name = model_comparison.loc[model_comparison['F1-Score'].idxmax(), 'Model']

        try:
            ticker_features = features[features['Ticker'] == ticker].sort_values('EarningsDate')
            if len(ticker_features) > 0:
                latest_row = ticker_features.iloc[[-1]]
                pred_label = trainer.predict(best_model_name, latest_row)[0]
                prediction = {
                    "model": best_model_name,
                    "predicted_reaction": int(pred_label),
                    "predicted_label": "Positive" if pred_label == 1 else "Negative",
                    "latest_surprise": float(latest.get('SurprisePercentage', 0.0)),
                }
        except Exception as e:
            logger.warning(f"Failed to compute model prediction: {e}")

    return {
        "ticker": ticker,
        "latest_earnings_date": str(latest['EarningsDate']),
        "surprise_percentage": float(latest.get('SurprisePercentage', 0.0)),
        "post_event_return": float(latest.get('PostEventReturn', 0.0)),
        "prediction": prediction
    }


@app.get("/data/events")
async def get_events(ticker: Optional[str] = None, limit: int = 100):
    """Get earnings events data"""
    async with state_manager._lock:
        data = state_manager.event_analysis

    if data is None or data.empty:
        raise HTTPException(status_code=400, detail="Run /analyze first")

    if ticker:
        data = data[data['Ticker'] == ticker]

    data = data.head(limit)

    records = []
    for _, row in data.iterrows():
        record = {}
        for col, val in row.items():
            if isinstance(val, pd.Timestamp):
                record[col] = str(val)
            elif isinstance(val, (np.floating, float)):
                record[col] = float(val) if not np.isnan(val) else None
            elif isinstance(val, (np.integer, int)):
                record[col] = int(val)
            else:
                record[col] = val
        records.append(record)

    return {"events": records, "count": len(records)}


def _safe_float(val, default: Optional[float] = 0.0) -> Optional[float]:
    """Ensure float value is JSON-compliant by replacing NaN and infinite values."""
    if val is None or pd.isna(val):
        return default
    try:
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


@app.get("/metrics")
async def get_metrics():
    """Get overall analysis metrics"""
    async with state_manager._lock:
        data = state_manager.event_analysis

    if data is None or data.empty:
        raise HTTPException(status_code=400, detail="Run /analyze first")

    corr = data['SurprisePercentage'].corr(data['PostEventReturn']) if len(data) > 1 else 0.0

    return {
        "total_events": len(data),
        "unique_tickers": data['Ticker'].nunique(),
        "date_range": {
            "start": str(data['EarningsDate'].min()),
            "end": str(data['EarningsDate'].max())
        },
        "returns": {
            "avg_pre_event": _safe_float(data['PreEventReturn'].mean()) if 'PreEventReturn' in data else 0.0,
            "avg_post_event": _safe_float(data['PostEventReturn'].mean()) if 'PostEventReturn' in data else 0.0,
            "avg_car": _safe_float(data['AbnormalReturnCAR'].mean()) if 'AbnormalReturnCAR' in data else 0.0
        },
        "surprise": {
            "avg": _safe_float(data['SurprisePercentage'].mean()) if 'SurprisePercentage' in data else 0.0,
            "std": _safe_float(data['SurprisePercentage'].std()) if 'SurprisePercentage' in data else 0.0,
            "correlation_with_return": _safe_float(corr)
        },
        "volatility": {
            "avg_pre": _safe_float(data['PreEventVolatility'].mean()) if 'PreEventVolatility' in data else 0.0,
            "avg_post": _safe_float(data['PostEventVolatility'].mean()) if 'PostEventVolatility' in data else 0.0
        },
        "volume": {
            "avg_spike_ratio": _safe_float(data['VolumeSpikeRatio'].mean()) if 'VolumeSpikeRatio' in data else 0.0
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.get('api_host', '0.0.0.0'), port=config.get('api_port', 8000))

