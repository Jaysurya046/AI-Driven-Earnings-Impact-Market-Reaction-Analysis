"""
Streamlit Web Interface for Interactive Earnings Analysis
Provides an interactive dashboard for analyzing earnings impact and ML predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data_collection import DataCollector
from src.event_analysis import EventAnalyzer
from src.feature_engineering import FeatureEngineer
from src.ml_models import ModelTrainer
from src.visualizations import Visualizer
from src.chart_data import (
    prepare_surprise_vs_reaction_data,
    prepare_model_comparison_data,
    prepare_volatility_analysis_data,
)


# Page configuration
st.set_page_config(
    page_title="Earnings Impact Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'price_data' not in st.session_state:
        st.session_state.price_data = None
    if 'earnings_data' not in st.session_state:
        st.session_state.earnings_data = None
    if 'event_analysis' not in st.session_state:
        st.session_state.event_analysis = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    if 'labels' not in st.session_state:
        st.session_state.labels = None
    if 'model_comparison' not in st.session_state:
        st.session_state.model_comparison = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = None


def sidebar_configuration():
    """Sidebar for user configuration"""
    st.sidebar.header("⚙️ Configuration")
    
    # Stock selection
    default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
    tickers_input = st.sidebar.text_input(
        "Stock Tickers (comma-separated)",
        value=", ".join(default_tickers),
        help="Enter stock ticker symbols separated by commas"
    )
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    lookback_years = st.sidebar.slider(
        "Lookback Period (Years)",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of years of historical data to analyze"
    )
    
    event_window = st.sidebar.slider(
        "Event Window (Days)",
        min_value=1,
        max_value=10,
        value=5,
        help="Trading days before and after earnings announcement"
    )
    
    estimation_window = st.sidebar.slider(
        "Estimation Window (Days)",
        min_value=30,
        max_value=120,
        value=60,
        help="Days used to estimate normal returns"
    )
    
    label_threshold = st.sidebar.number_input(
        "Label Threshold (%)",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
        help="Threshold for classifying positive/negative reactions"
    )
    
    # Sector analysis options
    st.sidebar.subheader("Sector Analysis")
    enable_sector = st.sidebar.checkbox(
        "Enable Sector Comparison",
        value=False,
        help="Compare performance across different sectors"
    )
    
    # Backtesting options
    st.sidebar.subheader("Backtesting")
    enable_backtest = st.sidebar.checkbox(
        "Enable Backtesting",
        value=False,
        help="Run backtesting on trading strategies"
    )
    
    return {
        'tickers': [t.strip().upper() for t in tickers_input.split(',') if t.strip()],
        'lookback_years': lookback_years,
        'event_window': event_window,
        'estimation_window': estimation_window,
        'label_threshold': label_threshold,
        'enable_sector': enable_sector,
        'enable_backtest': enable_backtest
    }


def run_analysis(config: Dict) -> Optional[Dict]:
    """Run the complete analysis pipeline"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = {}
    
    try:
        # Step 1: Data Collection
        status_text.text("📊 Collecting data from Yahoo Finance...")
        collector = DataCollector(lookback_years=config['lookback_years'])
        price_data, earnings_data = collector.fetch_multiple_stocks(config['tickers'])
        results['price_data'] = price_data
        results['earnings_data'] = earnings_data
        progress_bar.progress(20)
        
        # Step 2: Event Analysis
        status_text.text("🔍 Analyzing earnings events...")
        analyzer = EventAnalyzer(
            event_window=config['event_window'],
            estimation_window=config['estimation_window']
        )
        event_analysis = analyzer.analyze_multiple_events(price_data, earnings_data)
        results['event_analysis'] = event_analysis
        progress_bar.progress(40)
        
        # Step 3: Feature Engineering
        status_text.text("⚙️ Engineering features...")
        engineer = FeatureEngineer()
        features = engineer.create_features(event_analysis)
        labels = engineer.create_binary_labels(event_analysis, threshold=config['label_threshold'])
        results['features'] = features
        results['labels'] = labels
        progress_bar.progress(60)
        
        # Step 4: Model Training
        status_text.text("🤖 Training ML models...")
        X_train, X_test, y_train, y_test, feature_columns = engineer.prepare_train_test_data(
            features, labels, test_size=0.2, random_state=42, chronological=True
        )
        
        trainer = ModelTrainer()
        model_comparison = trainer.train_all_models(X_train, X_test, y_train, y_test)
        results['model_comparison'] = model_comparison
        results['trained_models'] = trainer
        progress_bar.progress(80)
        
        # Step 5: Sector Analysis (if enabled)
        if config.get('enable_sector', False):
            status_text.text("🏢 Performing sector comparison...")
            from src.sector_comparison import SectorAnalyzer
            sector_analyzer = SectorAnalyzer()
            sector_results = sector_analyzer.analyze_by_sector(event_analysis)
            results['sector_analysis'] = sector_results
            progress_bar.progress(90)
        
        # Step 6: Backtesting (if enabled)
        if config.get('enable_backtest', False):
            status_text.text("📈 Running backtesting...")
            from src.backtesting import Backtester
            backtester = Backtester(initial_capital=100000)
            backtest_results = backtester.run_strategy(event_analysis, strategy='momentum')
            results['backtest_results'] = backtest_results
            progress_bar.progress(95)
        
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)
        
        return results
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        return None


def display_overview_dashboard(results: Dict):
    """Display main overview dashboard"""
    st.markdown('<p class="main-header">📈 Earnings Impact Analysis Dashboard</p>', unsafe_allow_html=True)
    
    # Key metrics row
    event_analysis = results['event_analysis']
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_events = len(event_analysis)
        st.metric("Total Events Analyzed", f"{total_events:,}")
    
    with col2:
        avg_surprise = event_analysis['SurprisePercentage'].mean()
        st.metric("Avg Earnings Surprise", f"{avg_surprise:.2f}%")
    
    with col3:
        avg_return = event_analysis['PostEventReturn'].mean()
        st.metric("Avg Post-Event Return", f"{avg_return:.2f}%")
    
    with col4:
        correlation = event_analysis['SurprisePercentage'].corr(event_analysis['PostEventReturn'])
        st.metric("Surprise-Return Correlation", f"{correlation:.3f}")
    
    st.divider()
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        # Check if statsmodels is available for OLS trendline
        try:
            import statsmodels.api  # noqa: F401
            has_statsmodels = True
        except ImportError:
            has_statsmodels = False

        scatter_kwargs = {
            'x': 'SurprisePercentage',
            'y': 'PostEventReturn',
            'color': 'Ticker',
            'title': 'Earnings Surprise vs Market Reaction',
            'labels': {'SurprisePercentage': 'EPS Surprise (%)', 'PostEventReturn': 'Post-Event Return (%)'},
            'hover_data': ['Ticker', 'EarningsDate'],
        }
        if has_statsmodels:
            scatter_kwargs['trendline'] = 'ols'

        # Earnings Surprise vs Market Reaction (with fallback if OLS fitting fails)
        try:
            fig_scatter = px.scatter(
                event_analysis,
                **scatter_kwargs
            )
        except Exception:
            scatter_kwargs.pop('trendline', None)
            fig_scatter = px.scatter(
                event_analysis,
                **scatter_kwargs
            )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Returns distribution
        fig_hist = px.histogram(
            event_analysis,
            x='PostEventReturn',
            color='Ticker',
            title='Distribution of Post-Event Returns',
            labels={'PostEventReturn': 'Return (%)'},
            nbins=30,
            opacity=0.7
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Volatility analysis using shared data-preparation helper
        prep_vol = prepare_volatility_analysis_data(event_analysis)
        vol_data_melted = prep_vol['melted_df']

        if not vol_data_melted.empty:
            fig_vol = px.box(
                vol_data_melted,
                x='Period',
                y='Volatility',
                color='Period',
                title='Volatility: Pre vs Post Event',
                labels={'Volatility': 'Volatility (%)'}
            )
            st.plotly_chart(fig_vol, use_container_width=True)
    
    with col2:
        # Volume analysis
        fig_vol_volume = px.bar(
            event_analysis.groupby('Ticker')['VolumeSpikeRatio'].mean().reset_index(),
            x='Ticker',
            y='VolumeSpikeRatio',
            title='Average Volume Spike by Stock',
            labels={'VolumeSpikeRatio': 'Volume Spike Ratio', 'Ticker': 'Stock'}
        )
        st.plotly_chart(fig_vol_volume, use_container_width=True)


def display_model_performance(results: Dict):
    """Display ML model performance section"""
    st.header("🤖 Machine Learning Model Performance")
    
    model_comparison = results['model_comparison']
    prep_mc = prepare_model_comparison_data(model_comparison)

    # Model comparison table
    st.subheader("Model Comparison Metrics")
    st.dataframe(prep_mc['display_df'], use_container_width=True)

    # Best model highlight
    best_model = prep_mc['best_model']
    if best_model:
        st.success(f"🏆 Best Performing Model: **{best_model['Model']}** (F1-Score: {best_model['F1-Score']:.4f})")
    
    # Model performance chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig_metrics = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, row in model_comparison.iterrows():
            fig_metrics.add_trace(go.Bar(
                name=row['Model'],
                x=metrics,
                y=[row[m] for m in metrics],
                text=[f"{row[m]:.3f}" for m in metrics],
                textposition='auto'
            ))
        
        fig_metrics.update_layout(
            title='Model Performance Comparison',
            barmode='group',
            yaxis=dict(range=[0, 1]),
            showlegend=True
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with col2:
        # Feature importance (from best model)
        if results.get('features') is not None:
            trained_models = results['trained_models']
            best_model_name = best_model['Model']
            model = trained_models.models[best_model_name]
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_cols = [col for col in results['features'].columns if col not in 
                               ['Ticker', 'EarningsDate', 'PostEventReturn', 'PreEventReturn', 
                                'EPS', 'SurprisePercentage', 'DataPointsInWindow', 'DataPointsInEstimation']]
                if len(feature_cols) != len(importances):
                    feature_cols = [f"Feature_{i}" for i in range(len(importances))]
                
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                
                fig_importance = px.bar(
                    importance_df.tail(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f'Top 10 Features - {best_model_name}',
                    labels={'Importance': 'Importance Score', 'Feature': ''}
                )
                st.plotly_chart(fig_importance, use_container_width=True)


def display_sector_analysis(results: Dict):
    """Display sector comparison section"""
    if 'sector_analysis' not in results:
        st.warning("Sector analysis not enabled. Enable it in the sidebar.")
        return
    
    st.header("🏢 Sector Comparison Analysis")
    
    sector_results = results['sector_analysis']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average returns by sector
        returns_df = sector_results['sector_returns']
        ret_col = 'AvgReturn' if 'AvgReturn' in returns_df.columns else returns_df.columns[1]
        fig_sector_returns = px.bar(
            returns_df,
            x='Sector',
            y=ret_col,
            title='Average Post-Event Return by Sector',
            labels={ret_col: 'Return (%)', 'Sector': ''},
            color=ret_col,
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_sector_returns, use_container_width=True)
    
    with col2:
        # Sector volatility
        vol_df = sector_results['sector_volatility']
        vol_col = 'AvgVolatility' if 'AvgVolatility' in vol_df.columns else ('AvgPostVol' if 'AvgPostVol' in vol_df.columns else vol_df.columns[1])
        fig_sector_vol = px.bar(
            vol_df,
            x='Sector',
            y=vol_col,
            title='Average Volatility by Sector',
            labels={vol_col: 'Volatility (%)', 'Sector': ''},
            color=vol_col,
            color_continuous_scale='YlOrRd'
        )
        st.plotly_chart(fig_sector_vol, use_container_width=True)
    
    # Sector comparison table
    st.subheader("Detailed Sector Metrics")
    if 'sector_summary' in sector_results:
        st.dataframe(sector_results['sector_summary'].round(4), use_container_width=True)


def display_backtesting(results: Dict):
    """Display backtesting results section"""
    if 'backtest_results' not in results:
        st.warning("Backtesting not enabled. Enable it in the sidebar.")
        return
    
    st.header("📈 Strategy Backtesting Results")
    
    backtest = results['backtest_results']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'total_return' in backtest:
            total_return = backtest['total_return']
        elif len(backtest.get('portfolio_value', [])) > 0:
            total_return = backtest['portfolio_value'][-1] / backtest['initial_capital'] - 1
        else:
            total_return = 0.0
        st.metric("Total Return", f"{total_return:.2%}")
    
    with col2:
        sharpe_ratio = backtest.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    with col3:
        max_drawdown = backtest.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_drawdown:.2%}")
    
    with col4:
        win_rate = backtest.get('win_rate', 0)
        st.metric("Win Rate", f"{win_rate:.2%}")
    
    # Portfolio value over time
    fig_portfolio = go.Figure()
    fig_portfolio.add_trace(go.Scatter(
        x=backtest['dates'],
        y=backtest['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        fill='tozeroy'
    ))
    fig_portfolio.update_layout(
        title='Portfolio Value Over Time',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        showlegend=False
    )
    st.plotly_chart(fig_portfolio, use_container_width=True)
    
    # Trade history
    if 'trades' in backtest:
        st.subheader("Trade History")
        trades_df = pd.DataFrame(backtest['trades'])
        st.dataframe(trades_df, use_container_width=True)


def display_data_explorer(results: Dict):
    """Display data explorer section"""
    st.header("🔍 Data Explorer")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Price Data",
        "Earnings Data",
        "Event Analysis",
        "Features"
    ])
    
    with tab1:
        if results.get('price_data') is not None:
            st.dataframe(results['price_data'].head(100), use_container_width=True)
            st.write(f"**Shape:** {results['price_data'].shape}")
        else:
            st.info("No price data available")
    
    with tab2:
        if results.get('earnings_data') is not None:
            st.dataframe(results['earnings_data'].head(100), use_container_width=True)
            st.write(f"**Shape:** {results['earnings_data'].shape}")
        else:
            st.info("No earnings data available")
    
    with tab3:
        if results.get('event_analysis') is not None:
            st.dataframe(results['event_analysis'].head(100), use_container_width=True)
            st.write(f"**Shape:** {results['event_analysis'].shape}")
        else:
            st.info("No event analysis data available")
    
    with tab4:
        if results.get('features') is not None:
            st.dataframe(results['features'].head(100), use_container_width=True)
            st.write(f"**Shape:** {results['features'].shape}")
        else:
            st.info("No features data available")


def main():
    """Main application function"""
    initialize_session_state()
    
    # Sidebar configuration
    config = sidebar_configuration()
    
    # Main content area
    st.markdown('<p style="text-align: center; color: #666;">Interactive Earnings Impact & Market Reaction Analysis</p>', unsafe_allow_html=True)
    st.divider()
    
    # Run analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Run Analysis", use_container_width=True):
            with st.spinner("Running analysis... This may take a few minutes."):
                results = run_analysis(config)
                if results:
                    st.session_state.analysis_complete = True
                    for key, value in results.items():
                        st.session_state[key] = value
                    st.rerun()
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete:
        results = {key: st.session_state[key] for key in [
            'price_data', 'earnings_data', 'event_analysis', 
            'features', 'labels', 'model_comparison', 'trained_models'
        ] if st.session_state[key] is not None}
        
        # Add optional analyses
        if 'sector_analysis' in st.session_state:
            results['sector_analysis'] = st.session_state['sector_analysis']
        if 'backtest_results' in st.session_state:
            results['backtest_results'] = st.session_state['backtest_results']
        
        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "🤖 ML Models",
            "🏢 Sector Analysis",
            "📈 Backtesting",
            "🔍 Data Explorer"
        ])
        
        with tab1:
            display_overview_dashboard(results)
        
        with tab2:
            display_model_performance(results)
        
        with tab3:
            display_sector_analysis(results)
        
        with tab4:
            display_backtesting(results)
        
        with tab5:
            display_data_explorer(results)
    
    else:
        # Welcome message
        st.info("👈 Configure your analysis parameters in the sidebar and click 'Run Analysis' to get started!")
        
        # Example insights
        st.markdown("""
        ### What This Tool Does:
        
        - **📊 Event Analysis**: Analyzes stock price movements around earnings announcements
        - **🤖 ML Predictions**: Trains models to predict market reactions to earnings
        - **🏢 Sector Comparison**: Compares performance across different industry sectors
        - **📈 Backtesting**: Tests trading strategies based on earnings signals
        - **🔍 Interactive Exploration**: Explore raw data and engineered features
        
        ### Quick Start:
        1. Enter stock tickers (e.g., AAPL, MSFT, GOOGL)
        2. Adjust analysis parameters as needed
        3. Enable sector comparison and backtesting if desired
        4. Click "Run Analysis" and explore the results!
        """)


if __name__ == "__main__":
    main()
