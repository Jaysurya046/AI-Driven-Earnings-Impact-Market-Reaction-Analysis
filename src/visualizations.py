"""
Visualization Module
Creates charts and dashboards for analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class Visualizer:
    """Creates visualization for earnings analysis"""
    
    @staticmethod
    def plot_event_window(
        event_analysis_df: pd.DataFrame,
        ticker: str,
        save_path: str = None
    ) -> None:
        """
        Plot post-event returns for a single stock
        
        Args:
            event_analysis_df: DataFrame from event analysis
            ticker: Stock ticker to plot
            save_path: Optional path to save figure
        """
        ticker_data = event_analysis_df[event_analysis_df['Ticker'] == ticker]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Pre vs Post Returns
        ax1 = axes[0]
        x_pos = np.arange(len(ticker_data))
        width = 0.35
        
        ax1.bar(x_pos - width/2, ticker_data['PreEventReturn'], width, label='Pre-Event', alpha=0.8)
        ax1.bar(x_pos + width/2, ticker_data['PostEventReturn'], width, label='Post-Event', alpha=0.8)
        ax1.set_ylabel('Return (%)')
        ax1.set_title(f'{ticker} - Pre vs Post Earnings Return')
        ax1.legend()
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Abnormal Returns
        ax2 = axes[1]
        colors = ['green' if x > 0 else 'red' for x in ticker_data['AbnormalReturnCAR']]
        ax2.bar(x_pos, ticker_data['AbnormalReturnCAR'], color=colors, alpha=0.8)
        ax2.set_ylabel('Cumulative Abnormal Return (%)')
        ax2.set_title(f'{ticker} - Cumulative Abnormal Returns')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_surprise_vs_reaction(
        event_analysis_df: pd.DataFrame,
        save_path: str = None
    ) -> None:
        """
        Scatter plot: Earnings Surprise vs Market Reaction
        
        Args:
            event_analysis_df: DataFrame from event analysis
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = ['green' if x > 0 else 'red' for x in event_analysis_df['PostEventReturn']]
        
        scatter = ax.scatter(
            event_analysis_df['SurprisePercentage'],
            event_analysis_df['PostEventReturn'],
            c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=0.5
        )
        
        # Add trend line
        z = np.polyfit(
            event_analysis_df['SurprisePercentage'].dropna(),
            event_analysis_df[event_analysis_df['SurprisePercentage'].notna()]['PostEventReturn'],
            1
        )
        p = np.poly1d(z)
        x_range = np.linspace(
            event_analysis_df['SurprisePercentage'].min(),
            event_analysis_df['SurprisePercentage'].max(),
            100
        )
        ax.plot(x_range, p(x_range), "b--", alpha=0.8, linewidth=2, label='Trend')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Earnings Surprise (%)', fontsize=12)
        ax.set_ylabel('Post-Event Return (%)', fontsize=12)
        ax.set_title('Earnings Surprise vs Market Reaction', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_model_comparison(
        model_comparison_df: pd.DataFrame,
        save_path: str = None
    ) -> None:
        """
        Bar plot comparing model performance
        
        Args:
            model_comparison_df: DataFrame from ModelTrainer.train_all_models()
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        metrics_df = model_comparison_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
        
        # Metrics comparison
        metrics_df.plot(kind='bar', ax=axes[0], width=0.8)
        axes[0].set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Score')
        axes[0].set_xlabel('')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0, 1.1])
        
        # AUC-ROC comparison
        auc_data = model_comparison_df.set_index('Model')['AUC-ROC'].dropna()
        if len(auc_data) > 0:
            auc_data.plot(kind='bar', ax=axes[1], color='skyblue', edgecolor='black')
            axes[1].set_title('AUC-ROC Score Comparison', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('AUC-ROC')
            axes[1].set_xlabel('')
            axes[1].set_ylim([0, 1.1])
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_feature_importance(
        importance_df: pd.DataFrame,
        top_n: int = 15,
        save_path: str = None
    ) -> None:
        """
        Bar plot of top feature importances
        
        Args:
            importance_df: DataFrame with Feature and Importance columns
            top_n: Number of top features to display
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = importance_df.head(top_n)
        
        ax.barh(range(len(top_features)), top_features['Importance'].values, color='steelblue', edgecolor='black')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'].values)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_volatility_analysis(
        event_analysis_df: pd.DataFrame,
        save_path: str = None
    ) -> None:
        """
        Volatility before, during, and after earnings
        
        Args:
            event_analysis_df: DataFrame from event analysis
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(event_analysis_df))
        width = 0.25
        
        ax.bar(x_pos - width, event_analysis_df['EstimationVolatility'], width, 
               label='Estimation Period', alpha=0.8, color='blue')
        ax.bar(x_pos, event_analysis_df['PreEventVolatility'], width, 
               label='Pre-Event', alpha=0.8, color='orange')
        ax.bar(x_pos + width, event_analysis_df['PostEventVolatility'], width, 
               label='Post-Event', alpha=0.8, color='red')
        
        ax.set_ylabel('Volatility (%)')
        ax.set_title('Volatility Around Earnings Events', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_volume_analysis(
        event_analysis_df: pd.DataFrame,
        save_path: str = None
    ) -> None:
        """
        Volume before and after earnings
        
        Args:
            event_analysis_df: DataFrame from event analysis
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(event_analysis_df))
        width = 0.35
        
        ax.bar(x_pos - width/2, event_analysis_df['EstimationAvgVolume']/1e6, width, 
               label='Estimation Period', alpha=0.8, color='steelblue')
        ax.bar(x_pos + width/2, event_analysis_df['PostEventAvgVolume']/1e6, width, 
               label='Post-Event', alpha=0.8, color='coral')
        
        ax.set_ylabel('Average Volume (Millions)')
        ax.set_title('Volume Around Earnings Events', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix_array: np.ndarray,
        model_name: str = "Model",
        save_path: str = None
    ) -> None:
        """
        Plot confusion matrix heatmap
        
        Args:
            confusion_matrix_array: Confusion matrix from sklearn
            model_name: Name of model
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(confusion_matrix_array, annot=True, fmt='d', cmap='Blues', 
                   cbar=True, ax=ax, square=True)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
