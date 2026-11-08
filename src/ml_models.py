"""
Machine Learning Models Module
Trains and evaluates classification models for market reaction prediction
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import sklearn
import xgboost as xgb
from typing import Dict, Tuple, List, Optional, Union
import logging
import pickle
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates ML models for earnings prediction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_columns = {}
        self.metadata = {}
    
    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scale: bool = True,
        **kwargs
    ) -> Tuple[LogisticRegression, StandardScaler]:
        """
        Train Logistic Regression model
        
        Args:
            X_train: Training features
            y_train: Training labels
            scale: Whether to scale features
            **kwargs: Additional parameters for LogisticRegression
            
        Returns:
            Tuple of (trained_model, scaler)
        """
        logger.info("Training Logistic Regression...")
        
        scaler = None
        X_train_scaled = X_train
        
        if scale:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
        
        model = LogisticRegression(max_iter=1000, random_state=42, **kwargs)
        model.fit(X_train_scaled, y_train)
        
        logger.info("Logistic Regression training completed")
        return model, scaler
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100,
        **kwargs
    ) -> RandomForestClassifier:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of trees
            **kwargs: Additional parameters for RandomForestClassifier
            
        Returns:
            Trained RandomForestClassifier
        """
        logger.info("Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
        model.fit(X_train, y_train)
        
        logger.info("Random Forest training completed")
        return model
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scale: bool = True,
        **kwargs
    ) -> Tuple[xgb.XGBClassifier, StandardScaler]:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            scale: Whether to scale features
            **kwargs: Additional parameters for XGBClassifier
            
        Returns:
            Tuple of (trained_model, scaler)
        """
        logger.info("Training XGBoost...")
        
        scaler = None
        X_train_scaled = X_train
        
        if scale:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
        
        model = xgb.XGBClassifier(
            random_state=42,
            verbosity=0,
            **kwargs
        )
        model.fit(X_train_scaled, y_train)
        
        logger.info("XGBoost training completed")
        return model, scaler
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scaler = None,
        model_name: str = "Model"
    ) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            scaler: Optional scaler for feature scaling
            model_name: Name of model for logging
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Scale test data if scaler provided
        X_test_scaled = X_test
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ConfusionMatrix': confusion_matrix(y_test, y_pred),
            'ClassificationReport': classification_report(y_test, y_pred)
        }
        
        # Add AUC-ROC if probabilities available
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                results['AUC-ROC'] = roc_auc
                results['PredictionProbabilities'] = y_pred_proba
            except:
                logger.warning(f"Could not calculate AUC-ROC for {model_name}")
        
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        if 'AUC-ROC' in results:
            logger.info(f"  AUC-ROC:   {results['AUC-ROC']:.4f}")
        
        return results
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Train all models and generate comparison
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            DataFrame with comparison of all models
        """
        results_list = []
        
        # Logistic Regression
        lr_model, lr_scaler = self.train_logistic_regression(X_train, y_train)
        lr_results = self.evaluate_model(lr_model, X_test, y_test, lr_scaler, "Logistic Regression")
        self.models['Logistic Regression'] = lr_model
        self.scalers['Logistic Regression'] = lr_scaler
        results_list.append(lr_results)
        
        # Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        rf_results = self.evaluate_model(rf_model, X_test, y_test, None, "Random Forest")
        self.models['Random Forest'] = rf_model
        self.scalers['Random Forest'] = None
        results_list.append(rf_results)
        
        # XGBoost
        xgb_model, xgb_scaler = self.train_xgboost(X_train, y_train)
        xgb_results = self.evaluate_model(xgb_model, X_test, y_test, xgb_scaler, "XGBoost")
        self.models['XGBoost'] = xgb_model
        self.scalers['XGBoost'] = xgb_scaler
        results_list.append(xgb_results)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame([
            {
                'Model': r['Model'],
                'Accuracy': r['Accuracy'],
                'Precision': r['Precision'],
                'Recall': r['Recall'],
                'F1-Score': r['F1-Score'],
                'AUC-ROC': r.get('AUC-ROC', np.nan)
            }
            for r in results_list
        ])
        
        self.results = {r['Model']: r for r in results_list}
        
        return comparison_df
    
    def save_model(
        self,
        model_name: str,
        filepath: Union[str, Path],
        feature_columns: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save trained model to disk along with metadata (feature columns, scalers, library versions)
        
        Args:
            model_name: Name of model to save
            filepath: Path to save file (.pkl)
            feature_columns: List of feature column names used during training
            metadata: Optional additional metadata dict
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        cols = feature_columns or self.feature_columns.get(model_name)
        if cols is None and hasattr(self, '_global_feature_columns'):
            cols = self._global_feature_columns

        metrics = self.results.get(model_name, {})
        # Serialize only JSON-compatible metrics
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                clean_metrics[k] = v
            elif isinstance(v, (np.floating, np.integer)):
                clean_metrics[k] = float(v)

        library_versions = {
            'scikit-learn': sklearn.__version__,
            'xgboost': xgb.__version__,
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'python': platform.python_version()
        }

        model_data = {
            'model': self.models[model_name],
            'scaler': self.scalers.get(model_name),
            'feature_columns': cols,
            'model_name': model_name,
            'metrics': clean_metrics,
            'library_versions': library_versions,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'extra_metadata': metadata or {}
        }
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {path}")

            # Save readable JSON metadata sidecar
            json_meta_path = path.with_suffix('.json')
            meta_json = {
                'model_name': model_name,
                'model_class': type(self.models[model_name]).__name__,
                'has_scaler': self.scalers.get(model_name) is not None,
                'scaler_class': type(self.scalers.get(model_name)).__name__ if self.scalers.get(model_name) is not None else None,
                'feature_columns': cols,
                'metrics': clean_metrics,
                'library_versions': library_versions,
                'created_at': model_data['created_at'],
                'extra_metadata': metadata or {}
            }
            with open(json_meta_path, 'w') as jf:
                json.dump(meta_json, jf, indent=2)
            logger.info(f"Model metadata saved to {json_meta_path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self, filepath: Union[str, Path], model_name: str) -> Dict:
        """
        Load model from disk and restore model, scaler, and metadata
        
        Args:
            filepath: Path to model file (.pkl)
            model_name: Name to assign to loaded model
            
        Returns:
            Dictionary containing loaded model data
        """
        path = Path(filepath)
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models[model_name] = model_data['model']
            self.scalers[model_name] = model_data.get('scaler')
            self.feature_columns[model_name] = model_data.get('feature_columns')
            self.metadata[model_name] = model_data
            
            logger.info(f"Model {model_name} loaded from {path}")
            return model_data
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, model_name: str, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using a loaded model, automatically applying feature alignment and scaling.
        
        Args:
            model_name: Name of model to use
            X: Input features (DataFrame with feature columns or ndarray)
            
        Returns:
            Numpy array of predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' has not been trained or loaded.")
            
        model = self.models[model_name]
        scaler = self.scalers.get(model_name)
        
        if isinstance(X, pd.DataFrame):
            expected_cols = self.feature_columns.get(model_name)
            if expected_cols:
                missing = [c for c in expected_cols if c not in X.columns]
                if missing:
                    raise ValueError(f"Input DataFrame is missing required features: {missing}")
                X_vals = X[expected_cols].values
            else:
                X_vals = X.values
        else:
            X_vals = np.asarray(X)
            
        if scaler is not None:
            X_vals = scaler.transform(X_vals)
            
        return model.predict(X_vals)
