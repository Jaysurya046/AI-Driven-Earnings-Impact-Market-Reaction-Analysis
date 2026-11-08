"""
LSTM Networks for Time-Series Prediction
Advanced deep learning models for earnings impact prediction
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMPredictor:
    """LSTM-based time-series prediction for market reactions"""
    
    def __init__(self, 
                 sequence_length: int = 10,
                 lstm_units: int = 50,
                 dropout_rate: float = 0.2):
        """
        Initialize LSTM predictor
        
        Args:
            sequence_length: Number of time steps to look back
            lstm_units: Number of LSTM units in hidden layer
            dropout_rate: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model: Any = None
        self.scaler: Any = None
        self.history: Any = None
    
    def _check_tensorflow(self):
        """Check if TensorFlow is available"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping
            from sklearn.preprocessing import MinMaxScaler
            
            return {
                'tf': tf,
                'Sequential': Sequential,
                'LSTM': LSTM,
                'Dense': Dense,
                'Dropout': Dropout,
                'Adam': Adam,
                'EarlyStopping': EarlyStopping,
                'MinMaxScaler': MinMaxScaler
            }
        except ImportError as e:
            logger.error(f"TensorFlow not installed. Run: pip install tensorflow")
            logger.error(f"Error: {str(e)}")
            raise
    
    def prepare_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training
        
        Args:
            data: Input feature array
            target: Target values
            
        Returns:
            Tuple of (X_sequences, y_target)
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(target[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int], n_features: int) -> None:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            n_features: Number of input features
        """
        keras_libs = self._check_tensorflow()
        
        model = keras_libs['Sequential']([
            keras_libs['LSTM'](
                self.lstm_units,
                return_sequences=True,
                input_shape=input_shape
            ),
            keras_libs['Dropout'](self.dropout_rate),
            
            keras_libs['LSTM'](
                self.lstm_units // 2,
                return_sequences=False
            ),
            keras_libs['Dropout'](self.dropout_rate),
            
            keras_libs['Dense'](25, activation='relu'),
            keras_libs['Dense'](1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=keras_libs['Adam'](learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        self.model = model
        logger.info(f"LSTM model built with input shape: {input_shape}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32,
              early_stopping_patience: int = 10) -> Dict:
        """
        Train LSTM model
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dictionary with training history
        """
        keras_libs = self._check_tensorflow()
        
        # Scale features
        self.scaler = keras_libs['MinMaxScaler'](feature_range=(0, 1))
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Reshape for LSTM [samples, timesteps, features]
        X_train_reshaped = np.zeros((len(X_train_scaled) - self.sequence_length, 
                                      self.sequence_length, X_train_scaled.shape[1]))
        
        for i in range(len(X_train_scaled) - self.sequence_length):
            X_train_reshaped[i] = X_train_scaled[i:i + self.sequence_length]
        
        # Prepare targets (align with sequences)
        y_train_seq = y_train[self.sequence_length:]
        
        # Build model
        n_timesteps = X_train_reshaped.shape[1]
        n_features = X_train_reshaped.shape[2]
        self.build_model(input_shape=(n_timesteps, n_features), n_features=n_features)
        
        # Callbacks
        callbacks = [
            keras_libs['EarlyStopping'](
                monitor='val_loss' if X_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_reshaped = np.zeros((len(X_val_scaled) - self.sequence_length,
                                        self.sequence_length, X_val_scaled.shape[1]))
            for i in range(len(X_val_scaled) - self.sequence_length):
                X_val_reshaped[i] = X_val_scaled[i:i + self.sequence_length]
            y_val_seq = y_val[self.sequence_length:]
            validation_data = (X_val_reshaped, y_val_seq)
        
        # Train model
        logger.info(f"Training LSTM model for up to {epochs} epochs...")
        history = self.model.fit(
            X_train_reshaped, y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2 if validation_data is None else 0.0,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        
        logger.info("LSTM training completed")
        logger.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        if 'val_accuracy' in history.history:
            logger.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with trained model
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Array of predictions
        """
        if self.model is None or self.scaler is None:
            logger.error("Model not trained yet. Call train() first.")
            return None
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_reshaped = np.zeros((len(X_scaled) - self.sequence_length,
                               self.sequence_length, X_scaled.shape[1]))
        for i in range(len(X_scaled) - self.sequence_length):
            X_reshaped[i] = X_scaled[i:i + self.sequence_length]
        
        # Predict
        predictions = self.model.predict(X_reshaped, verbose=0)
        
        return predictions.flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Input features
            
        Returns:
            Array of probability scores
        """
        return self.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                                   f1_score, roc_auc_score, confusion_matrix
        
        predictions = self.predict(X_test)
        
        # Convert probabilities to binary predictions
        y_pred = (predictions > 0.5).astype(int)
        
        # Align test labels with predictions
        y_test_aligned = y_test[self.sequence_length:len(y_pred) + self.sequence_length]
        
        metrics = {
            'Accuracy': accuracy_score(y_test_aligned, y_pred),
            'Precision': precision_score(y_test_aligned, y_pred, zero_division=0),
            'Recall': recall_score(y_test_aligned, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test_aligned, y_pred, zero_division=0),
            'AUC-ROC': roc_auc_score(y_test_aligned, predictions),
            'ConfusionMatrix': confusion_matrix(y_test_aligned, y_pred).tolist()
        }
        
        logger.info("LSTM Model Evaluation:")
        for metric, value in metrics.items():
            if metric != 'ConfusionMatrix':
                logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            logger.error("No model to save")
            return
        
        keras_libs = self._check_tensorflow()
        
        # Save Keras model
        model_path = Path(filepath).parent / f"{Path(filepath).stem}_model.h5"
        self.model.save(str(model_path))
        
        # Save scaler and parameters
        import pickle
        model_data = {
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'history': self.history
        }
        
        params_path = Path(filepath).parent / f"{Path(filepath).stem}_params.pkl"
        with open(params_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Parameters saved to {params_path}")
    
    def load_model(self, model_path: str, params_path: str) -> None:
        """
        Load trained model from disk
        
        Args:
            model_path: Path to Keras model file
            params_path: Path to parameters file
        """
        keras_libs = self._check_tensorflow()
        
        # Load parameters
        import pickle
        with open(params_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.sequence_length = model_data['sequence_length']
        self.lstm_units = model_data['lstm_units']
        self.dropout_rate = model_data['dropout_rate']
        self.history = model_data['history']
        
        # Load Keras model
        self.model = keras_libs['tf'].keras.models.load_model(model_path)
        
        logger.info(f"Model loaded from {model_path}")


class HybridEnsemblePredictor:
    """Combines LSTM with traditional ML models for improved predictions"""
    
    def __init__(self):
        """Initialize hybrid ensemble"""
        self.lstm_predictor = None
        self.traditional_models = {}
        self.weights = {}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train hybrid ensemble
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with training results
        """
        from src.ml_models import ModelTrainer
        
        # Train LSTM
        logger.info("Training LSTM component...")
        self.lstm_predictor = LSTMPredictor(sequence_length=10, lstm_units=64)
        self.lstm_predictor.train(X_train, y_train, epochs=30, batch_size=32)
        lstm_eval = self.lstm_predictor.evaluate(X_test, y_test)
        
        # Train traditional models
        logger.info("Training traditional ML models...")
        trainer = ModelTrainer()
        model_comparison = trainer.train_all_models(X_train, X_test, y_train, y_test)
        
        # Select best traditional model
        best_traditional = model_comparison.loc[model_comparison['F1-Score'].idxmax()]
        best_model_name = best_traditional['Model']
        self.traditional_models[best_model_name] = trainer.models[best_model_name]
        
        # Calculate ensemble weights based on F1-scores
        lstm_f1 = lstm_eval['F1-Score']
        traditional_f1 = best_traditional['F1-Score']
        
        total_f1 = lstm_f1 + traditional_f1
        self.weights = {
            'lstm': lstm_f1 / total_f1,
            'traditional': traditional_f1 / total_f1
        }
        
        results = {
            'lstm_performance': lstm_eval,
            'traditional_performance': best_traditional.to_dict(),
            'ensemble_weights': self.weights,
            'trained_models': trainer
        }
        
        logger.info(f"Ensemble weights - LSTM: {self.weights['lstm']:.3f}, "
                   f"Traditional: {self.weights['traditional']:.3f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Input features
            
        Returns:
            Ensemble predictions
        """
        if self.lstm_predictor is None or not self.traditional_models:
            logger.error("Ensemble not trained yet")
            return None
        
        # Get predictions from both models
        lstm_pred = self.lstm_predictor.predict_proba(X)
        
        # Get traditional model prediction
        model_name = list(self.traditional_models.keys())[0]
        model = self.traditional_models[model_name]
        scaler = getattr(model_name, 'scalers', {}).get(model_name, None)
        
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        traditional_pred = model.predict_proba(X_scaled)[:, 1]
        
        # Align predictions (LSTM has sequence offset)
        min_len = min(len(lstm_pred), len(traditional_pred))
        lstm_pred = lstm_pred[-min_len:]
        traditional_pred = traditional_pred[-min_len:]
        
        # Weighted ensemble
        ensemble_pred = (
            self.weights['lstm'] * lstm_pred +
            self.weights['traditional'] * traditional_pred
        )
        
        return ensemble_pred


def prepare_time_series_features(event_analysis: pd.DataFrame, 
                                  feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features for time-series modeling
    
    Args:
        event_analysis: Event analysis DataFrame
        feature_columns: Columns to use as features
        
    Returns:
        Tuple of (features_array, labels_array)
    """
    # Sort by ticker and date
    data = event_analysis.sort_values(['Ticker', 'EarningsDate']).copy()
    
    # Extract features and labels
    X = data[feature_columns].values
    y = (data['PostEventReturn'] > 0).astype(int).values
    
    return X, y


if __name__ == "__main__":
    print("LSTM Time-Series Prediction Module")
    print("=" * 50)
    print("\nThis module provides LSTM-based deep learning for earnings prediction.")
    print("Requires TensorFlow 2.x to be installed.")
    print("\nKey Features:")
    print("- Sequence-based time-series modeling")
    print("- Automatic architecture building")
    print("- Early stopping and regularization")
    print("- Hybrid ensemble with traditional ML")
    print("- Model persistence and loading")
