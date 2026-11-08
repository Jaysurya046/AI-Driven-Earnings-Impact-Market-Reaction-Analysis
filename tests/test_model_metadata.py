"""Tests for saving and loading model metadata, scalers, and library versions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.ml_models import ModelTrainer


def test_save_and_load_model_with_metadata(tmp_path):
    trainer = ModelTrainer()
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
    y_train = np.array([0, 0, 1, 1, 1])
    feature_cols = ['feat_a', 'feat_b']

    model, scaler = trainer.train_logistic_regression(X_train, y_train, scale=True)
    trainer.models['Logistic Regression'] = model
    trainer.scalers['Logistic Regression'] = scaler
    trainer.results['Logistic Regression'] = {'Accuracy': 0.8, 'F1-Score': 0.75}

    pkl_path = tmp_path / "models" / "lr_test.pkl"
    trainer.save_model(
        model_name='Logistic Regression',
        filepath=pkl_path,
        feature_columns=feature_cols,
        metadata={'dataset_version': '1.0'}
    )

    assert pkl_path.exists()
    json_path = pkl_path.with_suffix('.json')
    assert json_path.exists()

    # Verify JSON content
    with open(json_path, 'r') as f:
        meta_json = json.load(f)
    assert meta_json['model_name'] == 'Logistic Regression'
    assert meta_json['feature_columns'] == feature_cols
    assert meta_json['has_scaler'] is True
    assert 'scikit-learn' in meta_json['library_versions']
    assert meta_json['extra_metadata']['dataset_version'] == '1.0'

    # Load into fresh trainer
    loader = ModelTrainer()
    loaded_data = loader.load_model(pkl_path, 'Logistic Regression')

    assert 'model' in loaded_data
    assert loader.feature_columns['Logistic Regression'] == feature_cols
    assert loader.scalers['Logistic Regression'] is not None

    # Test predict with DataFrame containing columns out of order
    df_test = pd.DataFrame({'feat_b': [2.5, 4.5], 'feat_a': [1.5, 3.5], 'unrelated': [99, 99]})
    preds = loader.predict('Logistic Regression', df_test)
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})


def test_predict_missing_feature_raises_error(tmp_path):
    trainer = ModelTrainer()
    X = np.array([[1.0, 2.0], [2.0, 3.0]])
    y = np.array([0, 1])
    model = trainer.train_random_forest(X, y, n_estimators=5)
    trainer.models['RF'] = model
    trainer.scalers['RF'] = None
    trainer.feature_columns['RF'] = ['col1', 'col2']

    df_missing = pd.DataFrame({'col1': [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing required features"):
        trainer.predict('RF', df_missing)
