"""
Configuration Module
Unified config loader merging config.ini, environment variables, and defaults.
Loads sensitive configurations and secrets from .env via python-dotenv.
"""

import configparser
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_CONFIG: Dict[str, Any] = {
    'lookback_years': 5,
    'tickers': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'PG',
        'V', 'UNH', 'HD', 'BAC', 'MA', 'XOM', 'DIS', 'CSCO', 'PFE', 'NFLX'
    ],
    'event_window': 5,
    'estimation_window': 60,
    'label_threshold': 0.0,
    'test_size': 0.2,
    'random_state': 42,
    'models_to_train': ['logistic_regression', 'random_forest', 'xgboost'],
    'data_dir': 'data',
    'models_dir': 'models',
    'viz_dir': 'visualizations',
    'notebooks_dir': 'notebooks',
    'figure_size': (14, 8),
    'dpi': 300,
    'color_scheme': 'seaborn',
    'api_host': '0.0.0.0',
    'api_port': 8000,
    'cors_origins': ['*'],
    'api_secret_key': '',
}


def _parse_list(val: Optional[str]) -> List[str]:
    """Parse comma-separated string into list of stripped strings."""
    if not val:
        return []
    return [item.strip() for item in val.split(',') if item.strip()]


def mask_secrets(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return a dictionary copy with sensitive keys masked."""
    masked: Dict[str, Any] = {}
    for k, v in config_dict.items():
        if any(secret in k.lower() for secret in ['secret', 'key', 'token', 'password']):
            masked[k] = '***' if v else ''
        else:
            masked[k] = v
    return masked


class AppConfig:
    """Unified configuration container with dict-like and attribute access."""

    def __init__(self, data: Optional[Dict[str, Any]] = None, **kwargs):
        self._data: Dict[str, Any] = dict(DEFAULT_CONFIG)
        if data:
            self._data.update(data)
        if kwargs:
            self._data.update(kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'AppConfig' object has no attribute '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:
        # Mask sensitive keys in representation
        masked = mask_secrets(self._data)
        return f"AppConfig({masked})"


def load_config(
    config_file: Optional[Union[str, Path]] = None,
    env_file: Optional[Union[str, Path]] = None,
    env_prefix: str = "APP_"
) -> AppConfig:
    """
    Load configuration by merging config.ini, environment variables, and defaults.

    Order of precedence:
    1. Environment variables (highest priority, loaded via python-dotenv if .env exists)
    2. config.ini settings
    3. Default fallback values (lowest priority)

    Args:
        config_file: Path to config.ini. If None, searches standard locations.
        env_file: Optional path to .env file to load.
        env_prefix: Optional prefix for environment variables (e.g., APP_LOOKBACK_YEARS).

    Returns:
        AppConfig object with merged configuration.
    """
    # Load custom env_file if provided
    if env_file is not None and Path(env_file).is_file():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_file, override=True)
        except ImportError:
            pass

    # 1. Start with defaults
    cfg: Dict[str, Any] = dict(DEFAULT_CONFIG)

    # 2. Locate config file
    if config_file is None:
        candidates = [
            Path.cwd() / "config.ini",
            Path(__file__).resolve().parent.parent / "config.ini",
        ]
        for c in candidates:
            if c.is_file():
                config_file = c
                break

    # 3. Read config.ini if available
    if config_file and Path(config_file).is_file():
        parser = configparser.ConfigParser()
        parser.read(config_file)

        if parser.has_section('DATA_COLLECTION'):
            sec = parser['DATA_COLLECTION']
            if 'lookback_years' in sec:
                cfg['lookback_years'] = sec.getint('lookback_years')
            if 'tickers' in sec:
                cfg['tickers'] = _parse_list(sec.get('tickers'))

        if parser.has_section('EVENT_ANALYSIS'):
            sec = parser['EVENT_ANALYSIS']
            if 'event_window' in sec:
                cfg['event_window'] = sec.getint('event_window')
            if 'estimation_window' in sec:
                cfg['estimation_window'] = sec.getint('estimation_window')

        if parser.has_section('FEATURE_ENGINEERING'):
            sec = parser['FEATURE_ENGINEERING']
            if 'label_threshold' in sec:
                cfg['label_threshold'] = sec.getfloat('label_threshold')

        if parser.has_section('MACHINE_LEARNING'):
            sec = parser['MACHINE_LEARNING']
            if 'test_size' in sec:
                cfg['test_size'] = sec.getfloat('test_size')
            if 'random_state' in sec:
                cfg['random_state'] = sec.getint('random_state')
            if 'models_to_train' in sec:
                cfg['models_to_train'] = _parse_list(sec.get('models_to_train'))

        if parser.has_section('DIRECTORIES'):
            sec = parser['DIRECTORIES']
            for d in ['data_dir', 'models_dir', 'viz_dir', 'notebooks_dir']:
                if d in sec:
                    cfg[d] = sec.get(d)

        if parser.has_section('VISUALIZATION'):
            sec = parser['VISUALIZATION']
            if 'figure_size' in sec:
                fig_val = sec.get('figure_size')
                if fig_val:
                    dims = [int(x.strip()) for x in fig_val.split(',')]
                    if len(dims) == 2:
                        cfg['figure_size'] = tuple(dims)
            if 'dpi' in sec:
                cfg['dpi'] = sec.getint('dpi')
            if 'color_scheme' in sec:
                cfg['color_scheme'] = sec.get('color_scheme')

        if parser.has_section('API'):
            sec = parser['API']
            if 'host' in sec:
                cfg['api_host'] = sec.get('host')
            if 'port' in sec:
                cfg['api_port'] = sec.getint('port')
            if 'cors_origins' in sec:
                cfg['cors_origins'] = _parse_list(sec.get('cors_origins'))
            if 'api_secret_key' in sec:
                cfg['api_secret_key'] = sec.get('api_secret_key')

    # 4. Override with Environment Variables
    env_mappings: Dict[str, Tuple[str, Any]] = {
        'LOOKBACK_YEARS': ('lookback_years', int),
        'TICKERS': ('tickers', _parse_list),
        'EVENT_WINDOW': ('event_window', int),
        'ESTIMATION_WINDOW': ('estimation_window', int),
        'LABEL_THRESHOLD': ('label_threshold', float),
        'TEST_SIZE': ('test_size', float),
        'RANDOM_STATE': ('random_state', int),
        'MODELS_TO_TRAIN': ('models_to_train', _parse_list),
        'DATA_DIR': ('data_dir', str),
        'MODELS_DIR': ('models_dir', str),
        'VIZ_DIR': ('viz_dir', str),
        'API_HOST': ('api_host', str),
        'API_PORT': ('api_port', int),
        'CORS_ORIGINS': ('cors_origins', _parse_list),
        'API_SECRET_KEY': ('api_secret_key', str),
    }

    for env_key, (cfg_key, converter) in env_mappings.items():
        # Check both prefixed (e.g. APP_LOOKBACK_YEARS) and bare (LOOKBACK_YEARS)
        val = os.getenv(f"{env_prefix}{env_key}") or os.getenv(env_key)
        if val is not None:
            try:
                cfg[cfg_key] = converter(val)
            except Exception:
                pass

    return AppConfig(cfg)
