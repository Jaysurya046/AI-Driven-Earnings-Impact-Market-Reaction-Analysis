"""
Unit tests for standardized packaging and exports
"""

import sys
from pathlib import Path
import pytest

import src

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def test_src_package_metadata():
    """Verify src is a proper package with version and author metadata"""
    assert hasattr(src, "__version__")
    assert isinstance(src.__version__, str)
    assert len(src.__version__) > 0
    assert hasattr(src, "__file__")
    assert Path(src.__file__).name == "__init__.py"


def test_src_top_level_exports():
    """Verify all primary domain classes and utilities are exposed at the top package level"""
    expected_exports = [
        "DataCollector",
        "validate_collected_data",
        "EventAnalyzer",
        "FeatureEngineer",
        "SAFE_FEATURE_COLUMNS",
        "NON_FEATURE_COLUMNS",
        "LEAKY_COLUMNS",
        "METADATA_COLUMNS",
        "ModelTrainer",
        "Backtester",
        "Visualizer",
        "SectorAnalyzer",
        "SentimentAnalyzer",
        "AppConfig",
        "load_config",
        "mask_secrets",
    ]
    for symbol in expected_exports:
        assert hasattr(src, symbol), f"Expected {symbol} in src exports"
        assert symbol in src.__all__, f"Expected {symbol} in src.__all__"


def test_pyproject_toml_structure_and_metadata():
    """Verify pyproject.toml conforms to PEP 517/621 requirements"""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml must exist in repo root"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    # Build system
    assert "build-system" in data
    assert "requires" in data["build-system"]
    assert "build-backend" in data["build-system"]
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"

    # Project metadata
    assert "project" in data
    project = data["project"]
    assert project["name"] == "earnings-impact-analysis"
    assert project["version"] == src.__version__
    assert "dependencies" in project
    assert len(project["dependencies"]) >= 10

    # Setuptools discovery & pytest configuration
    assert "tool" in data
    assert "setuptools" in data["tool"]
    assert "packages" in data["tool"]["setuptools"]
    assert "pytest" in data["tool"]
    assert "ini_options" in data["tool"]["pytest"]
