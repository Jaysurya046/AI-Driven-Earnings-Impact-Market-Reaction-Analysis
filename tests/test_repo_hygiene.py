"""
Unit tests for repository hygiene, CI/CD pipelines, lockfiles, and documentation integrity
"""

from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_ci_workflow_validity():
    """Verify GitHub Actions CI workflow exists and is properly structured"""
    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_path.exists(), ".github/workflows/ci.yml must exist"

    workflow_text = ci_path.read_text(encoding="utf-8")
    assert "name: CI" in workflow_text
    assert "\"on\":" in workflow_text
    assert "jobs:" in workflow_text
    assert "test:" in workflow_text
    assert "runs-on:" in workflow_text
    assert "steps:" in workflow_text

    # Verify key steps are present in workflow
    assert "actions/checkout" in workflow_text, "CI must check out repository"
    assert "actions/setup-python" in workflow_text, "CI must setup python"
    assert "pytest" in workflow_text, "CI must execute pytest"


def test_requirements_lock_file():
    """Verify requirements-lock.txt exists and contains strictly pinned versions"""
    lock_path = REPO_ROOT / "requirements-lock.txt"
    assert lock_path.exists(), "requirements-lock.txt must exist"

    lines = [
        line.strip()
        for line in lock_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(lines) >= 15, "Lockfile should contain all core dependencies"

    # All non-comment lines must be pinned with '=='
    for line in lines:
        assert "==" in line, f"Dependency line '{line}' in requirements-lock.txt must be pinned with '=='"


def test_requirements_txt_includes_test_and_api_deps():
    """Verify requirements.txt covers all essential packages for test & api execution"""
    req_path = REPO_ROOT / "requirements.txt"
    assert req_path.exists(), "requirements.txt must exist"

    req_content = req_path.read_text(encoding="utf-8")
    for expected_pkg in ["pytest", "httpx", "fastapi", "pandas", "scikit-learn", "xgboost"]:
        assert expected_pkg in req_content, f"requirements.txt must include {expected_pkg}"


def test_documentation_leakage_free():
    """Verify documentation does not advertise leaky post-event features as input features"""
    prohibited_features = ["PostEventMomentum: Post-event return", "PreVol_PostReturn: PreVol"]
    arch_path = REPO_ROOT / "ARCHITECTURE.md"
    assert arch_path.exists()

    arch_text = arch_path.read_text(encoding="utf-8")
    for prohibited in prohibited_features:
        assert prohibited not in arch_text, f"ARCHITECTURE.md contains leaky feature reference: {prohibited}"


def test_mit_license_exists_and_valid():
    """Verify that a proper MIT License file exists in the repo root"""
    license_path = REPO_ROOT / "LICENSE"
    assert license_path.exists(), "LICENSE file must exist in repo root"

    license_text = license_path.read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Copyright (c)" in license_text
    assert "Permission is hereby granted, free of charge" in license_text
    assert "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT" in license_text

