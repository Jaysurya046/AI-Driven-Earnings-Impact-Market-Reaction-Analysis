"""
Unit tests for repository hygiene, CI/CD pipelines, lockfiles, and documentation integrity
"""

from pathlib import Path
import yaml
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_ci_workflow_validity():
    """Verify GitHub Actions CI workflow exists and is properly structured"""
    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_path.exists(), ".github/workflows/ci.yml must exist"

    with open(ci_path, "r", encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    assert "name" in workflow
    assert "on" in workflow or True in workflow
    assert "jobs" in workflow
    assert "test" in workflow["jobs"]

    test_job = workflow["jobs"]["test"]
    assert "runs-on" in test_job
    assert "steps" in test_job

    step_uses_or_runs = [
        step.get("uses", "") + step.get("run", "")
        for step in test_job["steps"]
    ]

    # Verify key steps are present in workflow
    assert any("actions/checkout" in s for s in step_uses_or_runs), "CI must check out repository"
    assert any("actions/setup-python" in s for s in step_uses_or_runs), "CI must setup python"
    assert any("pytest" in s for s in step_uses_or_runs), "CI must execute pytest"


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
