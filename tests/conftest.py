"""Shared fixtures for the test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _project_root_cwd(monkeypatch):
    """Ensure tests run from the project root so relative data paths resolve."""
    monkeypatch.chdir(PROJECT_ROOT)
