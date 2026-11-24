from __future__ import annotations

import pytest

from docprep.semantic import HeuristicVerifier

from .helpers import make_config


@pytest.fixture
def cfg():
    return make_config()


@pytest.fixture
def verifier():
    return HeuristicVerifier()


@pytest.fixture(autouse=True)
def _no_env_output_root(monkeypatch):
    monkeypatch.delenv("DOCPREP_OUTPUT_ROOT", raising=False)
