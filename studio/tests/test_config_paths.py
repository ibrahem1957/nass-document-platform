"""Monorepo configuration: relative paths resolve against the Studio project folder, and "current" selects
the interpreter running the Studio (one shared virtual environment)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from studio.config import ConfigError, load_config


def _write(project: Path, **overrides) -> Path:
    (project / "config").mkdir(parents=True)
    value = {"schema_version": 1, "kind": "document_processing_studio_config", "jobs_root": "data/jobs",
             "nass_root": "../nass", "nass_python": "current", "docprep_root": "../docprep",
             "docprep_python": "current", "existing_documents": []}
    value.update(overrides)
    path = project / "config" / "studio.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_relative_paths_resolve_against_the_studio_project_not_the_working_directory(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "nass").mkdir(parents=True)
    (repo / "docprep").mkdir()
    config_path = _write(repo / "studio")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_config(config_path)
    assert config.nass_root == (repo / "nass").resolve()
    assert config.docprep_root == (repo / "docprep").resolve()
    assert config.jobs_root == (repo / "studio" / "data" / "jobs").resolve()


def test_current_python_is_the_running_interpreter(tmp_path):
    repo = tmp_path / "repo"
    (repo / "nass").mkdir(parents=True)
    (repo / "docprep").mkdir()
    config = load_config(_write(repo / "studio"))
    assert config.nass_python == Path(sys.executable).resolve()
    assert config.docprep_python == Path(sys.executable).resolve()


def test_absolute_paths_still_work_and_missing_paths_are_refused(tmp_path):
    repo = tmp_path / "repo"
    nass = tmp_path / "absolute_nass"
    nass.mkdir()
    (repo / "docprep").mkdir(parents=True)
    config = load_config(_write(repo / "studio", nass_root=str(nass), nass_python=sys.executable))
    assert config.nass_root == nass.resolve()
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(_write(tmp_path / "other" / "studio", nass_root="../missing"))


def test_shipped_config_is_portable():
    shipped = Path(__file__).resolve().parents[1] / "config" / "studio.json"
    value = json.loads(shipped.read_text(encoding="utf-8"))
    for key in ("nass_root", "docprep_root", "jobs_root"):
        assert not os.path.isabs(value[key]), key
    assert value["nass_python"] == value["docprep_python"] == "current"
    assert value["existing_documents"] == []
