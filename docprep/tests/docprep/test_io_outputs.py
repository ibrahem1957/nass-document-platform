"""Inputs (read-only, error handling), write-once outputs, manifest and CLI."""

from __future__ import annotations

import json
import os

import pytest

from docprep.config import ConfigError, load_config
from docprep.inputs import InputError, hash_inputs, load_run
from docprep.output import OutputExistsError
from docprep.structure import main, run

from .helpers import el, write_run


def simple_run(tmp_path, name="run_simple"):
    elements = [el("h", 1, 0, "Overview", type="heading", bbox=[72, 80, 200, 96]),
                el("b", 1, 1, "This synthetic paragraph explains the overview in a few words.", bbox=[72, 100, 540, 114])]
    return write_run(tmp_path, name, elements)


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): (p.stat().st_size, p.read_bytes()) for p in folder.rglob("*") if p.is_file()}


def test_invalid_json_is_rejected(tmp_path):
    run_dir = simple_run(tmp_path)
    (run_dir / "document.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(InputError, match="not valid JSON"):
        load_run(run_dir)


def test_invalid_optional_json_is_rejected(tmp_path):
    run_dir = simple_run(tmp_path)
    (run_dir / "validation_report.json").write_text("[1, 2", encoding="utf-8")
    with pytest.raises(InputError, match="validation_report.json"):
        load_run(run_dir)


@pytest.mark.parametrize("mutate,message", [
    (lambda d: d.pop("elements"), "'elements' must be a list"),
    (lambda d: d["elements"][0].pop("page"), "has no 'page'"),
    (lambda d: d["elements"].append(dict(d["elements"][0])), "duplicate element ids"),
    (lambda d: d["elements"][0].update(page=0), "invalid page number"),
])
def test_incomplete_documents_are_rejected(tmp_path, mutate, message):
    run_dir = simple_run(tmp_path)
    document = json.loads((run_dir / "document.json").read_text(encoding="utf-8"))
    mutate(document)
    (run_dir / "document.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(InputError, match=message):
        load_run(run_dir)


def test_missing_document_or_folder(tmp_path):
    with pytest.raises(InputError, match="not found"):
        load_run(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(InputError, match="document.json is missing"):
        load_run(tmp_path / "empty")


def test_inputs_are_not_modified(tmp_path):
    run_dir = simple_run(tmp_path)
    before, hashes = snapshot(run_dir), hash_inputs(run_dir)
    target, findings = run(run_dir, out_root=tmp_path / "out", stamp="20260101T000000Z")
    assert snapshot(run_dir) == before and hash_inputs(run_dir) == hashes
    assert next(f for f in findings if f["code"] == "inputs_unchanged")["severity"] == "ok"
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs_unchanged"] is True and manifest["input_hashes"]["document.json"] == hashes["document.json"]
    assert not (run_dir / "section_tree.json").exists()  # nothing written into the run folder


def test_outputs_are_write_once_and_read_only(tmp_path):
    run_dir = simple_run(tmp_path)
    target, _ = run(run_dir, out_root=tmp_path / "out", stamp="20260101T000000Z")
    assert target.name == "run_simple__20260101T000000Z"
    assert sorted(p.name for p in target.iterdir()) == ["assignments.json", "manifest.json", "section_tree.json",
                                                         "structure_report.html", "structure_report.md"]
    assert all(not os.access(p, os.W_OK) for p in target.iterdir())
    with pytest.raises(OutputExistsError):
        run(run_dir, out_root=tmp_path / "out", stamp="20260101T000000Z")
    assert not [p for p in (tmp_path / "out").iterdir() if p.name.startswith(".")]  # no partial folder left
    second, _ = run(run_dir, out_root=tmp_path / "out", stamp="20260101T000001Z")
    assert second != target and second.is_dir()


def test_manifest_records_hashes_configuration_and_thresholds(tmp_path):
    target, _ = run(simple_run(tmp_path), out_root=tmp_path / "out", stamp="20260101T000000Z")
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    import hashlib
    for name, digest in manifest["output_hashes"].items():
        assert hashlib.sha256((target / name).read_bytes()).hexdigest() == digest
    assert manifest["detection_thresholds"]["promote"] == manifest["configuration"]["thresholds"]["promote"]
    assert manifest["source_run_id"] == "run_simple" and manifest["docprep_version"]


def test_html_report_is_self_contained(tmp_path):
    target, _ = run(simple_run(tmp_path), out_root=tmp_path / "out", stamp="20260101T000000Z")
    page = (target / "structure_report.html").read_text(encoding="utf-8")
    assert "<script src" not in page and "<link" not in page and "http://" not in page and "https://" not in page
    assert "f-promoted" in page and "f-unassigned" in page and "f-page" in page and "f-route" in page
    assert "Overview" in page


def test_cli_exit_codes(tmp_path, capsys):
    run_dir = simple_run(tmp_path)
    assert main([str(run_dir), "--out", str(tmp_path / "out"), "--no-semantic"]) == 0
    assert "written:" in capsys.readouterr().out
    assert main([str(tmp_path / "missing"), "--out", str(tmp_path / "out")]) == 2
    bad = tmp_path / "bad.toml"
    bad.write_text("[thresholds]\nunknown_key = 1\n", encoding="utf-8")
    assert main([str(run_dir), "--out", str(tmp_path / "out"), "--config", str(bad)]) == 2


def test_config_validation(tmp_path):
    with pytest.raises(ConfigError, match="unknown configuration key"):
        load_config(None, {"weights": {"nope": 1}}, use_default_file=False)
    with pytest.raises(ConfigError, match="thresholds must satisfy"):
        load_config(None, {"thresholds": {"promote": 0.9}}, use_default_file=False)
    cfg = load_config(None, {"thresholds": {"promote": 0.5}}, use_default_file=False)
    assert cfg.thresholds.promote == 0.5


def test_project_config_file_matches_code_defaults():
    from docprep.config import DEFAULT_CONFIG_FILE

    assert DEFAULT_CONFIG_FILE.is_file()
    assert load_config(None).to_dict() == load_config(None, use_default_file=False).to_dict()


def test_output_root_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCPREP_OUTPUT_ROOT", str(tmp_path / "env_out"))
    assert load_config(None, use_default_file=False).output.root == str(tmp_path / "env_out")
