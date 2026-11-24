"""Regression tests for run_5fd02a57a0f937aa60: a native module blocked by Windows
application control surfaced only as `load_error: cannot import name
'NP_SUPPORTED_MODULES'` while preflight reported DEPENDENCIES_VERIFIED."""

from __future__ import annotations

import sys

from lab.core.e2 import airgap, appcontrol
from lab.core.e2 import runner as runner_module
from lab.core.e2.airgap import preflight_deployment
from lab.core.e2.runner import E2Runner
from lab.core.e2.specs import mock_spec
from tests.lab.test_airgap import _fixture
from tests.lab.test_airgap import _hardware as _airgap_hardware
from tests.lab.test_e2a import _hardware, _workspace_with_gt

# Verbatim shape of `wevtutil qe Microsoft-Windows-CodeIntegrity/Operational /f:text`.
WEVTUTIL_TEXT = r"""Event[0]
  Log Name: Microsoft-Windows-CodeIntegrity/Operational
  Source: Microsoft-Windows-CodeIntegrity
  Date: 2026-11-17T16:00:13.0690000Z
  Event ID: 3077
  Description:
Code Integrity determined that a process (\Device\HarddiskVolume3\Users\User\AppData\Local\Programs\Python\Python312\python.exe) attempted to load \Device\HarddiskVolume5\agenticdoc_e2\venv-paddleocr-vl\Lib\site-packages\numpy\random\mtrand.cp312-win_amd64.pyd that did not meet the Enterprise signing level requirements or violated code integrity policy (Policy ID:{0283ac0f-fff1-49ae-ada1-8a933130cad6}).

Event[1]
  Log Name: Microsoft-Windows-CodeIntegrity/Operational
  Date: 2026-11-17T15:57:56.3190000Z
  Event ID: 3077
  Description:
Code Integrity determined that a process (\Device\HarddiskVolume3\Users\User\AppData\Local\Programs\Python\Python312\python.exe) attempted to load \Device\HarddiskVolume5\agenticdoc_e2\venv-paddleocr-vl\Lib\site-packages\numpy\random\_mt19937.cp312-win_amd64.pyd that did not meet the Enterprise signing level requirements or violated code integrity policy (Policy ID:{0283ac0f-fff1-49ae-ada1-8a933130cad6}).

Event[2]
  Event ID: 3033
  Description:
Code Integrity determined that a process (\Device\HarddiskVolume3\x\python.exe) attempted to load \Device\HarddiskVolume5\agenticdoc_e2\venv-paddleocr-vl-fullpage\Lib\site-packages\ujson.cp312-win_amd64.pyd that did not meet the Enterprise signing level requirements.
"""


def test_code_integrity_events_are_parsed_per_environment() -> None:
    blocks = appcontrol.parse_wevtutil_text(WEVTUTIL_TEXT, r"D:\agenticdoc_e2\venv-paddleocr-vl")
    assert [item["file"] for item in blocks] == [
        r"Lib\site-packages\numpy\random\mtrand.cp312-win_amd64.pyd",
        r"Lib\site-packages\numpy\random\_mt19937.cp312-win_amd64.pyd",
    ]
    assert all(item["policy_id"] == "{0283ac0f-fff1-49ae-ada1-8a933130cad6}" for item in blocks)
    # The sibling "-fullpage" venv is a different environment, not a prefix match.
    fullpage = appcontrol.parse_wevtutil_text(
        WEVTUTIL_TEXT, r"D:\agenticdoc_e2\venv-paddleocr-vl-fullpage"
    )
    assert [item["file"] for item in fullpage] == [r"Lib\site-packages\ujson.cp312-win_amd64.pyd"]


def test_native_import_probe_reports_the_first_failing_module(monkeypatch) -> None:
    monkeypatch.setattr(appcontrol, "recent_blocks", lambda *_args, **_kwargs: [])
    ok = appcontrol.native_import_probe(["json", "csv"])
    assert ok["ok"] is True and [item["module"] for item in ok["results"]] == ["json", "csv"]
    bad = appcontrol.native_import_probe(["json", "agenticdoc_missing_module_xyz", "csv"])
    assert bad["ok"] is False
    assert bad["failed"]["module"] == "agenticdoc_missing_module_xyz"
    assert bad["failed"]["type"] == "ModuleNotFoundError"
    # Stops at the first failure, so a blocked module is attempted only once.
    assert [item["module"] for item in bad["results"]] == ["json", "agenticdoc_missing_module_xyz"]


def _probe_manifest(tmp_path):
    manifest = _fixture(tmp_path)
    lock = tmp_path / "requirements.lock"
    lock.write_text("pytest==8.4.2\n", encoding="utf-8")
    manifest["runtime_profiles"]["test"].update(
        {"lockfile": str(lock), "lock_complete": True, "runtime_import_probe": ["numpy.random"]}
    )
    return manifest


def test_preflight_fails_closed_when_native_modules_are_blocked(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(airgap, "_package_version", lambda _name: "8.4.2")
    blocked = {
        "ok": False,
        "failed": {
            "module": "numpy.random",
            "type": "ImportError",
            "message": "DLL load failed while importing mtrand: An Application Control policy has blocked this file.",
        },
        "process_error": None,
        "application_control_blocks": [
            {"file": r"Lib\site-packages\numpy\random\_mt19937.cp312-win_amd64.pyd", "policy_id": "{p}"}
        ],
    }
    monkeypatch.setattr(appcontrol, "native_import_probe", lambda modules, **_kw: blocked)
    report = preflight_deployment(_probe_manifest(tmp_path), "test", hardware=_airgap_hardware())
    # Before the fix this reported DEPENDENCIES_VERIFIED and inference then failed.
    assert report["status"] == "BLOCKED"
    assert any(error.startswith("native_import_failed: numpy.random") for error in report["errors"])
    assert any("application_control_blocked" in error for error in report["errors"])
    assert report["native_import_probe"] is blocked


def test_preflight_passes_when_the_probe_loads_and_skips_declared_blockers(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(airgap, "_package_version", lambda _name: "8.4.2")
    calls: list[list[str]] = []

    def fine(modules, **_kwargs):
        calls.append(modules)
        return {"ok": True, "failed": None, "process_error": None, "application_control_blocks": []}

    monkeypatch.setattr(appcontrol, "native_import_probe", fine)
    manifest = _probe_manifest(tmp_path)
    assert preflight_deployment(manifest, "test", hardware=_airgap_hardware())["status"] == "DEPENDENCIES_VERIFIED"
    manifest["deployments"][0]["deployment_status"] = "BLOCKED"
    assert preflight_deployment(manifest, "test", hardware=_airgap_hardware())["status"] == "BLOCKED"
    assert calls == [["numpy.random"]]  # never re-triggers a known-blocked deployment


def test_failed_run_records_traceback_and_application_control_evidence(
    settings, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    block = {"file": r"Lib\site-packages\numpy\random\_mt19937.cp312-win_amd64.pyd", "policy_id": "{p}"}
    monkeypatch.setattr(runner_module.appcontrol, "recent_blocks", lambda *_a, **_k: [block])
    manager, workspace_id, document, source, _gt = _workspace_with_gt(settings)
    run = E2Runner(manager, settings, hardware=_hardware()).run(
        workspace_id, document, source, mock_spec(fail=True), [1], allow_mock=True
    )
    errors = run.environment["e2"]["errors"]
    page_error = next(item for item in errors if item["code"] == "backend_error")
    assert "Traceback" in page_error["traceback"] and "Mock backend configured to fail" in page_error["traceback"]
    evidence = next(item for item in errors if item["code"] == "application_control_blocked")
    assert evidence["application_control_blocks"] == [block]
    assert "_mt19937" in evidence["message"]
