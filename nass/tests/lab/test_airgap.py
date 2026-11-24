from __future__ import annotations

import hashlib
import json
import platform
import socket
from pathlib import Path

import pytest

from lab.core.e2 import airgap
from lab.core.e2.airgap import (
    ManifestError,
    OutboundNetworkBlocked,
    load_manifest,
    outbound_network_guard,
    preflight_deployment,
    validate_manifest,
    validate_registry_alignment,
)
from lab.core.e2.hardware import HardwareProfile
from lab.core.e2.specs import load_model_specs


def _hardware(*, ram: float = 64.0, vram: float = 24.0) -> HardwareProfile:
    return HardwareProfile(
        cpu="test",
        logical_cpus=8,
        ram_total_gb=ram,
        ram_available_gb=ram,
        gpus=[{"name": "test", "vram_gb": vram}],
        cuda_available=True,
    )


def _fixture(tmp_path: Path, roles: tuple[str, ...] = ("weights",)) -> dict:
    revision = "a" * 40
    root = tmp_path / revision
    root.mkdir()
    artifacts = []
    for role in roles:
        path = root / f"{role}.bin"
        path.write_bytes(role.encode())
        artifacts.append(
            {
                "filename": path.name,
                "role": role,
                "required": True,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "checksum_status": "HASH_VERIFIED",
                "source_url": f"https://example.invalid/resolve/{revision}/{path.name}",
                "local_relative_path": path.name,
                "cluster_relative_path": path.name,
                "status": "HASH_VERIFIED",
            }
        )
    flags = {
        "artifacts_present": True,
        "hashes_verified": True,
        "dependencies_verified": True,
        "successfully_loaded": True,
        "network_isolated_inference_tested": False,
        "compatible_target_tested": False,
    }
    return {
        "schema_version": 1,
        "manifest_id": "test",
        "policy": {},
        "status_definitions": {name: name for name in airgap.DEPLOYMENT_STATUSES},
        "runtime_profiles": {
            "test": {
                "profile_id": "test",
                "python": platform.python_version(),
                "packages": {},
                "native_dependencies": [],
            }
        },
        "components": [
            {
                "component_id": "model",
                "name": "model",
                "repo_id": "org/model",
                "revision": revision,
                "source_repository": "https://example.invalid/org/model",
                "architecture": "test",
                "license": {"spdx": "Apache-2.0"},
                "local_root": str(root),
                "cluster_root": f"/models/{revision}",
                "artifacts": artifacts,
                "declared_status": "HASH_VERIFIED",
            }
        ],
        "deployments": [
            {
                "deployment_id": "test",
                "model_name": "test",
                "variant": "test",
                "pipeline_mode": "test",
                "primary_component": "model",
                "components": ["model"],
                "runtime_profile": "test",
                "inference_framework": "test",
                "quantization": "none",
                "hardware": {"min_ram_gb": 1.0, "min_vram_gb": 1.0, "cpu_feasible": False},
                "os_arch_constraints": {
                    "allowed_os": [platform.system()],
                    "allowed_arch": [platform.machine()],
                },
                "offline_startup": [],
                "offline_validation": [],
                "verification": flags,
                "deployment_status": "HASH_VERIFIED",
                "missing_or_unverified": [],
            }
        ],
    }


def _report(manifest: dict) -> dict:
    return preflight_deployment(manifest, "test", hardware=_hardware())


@pytest.mark.parametrize("role", ["weights", "tokenizer", "processor"])
def test_preflight_detects_missing_required_artifact(tmp_path: Path, role: str) -> None:
    manifest = _fixture(tmp_path, (role,))
    Path(manifest["components"][0]["local_root"], f"{role}.bin").unlink()
    report = _report(manifest)
    assert report["status"] == "NOT_DOWNLOADED"
    assert any(f"missing_{role}" in error for error in report["errors"])


def test_preflight_detects_corrupt_artifact(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    Path(manifest["components"][0]["local_root"], "weights.bin").write_bytes(b"broken")
    report = _report(manifest)
    assert report["status"] == "BLOCKED"
    assert any("size_mismatch" in error or "hash_mismatch" in error for error in report["errors"])


def test_preflight_detects_incorrect_revision_path(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    component = manifest["components"][0]
    wrong = tmp_path / "wrong-revision"
    Path(component["local_root"]).rename(wrong)
    component["local_root"] = str(wrong)
    report = _report(manifest)
    assert report["status"] == "BLOCKED"
    assert any("revision_path_mismatch" in error for error in report["errors"])


@pytest.mark.parametrize("role", ["base_weights", "layout_weights"])
def test_preflight_detects_missing_linked_component_asset(tmp_path: Path, role: str) -> None:
    manifest = _fixture(tmp_path, (role,))
    Path(manifest["components"][0]["local_root"], f"{role}.bin").unlink()
    report = _report(manifest)
    assert any(f"missing_{role}" in error for error in report["errors"])


def test_preflight_detects_dependency_version_mismatch(tmp_path: Path, monkeypatch) -> None:
    manifest = _fixture(tmp_path)
    manifest["runtime_profiles"]["test"]["packages"] = {"example": "1.0"}
    monkeypatch.setattr(airgap, "_package_version", lambda _name: "2.0")
    report = _report(manifest)
    assert report["status"] == "HASH_VERIFIED"
    assert any("dependency_version_mismatch" in error for error in report["errors"])


def test_unverified_dependency_cannot_be_promoted(tmp_path: Path, monkeypatch) -> None:
    manifest = _fixture(tmp_path)
    manifest["runtime_profiles"]["test"]["packages"] = {"example": "UNVERIFIED"}
    monkeypatch.setattr(airgap, "_package_version", lambda _name: "2.0")
    report = _report(manifest)
    assert report["status"] == "HASH_VERIFIED"
    assert any("dependency_unverified" in warning for warning in report["warnings"])


def test_preflight_detects_unsupported_hardware(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    report = preflight_deployment(manifest, "test", hardware=_hardware(ram=0.5, vram=0.0))
    assert report["status"] == "BLOCKED"
    assert any("unsupported_hardware" in error for error in report["errors"])


def test_outbound_network_attempt_is_blocked() -> None:
    with outbound_network_guard(), pytest.raises(OutboundNetworkBlocked):
        socket.create_connection(("example.com", 443), timeout=0.01)


def test_incomplete_manifest_is_rejected(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    del manifest["components"][0]["revision"]
    with pytest.raises(ManifestError, match="missing fields"):
        validate_manifest(manifest)


def test_manifest_rejects_artifact_path_traversal(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    manifest["components"][0]["artifacts"][0]["local_relative_path"] = "../escape.bin"
    with pytest.raises(ManifestError, match="unsafe relative path"):
        validate_manifest(manifest)


def test_ready_status_requires_file_backed_evidence_not_flags(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    manifest["deployments"][0]["deployment_status"] = "READY_FOR_AIRGAP"
    with pytest.raises(ManifestError, match="file-backed evidence"):
        validate_manifest(manifest)


def test_preflight_detects_unexpected_executable(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    root = Path(manifest["components"][0]["local_root"])
    (root / "surprise.exe").write_bytes(b"MZ")
    report = _report(manifest)
    assert report["status"] == "BLOCKED"
    assert report["components"][0]["unexpected_executable_files"] == [
        str(root / "surprise.exe")
    ]


def test_lockfile_is_read_and_every_installed_version_is_checked(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = _fixture(tmp_path)
    lock = tmp_path / "requirements.lock"
    lock.write_text("alpha==1.0\nbeta==2.0\n", encoding="utf-8")
    profile = manifest["runtime_profiles"]["test"]
    profile.update({"lockfile": str(lock), "lock_complete": True})
    versions = {"alpha": "1.0", "beta": "9.0"}
    monkeypatch.setattr(airgap, "_package_version", versions.get)
    report = _report(manifest)
    assert report["runtime"]["locked_requirement_count"] == 2
    assert any("beta expected 2.0, got 9.0" in error for error in report["errors"])
    assert report["runtime"]["wheel_hashes_verified"] is False


def test_additional_lockfiles_are_merged_and_conflicts_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = _fixture(tmp_path)
    base_lock = tmp_path / "base.lock"
    base_lock.write_text("alpha==1.0\n", encoding="utf-8")
    ui_lock = tmp_path / "ui.lock"
    ui_lock.write_text("beta==2.0 --hash=sha256:" + "0" * 64 + "\n", encoding="utf-8")
    profile = manifest["runtime_profiles"]["test"]
    profile.update(
        {"lockfile": str(base_lock), "additional_lockfiles": [str(ui_lock)], "lock_complete": True}
    )
    monkeypatch.setattr(airgap, "_package_version", {"alpha": "1.0", "beta": "2.0"}.get)
    report = _report(manifest)
    assert report["runtime"]["locked_requirement_count"] == 2
    assert report["runtime"]["ok"] is True
    ui_lock.write_text("alpha==9.9\n", encoding="utf-8")
    report = _report(manifest)
    assert any(error.startswith("conflicting_lock_entries: alpha") for error in report["errors"])
    assert report["status"] == "HASH_VERIFIED"


def test_manual_verification_flags_cannot_promote_without_evidence(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    manifest["deployments"][0]["verification"] = {
        name: True for name in manifest["deployments"][0]["verification"]
    }
    lock = tmp_path / "requirements.lock"
    lock.write_text("pytest==8.4.2\n", encoding="utf-8")
    manifest["runtime_profiles"]["test"].update(
        {"lockfile": str(lock), "lock_complete": True}
    )
    report = _report(manifest)
    assert report["status"] == "DEPENDENCIES_VERIFIED"
    assert report["ready_for_airgap"] is False
    assert not report["evidence"]["offline_inference"]["ok"]


@pytest.mark.parametrize("operation", ["connect_ex", "sendto", "dns"])
def test_all_additional_outbound_socket_paths_are_blocked(operation: str) -> None:
    with outbound_network_guard(), pytest.raises(OutboundNetworkBlocked):
        if operation == "connect_ex":
            socket.socket().connect_ex(("8.8.8.8", 53))
        elif operation == "sendto":
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("8.8.8.8", 53))
        else:
            socket.getaddrinfo("example.com", 443)


@pytest.mark.parametrize(
    "call",
    [
        lambda: socket.gethostbyname("example.com"),
        lambda: socket.gethostbyname_ex("example.com"),
        lambda: socket.gethostbyaddr("8.8.8.8"),
    ],
    ids=["gethostbyname", "gethostbyname_ex", "gethostbyaddr"],
)
def test_legacy_name_resolution_paths_are_blocked(call) -> None:
    with outbound_network_guard(), pytest.raises(OutboundNetworkBlocked):
        call()


def test_guard_allows_loopback_and_restores_every_original() -> None:
    originals = {name: getattr(owner, name) for owner, name in airgap._GUARDED_ATTRIBUTES}
    with outbound_network_guard():
        assert airgap.guard_installed()
        assert socket.gethostbyname("localhost")
        assert socket.getaddrinfo("127.0.0.1", 80)
    assert not airgap.guard_installed()
    for owner, name in airgap._GUARDED_ATTRIBUTES:
        assert getattr(owner, name) is originals[name]


def _write_json(path: Path, payload: dict) -> dict:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _evidence_manifest(tmp_path: Path, *, target: dict | None, run_adapter: str = "e2.test") -> dict:
    """A fixture whose artifacts, lock and evidence are all genuinely file-backed."""
    from datetime import datetime, timezone

    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = _fixture(tmp_path)
    lock = tmp_path / "requirements.lock"
    lock.write_text("pytest==8.4.2\n", encoding="utf-8")
    manifest["runtime_profiles"]["test"].update({"lockfile": str(lock), "lock_complete": True})
    now = datetime.now(timezone.utc).isoformat()
    preflight = _write_json(
        tmp_path / "preflight.json",
        {"deployment_id": "test", "manifest_id": "test", "status": "DEPENDENCIES_VERIFIED"},
    )
    run_record = _write_json(
        tmp_path / "run.json", {"adapter": {"adapter_id": run_adapter}, "status": "ok"}
    )
    common = {"deployment_id": "test", "manifest_id": "test", "status": "passed", "observed_at": now}
    manifest["deployments"][0]["evidence"] = {
        "offline_inference": _write_json(
            tmp_path / "offline.json",
            common
            | {
                "successful_inference": True,
                "raw_output_preserved": True,
                "preflight_report": preflight,
                "run_record": run_record,
            },
        ),
        "network_isolation": _write_json(
            tmp_path / "isolation.json",
            common
            | {
                "enforcement_scope": "container_network_none",
                "observed_outbound_attempts_blocked": True,
            },
        ),
        "target_compatibility": _write_json(
            tmp_path / "target.json",
            common
            | {
                "compatible": True,
                "os": platform.system(),
                "architecture": platform.machine(),
                "accelerator": "cpu",
                "runtime_identity_sha256": "f" * 64,
            },
        ),
    }
    if target is not None:
        manifest["target_cluster_requirements"] = {"status": "VERIFIED", "target": target}
    return manifest


def test_laptop_evidence_cannot_satisfy_an_unverified_or_different_target(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(airgap, "_package_version", lambda _name: "8.4.2")
    unverified = _report(_evidence_manifest(tmp_path / "a", target=None))
    assert unverified["status"] == "OFFLINE_TESTED"
    assert "target_specification_unverified" in unverified["evidence"]["target_compatibility"]["errors"]
    linux_gpu = {"os": "Linux", "architecture": "x86_64", "accelerator": "cuda"}
    cluster = _report(_evidence_manifest(tmp_path / "b", target=linux_gpu))
    assert cluster["status"] == "OFFLINE_TESTED" and cluster["ready_for_airgap"] is False
    errors = cluster["evidence"]["target_compatibility"]["errors"]
    assert any(error.startswith("target_os_mismatch") for error in errors)
    assert any(error.startswith("target_accelerator_mismatch") for error in errors)


def test_matching_verified_evidence_is_the_only_path_to_ready(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(airgap, "_package_version", lambda _name: "8.4.2")
    target = {
        "os": platform.system(),
        "architecture": platform.machine(),
        "accelerator": "cpu",
        "runtime_identity_sha256": "f" * 64,
    }
    report = _report(_evidence_manifest(tmp_path, target=target))
    assert report["status"] == "READY_FOR_AIRGAP", report["evidence"]


def test_offline_evidence_must_link_a_run_of_the_same_deployment(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(airgap, "_package_version", lambda _name: "8.4.2")
    target = {"os": platform.system(), "architecture": platform.machine()}
    report = _report(_evidence_manifest(tmp_path, target=target, run_adapter="e2.other-model"))
    assert report["status"] == "DEPENDENCIES_VERIFIED"
    assert "run_record_deployment_mismatch" in report["evidence"]["offline_inference"]["errors"]


def test_real_manifest_is_complete_and_registry_aligned() -> None:
    manifest = load_manifest()
    assert validate_registry_alignment(load_model_specs(), manifest) == []
    assert {item["deployment_id"] for item in manifest["deployments"]} == {
        spec.id for spec in load_model_specs()
    }
    assert not any(
        item["deployment_status"] == "READY_FOR_AIRGAP"
        for item in manifest["deployments"]
    )


def test_declared_blocker_cannot_be_promoted_by_empty_artifact_list(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    manifest["components"][0]["artifacts"] = []
    manifest["components"][0]["declared_status"] = "BLOCKED"
    manifest["deployments"][0]["deployment_status"] = "BLOCKED"
    assert _report(manifest)["status"] == "BLOCKED"


def test_manifest_json_round_trips() -> None:
    manifest = load_manifest()
    assert json.loads(json.dumps(manifest))["manifest_id"] == manifest["manifest_id"]
