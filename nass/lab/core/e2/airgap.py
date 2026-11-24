"""Air-gap manifest validation, artifact hashing, and outbound-network blocking.

The manifest is evidence, not a downloader. This module never fetches a model,
package, container, or metadata. Missing checksums stay explicitly unverified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import ipaddress
import json
import os
import platform
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from lab.core.e2.hardware import HardwareProfile, probe_hardware
from lab.core.settings import LAB_DIR

DEFAULT_MANIFEST = LAB_DIR.parent / "docs" / "local" / "AIRGAP_DEPLOYMENT_MANIFEST.json"
DEPLOYMENT_STATUSES = {
    "NOT_DOWNLOADED",
    "DOWNLOADED",
    "HASH_VERIFIED",
    "DEPENDENCIES_VERIFIED",
    "OFFLINE_TESTED",
    "READY_FOR_AIRGAP",
    "BLOCKED",
}
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    "PADDLE_PDX_MODEL_SOURCE": "huggingface",
    # Any PaddleX hub fallback (a sub-model without a local directory) is pointed at a
    # closed loopback port, so it fails locally instead of resolving a public host.
    "PADDLE_PDX_HUGGING_FACE_ENDPOINT": "http://127.0.0.1:9",
}


class ManifestError(ValueError):
    pass


class OutboundNetworkBlocked(OSError):
    pass


def _require(mapping: dict, fields: set[str], location: str) -> None:
    missing = sorted(field for field in fields if field not in mapping)
    if missing:
        raise ManifestError(f"{location}: missing fields {missing}")


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    source = Path(path or DEFAULT_MANIFEST)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"Cannot read manifest {source}: {exc}") from exc
    validate_manifest(payload)
    payload["_manifest_path"] = str(source.resolve())
    payload["_manifest_sha256"] = _sha256(source)
    return payload


def validate_manifest(payload: dict[str, Any]) -> None:
    _require(
        payload,
        {
            "schema_version",
            "manifest_id",
            "policy",
            "status_definitions",
            "runtime_profiles",
            "components",
            "deployments",
        },
        "manifest",
    )
    if payload["schema_version"] != 1:
        raise ManifestError("manifest: unsupported schema_version")
    if set(payload["status_definitions"]) != DEPLOYMENT_STATUSES:
        raise ManifestError("manifest: status_definitions must contain every required status")
    component_ids: set[str] = set()
    for index, component in enumerate(payload["components"]):
        location = f"components[{index}]"
        _require(
            component,
            {
                "component_id",
                "name",
                "repo_id",
                "revision",
                "source_repository",
                "architecture",
                "license",
                "local_root",
                "cluster_root",
                "artifacts",
                "declared_status",
            },
            location,
        )
        component_id = component["component_id"]
        if component_id in component_ids:
            raise ManifestError(f"{location}: duplicate component_id {component_id}")
        component_ids.add(component_id)
        revision = str(component["revision"])
        if revision.lower() in {"main", "master", "latest"} or not revision:
            raise ManifestError(f"{location}: floating or empty revision {revision!r}")
        if component["declared_status"] not in DEPLOYMENT_STATUSES:
            raise ManifestError(f"{location}: invalid declared_status")
        filenames: set[str] = set()
        for artifact_index, artifact in enumerate(component["artifacts"]):
            artifact_location = f"{location}.artifacts[{artifact_index}]"
            _require(
                artifact,
                {
                    "filename",
                    "role",
                    "required",
                    "size_bytes",
                    "sha256",
                    "checksum_status",
                    "source_url",
                    "local_relative_path",
                    "cluster_relative_path",
                    "status",
                },
                artifact_location,
            )
            filename = artifact["filename"]
            if filename in filenames:
                raise ManifestError(f"{artifact_location}: duplicate filename {filename}")
            filenames.add(filename)
            checksum = artifact["sha256"]
            if checksum is not None and (
                len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum)
            ):
                raise ManifestError(f"{artifact_location}: invalid SHA-256")
            if artifact["required"] and artifact["size_bytes"] is None:
                raise ManifestError(f"{artifact_location}: required artifact needs an exact size")
            for path_field in ("local_relative_path", "cluster_relative_path"):
                relative = Path(str(artifact[path_field]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ManifestError(
                        f"{artifact_location}: unsafe relative path in {path_field}"
                    )
            if revision not in artifact["source_url"] and artifact["source_url"] != "UNVERIFIED":
                raise ManifestError(f"{artifact_location}: source URL is not revision-pinned")
    deployment_ids: set[str] = set()
    runtime_profiles = set(payload["runtime_profiles"])
    for index, deployment in enumerate(payload["deployments"]):
        location = f"deployments[{index}]"
        _require(
            deployment,
            {
                "deployment_id",
                "model_name",
                "variant",
                "pipeline_mode",
                "components",
                "runtime_profile",
                "inference_framework",
                "quantization",
                "hardware",
                "os_arch_constraints",
                "offline_startup",
                "offline_validation",
                "verification",
                "deployment_status",
                "missing_or_unverified",
            },
            location,
        )
        deployment_id = deployment["deployment_id"]
        if deployment_id in deployment_ids:
            raise ManifestError(f"{location}: duplicate deployment_id {deployment_id}")
        deployment_ids.add(deployment_id)
        unknown = sorted(set(deployment["components"]) - component_ids)
        if unknown:
            raise ManifestError(f"{location}: unknown components {unknown}")
        if deployment["runtime_profile"] not in runtime_profiles:
            raise ManifestError(f"{location}: unknown runtime_profile")
        if deployment["deployment_status"] not in DEPLOYMENT_STATUSES:
            raise ManifestError(f"{location}: invalid deployment_status")
        verification = deployment["verification"]
        _require(
            verification,
            {
                "artifacts_present",
                "hashes_verified",
                "dependencies_verified",
                "successfully_loaded",
                "network_isolated_inference_tested",
                "compatible_target_tested",
            },
            f"{location}.verification",
        )
        evidence = deployment.get("evidence") or {}
        if deployment["deployment_status"] == "READY_FOR_AIRGAP" and not all(
            evidence.get(name)
            for name in ("offline_inference", "network_isolation", "target_compatibility")
        ):
            raise ManifestError(
                f"{location}: READY_FOR_AIRGAP requires file-backed evidence, not flags"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _guarded_address(address: Any) -> None:
    if isinstance(address, tuple) and address:
        host = str(address[0])
        if _is_loopback_host(host):
            return
    raise OutboundNetworkBlocked(f"Air-gap policy blocked outbound network address: {address!r}")


# (owner, attribute) pairs patched by the in-process guard. Every Python-level path
# that opens an outbound connection, sends a datagram or resolves a name is listed.
# Native code (Rust/C extensions such as hf-xet), subprocesses and other processes
# are NOT covered: only OS/container isolation can prove those are offline.
_GUARDED_ATTRIBUTES = (
    (socket.socket, "connect"),
    (socket.socket, "connect_ex"),
    (socket.socket, "sendto"),
    (socket, "create_connection"),
    (socket, "getaddrinfo"),
    (socket, "gethostbyname"),
    (socket, "gethostbyname_ex"),
    (socket, "gethostbyaddr"),
)


def _blocked_host(host: Any, what: str) -> None:
    if host is not None and not _is_loopback_host(str(host)):
        raise OutboundNetworkBlocked(f"Air-gap policy blocked {what} for host: {host!r}")


def _guarded_replacements(originals: dict[str, Any]) -> dict[str, Any]:
    def connect(sock, address):
        _guarded_address(address)
        return originals["connect"](sock, address)

    def connect_ex(sock, address):
        _guarded_address(address)
        return originals["connect_ex"](sock, address)

    def sendto(sock, data, *args):
        if not args:
            raise OutboundNetworkBlocked("Air-gap policy blocked sendto without an address")
        _guarded_address(args[-1])
        return originals["sendto"](sock, data, *args)

    def create_connection(address, *args, **kwargs):
        _guarded_address(address)
        return originals["create_connection"](address, *args, **kwargs)

    def getaddrinfo(host, *args, **kwargs):
        _blocked_host(host, "DNS resolution")
        return originals["getaddrinfo"](host, *args, **kwargs)

    def gethostbyname(host):
        _blocked_host(host, "DNS resolution")
        return originals["gethostbyname"](host)

    def gethostbyname_ex(host):
        _blocked_host(host, "DNS resolution")
        return originals["gethostbyname_ex"](host)

    def gethostbyaddr(host):
        _blocked_host(host, "reverse DNS resolution")
        return originals["gethostbyaddr"](host)

    replacements = {
        "connect": connect,
        "connect_ex": connect_ex,
        "sendto": sendto,
        "create_connection": create_connection,
        "getaddrinfo": getaddrinfo,
        "gethostbyname": gethostbyname,
        "gethostbyname_ex": gethostbyname_ex,
        "gethostbyaddr": gethostbyaddr,
    }
    for function in replacements.values():
        function.__airgap_guard__ = True
    return replacements


def _install_guard() -> dict[str, Any]:
    originals = {name: getattr(owner, name) for owner, name in _GUARDED_ATTRIBUTES}
    replacements = _guarded_replacements(originals)
    for owner, name in _GUARDED_ATTRIBUTES:
        setattr(owner, name, replacements[name])
    return originals


def _restore_guard(originals: dict[str, Any]) -> None:
    for owner, name in _GUARDED_ATTRIBUTES:
        setattr(owner, name, originals[name])


def guard_installed() -> bool:
    return all(
        getattr(getattr(owner, name), "__airgap_guard__", False)
        for owner, name in _GUARDED_ATTRIBUTES
    )


@contextmanager
def outbound_network_guard() -> Iterator[None]:
    """Block non-loopback connections, datagrams and name resolution in this process."""
    if guard_installed():
        yield
        return
    originals = _install_guard()
    try:
        yield
    finally:
        _restore_guard(originals)


def enforce_offline_runtime() -> None:
    """Set offline controls and install the guard for a worker before model imports."""
    os.environ.update(OFFLINE_ENVIRONMENT)
    if not guard_installed():
        _install_guard()


def _component_result(component: dict, environment: str) -> dict[str, Any]:
    root_key = "local_root" if environment == "local" else "cluster_root"
    root = Path(component[root_key])
    errors: list[str] = []
    warnings: list[str] = []
    artifacts: list[dict] = []
    all_required_present = True
    all_required_hashes = True
    allowed_paths: set[str] = set()
    expected_executable_files: list[str] = []
    if component.get("root_must_end_with_revision", True) and root.name != component["revision"]:
        errors.append(
            f"revision_path_mismatch: root {root} does not end with {component['revision']}"
        )
    for artifact in component["artifacts"]:
        path = root / artifact["local_relative_path" if environment == "local" else "cluster_relative_path"]
        relative = artifact[
            "local_relative_path" if environment == "local" else "cluster_relative_path"
        ]
        allowed_paths.add(str(relative).replace("\\", "/"))
        if Path(relative).suffix.lower() in {".py", ".exe", ".dll", ".so", ".dylib"}:
            expected_executable_files.append(str(path))
        record = {
            "filename": artifact["filename"],
            "role": artifact["role"],
            "required": artifact["required"],
            "path": str(path),
            "exists": path.is_file(),
            "expected_size": artifact["size_bytes"],
            "expected_sha256": artifact["sha256"],
            "actual_size": None,
            "actual_sha256": None,
            "ok": False,
        }
        if not path.is_file():
            if artifact["required"]:
                all_required_present = False
                errors.append(f"missing_{artifact['role']}: {path}")
            artifacts.append(record)
            continue
        record["actual_size"] = path.stat().st_size
        if record["actual_size"] != artifact["size_bytes"]:
            errors.append(
                f"size_mismatch: {path} expected {artifact['size_bytes']} got {record['actual_size']}"
            )
        expected_hash = artifact["sha256"]
        if expected_hash is None:
            if artifact["required"]:
                all_required_hashes = False
            warnings.append(f"checksum_unverified: {path}")
        else:
            record["actual_sha256"] = _sha256(path)
            if record["actual_sha256"] != expected_hash:
                errors.append(f"hash_mismatch: {path}")
        record["ok"] = not any(str(path) in error for error in errors)
        artifacts.append(record)
    unexpected_files: list[str] = []
    unexpected_executable_files: list[str] = []
    if root.is_dir():
        for found in root.rglob("*"):
            if not found.is_file():
                continue
            relative = found.relative_to(root).as_posix()
            if relative in allowed_paths:
                continue
            unexpected_files.append(str(found))
            if found.suffix.lower() in {
                ".bat", ".cmd", ".com", ".dll", ".dylib", ".exe", ".js",
                ".msi", ".pkl", ".pickle", ".ps1", ".pth", ".py", ".so",
            }:
                unexpected_executable_files.append(str(found))
        if unexpected_files:
            errors.append(f"unexpected_model_files: {unexpected_files}")
    if component["declared_status"] == "BLOCKED":
        status = "BLOCKED"
    elif errors and any(
        error.startswith(
            ("hash_mismatch", "size_mismatch", "revision_path", "unexpected_model_files")
        )
        for error in errors
    ):
        status = "BLOCKED"
    elif not all_required_present:
        status = "NOT_DOWNLOADED"
    elif all_required_hashes:
        status = "HASH_VERIFIED"
    else:
        status = "DOWNLOADED"
    return {
        "component_id": component["component_id"],
        "root": str(root),
        "status": status,
        "artifacts": artifacts,
        "errors": errors,
        "warnings": warnings,
        "unexpected_files": unexpected_files,
        "unexpected_executable_files": unexpected_executable_files,
        "expected_executable_files": expected_executable_files,
    }


def _locked_requirements(profile: dict) -> tuple[list[tuple[str, str]], list[str], bool]:
    """Parse the profile's lock plus any `additional_lockfiles` (e.g. the UI layer)."""
    lockfile = profile.get("lockfile")
    if not lockfile or lockfile in {"MISSING", "NOT_APPLICABLE"}:
        return [], [f"dependency_lock_missing: {lockfile or 'MISSING'}"], False
    requirements: list[tuple[str, str]] = []
    errors: list[str] = []
    every_entry_hashed = True
    seen: dict[str, str] = {}
    for entry in [lockfile, *profile.get("additional_lockfiles", [])]:
        found, file_errors, hashed = _parse_lockfile(entry)
        errors.extend(file_errors)
        every_entry_hashed = every_entry_hashed and hashed
        for name, version in found:
            key = name.lower().replace("_", "-")
            if key in seen and seen[key] != version:
                errors.append(f"conflicting_lock_entries: {name} {seen[key]} vs {version}")
                continue
            if key not in seen:
                seen[key] = version
                requirements.append((name, version))
    if not requirements and not errors:
        errors.append("empty_dependency_lock")
    return requirements, errors, every_entry_hashed and bool(requirements)


def _parse_lockfile(lockfile: str) -> tuple[list[tuple[str, str]], list[str], bool]:
    path = Path(lockfile)
    if not path.is_absolute():
        path = LAB_DIR.parent / path
    if not path.is_file():
        return [], [f"dependency_lock_missing: {path}"], False
    requirements: list[tuple[str, str]] = []
    errors: list[str] = []
    every_entry_hashed = True
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirement, _, hash_part = line.partition(" --hash=sha256:")
        if "==" not in requirement or requirement.startswith(("-", "git+")):
            errors.append(f"unpinned_lock_entry: {path}:{number}: {line}")
            continue
        name, version = requirement.split("==", 1)
        if not name or not version or any(token in version for token in ("*", ">", "<", "~")):
            errors.append(f"unpinned_lock_entry: {path}:{number}: {line}")
            continue
        requirements.append((name.strip(), version.strip()))
        if not hash_part:
            every_entry_hashed = False
    if not requirements:
        errors.append(f"empty_dependency_lock: {path}")
    return requirements, errors, every_entry_hashed


def _runtime_result(profile: dict) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    packages: list[dict] = []
    expected_python = profile.get("python")
    actual_python = platform.python_version()
    if expected_python not in {None, "UNVERIFIED"} and actual_python != expected_python:
        errors.append(f"python_version_mismatch: expected {expected_python}, got {actual_python}")
    requirements, lock_errors, wheel_hashes_verified = _locked_requirements(profile)
    errors.extend(lock_errors)
    if profile.get("lock_complete") is not True:
        errors.append("dependency_lock_declared_incomplete")
    locked = {name.lower().replace("_", "-"): version for name, version in requirements}
    declared = profile.get("packages", {})
    for name, expected in declared.items():
        normalized = name.lower().replace("_", "-")
        if expected != "UNVERIFIED" and locked and locked.get(normalized) != expected:
            errors.append(
                f"lock_profile_mismatch: {name} profile={expected} lock={locked.get(normalized)}"
            )
    checks = requirements or list(declared.items())
    for name, expected in checks:
        actual = _package_version(name)
        if expected == "UNVERIFIED":
            ok = False
            warnings.append(f"dependency_unverified: {name} installed={actual}")
        else:
            ok = actual == expected
        packages.append({"name": name, "expected": expected, "actual": actual, "ok": ok})
        if not ok:
            errors.append(f"dependency_version_mismatch: {name} expected {expected}, got {actual}")
    if not wheel_hashes_verified and requirements:
        warnings.append("wheel_hashes_unverified: lock pins versions but carries no wheel hashes")
    return {
        "profile_id": profile["profile_id"],
        "python_expected": expected_python,
        "python_actual": actual_python,
        "packages": packages,
        "native_dependencies": profile.get("native_dependencies", []),
        "lockfile": profile.get("lockfile"),
        "locked_requirement_count": len(requirements),
        "wheel_hashes_verified": wheel_hashes_verified,
        "lock_complete": profile.get("lock_complete") is True,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


TARGET_MATCH_FIELDS = ("os", "architecture", "accelerator", "runtime_identity_sha256")


def _target_mismatches(payload: dict, manifest: dict[str, Any]) -> list[str]:
    """Compatibility evidence must describe the operator-verified production target.

    Evidence gathered on any other machine (e.g. this Windows CPU laptop) cannot
    satisfy a Linux/GPU cluster: every declared target field must match exactly.
    """
    requirements = manifest.get("target_cluster_requirements") or {}
    target = requirements.get("target")
    if requirements.get("status") != "VERIFIED" or not isinstance(target, dict):
        return ["target_specification_unverified"]
    errors = []
    for field in TARGET_MATCH_FIELDS:
        expected = target.get(field)
        if expected is None:
            if field in {"os", "architecture"}:
                errors.append(f"target_specification_missing_{field}")
            continue
        if str(payload.get(field)) != str(expected):
            errors.append(f"target_{field}_mismatch: evidence={payload.get(field)!r} target={expected!r}")
    return errors


def _evidence_result(deployment: dict, manifest: dict[str, Any]) -> dict[str, Any]:
    evidence = deployment.get("evidence") or {}
    results: dict[str, Any] = {}
    for name in ("offline_inference", "network_isolation", "target_compatibility"):
        declaration = evidence.get(name)
        record = {"declared": declaration, "ok": False, "errors": []}
        if not isinstance(declaration, dict):
            record["errors"].append("missing_file_backed_evidence")
            results[name] = record
            continue
        path = Path(str(declaration.get("path") or ""))
        expected_hash = declaration.get("sha256")
        if not path.is_file() or not isinstance(expected_hash, str):
            record["errors"].append("evidence_file_or_sha256_missing")
            results[name] = record
            continue
        actual_hash = _sha256(path)
        record["actual_sha256"] = actual_hash
        if actual_hash != expected_hash:
            record["errors"].append("evidence_hash_mismatch")
            results[name] = record
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record["errors"].append("evidence_json_invalid")
            results[name] = record
            continue
        if payload.get("deployment_id") != deployment["deployment_id"]:
            record["errors"].append("evidence_deployment_mismatch")
        if payload.get("manifest_id") != manifest["manifest_id"]:
            record["errors"].append("evidence_manifest_mismatch")
        if payload.get("status") not in {"passed", "verified"}:
            record["errors"].append("evidence_status_not_verified")
        observed_at = payload.get("observed_at")
        try:
            age_days = (
                datetime.now(timezone.utc) - datetime.fromisoformat(observed_at)
            ).total_seconds() / 86400
        except (TypeError, ValueError):
            age_days = 31
        if age_days < 0 or age_days > 30:
            record["errors"].append("evidence_missing_or_stale_over_30_days")
        if name == "offline_inference":
            if not payload.get("successful_inference") or not payload.get(
                "raw_output_preserved"
            ):
                record["errors"].append("offline_inference_success_not_proven")
            for linked_name in ("preflight_report", "run_record"):
                linked = payload.get(linked_name) or {}
                linked_path = Path(str(linked.get("path") or ""))
                linked_hash = linked.get("sha256")
                if not linked_path.is_file() or not isinstance(linked_hash, str):
                    record["errors"].append(f"{linked_name}_evidence_missing")
                    continue
                if _sha256(linked_path) != linked_hash:
                    record["errors"].append(f"{linked_name}_hash_mismatch")
                    continue
                if linked_name == "run_record":
                    try:
                        linked_payload = json.loads(linked_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        record["errors"].append("run_record_invalid")
                    else:
                        adapter = (linked_payload.get("adapter") or {}).get(
                            "adapter_id"
                        ) or linked_payload.get("adapter_id")
                        if adapter != f"e2.{deployment['deployment_id']}":
                            record["errors"].append("run_record_deployment_mismatch")
                        if linked_payload.get("status") not in {"ok", "completed"}:
                            record["errors"].append("run_record_not_successful")
                if linked_name == "preflight_report":
                    try:
                        linked_payload = json.loads(linked_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        record["errors"].append("preflight_report_invalid")
                    else:
                        if linked_payload.get("deployment_id") != deployment["deployment_id"]:
                            record["errors"].append("preflight_deployment_mismatch")
                        if linked_payload.get("manifest_id") != manifest["manifest_id"]:
                            record["errors"].append("preflight_manifest_mismatch")
                        if linked_payload.get("status") not in {
                            "DEPENDENCIES_VERIFIED", "OFFLINE_TESTED", "READY_FOR_AIRGAP"
                        }:
                            record["errors"].append("preflight_status_insufficient")
        elif name == "network_isolation":
            if payload.get("enforcement_scope") not in {
                "container_network_none",
                "os_firewall_denied",
                "physically_isolated_host",
            }:
                record["errors"].append("network_isolation_not_os_or_container_level")
            if payload.get("observed_outbound_attempts_blocked") is not True:
                record["errors"].append("network_blocking_not_observed")
        elif name == "target_compatibility":
            if payload.get("compatible") is not True:
                record["errors"].append("target_not_compatible")
            for field in ("os", "architecture", "runtime_identity_sha256"):
                if not payload.get(field):
                    record["errors"].append(f"target_{field}_missing")
            record["errors"].extend(_target_mismatches(payload, manifest))
        record["ok"] = not record["errors"]
        results[name] = record
    return results


def _hardware_result(deployment: dict, hardware: HardwareProfile) -> dict[str, Any]:
    requirements = deployment["hardware"]
    errors: list[str] = []
    total_ram = hardware.ram_total_gb
    min_ram = requirements.get("min_ram_gb")
    if total_ram is not None and min_ram is not None and total_ram < min_ram:
        errors.append(f"insufficient_ram: {total_ram:.2f} GB < {min_ram} GB")
    min_vram = requirements.get("min_vram_gb") or 0
    if not requirements.get("cpu_feasible", False) and hardware.max_vram_gb < min_vram:
        errors.append(f"unsupported_hardware: VRAM {hardware.max_vram_gb:.2f} GB < {min_vram} GB")
    constraints = deployment["os_arch_constraints"]
    current_os = platform.system()
    current_arch = platform.machine()
    allowed_os = constraints.get("allowed_os", [])
    allowed_arch = constraints.get("allowed_arch", [])
    if allowed_os and current_os not in allowed_os:
        errors.append(f"unsupported_os: {current_os} not in {allowed_os}")
    if allowed_arch and current_arch not in allowed_arch:
        errors.append(f"unsupported_arch: {current_arch} not in {allowed_arch}")
    return {
        "cpu": hardware.cpu,
        "ram_total_gb": total_ram,
        "max_vram_gb": hardware.max_vram_gb,
        "os": current_os,
        "architecture": current_arch,
        "errors": errors,
        "ok": not errors,
    }


def preflight_deployment(
    manifest: dict[str, Any],
    deployment_id: str,
    *,
    environment: str = "local",
    hardware: HardwareProfile | None = None,
    native_probe: bool = True,
) -> dict[str, Any]:
    if environment not in {"local", "cluster"}:
        raise ValueError("environment must be 'local' or 'cluster'")
    deployment = next(
        (item for item in manifest["deployments"] if item["deployment_id"] == deployment_id),
        None,
    )
    if deployment is None:
        raise ManifestError(f"Unknown deployment_id {deployment_id!r}")
    components_by_id = {item["component_id"]: item for item in manifest["components"]}
    component_results = [
        _component_result(components_by_id[component_id], environment)
        for component_id in deployment["components"]
    ]
    profile = manifest["runtime_profiles"][deployment["runtime_profile"]]
    runtime = _runtime_result(profile)
    hardware_result = _hardware_result(deployment, hardware or probe_hardware())
    component_statuses = {item["status"] for item in component_results}
    errors = [error for item in component_results for error in item["errors"]]
    errors += runtime["errors"] + hardware_result["errors"]
    # Hashes cannot show that native code will load: application control decides at
    # load time, by reputation, and its verdict can change. Import the runtime's native
    # modules in a subprocess (no weights). Skipped for deployments already declared
    # BLOCKED so known-blocked modules are not loaded again.
    probe = None
    probe_modules = profile.get("runtime_import_probe") or []
    if (
        native_probe
        and environment == "local"
        and probe_modules
        and deployment["deployment_status"] != "BLOCKED"
    ):
        from lab.core.e2 import appcontrol  # noqa: PLC0415

        probe = appcontrol.native_import_probe(list(probe_modules))
        if not probe["ok"]:
            failed = probe["failed"] or {}
            errors.append(
                f"native_import_failed: {failed.get('module')}: {failed.get('type')}: "
                f"{failed.get('message') or probe['process_error']}"
            )
            errors.extend(
                f"application_control_blocked: {item['file']} (policy {item['policy_id']})"
                for item in probe["application_control_blocks"]
            )
    warnings = [warning for item in component_results for warning in item["warnings"]]
    warnings += runtime["warnings"]
    verification = deployment["verification"]
    evidence = _evidence_result(deployment, manifest)
    if (
        deployment["deployment_status"] == "BLOCKED"
        or "BLOCKED" in component_statuses
        or hardware_result["errors"]
        or (probe is not None and not probe["ok"])
    ):
        status = "BLOCKED"
    elif "NOT_DOWNLOADED" in component_statuses:
        status = "NOT_DOWNLOADED"
    elif "DOWNLOADED" in component_statuses:
        status = "DOWNLOADED"
    elif not runtime["ok"]:
        status = "HASH_VERIFIED"
    elif not evidence["offline_inference"]["ok"] or not evidence["network_isolation"]["ok"]:
        status = "DEPENDENCIES_VERIFIED"
    elif not evidence["target_compatibility"]["ok"]:
        status = "OFFLINE_TESTED"
    elif all(item["ok"] for item in evidence.values()):
        status = "READY_FOR_AIRGAP"
    else:
        status = "BLOCKED"
    return {
        "schema_version": 1,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest.get("_manifest_sha256"),
        "deployment_id": deployment_id,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "offline_controls": OFFLINE_ENVIRONMENT,
        "status": status,
        "declared_status": deployment["deployment_status"],
        "components": component_results,
        "runtime": runtime,
        "native_import_probe": probe,
        "hardware": hardware_result,
        "declared_verification": verification,
        "evidence": evidence,
        "errors": errors,
        "warnings": warnings,
        "missing_or_unverified": deployment["missing_or_unverified"],
        "ready_for_airgap": status == "READY_FOR_AIRGAP",
    }


def write_preflight_report(report: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"airgap_preflight_{report['deployment_id']}_{stamp}.json"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def deployment_record(manifest: dict[str, Any], deployment_id: str) -> dict[str, Any]:
    try:
        return next(item for item in manifest["deployments"] if item["deployment_id"] == deployment_id)
    except StopIteration as exc:
        raise ManifestError(f"No air-gap record for {deployment_id}") from exc


def validate_registry_alignment(specs: list[Any], manifest: dict[str, Any]) -> list[str]:
    """Return exact registry/manifest drift without loading a model."""
    components = {item["component_id"]: item for item in manifest["components"]}
    deployments = {item["deployment_id"]: item for item in manifest["deployments"]}
    errors: list[str] = []
    for spec in specs:
        deployment = deployments.get(spec.id)
        if deployment is None:
            errors.append(f"missing_deployment: {spec.id}")
            continue
        primary_id = deployment.get("primary_component")
        primary = components.get(primary_id)
        if primary is None:
            errors.append(f"missing_primary_component: {spec.id} -> {primary_id}")
            continue
        if primary["repo_id"] != spec.repo_id or primary["revision"] != spec.revision:
            errors.append(f"primary_revision_mismatch: {spec.id}")
        if spec.base_repo_id:
            base = next(
                (
                    components[item]
                    for item in deployment["components"]
                    if components[item]["repo_id"] == spec.base_repo_id
                ),
                None,
            )
            if base is None or base["revision"] != spec.base_revision:
                errors.append(f"base_revision_mismatch: {spec.id}")
        if spec.layout_repo_id:
            layout = next(
                (
                    components[item]
                    for item in deployment["components"]
                    if components[item]["repo_id"] == spec.layout_repo_id
                ),
                None,
            )
            if layout is None or layout["revision"] != spec.layout_revision:
                errors.append(f"layout_revision_mismatch: {spec.id}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate one deployment without network access")
    parser.add_argument("deployment_id")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--environment", choices=["local", "cluster"], default="local")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    with outbound_network_guard():
        report = preflight_deployment(
            manifest, args.deployment_id, environment=args.environment
        )
    if args.output:
        if args.output.exists():
            raise ManifestError(f"Refusing to overwrite {args.output}")
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_airgap"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
