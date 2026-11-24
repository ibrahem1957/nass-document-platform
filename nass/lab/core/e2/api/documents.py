"""Exact-hash document authorization for hosted API evaluation.

Two independent gates, both by exact SHA-256 of the document's bytes:

1. Eligibility: the hash is listed in `lab/config/api_synthetic_documents.toml`
   (reviewed entries; a file's name or folder never makes it eligible). Each entry
   has a reviewed category that fixes which declaration the user must make:
   `synthetic` (fully synthetic benchmark) or `user_owned_test` (the user's own
   evaluation document whose synthetic status is not established).
   A PDF can also be made eligible from the UI, without editing the TOML: a
   write-once *registration* record (results/api_document_registry/<sha>/) made
   by a named person after the bytes and page count are verified. A registered
   document is always `user_owned_test` (the UI cannot establish that a file is
   synthetic), entries in the reviewed TOML always win, and a newer record can
   withdraw the registration. Registration is eligibility only, never permission.
2. Authorization: a separate, explicit, write-once user record for that hash, with
   reviewer, declaration, approval time, expiry and revocation. The newest record
   for a hash wins, so a revocation takes effect immediately; nothing is deleted.

Checks and writes share one cross-process lock, so a revocation cannot interleave
with the authorization check that precedes a submission.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from lab.core.e2.api.authorization import provider_usage_lock
from lab.core.e2.api.records import ordered, write_sequenced
from lab.core.e2.store import results_root
from lab.core.models import ContractModel
from lab.core.settings import LAB_DIR

DOCUMENT_REGISTRY = LAB_DIR / "config" / "api_synthetic_documents.toml"
MAX_AUTHORIZATION_DAYS = 30
DOCUMENT_LOCK_ID = "_document_authorizations"
SYNTHETIC_DECLARATION = (
    "I reviewed this exact document (identified by its SHA-256). It is fully synthetic: "
    "it contains no real person, customer, company, financial or confidential data, and "
    "I approve sending its rendered pages to external hosted APIs for evaluation."
)
# For a user-owned evaluation document whose synthetic status is NOT established. It
# does not claim the document is synthetic; it makes the user state, page by page,
# what will actually be transmitted. Exactly as strict a gate (same hash, record,
# expiry, revocation and send-time checks); only the truthful statement differs.
USER_OWNED_DECLARATION = (
    "I reviewed every page of this exact document (identified by its SHA-256). It is my "
    "own evaluation document, not a company file, and I am entitled to share it. Its pages "
    "contain no personal data, confidential information or restricted data, except data I "
    "have explicit permission to transmit to these providers, and I approve sending its "
    "rendered pages to the external hosted API providers I enable for this evaluation."
)
CATEGORY_SYNTHETIC = "synthetic"
CATEGORY_USER_OWNED = "user_owned_test"
DECLARATIONS = {CATEGORY_SYNTHETIC: SYNTHETIC_DECLARATION, CATEGORY_USER_OWNED: USER_OWNED_DECLARATION}
CATEGORY_LABELS = {
    CATEGORY_SYNTHETIC: "fully synthetic benchmark",
    CATEGORY_USER_OWNED: "user-owned test document (synthetic status not established)",
}
_SHA = re.compile(r"^[0-9a-f]{64}$")


class DocumentAuthorizationError(PermissionError):
    pass


class SyntheticDocument(ContractModel):
    id: str
    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)
    page_count: int = Field(gt=0)
    title: str
    provenance: str
    ground_truth_filename: str | None = None
    ground_truth_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    verified_on: str
    # Which declaration the user must make. Set per reviewed registry entry, never
    # inferred; changing it invalidates existing authorizations (identity changes).
    authorization_category: Literal["synthetic", "user_owned_test"] = CATEGORY_SYNTHETIC

    @property
    def declaration(self) -> str:
        return DECLARATIONS[self.authorization_category]

    def identity(self) -> dict[str, Any]:
        return self.model_dump()


def load_registry(path: Path | None = None, *, root: Path | None = None) -> dict[str, SyntheticDocument]:
    """Eligible documents keyed by SHA-256. Read at call time (never cached).

    With no explicit `path`, documents registered from the UI under `root` are
    included; an entry in the reviewed TOML always takes precedence over them.
    """
    documents = _load_toml_registry(path)
    if path is None:
        for sha, item in registered_documents(root).items():
            documents.setdefault(sha, item)
    return documents


def _load_toml_registry(path: Path | None = None) -> dict[str, SyntheticDocument]:
    source = Path(path or DOCUMENT_REGISTRY)
    if not source.is_file():
        return {}
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    documents: dict[str, SyntheticDocument] = {}
    for entry in raw.get("documents", []):
        item = SyntheticDocument.model_validate(entry)
        if item.sha256 in documents:
            raise DocumentAuthorizationError(f"Duplicate registry hash {item.sha256}")
        documents[item.sha256] = item
    return documents


# ------------------------------------------------------------ UI registration

REGISTRATION_STATEMENT = (
    "I am registering this exact PDF (identified by its SHA-256) as my own user-owned "
    "evaluation document. Registration only makes it eligible: nothing is sent until I "
    "also authorize it under the user-owned declaration and approve a run plan."
)


def _registration_dir(root: Path | None, sha256: str) -> Path:
    if not _SHA.match(sha256 or ""):
        raise DocumentAuthorizationError("A document is identified only by a lowercase SHA-256")
    return _root(root) / "api_document_registry" / sha256


def _registration_id(display_name: str, sha256: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", Path(display_name).stem.lower()).strip("-")[:40] or "document"
    return f"registered-{stem}-{sha256[:8]}"


def registration_records(sha256: str, root: Path | None = None) -> list[dict]:
    records = []
    for path, payload in ordered(_registration_dir(root, sha256), "registration"):
        # An unreadable record fails closed: it withdraws the registration.
        records.append((payload or {"registered": False, "corrupt": True}) | {"path": str(path)})
    return [item for item in records if not item.get("corrupt")] + [
        item for item in records if item.get("corrupt")]


def registered_documents(root: Path | None = None) -> dict[str, SyntheticDocument]:
    """Documents whose newest registration record is an active registration."""
    base = _root(root) / "api_document_registry"
    found: dict[str, SyntheticDocument] = {}
    if not base.is_dir():
        return found
    for directory in sorted(base.iterdir()):
        if not directory.is_dir() or not _SHA.match(directory.name):
            continue
        records = registration_records(directory.name, root)
        latest = records[-1] if records else None
        if not latest or not latest.get("registered"):
            continue
        try:
            item = SyntheticDocument.model_validate(latest.get("entry") or {})
        except ValueError:
            continue
        if item.sha256 == directory.name and item.authorization_category == CATEGORY_USER_OWNED:
            found[item.sha256] = item
    return found


def register_document(
    file_path: Path, *, display_name: str, registered_by: str, statement: str,
    root: Path | None = None,
) -> SyntheticDocument:
    """Make one exact PDF eligible (user-owned category) after verifying its bytes."""
    import pypdfium2 as pdfium  # noqa: PLC0415

    if statement != REGISTRATION_STATEMENT:
        raise DocumentAuthorizationError("The registration statement was not accepted")
    if not registered_by.strip():
        raise DocumentAuthorizationError("Enter the name of the person registering the document")
    data = Path(file_path).read_bytes()
    if not data.startswith(b"%PDF-"):
        raise DocumentAuthorizationError("Only PDF files can be registered")
    sha = hashlib.sha256(data).hexdigest()
    if sha in _load_toml_registry():
        raise DocumentAuthorizationError("This document is already in the reviewed registry")
    try:
        pdf = pdfium.PdfDocument(data)
        try:
            page_count = len(pdf)
        finally:
            pdf.close()
    except Exception as exc:  # noqa: BLE001 - any parse failure refuses registration
        raise DocumentAuthorizationError(f"The PDF cannot be parsed: {type(exc).__name__}") from exc
    if page_count < 1:
        raise DocumentAuthorizationError("The PDF has no pages")
    now = datetime.now(timezone.utc)
    name = Path(display_name).name or "document.pdf"
    entry = SyntheticDocument(
        id=_registration_id(name, sha), filename=name, sha256=sha, bytes=len(data),
        page_count=page_count, title=f"{name} (registered from the OCR/VLM Evaluation page)",
        provenance=(f"Registered by {registered_by.strip()} on {now.date().isoformat()} from the "
                    "OCR/VLM Evaluation page; user-owned test document, synthetic status not established"),
        verified_on=now.date().isoformat(), authorization_category=CATEGORY_USER_OWNED,
    )
    with provider_usage_lock(DOCUMENT_LOCK_ID, root):
        records = registration_records(sha, root)
        if records and records[-1].get("registered"):
            return SyntheticDocument.model_validate(records[-1]["entry"])
        write_sequenced(_registration_dir(root, sha), "registration", {
            "schema_version": 1, "registered": True, "document_sha256": sha,
            "entry": entry.identity(), "registered_by": registered_by.strip(),
            "statement": statement, "created_at": now.isoformat(), "verified_file_sha256": sha,
        })
    return entry


def withdraw_registration(sha256: str, *, withdrawn_by: str, reason: str, root: Path | None = None) -> Path:
    if not withdrawn_by.strip() or not reason.strip():
        raise DocumentAuthorizationError("Give your name and a reason to withdraw the registration")
    with provider_usage_lock(DOCUMENT_LOCK_ID, root):
        return write_sequenced(_registration_dir(root, sha256), "registration", {
            "schema_version": 1, "registered": False, "document_sha256": sha256,
            "withdrawn_by": withdrawn_by.strip(), "reason": reason.strip(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root(root: Path | None) -> Path:
    return Path(root or results_root())


def _auth_dir(root: Path | None, sha256: str) -> Path:
    if not _SHA.match(sha256 or ""):
        raise DocumentAuthorizationError("A document is identified only by a lowercase SHA-256")
    return _root(root) / "api_document_authorizations" / sha256


def _write_once(directory: Path, payload: dict) -> Path:
    # Strictly sequenced: a same-tick approval and revocation can never be reordered.
    return write_sequenced(directory, "authorization", payload)


def authorization_records(sha256: str, root: Path | None = None) -> list[dict]:
    records = []
    for path, payload in ordered(_auth_dir(root, sha256), "authorization"):
        if payload is None:
            # A corrupt audit record fails closed: it is treated as a revocation.
            payload = {"approved": False, "corrupt": True}
        records.append(payload | {"path": str(path)})
    # Unreadable records sort last, so the latest state is "revoked" (fail closed).
    return [record for record in records if not record.get("corrupt")] + [
        record for record in records if record.get("corrupt")]


def authorize_document(
    document: SyntheticDocument, *, file_path: Path, reviewer: str, declaration: str,
    days_valid: int, root: Path | None = None, registry: dict[str, SyntheticDocument] | None = None,
) -> Path:
    """Record an explicit approval for one exact hash after re-verifying the bytes.

    The declaration must be exactly the one for the entry's reviewed category; the
    synthetic declaration is never accepted for a user-owned entry, or vice versa.
    """
    if declaration != document.declaration:
        raise DocumentAuthorizationError(
            f"The declaration for a {CATEGORY_LABELS[document.authorization_category]} was not accepted")
    if not reviewer.strip():
        raise DocumentAuthorizationError("Enter the reviewer's name")
    if not 1 <= int(days_valid) <= MAX_AUTHORIZATION_DAYS:
        raise DocumentAuthorizationError(f"Validity must be 1..{MAX_AUTHORIZATION_DAYS} days")
    eligible = registry if registry is not None else load_registry(root=root)
    if eligible.get(document.sha256) != document:
        raise DocumentAuthorizationError("This document is not an eligible registry entry")
    actual = file_sha256(file_path)
    if actual != document.sha256 or Path(file_path).stat().st_size != document.bytes:
        raise DocumentAuthorizationError("The file's bytes do not match the registered SHA-256/size")
    now = datetime.now(timezone.utc)
    with provider_usage_lock(DOCUMENT_LOCK_ID, root):
        return _write_once(_auth_dir(root, document.sha256), {
            "schema_version": 1,
            "document_sha256": document.sha256,
            "document": document.identity(),
            "approved": True,
            "reviewer": reviewer.strip(),
            "authorization_category": document.authorization_category,
            "declaration": declaration,
            "approved_at": now.isoformat(),
            "expires_at": (now + timedelta(days=int(days_valid))).isoformat(),
            "created_at": now.isoformat(),
            "verified_file_sha256": actual,
        })


def revoke_document(sha256: str, *, revoked_by: str, reason: str, root: Path | None = None) -> Path:
    if not revoked_by.strip() or not reason.strip():
        raise DocumentAuthorizationError("Give your name and a reason to revoke")
    now = datetime.now(timezone.utc)
    with provider_usage_lock(DOCUMENT_LOCK_ID, root):
        return _write_once(_auth_dir(root, sha256), {
            "schema_version": 1, "document_sha256": sha256, "approved": False,
            "revoked_by": revoked_by.strip(), "reason": reason.strip(),
            "created_at": now.isoformat(),
        })


def authorization_state_unlocked(
    sha256: str, *, root: Path | None = None, registry: dict[str, SyntheticDocument] | None = None,
) -> dict[str, Any]:
    """Whether this exact hash may be sent now. Call under the document lock when the
    answer gates a transmission."""
    eligible = registry if registry is not None else load_registry(root=root)
    reasons: list[str] = []
    entry = eligible.get(sha256)
    latest = None
    if entry is None:
        reasons.append(
            "This document's SHA-256 is not in the synthetic document registry (the "
            "reviewed list of eligible documents); hosted API "
            "transmission is refused (its name or folder does not matter)"
        )
    else:
        records = authorization_records(sha256, root)
        latest = records[-1] if records else None
        if latest is None:
            reasons.append("No explicit authorization exists for this document hash")
        elif not latest.get("approved"):
            reasons.append("Authorization for this document hash was revoked" + (
                " (unreadable audit record treated as revoked)" if latest.get("corrupt") else ""))
        else:
            try:
                expired = datetime.fromisoformat(latest["expires_at"]) <= datetime.now(timezone.utc)
            except (KeyError, ValueError):
                expired = True
            if expired:
                reasons.append("Authorization for this document hash has expired")
            if latest.get("document") != entry.identity():
                reasons.append("The registry entry changed after authorization; authorize again")
            if latest.get("declaration") != entry.declaration:
                reasons.append("The authorization lacks the required declaration")
            if latest.get("authorization_category", CATEGORY_SYNTHETIC) != entry.authorization_category:
                reasons.append("The authorization was made for a different document category; authorize again")
    return {
        "allowed": not reasons,
        "reasons": reasons,
        "document": entry.identity() if entry else None,
        "authorization": {key: value for key, value in (latest or {}).items() if key != "path"}
        | ({"path": latest["path"]} if latest else {}),
    }


def authorization_state(sha256: str, *, root: Path | None = None,
                        registry: dict[str, SyntheticDocument] | None = None) -> dict[str, Any]:
    with provider_usage_lock(DOCUMENT_LOCK_ID, root):
        return authorization_state_unlocked(sha256, root=root, registry=registry)
