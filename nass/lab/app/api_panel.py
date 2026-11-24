"""Streamlit panel for EXPERIMENTAL hosted API evaluation (one confirmed page at a time).

Nothing here runs unless the user enables a provider, supplies a credential for the
session and confirms the exact image to be sent. The credential lives only in the
browser session's memory (or the process environment) and is never written.
"""

from __future__ import annotations

import os

import streamlit as st

from lab.core.e2.api.authorization import (
    REQUIRED_STATEMENT,
    AuthorizationError,
    check_authorization,
    latest_enablement,
    record_enablement,
    usage_policy,
)
from lab.core.e2.api.jobs import (
    HOST_APPROVAL_STATEMENT,
    RESUMABLE,
    JobJournalError,
    approve_result_host,
    open_journals_for_image,
    revoke_result_host,
)
from lab.core.e2.api import documents
from lab.core.e2.api.providers import load_providers
from lab.core.e2.api.runner import APIEvaluationRunner, APIRunRefused


def render_api_panel(*, manager, settings, store, workspace_id, document, source, page_number, ground_truth) -> None:
    st.error(
        "Hosted API mode is an EXPERIMENTAL evaluation outside the air-gapped system: the "
        "selected page image is sent to the provider over the internet. The runner refuses "
        "any PDF whose exact SHA-256 is not an eligible registry entry with your explicit, "
        "unexpired authorization. Local inference is unaffected."
    )
    _show_flash()
    providers = load_providers()
    labels = {f"{item.label} [{item.status}]": item for item in providers}
    spec = labels[st.selectbox("API provider and model", list(labels))]
    with st.expander("Exact API identity (what is and is not verified)", expanded=True):
        st.json(spec.identity(), expanded=True)
        st.caption(
            "Exact equivalence to a downloadable model is NOT claimed unless "
            "`exact_revision_verified` is true."
        )
    if not spec.runnable:
        st.info(f"{spec.provider} is planned, not implemented: no request can be sent.")
        return

    runner = APIEvaluationRunner(manager, settings, store)
    state = check_authorization(spec, pages=1, root=store.root)
    latest = latest_enablement(spec, root=store.root)
    with st.expander("Provider enablement (disabled by default)", expanded=not state.allowed):
        if latest:
            st.text(
                f"Latest record: {'ENABLED' if latest.get('enabled') else 'DISABLED'} by "
                f"{latest.get('enabled_by') or '-'} at {latest.get('created_at')}; expires "
                f"{latest.get('expires_at')}\nUsage: {state.describe()}\n"
                f"History (all enablements): {state.pages_used_total} page(s)"
            )
        policy = usage_policy(spec)
        st.caption(f"Limit source: {policy['source']}. {policy['provider_quota']}.")
        with st.form(f"api-enable-{spec.id}"):
            enabled_by = st.text_input("Your name")
            max_total = st.number_input(f"Maximum pages for this enablement (ceiling {policy['max_pages_per_enablement']})",
                                        1, policy["max_pages_per_enablement"], min(16, policy["max_pages_per_enablement"]))
            max_day = st.number_input(f"Maximum pages per day (ceiling {policy['max_pages_per_day_ceiling']})",
                                      1, policy["max_pages_per_day_ceiling"], min(16, policy["max_pages_per_day_ceiling"]))
            days = st.number_input("Valid for (days)", 1, 30, policy["default_enablement_days"])
            agree = st.checkbox(REQUIRED_STATEMENT)
            enable = st.form_submit_button("Enable this provider")
        if enable:
            if not agree:
                st.error("Tick the confirmation statement to enable the provider.")
            else:
                try:
                    record_enablement(
                        spec, enabled=True, enabled_by=enabled_by, statement=REQUIRED_STATEMENT,
                        max_pages_total=int(max_total), max_pages_per_day=int(max_day),
                        days_valid=int(days), root=store.root,
                    )
                    st.rerun()
                except AuthorizationError as exc:
                    st.error(str(exc))
        if latest and latest.get("enabled") and st.button("Disable this provider", key=f"api-disable-{spec.id}"):
            record_enablement(spec, enabled=False, enabled_by=latest.get("enabled_by") or "user",
                              statement="", max_pages_total=0, max_pages_per_day=0, days_valid=1,
                              root=store.root)
            st.rerun()
    if not state.allowed:
        st.warning("No request can be sent:\n- " + "\n- ".join(state.reasons))
        return

    typed = st.text_input(
        f"{spec.credential_label} (kept only in Streamlit session memory; never shown or stored on disk)",
        type="password", key=f"api-credential-{spec.id}",
    )
    from_env = os.environ.get(spec.credential_env, "")
    credential = typed or from_env
    st.caption(
        "Credential source: "
        + ("this session's field" if typed else f"environment variable {spec.credential_env}" if from_env else "none")
    )

    _render_unfinished_jobs(runner, spec, store, credential, ground_truth)
    _render_document_authorization(document, source, store)

    try:
        preview = runner.preview(spec, workspace_id, document, source, page_number)
    except APIRunRefused as exc:
        st.error(f"Hosted API unavailable for this document: {exc}")
        return
    st.subheader("Exactly what will be transmitted")
    left, right = st.columns([1, 2])
    with left:
        st.image(preview["image_path"], caption=f"page_{page_number:04d}.png", width="stretch")
    with right:
        st.json({key: value for key, value in preview.items() if key != "image_path"}, expanded=True)
    sha = preview["file"]["sha256"]
    pending = open_journals_for_image(spec.id, sha, store.root)
    if pending:
        st.warning(
            f"This exact page image already has an unfinished job ({pending[0].state}, job "
            f"{pending[0].job_id}). Resume or abandon it above; it is never submitted twice."
        )
    confirm = st.checkbox(
        f"I confirm sending page {page_number} of {document.display_name} "
        f"(PNG, {preview['file']['bytes']:,} bytes, SHA-256 {sha[:16]}…) to {spec.provider}.",
        key=f"api-confirm-{spec.id}-{document.doc_id}-{page_number}-{sha}",
    )
    inflight_key = f"api-inflight-{spec.id}-{sha}"
    send = st.button(
        "Send this one page to the API",
        disabled=not (confirm and credential) or bool(pending) or st.session_state.get(inflight_key, False),
    )
    if send:
        st.session_state[inflight_key] = True  # a rerun mid-request cannot click again
        try:
            with st.spinner(f"Waiting for {spec.provider} (one page)..."):
                outcome = runner.run(
                    spec, workspace_id, document, source, page_number,
                    credential=credential, confirmed_sha256=sha, ground_truth=ground_truth,
                )
        except APIRunRefused as exc:
            st.error(f"Refused before sending: {exc}")
            return
        finally:
            st.session_state[inflight_key] = False
        _report_outcome(outcome)


def _report_outcome(outcome) -> None:
    """Remember the outcome and rerun, so every panel shows the new job state at once."""
    if outcome.run is not None:
        st.session_state.e2_selected_run = outcome.run.run_id
    if outcome.state == "COMPLETED":
        flash = ("success", f"Saved write-once API run {outcome.journal.journal_id} at {outcome.directory}")
    elif outcome.state == "AWAITING_RESULT_HOST_APPROVAL":
        host = (outcome.journal.latest().get("result_url") or {}).get("host")
        flash = ("warning",
                 f"The job finished, but its result is on host {host!r}, which you have not approved. "
                 "Nothing was downloaded and nothing will be resubmitted. Review it under Unfinished jobs.")
    elif outcome.state in {"RESUMABLE_ERROR", "SUBMISSION_UNCONFIRMED"}:
        flash = ("warning", f"Job paused ({outcome.state}). See Unfinished jobs to resume or abandon it.")
    else:
        error = outcome.run.error if outcome.run else outcome.state
        flash = ("error", f"API job ended: {error}. The run was saved at {outcome.directory}.")
    st.session_state["api_flash"] = flash
    st.rerun()


def _show_flash() -> None:
    flash = st.session_state.pop("api_flash", None)
    if flash:
        {"success": st.success, "warning": st.warning, "error": st.error}[flash[0]](flash[1])


def _render_host_approvals(runner, spec, store) -> None:
    hosts = sorted(set(runner.result_hosts(spec)) - {host.lower() for host in spec.result_hosts})
    if not hosts:
        return
    with st.expander(f"Approved result hosts ({len(hosts)})"):
        for host in hosts:
            st.text(host)
        with st.form(f"api-revoke-host-{spec.id}"):
            host = st.selectbox("Host to revoke", hosts)
            by = st.text_input("Your name", key=f"api-revoke-by-{spec.id}")
            revoke = st.form_submit_button("Revoke approval")
        if revoke:
            try:
                revoke_result_host(spec, host=host, revoked_by=by, root=store.root)
                st.rerun()
            except JobJournalError as exc:
                st.error(str(exc))


def _render_document_authorization(document, source, store) -> None:
    """Exact-hash document authorization. The runner enforces it again server-side."""
    sha = document.sha256
    state = documents.authorization_state(sha, root=store.root)
    entry = documents.load_registry().get(sha)
    with st.expander("Document authorization (exact SHA-256)", expanded=not state["allowed"]):
        st.text(f"Document: {document.display_name}\nSHA-256: {sha}")
        if entry is None:
            st.error(
                "Not eligible: this exact hash is not in lab/config/api_synthetic_documents.toml. "
                "Its name or folder does not matter; it can never be sent."
            )
            return
        st.text(f"Registry entry: {entry.id} ({entry.page_count} pages, {entry.bytes:,} bytes)\n{entry.title}\n"
                f"Category: {documents.CATEGORY_LABELS[entry.authorization_category]}\n"
                f"Provenance: {entry.provenance}")
        if entry.authorization_category == documents.CATEGORY_USER_OWNED:
            st.info(
                "This is NOT certified as synthetic. Authorize it only if the declaration below is true for "
                "every page: review the pages first. If you cannot make it truthfully, leave it unauthorized "
                "and nothing from this document can be sent."
            )
        if state["allowed"]:
            auth = state["authorization"]
            st.success(f"Authorized by {auth.get('reviewer')} until {auth.get('expires_at')}")
            with st.form(f"doc-revoke-{sha}"):
                by = st.text_input("Your name", key=f"doc-revoke-by-{sha}")
                reason = st.text_input("Reason", key=f"doc-revoke-reason-{sha}")
                revoke = st.form_submit_button("Revoke this document's authorization")
            if revoke:
                try:
                    documents.revoke_document(sha, revoked_by=by, reason=reason, root=store.root)
                    st.rerun()
                except documents.DocumentAuthorizationError as exc:
                    st.error(str(exc))
            return
        st.warning("Not authorized: " + "; ".join(state["reasons"]))
        with st.form(f"doc-authorize-{sha}"):
            reviewer = st.text_input("Reviewer name", key=f"doc-auth-reviewer-{sha}")
            typed = st.text_input("Type the first 12 characters of the SHA-256 above", key=f"doc-auth-sha-{sha}")
            days = st.number_input("Valid for (days)", 1, documents.MAX_AUTHORIZATION_DAYS, 7, key=f"doc-auth-days-{sha}")
            agree = st.checkbox(entry.declaration, key=f"doc-auth-agree-{sha}")
            approve = st.form_submit_button("Authorize this exact document")
        if approve:
            if not agree or typed.strip().lower() != sha[:12]:
                st.error("Tick the declaration and type the first 12 characters of the SHA-256 exactly.")
                return
            try:
                documents.authorize_document(entry, file_path=source, reviewer=reviewer,
                                             declaration=entry.declaration,
                                             days_valid=int(days), root=store.root)
                st.rerun()
            except documents.DocumentAuthorizationError as exc:
                st.error(str(exc))


def _render_unfinished_jobs(runner, spec, store, credential, ground_truth) -> None:
    _render_host_approvals(runner, spec, store)
    unfinished = [item for item in runner.journals(spec) if item.open]
    if not unfinished:
        return
    st.subheader(f"Unfinished jobs ({len(unfinished)}) - resumed from their Job ID, never resubmitted")
    for journal in unfinished:
        summary = journal.summary()
        with st.expander(
            f"{summary['state']} · page {summary['page']} · job {summary['job_id']} · "
            f"image {str(summary['image_sha256'])[:12]}…",
            expanded=True,
        ):
            st.json({key: value for key, value in summary.items() if key != "latest"}, expanded=False)
            host = summary["awaiting_host"]
            journal_hosts = runner.result_hosts(spec, journal.journal_id)
            if host:
                result_url = journal.latest().get("result_url") or {}
                st.error(f"Result host awaiting your review: {host}")
                st.text(
                    f"Path: {result_url.get('path')}\nURL expires: {result_url.get('expires_at') or 'not stated'}\n"
                    "Query values are never stored (signatures)."
                )
                if host in journal_hosts:
                    st.success("Approved for this job for up to 24 hours. New jobs require new approval.")
                else:
                    with st.form(f"api-approve-{journal.journal_id}"):
                        approver = st.text_input("Your name")
                        typed_host = st.text_input("Type the hostname exactly to approve it")
                        agree = st.checkbox(HOST_APPROVAL_STATEMENT)
                        approve = st.form_submit_button("Approve this result host")
                    if approve:
                        if not agree or typed_host.strip().lower() != host:
                            st.error("Tick the statement and type the exact hostname.")
                        else:
                            try:
                                approve_result_host(spec, host=typed_host, approved_by=approver,
                                                    statement=HOST_APPROVAL_STATEMENT, journal=journal,
                                                    root=store.root)
                                st.rerun()
                            except JobJournalError as exc:
                                st.error(str(exc))
            can_resume = (
                summary["state"] in RESUMABLE and bool(summary["job_id"])
                and (not host or host in journal_hosts)
            )
            key = f"api-resume-inflight-{journal.journal_id}"
            if st.button(
                "Resume this job (no resubmission)", key=f"api-resume-{journal.journal_id}",
                disabled=not (can_resume and credential) or st.session_state.get(key, False),
            ):
                st.session_state[key] = True
                try:
                    with st.spinner("Resuming from the existing Job ID..."):
                        outcome = runner.resume(spec, journal.journal_id, credential=credential,
                                                ground_truth=ground_truth)
                    _report_outcome(outcome)
                except APIRunRefused as exc:
                    st.error(f"Resume refused: {exc}")
                finally:
                    st.session_state[key] = False
            with st.form(f"api-abandon-{journal.journal_id}"):
                by = st.text_input("Your name", key=f"api-abandon-by-{journal.journal_id}")
                reason = st.text_input("Reason", key=f"api-abandon-reason-{journal.journal_id}")
                abandon = st.form_submit_button("Abandon this job")
            if abandon:
                try:
                    runner.abandon(spec, journal.journal_id, by=by, reason=reason)
                    st.rerun()
                except APIRunRefused as exc:
                    st.error(str(exc))
