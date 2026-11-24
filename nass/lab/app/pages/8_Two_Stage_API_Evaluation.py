"""E2-D: two-stage hosted API evaluation dashboard (EXPERIMENTAL, user-operated).

Nothing on this page sends anything by itself. A stage runs only after the user has
(1) authorized each exact document hash, (2) enabled each provider, (3) entered a
session credential, (4) approved the exact stage plan with spending limits, and
(5) pressed that stage's Run button. Stage 2 is never started automatically.
"""

from __future__ import annotations

import html
import os
import re
import threading
from pathlib import Path

import streamlit as st

from lab.app.api_panel import _render_document_authorization
from lab.core.e2.api import batch as batch_mod
from lab.core.e2.api import batch_export
from lab.core.e2.api import batch_report
from lab.core.e2.api import documents
from lab.core.e2.api import host_policy
from lab.core.e2.api import jobs as job_store
from lab.core.e2.api.authorization import (
    REQUIRED_STATEMENT,
    AuthorizationError,
    check_authorization,
    latest_enablement,
    record_enablement,
    usage_policy,
)
from lab.core.e2.api.jobs import HOST_APPROVAL_HOURS, HOST_APPROVAL_STATEMENT, JobJournalError
from lab.core.e2.api.providers import load_providers
from lab.core.e2.api.runner import APIEvaluationRunner, APIRunRefused
from lab.core.e2.store import ResultStore
from lab.core.ingest import IngestService
from lab.core.settings import load_settings
from lab.core.workspace import WorkspaceManager

STATUS_ICON = {
    batch_mod.COMPLETED: "✅", batch_mod.REUSED: "♻️", batch_mod.PENDING: "⏳", batch_mod.IN_PROGRESS: "🔄",
    batch_mod.NEEDS_HOST_APPROVAL: "✋", batch_mod.READY_TO_RESUME: "▶️", batch_mod.NEEDS_DECISION: "❓",
    batch_mod.FAILED: "❌", batch_mod.ABANDONED: "🚫", batch_mod.REFUSED: "⛔", batch_mod.BLOCKED: "🧱",
    batch_mod.INTERRUPTED_BEFORE_SEND: "⚠️", batch_mod.RETRY_AUTHORIZED: "🔁",
    batch_mod.INCOMPLETE_ARTIFACT: "⚠️",
}


@st.cache_resource
def _threads() -> dict[str, threading.Thread]:
    """Background executions of this server process (credentials live only in them)."""
    return {}


def _table(rows: list[dict], limit: int = 400) -> None:
    """Markdown table (no pandas/numpy: Smart App Control blocks numpy here)."""
    if not rows:
        st.caption("(none)")
        return
    columns = list(rows[0])

    def cell(value) -> str:
        text = "" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))
        text = " ".join(text.split())[:300]
        # Model output can appear here: escape every Markdown control character.
        return _MD_SPECIAL.sub(lambda match: "\\" + match.group(0), text)

    lines = ["| " + " | ".join(cell(c) for c in columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(row.get(c)) for c in columns) + " |" for row in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"\n_{len(rows) - limit} more rows not shown; see the exports._")
    st.markdown("\n".join(lines))


_MD_SPECIAL = re.compile(r"[\`*_{}\[\]()<>#+\-.!|~$]")


def _flash() -> None:
    message = st.session_state.pop("e2d_flash", None)
    if message:
        {"success": st.success, "warning": st.warning, "error": st.error}[message[0]](message[1])


def _set_flash(kind: str, text: str) -> None:
    st.session_state["e2d_flash"] = (kind, text)


st.set_page_config(page_title="Two-stage API evaluation - Extraction Lab", layout="wide")
settings = load_settings()
manager = WorkspaceManager(settings)
if "workspace_id" not in st.session_state:
    st.warning("Create a workspace on Home first.")
    st.stop()
workspace_id = st.session_state.workspace_id
ingest = IngestService(manager, settings)
store = ResultStore()
root = store.root

definition, definition_sha = batch_mod.load_definition()
st.title("E2-D two-stage hosted API evaluation")
st.caption(definition.title)
st.error(
    "EXPERIMENTAL, outside the air-gapped system: approved page images are sent over the internet to "
    "the hosted providers you select. You are the only person who can start a stage. Nothing runs "
    "until every gate below is satisfied and you press Run."
)
_flash()

# ------------------------------------------------------------------ documents
workspace_pdfs = [
    (item, ingest.original_path(workspace_id, item.doc_id))
    for item in ingest.list_documents(workspace_id) if item.kind == "pdf"
]
sources, problems = batch_mod.resolve_sources(definition, workspace_pdfs)
registry = {entry.id: entry for entry in documents.load_registry().values()}

st.header("1. Documents (matched by exact SHA-256 only)")
st.warning(
    "These reports are registered as user-owned test documents. They are NOT certified as synthetic "
    "(Word exports without a synthetic marking), so the synthetic declaration is refused for them. "
    "Authorize a document only if the user-owned declaration is true for every page; otherwise leave it "
    "blocked."
)
for wanted in definition.documents:
    entry = registry.get(wanted.registry_id)
    source = sources.get(wanted.registry_id)
    with st.container(border=True):
        if entry is None:
            st.error(f"{wanted.registry_id}: not in the registry")
            continue
        state = documents.authorization_state(entry.sha256, root=root)
        st.markdown(
            f"**{wanted.language}** · `{entry.id}` · {entry.page_count} pages · {entry.bytes:,} bytes  \n"
            f"SHA-256 `{entry.sha256}`  \n"
            f"In this workspace: {'yes' if source else 'NO - upload it on Documents'} · "
            f"Eligible: yes (registry) · Authorization: "
            + (f"valid until {state['authorization'].get('expires_at')}" if state["allowed"]
               else "; ".join(state["reasons"]))
        )
        if source:
            _render_document_authorization(source.document, source.pdf_path, store)
for problem in problems:
    st.info(problem)

# ------------------------------------------------------------------ providers
specs = {item.id: item for item in load_providers()}
st.header("2. Providers and credentials")
option_ids = [item for item in definition.options if item in specs]
selected_providers = st.multiselect(
    "Providers for new plans (each has its own credential; keys are never shared between providers)",
    option_ids, default=[item for item in definition.providers if item in option_ids],
    format_func=lambda item: specs[item].label, key="e2d-providers",
)
credentials: dict[str, str] = {}
columns = st.columns(max(len(selected_providers), 1))
for column, provider_id in zip(columns, selected_providers):
    spec = specs.get(provider_id)
    with column, st.container(border=True):
        if spec is None:
            st.error(f"{provider_id} is not in the provider registry")
            continue
        st.markdown(f"**{spec.label}**  \n{spec.classification} · status `{spec.status}`  \n"
                    f"Destination `{spec.endpoint}`")
        state = check_authorization(spec, pages=1, root=root)
        latest = latest_enablement(spec, root=root) or {}
        if state.allowed:
            st.success(f"Enabled by {latest.get('enabled_by')} until {latest.get('expires_at')}. "
                       f"{state.describe()}. History (all enablements): {state.pages_used_total} page(s)")
        else:
            st.warning("Not ready: " + "; ".join(state.reasons))
        with st.expander("Enable / disable this provider", expanded=False):
            with st.form(f"e2d-enable-{spec.id}"):
                by = st.text_input("Your name", key=f"e2d-enable-by-{spec.id}")
                policy = usage_policy(spec)
                st.caption(f"Limit source: {policy['source']}. {policy['provider_quota']}.")
                total = st.number_input(f"Maximum pages for this enablement (ceiling {policy['max_pages_per_enablement']})",
                                        1, policy["max_pages_per_enablement"], min(40, policy["max_pages_per_enablement"]),
                                        key=f"e2d-enable-total-{spec.id}")
                per_day = st.number_input(f"Maximum pages per day (ceiling {policy['max_pages_per_day_ceiling']})",
                                          1, policy["max_pages_per_day_ceiling"],
                                          min(40, policy["max_pages_per_day_ceiling"]), key=f"e2d-enable-day-{spec.id}")
                days = st.number_input("Valid for (days)", 1, 30, policy["default_enablement_days"],
                                       key=f"e2d-enable-days-{spec.id}")
                agree = st.checkbox(REQUIRED_STATEMENT, key=f"e2d-enable-agree-{spec.id}")
                enable = st.form_submit_button("Enable this provider")
            if enable:
                if not agree:
                    st.error("Tick the statement to enable the provider.")
                else:
                    try:
                        record_enablement(spec, enabled=True, enabled_by=by, statement=REQUIRED_STATEMENT,
                                          max_pages_total=int(total), max_pages_per_day=int(per_day),
                                          days_valid=int(days), root=root)
                        st.rerun()
                    except AuthorizationError as exc:
                        st.error(str(exc))
            if latest.get("enabled") and st.button("Disable this provider", key=f"e2d-disable-{spec.id}"):
                record_enablement(spec, enabled=False, enabled_by=latest.get("enabled_by") or "user", statement="",
                                  max_pages_total=0, max_pages_per_day=0, days_valid=1, root=root)
                st.rerun()
        typed = st.text_input(f"{spec.credential_label} (session memory only; never stored)", type="password",
                              key=f"e2d-credential-{spec.id}")
        from_env = os.environ.get(spec.credential_env, "")
        credentials[spec.id] = typed or from_env
        st.caption("Credential: " + ("entered in this session" if typed else
                                     f"from {spec.credential_env}" if from_env else "missing"))


# ---------------------------------------------------------------------- stages


def _plan_table(plan: dict) -> list[dict]:
    return [{
        "document": item["registry_id"], "page": item["page"], "role": item.get("page_role") or "",
        "content (heuristic)": ", ".join((item.get("page_features") or {}).get("heuristic_content") or []),
        "provider": item["provider_id"], "destination": item["destination"],
        "image SHA-256": item["image_sha256"][:16] + "…", "PNG bytes": item["image_bytes"],
        "action": item["action"], "max USD": item["max_cost_usd"], "note": item.get("reason") or "",
    } for item in plan["items"]]


def _start(batch_dir: Path, by: str) -> None:
    registry_threads = _threads()
    key = str(batch_dir)
    current = registry_threads.get(key)
    if current is not None and current.is_alive():
        _set_flash("warning", "This stage is already running.")
        return
    batch_mod.request_control(batch_dir, "RUN", by=by)
    runner = APIEvaluationRunner(manager, settings, store)
    executor = batch_mod.BatchRunner(runner, specs)
    session_credentials = {key_: value for key_, value in credentials.items() if value}

    def work() -> None:
        try:
            executor.execute(batch_dir, credentials=session_credentials, sources=sources)
        except (batch_mod.BatchError, APIRunRefused) as exc:
            batch_mod._record(batch_dir, "EXECUTION_REFUSED", reason=str(exc))
        except Exception as exc:  # noqa: BLE001 - recorded without credentials
            batch_mod._record(batch_dir, "EXECUTION_ERROR", error=type(exc).__name__)

    thread = threading.Thread(target=work, name=f"e2d-{batch_dir.name}", daemon=True)
    registry_threads[key] = thread
    thread.start()


def _render_progress(batch_dir: Path, plan: dict) -> None:
    status = batch_mod.batch_status(batch_dir, root)
    st.progress(status["done"] / max(status["total"], 1),
                text=f"{status['overall']} · {status['done']}/{status['total']} items done · control {status['control']}")
    approval = batch_mod.approval_state(batch_dir)["approval"] or {}
    limits = approval.get("limits") or {}
    cols = st.columns(len(status["providers"]) or 1)
    for col, (provider_id, info) in zip(cols, status["providers"].items()):
        limit = limits.get(provider_id) or {}
        col.metric(provider_id, f"{info['done']}/{info['total']}",
                   help="done (completed or reused) / planned")
        col.caption(f"Committed {info['committed_usd']:.4f} USD of budget {limit.get('budget_usd', 0):.4f}; "
                    f"submissions {info['submissions']}/{limit.get('max_submissions', 0)}")
    _table([{
        "": STATUS_ICON.get(item["status"], ""), "status": item["status"], "document": item["registry_id"],
        "page": item["page"], "provider": item["provider_id"], "attempts": item["attempts"],
        "job": item["job_id"] or "", "cost USD": item["cost_usd"],
        "saved": "reusable" if item["reusable_saved_result"] else (
            "complete" if item["artifact_complete"] else "not complete"),
        "artifact path": item["artifact_directory"] or "", "detail": item["detail"] or "",
    } for item in status["items"]])
    events = [event for event in batch_mod.control_events(batch_dir)
              if event["state"] in {"EXECUTION_REFUSED", "EXECUTION_ERROR", "EXECUTION_ENDED"}]
    if events:
        latest = events[-1]
        if latest["state"] != "EXECUTION_ENDED":
            st.error(f"Last execution: {latest['state']}: {latest.get('reason') or latest.get('error')}")
        else:
            stopped = {key: value["stopped"] for key, value in (latest.get("providers") or {}).items() if value["stopped"]}
            if stopped:
                st.warning("Last execution stopped submitting to: " +
                           "; ".join(f"{key}: {value}" for key, value in stopped.items()))


def _result_url_text(info: dict) -> str:
    """The sanitized result URL as recorded (query values were never stored)."""
    if not info.get("host"):
        return "not recorded"
    port = f":{info['port']}" if info.get("port") else ""
    query = "&".join(f"{name}=<omitted>" for name in info.get("query_parameter_names") or [])
    return f"{info.get('scheme', 'https')}://{info['host']}{port}{info.get('path') or ''}" + (f"?{query}" if query else "")


def _batch_journals(batch_dir: Path) -> set[str]:
    return {item["journal_id"] for item in batch_mod.batch_status(batch_dir, root)["items"] if item["journal_id"]}


def _render_host_policy(batch_dir: Path, plan: dict) -> None:
    """Opt-in per-batch automatic approval of Baidu result hosts (OFF by default)."""
    for provider in plan["providers"]:
        spec = specs.get(provider["provider_id"])
        if spec is None or not host_policy.supported(spec):
            continue
        state = host_policy.policy_state(batch_dir, spec)
        with st.expander(f"Trust approved Baidu result hosts for this batch ({spec.id})", expanded=False):
            st.markdown(
                "**OFF by default.** When on, a paused Baidu job of this batch gets its own job-scoped result-host "
                "approval automatically, but only for an HTTPS result URL (default port, `/v1/job/` path) on a "
                "public host that is listed in the registry, was approved by you for another job of this batch, "
                f"or matches `{host_policy.BAIDU_RESULT_HOST_PATTERN}`. Anything else still needs your manual "
                "approval. Downloads keep every existing check (HTTPS only, exact approved host, redirects "
                "re-validated, non-public resolved addresses refused). Submission, budgets and Oxen are unaffected."
            )
            st.caption("Registry-listed hosts: " + (", ".join(f"`{host}`" for host in spec.auto_result_hosts)
                                                    or "none"))
            record = state["record"] or {}
            if state["active"]:
                st.success(
                    f"ON for this batch: enabled by {record.get('enabled_by')} until {record.get('expires_at')}; "
                    + ("includes jobs that were already pending" if record.get("include_pending")
                       else "only jobs that pause after it was enabled")
                    + f". Automatic approvals so far: {len(host_policy.auto_approvals(batch_dir))}."
                )
                with st.form(f"e2d-hostpolicy-off-{plan['batch_id']}-{spec.id}"):
                    by = st.text_input("Your name", key=f"e2d-hostpolicy-offby-{plan['batch_id']}-{spec.id}")
                    off = st.form_submit_button("Turn automatic approval OFF")
                if off:
                    try:
                        host_policy.revoke_policy(batch_dir, spec, revoked_by=by)
                        st.rerun()
                    except host_policy.HostPolicyError as exc:
                        st.error(str(exc))
            else:
                st.info("Status: " + "; ".join(state["reasons"]))
                with st.form(f"e2d-hostpolicy-on-{plan['batch_id']}-{spec.id}"):
                    by = st.text_input("Your name", key=f"e2d-hostpolicy-by-{plan['batch_id']}-{spec.id}")
                    hours = st.number_input("Valid for (hours)", 1, host_policy.MAX_POLICY_HOURS, 24,
                                            key=f"e2d-hostpolicy-hours-{plan['batch_id']}-{spec.id}")
                    pending = st.checkbox(
                        "Also apply to jobs of this batch that are already waiting for approval",
                        value=False, key=f"e2d-hostpolicy-pending-{plan['batch_id']}-{spec.id}")
                    agree = st.checkbox(host_policy.POLICY_STATEMENT,
                                        key=f"e2d-hostpolicy-agree-{plan['batch_id']}-{spec.id}")
                    on = st.form_submit_button("Turn automatic approval ON for this batch")
                if on:
                    try:
                        host_policy.enable_policy(
                            batch_dir, spec, enabled_by=by, hours_valid=int(hours), include_pending=bool(pending),
                            statement=host_policy.POLICY_STATEMENT if agree else "")
                        _set_flash("success", "Automatic Baidu result-host approval is ON for this batch. "
                                              "It takes effect when you press Run/Resume.")
                        st.rerun()
                    except host_policy.HostPolicyError as exc:
                        st.error(str(exc))


def _render_host_approvals(batch_dir: Path, stage: int, waiting: list[dict]) -> None:
    st.subheader(f"Approval needed: result-storage host ({len(waiting)} job(s))")
    st.info(
        "These jobs finished at the provider, but each result is stored on a host that is not in the registry. "
        "Nothing was downloaded and nothing will be resubmitted. Approve each job separately: the approval "
        f"names only that job and that exact host and expires after {HOST_APPROVAL_HOURS} h. When the jobs "
        "you want are approved, press Resume: it downloads the results using the existing Job IDs."
    )
    journals: set[str] | None = None
    for item in waiting:
        spec = specs.get(item["provider_id"])
        host = item["awaiting_host"] or ""
        info = item.get("result_url") or {}
        key = f"{stage}-{item['journal_id']}"
        with st.container(border=True):
            st.markdown(
                f"**{item['registry_id']} · page {item['page']} · `{item['provider_id']}`**\n\n"
                f"- Job ID: `{item['job_id']}` (run `{item['journal_id']}`)\n"
                f"- Reported result host: `{host}`\n"
                f"- Result URL (HTTPS, signature omitted): `{_result_url_text(info)}`\n"
                f"- Result link expires: {info.get('expires_at') or 'not stated'} (refreshed from the job on resume)"
            )
            if spec is None:
                st.error("Provider not in the registry; cannot approve.")
                continue
            if info.get("scheme") not in (None, "https"):
                st.error("The reported result URL is not HTTPS; it cannot be approved.")
                continue
            if host_policy.supported(spec) and host_policy.policy_state(batch_dir, spec)["active"]:
                journals = journals if journals is not None else _batch_journals(batch_dir)
                checked = host_policy.check_job(
                    batch_dir, spec, job_store.JobJournal(job_store.journal_dir(spec.id, item["journal_id"], root)),
                    batch_journals=journals, root=root)
                if checked["eligible"]:
                    st.info(f"Automatic policy: will be approved for this job on Resume ({checked['basis']}).")
                else:
                    st.caption("Automatic policy does not apply (manual approval needed): "
                               + "; ".join(checked["reasons"]))
            if batch_mod.item_host_approved(spec, item, root):
                st.success(f"`{host}` is approved for this job only. Press Resume to download the result.")
                continue
            with st.form(f"e2d-hostjob-{key}"):
                by = st.text_input("Your name", key=f"e2d-hostjob-by-{key}")
                typed = st.text_input("Type the hostname exactly", key=f"e2d-hostjob-typed-{key}")
                job = st.text_input("Type the Job ID exactly", key=f"e2d-hostjob-job-{key}")
                agree = st.checkbox(HOST_APPROVAL_STATEMENT, key=f"e2d-hostjob-agree-{key}")
                approve = st.form_submit_button(f"Approve {host} for job {item['job_id']} only")
            if approve:
                if not agree:
                    st.error("Tick the statement to approve this host for this job.")
                    continue
                try:
                    batch_mod.approve_item_result_host(
                        batch_dir, spec, item["item_id"], host=typed, job_id=job, approved_by=by,
                        statement=HOST_APPROVAL_STATEMENT, root=root)
                    _set_flash("success", f"Approved {host} for job {item['job_id']} only. Press Resume when ready.")
                    st.rerun()
                except (batch_mod.BatchError, JobJournalError) as exc:
                    st.error(str(exc))


def _render_user_actions(batch_dir: Path, stage: int) -> None:
    status = batch_mod.batch_status(batch_dir, root)
    if status["active"]:
        return
    waiting = [item for item in status["items"] if item["status"] == batch_mod.NEEDS_HOST_APPROVAL]
    if waiting:
        _render_host_approvals(batch_dir, stage, waiting)
    undecided = [item for item in status["items"] if item["status"] == batch_mod.NEEDS_DECISION]
    for item in undecided:
        with st.expander(f"❓ Decision needed: {item['item_id']}", expanded=True):
            st.warning(item["detail"])
            facts = item.get("diagnostics") or {}
            if facts:
                elapsed = facts.get("stage_elapsed_ms") or {}
                st.markdown(
                    f"- Run `{item['journal_id']}` · provider `{item['provider_id']}` · "
                    f"{item['registry_id']} page {item['page']}\n"
                    f"- Error: `{facts.get('code')}`: {facts.get('message')}\n"
                    f"- Stalled stage: `{facts.get('stalled_stage') or 'unknown'}`; upload completed: "
                    f"{'yes' if facts.get('upload_completed') else 'no/unknown'}\n"
                    "- Stage times (ms): " + ", ".join(
                        f"{name} {value:.0f}" for name, value in elapsed.items() if isinstance(value, (int, float)))
                    + f"\n- HTTP status: {facts.get('http_status')}; response bytes: {facts.get('response_bytes')}; "
                    f"request id: {facts.get('request_id_headers') or 'none'}; streamed characters: "
                    f"{facts.get('streamed_characters') or 0}"
                )
                st.caption(
                    "The upload reached the provider, so it may have been processed or billed. Check your provider "
                    "dashboard first. Abandoning records your decision only; a new attempt needs a separate, "
                    "explicit retry authorization afterwards."
                )
            with st.form(f"e2d-abandon-{item['item_id']}-{item['journal_id']}"):
                by = st.text_input("Your name", key=f"e2d-abandon-by-{item['journal_id']}")
                reason = st.text_input("Reason (e.g. checked the provider dashboard)",
                                       key=f"e2d-abandon-reason-{item['journal_id']}")
                abandon = st.form_submit_button("Abandon this job (nothing is resent)")
            if abandon:
                try:
                    APIEvaluationRunner(manager, settings, store).abandon(
                        specs[item["provider_id"]], item["journal_id"], by=by, reason=reason)
                    st.rerun()
                except APIRunRefused as exc:
                    st.error(str(exc))
    retryable = [item for item in status["items"] if item["status"] in {batch_mod.FAILED, batch_mod.ABANDONED}]
    for item in retryable:
        with st.expander(f"{STATUS_ICON[item['status']]} {item['status']}: {item['item_id']}"):
            st.text(item["detail"] or "")
            with st.form(f"e2d-retry-{item['item_id']}-{item['journal_id']}"):
                by = st.text_input("Your name", key=f"e2d-retry-by-{item['journal_id']}")
                agree = st.checkbox(batch_mod.RETRY_STATEMENT, key=f"e2d-retry-agree-{item['journal_id']}")
                retry = st.form_submit_button("Authorize one more attempt for this item")
            if retry:
                try:
                    batch_mod.authorize_item_retry(batch_dir, item["item_id"], by=by,
                                                   statement=batch_mod.RETRY_STATEMENT if agree else "", root=root)
                    st.rerun()
                except batch_mod.BatchError as exc:
                    st.error(str(exc))


def _render_stage(stage: int) -> None:
    label = "Stage 1: 6 representative pages per provider" if stage == 1 else "Stage 2: all pages per provider"
    st.subheader(label)
    if problems:
        st.info("Upload both documents to this workspace to plan a stage.")
        return
    existing = batch_mod.list_batches(root, definition.id, stage=stage)
    prior_dir = None
    if stage == 2:
        stage1 = batch_mod.list_batches(root, definition.id, stage=1)
        if stage1:
            names = ["(no reuse)"] + [path.name for path in stage1]
            choice = st.selectbox("Reuse exact-match results from Stage 1 batch", names, index=len(names) - 1,
                                  key="e2d-stage2-prior")
            prior_dir = None if choice == "(no reuse)" else next(path for path in stage1 if path.name == choice)
        else:
            st.caption("No Stage 1 batch exists; every Stage 2 page would be submitted.")
    if st.button(f"Create a new Stage {stage} plan (renders pages locally; sends nothing)", key=f"e2d-plan-{stage}"):
        try:
            with st.spinner("Rendering and hashing pages locally..."):
                plan = batch_mod.build_plan(definition, definition_sha, stage, sources=sources, specs=specs,
                                            manager=manager, settings=settings, workspace_id=workspace_id,
                                            root=root, stage1_batch=prior_dir, providers=selected_providers)
                batch_mod.save_plan(plan, root)
            _set_flash("success", f"Stage {stage} plan {plan['batch_id']} created. Review and approve it below.")
            st.rerun()
        except batch_mod.BatchError as exc:
            st.error(str(exc))
    existing = batch_mod.list_batches(root, definition.id, stage=stage)
    if not existing:
        st.info(f"No Stage {stage} plan yet.")
        return
    names = [path.name for path in existing]
    # The selected plan is kept in the page URL (a batch name, never a secret), so a
    # browser refresh (a new Streamlit session) reopens the same plan instead of
    # silently switching to the newest one. Credentials are never kept this way.
    param = f"stage{stage}_batch"
    remembered = st.query_params.get(param)
    default = names.index(remembered) if remembered in names else len(names) - 1
    chosen = st.selectbox(f"Stage {stage} plan", names, index=default, key=f"e2d-batch-{stage}")
    if st.query_params.get(param) != chosen:
        st.query_params[param] = chosen
    batch_dir = next(path for path in existing if path.name == chosen)
    try:
        plan = batch_mod.load_plan(batch_dir)
    except batch_mod.BatchError as exc:
        st.error(str(exc))
        return
    st.markdown(f"Plan SHA-256 `{plan['plan_sha256']}` · created {plan['created_at']} · providers "
                + ", ".join(f"`{item['provider_id']}`" for item in plan["providers"])
                + (f" · reuses Stage 1 `{plan['prior_stage_batch']}`" if plan.get("prior_stage_batch") else ""))
    cost_rows = [{
        "provider": provider["provider_id"], "submit": provider["submit_count"], "reuse": provider["reuse_count"],
        "blocked": provider["blocked_count"], "destination hosts": ", ".join(provider["destination_hosts"]),
        "typical USD (assumption)": provider["typical_total_cost_usd"], "maximum USD": provider["max_total_cost_usd"],
        "max per page USD": provider["max_cost_per_page_usd"],
    } for provider in plan["providers"]]
    _table(cost_rows)
    st.caption("Typical cost is an assumption (configured tokens or compute seconds); the enforced budget uses "
               "the maximum per page until the provider reports the actual charge, which then replaces it. "
               "Baidu reports no price (free daily quota). Oxen: bound = max billable seconds x rate; a reported "
               "charge above the bound stops submissions to that provider.")
    with st.expander(f"Exact pages and destinations ({len(plan['items'])} items)", expanded=stage == 1):
        _table(_plan_table(plan))
        st.caption("Never sent: " + "; ".join(plan["not_sent"]))

    approval = batch_mod.approval_state(batch_dir)
    with st.container(border=True):
        st.markdown(f"**Batch approval for Stage {stage}** (separate for each stage)")
        if approval["allowed"]:
            record = approval["approval"]
            st.success(f"Approved by {record['approved_by']} until {record['expires_at']}; limits "
                       + "; ".join(f"{key}: ≤{value['budget_usd']:.4f} USD, ≤{value['max_submissions']} submissions"
                                   for key, value in record["limits"].items()))
            with st.form(f"e2d-revoke-{plan['batch_id']}"):
                by = st.text_input("Your name", key=f"e2d-revoke-by-{plan['batch_id']}")
                reason = st.text_input("Reason", key=f"e2d-revoke-reason-{plan['batch_id']}")
                revoke = st.form_submit_button("Revoke this batch approval")
            if revoke:
                try:
                    batch_mod.revoke_plan_approval(batch_dir, revoked_by=by, reason=reason)
                    st.rerun()
                except batch_mod.BatchError as exc:
                    st.error(str(exc))
        else:
            st.warning("Not approved: " + "; ".join(approval["reasons"]))
            cap = batch_mod.MAX_BATCH_BUDGET_USD
            for provider in plan["providers"]:
                if provider["submit_count"] and provider["max_total_cost_usd"] > cap + 1e-9:
                    bound = provider["max_cost_per_page_usd"] or 0.0
                    fits = min(provider["submit_count"], int((cap + 1e-9) // bound)) if bound > 0 else 0
                    st.info(
                        f"**Staged execution for `{provider['provider_id']}`.** This plan's worst case is "
                        f"{provider['max_total_cost_usd']:.4f} USD, above the fixed safety cap of {cap:.2f} USD "
                        "per batch (the cap is not raised). Approve up to the cap: pages start only while the "
                        f"committed spend plus one page bound ({bound:.4f} USD) fits, so at most {fits} of "
                        f"{provider['submit_count']} pages can be sent in this batch if no lower actual charge is "
                        "reported. Unsent pages stay pending and are never sent above the limit. Afterwards, create "
                        "a new plan for this stage: every completed page is reused (no new request) and only the "
                        "remaining pages need a new, separate approval."
                    )
            with st.form(f"e2d-approve-{plan['batch_id']}"):
                by = st.text_input("Your name", key=f"e2d-approve-by-{plan['batch_id']}")
                typed = st.text_input("Type the first 12 characters of the plan SHA-256",
                                      key=f"e2d-approve-sha-{plan['batch_id']}")
                budgets = {}
                for provider in plan["providers"]:
                    if provider["submit_count"]:
                        budgets[provider["provider_id"]] = st.number_input(
                            f"Spending limit for {provider['provider_id']} (USD)", 0.0,
                            batch_mod.MAX_BATCH_BUDGET_USD,
                            float(round(min(provider["max_total_cost_usd"], batch_mod.MAX_BATCH_BUDGET_USD), 4)),
                            step=0.01, format="%.4f", key=f"e2d-budget-{plan['batch_id']}-{provider['provider_id']}")
                hours = st.number_input("Approval valid for (hours)", 1, batch_mod.MAX_APPROVAL_HOURS, 24,
                                        key=f"e2d-hours-{plan['batch_id']}")
                retries = st.number_input("Extra submissions allowed for explicitly authorized retries", 0,
                                          batch_mod.MAX_RETRY_ALLOWANCE, 0, key=f"e2d-retries-{plan['batch_id']}")
                agree = st.checkbox(batch_mod.BATCH_APPROVAL_STATEMENT, key=f"e2d-approve-agree-{plan['batch_id']}")
                submit = st.form_submit_button(f"Approve this exact Stage {stage} plan")
            if submit:
                try:
                    batch_mod.approve_plan(batch_dir, approved_by=by,
                                           statement=batch_mod.BATCH_APPROVAL_STATEMENT if agree else "",
                                           typed_plan_prefix=typed, budgets_usd=budgets, hours_valid=int(hours),
                                           retry_allowance=int(retries))
                    st.rerun()
                except batch_mod.BatchError as exc:
                    st.error(str(exc))

    _render_host_policy(batch_dir, plan)

    checks = batch_mod.stage_readiness(batch_dir, specs=specs, sources=sources, root=root,
                                       credentials_present={key: bool(value) for key, value in credentials.items()})
    ready = all(check["ok"] for check in checks)
    with st.expander("Readiness checklist", expanded=not ready):
        for check in checks:
            st.markdown(f"{'✅' if check['ok'] else '❌'} **{check['check']}**: {check['detail']}")

    status = batch_mod.batch_status(batch_dir, root)
    started = status["overall"] != "NOT_STARTED"
    operator = st.text_input("Operator name (recorded with Run/Pause/Stop)", key=f"e2d-operator-{stage}")
    buttons = st.columns(4)
    run_label = f"Run Stage {stage}" if not started else f"Resume Stage {stage}"
    if buttons[0].button(run_label, type="primary", disabled=not (ready and operator.strip()) or status["active"],
                         key=f"e2d-run-{stage}"):
        _start(batch_dir, operator)
        st.rerun()
    if buttons[1].button("Pause", disabled=not status["active"], key=f"e2d-pause-{stage}"):
        batch_mod.request_control(batch_dir, "PAUSE", by=operator or "operator")
        _set_flash("warning", "Pausing: requests already in flight finish (they cannot be recalled); no new page starts.")
        st.rerun()
    if buttons[2].button("Stop", disabled=not status["active"], key=f"e2d-stop-{stage}"):
        batch_mod.request_control(batch_dir, "STOP", by=operator or "operator")
        _set_flash("warning", "Stopping: in-flight requests finish; nothing new is sent until you press Resume.")
        st.rerun()
    buttons[3].button("Refresh", key=f"e2d-refresh-{stage}")
    if not ready:
        st.caption("Run stays disabled until every readiness check passes and you enter your name.")

    was_active = status["active"]

    @st.fragment(run_every=2 if was_active else None)
    def live() -> None:
        # Only this fragment refreshes while a stage runs. When execution stops, redraw
        # the whole page once so the approval/decision controls and the Resume button
        # (rendered outside the fragment) reflect the new state.
        if was_active and not batch_mod.executor_active(batch_dir):
            st.rerun()
        _render_progress(batch_dir, plan)
        if was_active:
            st.caption("Approval and decision controls appear here when execution stops.")

    live()
    _render_user_actions(batch_dir, stage)


st.header("3. Stages")
stage_tabs = st.tabs(["Stage 1", "Stage 2", "Results, review and export"])
with stage_tabs[0]:
    _render_stage(1)
with stage_tabs[1]:
    st.info("Stage 2 has its own plan, approval, budget and Run button. It never starts automatically.")
    _render_stage(2)

# ------------------------------------------------------ all pages + exports
_TEXT_BOX = ("white-space:pre-wrap; unicode-bidi:plaintext; max-height:480px; overflow-y:auto; "
             "border:1px solid rgba(128,128,128,.35); border-radius:6px; padding:8px; font-size:0.9rem; "
             "line-height:1.5;")


def _render_page_image(registry_id: str, page: int) -> Path | None:
    """Fallback only: local render (same as the single-page view) when no saved image exists."""
    source = sources.get(registry_id)
    if source is None:
        return None
    return Path(batch_mod.render_page(manager, settings, workspace_id, source, page).path)


def _render_downloads_all(collected: dict) -> None:
    st.subheader("Download all results")
    counts = collected["counts"]
    st.caption(f"Built from existing saved results only ({counts['available']} of {counts['outputs']} outputs "
               f"available across {counts['pages']} pages); nothing is sent or written to the results store. "
               "Missing outputs are marked NOT AVAILABLE.")
    batch_id = collected["batch_id"]
    columns = st.columns(3)
    columns[0].download_button("Download all as CSV", batch_export.to_csv(collected),
                               file_name=f"e2d_all_pages_{batch_id}.csv", mime="text/csv",
                               key=f"e2d-all-csv-{batch_id}")
    zip_key, html_key = f"e2d-all-zip-{batch_id}", f"e2d-all-html-{batch_id}"
    with columns[1]:
        if st.button("Prepare ZIP (images + texts + metadata)", key=f"{zip_key}-build"):
            with st.spinner("Packing existing results..."):
                st.session_state[zip_key] = batch_export.to_zip(collected, _render_page_image)
        if st.session_state.get(zip_key):
            st.download_button("Download ZIP", st.session_state[zip_key], file_name=f"e2d_all_pages_{batch_id}.zip",
                               mime="application/zip", key=f"{zip_key}-download")
    with columns[2]:
        embed = st.checkbox("Embed page images (offline report)", value=True, key=f"{html_key}-embed")
        if st.button("Prepare HTML comparison report", key=f"{html_key}-build"):
            with st.spinner("Building the report..."):
                st.session_state[html_key] = batch_export.to_html(collected, _render_page_image, embed_images=embed)
        if st.session_state.get(html_key):
            st.download_button("Download HTML report", st.session_state[html_key],
                               file_name=f"e2d_comparison_{batch_id}.html", mime="text/html",
                               key=f"{html_key}-download")


def _render_all_pages(collected: dict) -> None:
    providers = collected["providers"]
    labels = {provider["provider_id"]: provider["label"] for provider in providers}
    documents_order = [document["registry_id"] for document in collected["documents"]]
    statuses = sorted({output["status"] for page in collected["pages"] for output in page["outputs"].values()})
    filters = st.columns(3)
    chosen_docs = filters[0].multiselect("Documents", documents_order, default=documents_order,
                                         key="e2d-all-docs")
    chosen_providers = filters[1].multiselect("Providers", [p["provider_id"] for p in providers],
                                              default=[p["provider_id"] for p in providers],
                                              format_func=lambda key: labels.get(key, key), key="e2d-all-providers")
    chosen_status = filters[2].multiselect("Result status", statuses, default=statuses, key="e2d-all-status")
    pages = [
        page for page in collected["pages"]
        if page["registry_id"] in chosen_docs and any(
            (page["outputs"].get(provider_id) or {}).get("status") in chosen_status
            for provider_id in chosen_providers)
    ]
    if not pages or not chosen_providers:
        st.info("No pages match these filters.")
        return
    per_screen = st.select_slider("Pages per screen", options=[3, 5, 10, 20, 31], value=5, key="e2d-all-per")
    screens = max(1, -(-len(pages) // per_screen))
    screen = int(st.number_input(f"Screen (1-{screens})", 1, screens, 1, key="e2d-all-screen"))
    screen = min(screen, screens)
    shown = pages[(screen - 1) * per_screen: screen * per_screen]
    st.caption(f"Showing pages {(screen - 1) * per_screen + 1}-{(screen - 1) * per_screen + len(shown)} of "
               f"{len(pages)} matching pages. Images are loaded only for the pages on this screen.")
    totals = {registry_id: sum(page["registry_id"] == registry_id for page in collected["pages"])
              for registry_id in documents_order}
    current = None
    for page in shown:
        if page["registry_id"] != current:
            current = page["registry_id"]
            st.subheader(f"{page['language']} Document — {totals.get(current, 0)} pages")
            st.caption(f"{current} · {page.get('document_title') or ''}")
        direction = batch_export.text_direction(page["language"])
        with st.container(border=True):
            st.markdown(f"**{page['registry_id']} · page {page['page']}**")
            columns = st.columns(1 + len(chosen_providers))
            with columns[0]:
                st.markdown("**Original PDF page**")
                image = batch_export.page_image(page, _render_page_image)
                if image is not None:
                    st.image(str(image), width="stretch")
                else:
                    st.warning("Original page image not available")
            for column, provider_id in zip(columns[1:], chosen_providers):
                output = page["outputs"].get(provider_id)
                with column:
                    st.markdown(f"**{labels.get(provider_id, provider_id)}**")
                    if output is None:
                        st.caption("Not part of this batch")
                    elif output["text"] is not None:
                        st.caption(f"{output['status']} · RAW output · {len(output['text'])} characters")
                        st.markdown(f'<div dir="{direction}" style="{_TEXT_BOX}">{html.escape(output["text"])}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.error(output["availability"])
                        if output.get("detail"):
                            st.caption(str(output["detail"]))


# ------------------------------------------------------------------- results
with stage_tabs[2]:
    all_batches = batch_mod.list_batches(root, definition.id)
    if not all_batches or problems:
        st.info("Results appear here after a stage has run (documents must be in this workspace).")
    else:
        labels = {f"Stage {batch_mod.load_plan(path)['stage']} · {path.name}": path for path in all_batches}
        chosen = st.selectbox("Batch", list(labels), index=len(labels) - 1, key="e2d-results-batch")
        batch_dir = labels[chosen]
        analysis = batch_report.analyze_batch(batch_dir, root, sources)
        rec = analysis["recommendation"]
        st.subheader(f"Recommendation: {rec['verdict']}")
        st.markdown(rec["guidance"])
        for note in analysis["notes"]:
            st.caption(note)
        _table([{"provider": key} | {k: v for k, v in value.items() if k != "by_category"}
                      for key, value in analysis["provider_summary"].items()])
        with st.expander("Evidence and limits"):
            for item in rec["evidence"] + rec["limits"]:
                st.markdown(f"- {item}")
        downloads = st.columns(3)
        downloads[0].download_button("Download JSON", batch_report.to_json(analysis),
                                     file_name=f"e2d_{analysis['batch_id']}.json", mime="application/json")
        downloads[1].download_button("Download CSV", batch_report.to_csv(analysis),
                                     file_name=f"e2d_{analysis['batch_id']}.csv", mime="text/csv")
        markdown = batch_report.to_markdown(analysis)
        downloads[2].download_button("Download report (Markdown)", markdown,
                                     file_name=f"e2d_{analysis['batch_id']}.md", mime="text/markdown")
        if st.button("Save this report to the results store (write-once)", key="e2d-save-report"):
            path = store.write_report(f"e2d_stage{analysis['stage']}_{analysis['batch_id']}", markdown)
            _set_flash("success", f"Report saved: {path}")
            st.rerun()

        collected = batch_export.collect(batch_dir, root)
        _render_downloads_all(collected)
        view = st.radio("View", ["Single Page", "All Pages"], horizontal=True, key="e2d-results-view")
        if view == "All Pages":
            _render_all_pages(collected)
        else:
            st.subheader("Side-by-side per page")
            page_keys = [(page["registry_id"], page["page"]) for page in analysis["pages"]]
            picked = st.selectbox("Page", page_keys, format_func=lambda key: f"{key[0]} · page {key[1]}",
                                  key="e2d-results-page")
            page_info = next(page for page in analysis["pages"] if (page["registry_id"], page["page"]) == picked)
            rows = [row for row in analysis["rows"] if (row["registry_id"], row["page"]) == picked]
            source = sources[picked[0]]
            plan = batch_mod.load_plan(batch_dir)
            st.caption(f"Type (heuristic): {page_info['category']} · {page_info.get('page_role') or ''} · "
                       f"features {page_info['features']}")
            image_col, *output_cols = st.columns(1 + len(rows))
            with image_col:
                image = batch_mod.render_page(manager, settings, workspace_id, source, picked[1])
                st.image(image.path, caption=f"page {picked[1]} (local render)", width="stretch")
                with st.expander(batch_report.REFERENCE_LABEL):
                    st.text(batch_mod.text_layer(source.pdf_path, picked[1]))
            for column, row in zip(output_cols, rows):
                with column:
                    st.markdown(f"**{row['provider_id']}** · {row['status']}")
                    st.caption(batch_report.API_LABEL)
                    run = batch_report._run_payload(row["run_directory"])
                    if run and run.get("pages"):
                        st.text_area("RAW output", run["pages"][0].get("text") or "", height=420,
                                     key=f"e2d-raw-{row['item_id']}", disabled=True)
                        if row.get("analysis"):
                            result = row["analysis"]
                            st.json({"text_layer_word_recall": result["text_layer_word_recall"],
                                     "numbers": result["values"]["numbers"], "percentages": result["values"]["percentages"],
                                     "dates": result["values"]["dates"], "tables": result["tables"],
                                     "reading_order_tau": result["reading_order_tau_vs_text_layer"],
                                     "repetition": result["repetition"], "latency_ms": row["latency_ms"],
                                     "tokens": [row["prompt_tokens"], row["completion_tokens"]],
                                     "cost_usd": row["cost_usd"], "finish_reason": row["finish_reason"]}, expanded=False)
                        with st.expander("Structured/raw provider response"):
                            raw = (((run.get("environment") or {}).get("e2") or {}).get("page_raw") or {})
                            st.json(raw.get(str(picked[1])) or {}, expanded=False)
                        with st.form(f"e2d-review-{row['item_id']}-{row['run_id']}"):
                            decision = st.radio("Decision", ["PASS", "FAIL", "NEEDS_REVIEW"], horizontal=True,
                                                key=f"e2d-review-decision-{row['run_id']}")
                            categories = st.multiselect(
                                "Error categories", ["missing text", "wrong numbers", "wrong dates/percentages",
                                                     "table structure", "chart text", "hallucination", "repetition",
                                                     "reading order", "Arabic shaping/RTL", "other"],
                                key=f"e2d-review-categories-{row['run_id']}")
                            notes = st.text_area("Notes", key=f"e2d-review-notes-{row['run_id']}")
                            save = st.form_submit_button("Save manual review (write-once)")
                        if save:
                            store.write_manual_review({
                                "run_id": row["run_id"], "page": row["page"], "decision": decision, "notes": notes,
                                "error_categories": categories, "adapter_id": run["adapter"]["adapter_id"],
                                "model_revision": "hosted API; revision not reported",
                                "inference_config": {"provider_id": row["provider_id"], "batch_id": plan["batch_id"],
                                                     "evidence_class": batch_report.API_LABEL},
                            })
                            _set_flash("success", f"Review saved for {row['item_id']}")
                            st.rerun()
                        for review in store.manual_reviews(run_id=row["run_id"], page=row["page"]):
                            st.caption(f"Review #{review.get('sequence')}: {review['decision']} "
                                       f"{review.get('error_categories')} {review.get('notes')}")
                    else:
                        st.info(row["detail"] or "No output")

        st.subheader(f"Human review tasks ({len(analysis['review_tasks'])})")
        _table(analysis["review_tasks"])
        st.subheader("Errors and incomplete items")
        _table([{k: row[k] for k in ("item_id", "status", "detail")} for row in analysis["errors"]])
