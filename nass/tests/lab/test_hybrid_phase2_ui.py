"""Regression: the Phase 2 send section renders (timeout table, crop details) and the panel only
uses functions that exist. Synthetic run only; nothing is sent (the send button is never clicked)."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

from lab.app import hybrid_phase2_panel as p2
from lab.core.hybrid import enrich
from tests.lab.test_hybrid_phase2 import _set, ctx  # noqa: F401 - shared synthetic run fixture

PANEL = Path(__file__).resolve().parents[2] / "lab" / "app" / "hybrid_phase2_panel.py"
ALIASES = {"crop_options": "lab.core.hybrid.crops", "enrich": "lab.core.hybrid.enrich", "qari": "lab.core.hybrid.qari",
           "panel": "lab.app.full_pdf_panel", "documents": "lab.core.e2.api.documents",
           "layout_regions": "lab.core.hybrid.layout_regions"}


def test_panel_uses_only_existing_module_attributes() -> None:
    source = PANEL.read_text(encoding="utf-8")
    missing = []
    for alias, module_name in ALIASES.items():
        module = importlib.import_module(module_name)
        missing += [f"{alias}.{name}" for name in sorted(set(re.findall(rf"\b{alias}\.([A-Za-z_]\w*)", source)))
                    if not hasattr(module, name)]
    assert missing == []
    assert p2.missing_api() == []


def _send_script(set_dir: str, root: str) -> None:
    from pathlib import Path
    from types import SimpleNamespace

    from lab.app import hybrid_phase2_panel as panel
    from lab.core.hybrid import enrich as phase2

    data = phase2.load_set(Path(set_dir))
    panel._send_section(Path(set_dir), data, panel._specs(), document=None, source=None,
                        store=SimpleNamespace(root=Path(root)), runner=None, workspace_id="test")


def test_send_section_renders_timeouts_and_crop_details_for_all_providers(ctx) -> None:  # noqa: F811
    from streamlit.testing.v1 import AppTest

    set_dir = _set(ctx)
    app = AppTest.from_function(_send_script, args=(str(set_dir), str(ctx["root"])), default_timeout=60)
    app.run()
    assert not app.exception
    app.multiselect(key=f"p2-providers-{set_dir.name}").set_value(list(p2.PROVIDERS)).run()
    assert not app.exception
    tables = " ".join(item.value for item in app.markdown)
    assert "Baidu default" in tables and "Oxen default" in tables and "Qari default" in tables
    first = enrich.load_set(set_dir)["regions"][0]
    assert first["region_id"] in tables and "px" in tables  # per-region crop details are shown
    send = next(item for item in app.button if item.label == "Approve & send selected crops")
    assert send.disabled  # no approval, no key: nothing can be sent
    app.number_input(key=f"p2-timeout-{set_dir.name}").set_value(120).run()
    assert not app.exception and "120 s" in " ".join(item.value for item in app.markdown)


def test_stale_module_shows_a_restart_message_instead_of_crashing(monkeypatch) -> None:
    monkeypatch.delattr(p2.crop_options, "size_based_timeout")
    assert p2.missing_api() == ["crops.size_based_timeout"]


def test_every_keyword_the_panel_passes_is_accepted_by_the_called_function() -> None:
    import ast
    import inspect

    problems = []
    modules = {alias: importlib.import_module(name) for alias, name in ALIASES.items()}
    for node in ast.walk(ast.parse(PANEL.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id in modules):
            continue
        function = getattr(modules[node.func.value.id], node.func.attr, None)
        if function is None or not callable(function):
            continue
        try:
            parameters = inspect.signature(function).parameters
        except (TypeError, ValueError):
            continue
        if any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
            continue
        problems += [f"line {node.lineno}: {node.func.value.id}.{node.func.attr}({kw.arg}=)"
                     for kw in node.keywords if kw.arg and kw.arg not in parameters]
    assert problems == []


def test_propose_accepts_expansions_and_a_stale_signature_is_reported(monkeypatch) -> None:
    import inspect

    from lab.core.hybrid import layout_regions

    assert "expansions" in inspect.signature(layout_regions.propose).parameters
    assert "expansions" in inspect.signature(layout_regions.create_layout_region_set).parameters

    def old_propose(document, blocks, *, labels, margin_px=15, titles=True, captions_below=False):
        return []

    monkeypatch.setattr(p2.layout_regions, "propose", old_propose)
    assert p2.missing_api() == ["layout_regions.propose(expansions=)"]
