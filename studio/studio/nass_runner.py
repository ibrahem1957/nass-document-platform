"""Isolated adapter executed with the existing Nass Python environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _emit(value: dict) -> None:
    print("STUDIO_RESULT=" + json.dumps(value, ensure_ascii=True, separators=(",", ":")))


def _imports(root: Path):
    sys.path.insert(0, str(root))
    from lab.core.auto import assemble, pipeline, qari_client, regions
    from lab.core.auto import config as auto_config
    return assemble, pipeline, qari_client, regions, auto_config


def readiness(root: Path) -> int:
    _assemble, pipeline, _client, _regions, auto_config = _imports(root)
    try:
        config = auto_config.load_config()
        endpoint = pipeline.check_service(config)
        layout = auto_config.layout_model_status(config)
        _emit({
            "ready": bool(layout.get("ok")),
            "layout": {"ready": bool(layout.get("ok")), "model_name": layout.get("model_name"),
                       "problems": layout.get("problems", [])},
            "external_processing": bool(endpoint.get("external")),
            "deployment_mode": endpoint.get("deployment_mode"),
            "service_display": endpoint.get("display"),
            "automatic_enabled": config.qari.automatic_enabled,
            "token_present": bool(config.qari_token()),
            "max_pdf_mb": config.app.max_pdf_mb,
            "max_pages": config.app.max_pages,
            "note": "Configuration inspection only; no network request, model load, or download was performed.",
        })
        return 0
    except Exception as exc:  # noqa: BLE001
        _emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 2


def run(args: argparse.Namespace) -> int:
    assemble, pipeline, qari_client, regions, auto_config = _imports(args.nass_root)
    run_dir = None
    try:
        config = auto_config.load_config()
        pipeline.check_service(config)
        layout = auto_config.layout_model_status(config)
        if not layout.get("ok"):
            raise RuntimeError("PP-DocLayout is not ready: " + "; ".join(layout.get("problems", [])))
        detector = regions.LayoutDetector(
            model_name=config.proven.layout.model_name, model_dir=config.layout.model_dir,
            device=config.layout.device, threshold=config.layout.score_threshold,
        )
        run_dir = pipeline.create_run(
            args.pdf, display_name=args.display_name, config=config, detector=detector,
            root=args.output_root,
        )
        pending = pipeline.pending_jobs(run_dir)
        if pending:
            with qari_client.QariClient(
                config.qari.service_url, config.qari_token(), timeout_s=config.qari.request_timeout_s,
                connect_retries=config.qari.max_connect_retries,
                deployment_mode=config.qari.deployment_mode,
                trusted_development_hosts=config.qari.trusted_development_hosts,
            ) as client:
                summary = pipeline.execute(run_dir, config, client)
                if summary.get("stopped"):
                    raise RuntimeError(summary["stopped"])
                try:
                    pipeline.verify_digits(run_dir, config, client)
                except Exception as exc:  # noqa: BLE001 - Nass treats this second read as report-only
                    print("STUDIO_EVENT=" + json.dumps({"digit_verification": "skipped",
                                                        "error_type": type(exc).__name__}))
        counts = pipeline.status(run_dir)["counts"]
        if counts["pending"] + counts["unconfirmed"] + counts["failed"]:
            raise RuntimeError(f"Nass is incomplete: {counts}")
        assemble.finalize(run_dir, config)
        _emit({"ok": True, "run_dir": str(run_dir.resolve()), "counts": counts})
        return 0
    except Exception as exc:  # noqa: BLE001
        _emit({"ok": False, "run_dir": str(run_dir.resolve()) if run_dir else None,
               "error": f"{type(exc).__name__}: {exc}"})
        return 3


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--nass-root", type=Path, required=True)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("readiness")
    execute = commands.add_parser("run")
    execute.add_argument("--pdf", type=Path, required=True)
    execute.add_argument("--display-name", required=True)
    execute.add_argument("--output-root", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "readiness":
        return readiness(args.nass_root)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
