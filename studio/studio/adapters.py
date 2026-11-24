from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import StudioConfig


class AdapterError(RuntimeError):
    def __init__(self, message: str, *, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True)
class CommandResult:
    command_name: str
    returncode: int
    stdout: str
    stderr: str


def redact(value: str) -> str:
    value = re.sub(r"(?i)(authorization\s*[:=]\s*)([^\s,;]+)", r"\1[REDACTED]", value)
    value = re.sub(r"(?i)(token|api[_-]?key|secret)(\s*[:=]\s*)([^\s,;]+)", r"\1\2[REDACTED]", value)
    return re.sub(r"([?&](?:token|key|signature|sig)=)[^&\s]+", r"\1[REDACTED]", value, flags=re.IGNORECASE)


class CommandRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen[str]] = {}

    def run(self, job_id: str, command_name: str, command: list[str], *, cwd: Path,
            timeout_s: int, cancel: threading.Event) -> CommandResult:
        if cancel.is_set():
            raise AdapterError(f"{command_name} cancelled before start")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        environment = os.environ.copy()
        environment.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", shell=False, creationflags=flags, env=environment,
        )
        with self._lock:
            self._processes[job_id] = process
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            raise AdapterError(
                f"{command_name} timed out after {timeout_s} seconds",
                stdout=redact(stdout), stderr=redact(stderr),
            ) from exc
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
        result = CommandResult(command_name, process.returncode, redact(stdout), redact(stderr))
        if cancel.is_set():
            raise AdapterError(f"{command_name} cancelled", stdout=result.stdout, stderr=result.stderr)
        if process.returncode != 0:
            raise AdapterError(
                f"{command_name} failed with exit code {process.returncode}",
                stdout=result.stdout, stderr=result.stderr,
            )
        return result

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            process = self._processes.get(job_id)
        if process is None or process.poll() is not None:
            return False
        process.terminate()
        return True


def _parse_result(stdout: str, prefix: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith(prefix):
            try:
                value = json.loads(line[len(prefix):])
            except json.JSONDecodeError as exc:
                raise AdapterError("adapter returned malformed result JSON", stdout=stdout) from exc
            if not isinstance(value, dict):
                raise AdapterError("adapter result must be an object", stdout=stdout)
            return value
    raise AdapterError("adapter did not return a result record", stdout=stdout)


class NassAdapter:
    def __init__(self, config: StudioConfig, runner: CommandRunner) -> None:
        self.config = config
        self.runner = runner
        self.script = config.project_root / "studio" / "nass_runner.py"

    def _base_command(self) -> list[str]:
        return [str(self.config.nass_python), str(self.script), "--nass-root", str(self.config.nass_root)]

    def readiness(self) -> dict:
        completed = subprocess.run(
            [*self._base_command(), "readiness"], cwd=self.config.project_root,
            capture_output=True, text=True, encoding="utf-8", errors="replace", shell=False,
            timeout=60, check=False,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        if completed.returncode != 0:
            raise AdapterError("Nass readiness check failed", stdout=redact(completed.stdout),
                               stderr=redact(completed.stderr))
        return _parse_result(redact(completed.stdout), "STUDIO_RESULT=")

    def run(self, job_id: str, pdf_path: Path, display_name: str, output_root: Path,
            cancel: threading.Event) -> Path:
        command = [*self._base_command(), "run", "--pdf", str(pdf_path), "--display-name", display_name,
                   "--output-root", str(output_root)]
        result = self.runner.run(job_id, "Nass extraction", command, cwd=self.config.project_root,
                                 timeout_s=self.config.stage_timeout_s, cancel=cancel)
        record = _parse_result(result.stdout, "STUDIO_RESULT=")
        run_dir = Path(record.get("run_dir", "")).resolve()
        expected = (output_root / "automatic_extraction").resolve()
        if expected not in run_dir.parents or not (run_dir / "document.json").is_file():
            raise AdapterError("Nass returned an invalid or incomplete run directory", stdout=result.stdout)
        return run_dir


class DocPrepAdapter:
    STAGES = (
        ("structure", "docprep.structure"),
        ("relationships", "docprep.relations"),
        ("quality", "docprep.quality"),
        ("prepared", "docprep.prepare"),
    )

    def __init__(self, config: StudioConfig, runner: CommandRunner) -> None:
        self.config = config
        self.runner = runner

    @staticmethod
    def _written_path(stdout: str, output_root: Path) -> Path:
        matches = [line.split("written:", 1)[1].strip() for line in stdout.splitlines() if line.startswith("written:")]
        if len(matches) != 1:
            raise AdapterError("DocPrep did not report exactly one output directory", stdout=stdout)
        path = Path(matches[0]).resolve()
        root = output_root.resolve()
        if root not in path.parents or not path.is_dir():
            raise AdapterError("DocPrep reported an output outside its job-specific root", stdout=stdout)
        return path

    def run_sequence(self, job_id: str, nass_run: Path, output_base: Path, cancel: threading.Event,
                     on_stage: Callable[[str], None]) -> dict[str, Path]:
        outputs: dict[str, Path] = {}
        roots = {stage: output_base / stage for stage, _module in self.STAGES}
        commands = {
            "structure": [str(self.config.docprep_python), "-m", "docprep.structure", str(nass_run),
                          "--out", str(roots["structure"])],
            "relationships": None,
            "quality": None,
            "prepared": None,
        }
        for stage, module in self.STAGES:
            if stage == "relationships":
                commands[stage] = [str(self.config.docprep_python), "-m", module, str(nass_run),
                                   "--structure", str(outputs["structure"]), "--out", str(roots[stage])]
            elif stage == "quality":
                commands[stage] = [str(self.config.docprep_python), "-m", module, str(nass_run),
                                   "--structure", str(outputs["structure"]), "--relations",
                                   str(outputs["relationships"]), "--out", str(roots[stage])]
            elif stage == "prepared":
                commands[stage] = [str(self.config.docprep_python), "-m", module, str(nass_run),
                                   "--structure", str(outputs["structure"]), "--relations",
                                   str(outputs["relationships"]), "--quality", str(outputs["quality"]),
                                   "--out", str(roots[stage])]
            on_stage(stage)
            result = self.runner.run(job_id, f"DocPrep {stage}", commands[stage] or [],
                                     cwd=self.config.docprep_root,
                                     timeout_s=self.config.stage_timeout_s, cancel=cancel)
            outputs[stage] = self._written_path(result.stdout, roots[stage])
        return outputs
