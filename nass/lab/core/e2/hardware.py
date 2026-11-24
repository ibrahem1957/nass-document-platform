"""Local hardware/environment probe. Read-only, offline, standard library only."""

from __future__ import annotations

import ctypes
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class HardwareProfile:
    cpu: str
    logical_cpus: int
    ram_total_gb: float | None
    ram_available_gb: float | None
    gpus: list[dict] = field(default_factory=list)
    cuda_available: bool = False
    torch_version: str | None = None
    python: str = ""
    os: str = ""
    hf_cache_dir: str = ""
    hf_cache_free_gb: float | None = None
    packages: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def max_vram_gb(self) -> float:
        return max((float(gpu.get("vram_gb") or 0) for gpu in self.gpus), default=0.0)


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _memory_gb() -> tuple[float | None, float | None]:
    gib = 1024**3
    if sys.platform == "win32":
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / gib, status.ullAvailPhys / gib
        return None, None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        available = os.sysconf("SC_AVPHYS_PAGES")
        return pages * size / gib, available * size / gib
    except (ValueError, OSError, AttributeError):
        return None, None


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _nvidia_gpus() -> list[dict]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        output = subprocess.run(
            [executable, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in output.splitlines():
        name, _, memory = line.partition(",")
        try:
            vram = float(memory.strip()) / 1024
        except ValueError:
            vram = None
        gpus.append({"name": name.strip(), "vram_gb": vram, "vendor": "nvidia"})
    return gpus


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def hf_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME")
    base = Path(home) if home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def probe_hardware() -> HardwareProfile:
    total, available = _memory_gb()
    cache = hf_cache_dir()
    free = None
    probe = cache if cache.exists() else cache.anchor or Path.home()
    try:
        free = shutil.disk_usage(probe).free / 1024**3
    except OSError:
        pass
    packages = {
        name: package_version(name)
        for name in (
            "torch",
            "transformers",
            "peft",
            "accelerate",
            "paddlepaddle",
            "paddleocr",
            "paddlex",
        )
    }
    cuda = False
    torch_version = packages.get("torch")
    gpus = _nvidia_gpus()
    if torch_version and importlib.util.find_spec("torch") is not None:
        try:
            import torch  # noqa: PLC0415 - optional, only when installed

            cuda = bool(torch.cuda.is_available())
        except Exception:  # broken installs must not break the probe
            cuda = False
    return HardwareProfile(
        cpu=_cpu_name(),
        logical_cpus=os.cpu_count() or 0,
        ram_total_gb=total,
        ram_available_gb=available,
        gpus=gpus,
        cuda_available=cuda,
        torch_version=torch_version,
        python=sys.version.split()[0],
        os=platform.platform(),
        hf_cache_dir=str(cache),
        hf_cache_free_gb=free,
        packages=packages,
    )


def peak_rss_mb() -> float | None:
    """Peak resident memory of the current process, in MiB."""
    if sys.platform == "win32":

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi")
        # Explicit types: the pseudo-handle (-1) is truncated without them on 64-bit.
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_Counters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        if psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return counters.PeakWorkingSetSize / 1024**2
        return None
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1024 if sys.platform != "darwin" else peak / 1024**2
    except (ImportError, OSError):
        return None
