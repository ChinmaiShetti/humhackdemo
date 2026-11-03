"""`ha doctor`: environment checks required before any phase (plan P0.2)."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ha.paths import get_paths

MIN_FREE_GB = 8.0
FORBIDDEN_PACKAGES = ("onnxruntime", "onnxruntime_gpu")  # native DLL fails on this machine


def _check_python() -> tuple[bool, str]:
    v = sys.version_info
    return (v.major, v.minor) == (3, 11), sys.version.split()[0]


def _check_utf8() -> tuple[bool, str]:
    return bool(sys.flags.utf8_mode), f"utf8_mode={sys.flags.utf8_mode} (set user env PYTHONUTF8=1)"


def _check_cuda() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch not installed"
    if not torch.cuda.is_available():
        return False, f"torch {torch.__version__}: CUDA not available"
    props = torch.cuda.get_device_properties(0)
    return True, f"{props.name}, {props.total_memory / 2**30:.1f} GiB, torch {torch.__version__}"


def _check_ollama() -> tuple[bool, str]:
    import requests

    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if not host.startswith("http"):
        host = f"http://{host}"
    try:
        r = requests.get(f"{host}/api/version", timeout=3)
        r.raise_for_status()
        return True, f"{host} version {r.json().get('version')}"
    except Exception as e:  # noqa: BLE001 - any failure means unreachable
        return False, f"{host} unreachable ({type(e).__name__})"


def _check_paths() -> list[tuple[str, bool, str]]:
    p = get_paths()
    rows = []
    for name, path in [("HA_ROOT", p.root), ("HA_DATA", p.data), ("HA_CACHE", p.cache)]:
        path.mkdir(parents=True, exist_ok=True)
        on_d = path.drive.upper() == "D:" or os.name != "nt"
        rows.append((name, on_d, f"{path}" + ("" if on_d else "  (must be on D:)")))
    hf = os.environ.get("HF_HOME")
    ok = bool(hf) and (Path(hf).drive.upper() == "D:" or os.name != "nt")
    rows.append(("HF_HOME", ok, hf or "unset"))
    return rows


def _check_disk() -> list[tuple[str, bool, str]]:
    rows = []
    drives = ["C:\\", "D:\\"] if os.name == "nt" else ["/"]
    for d in drives:
        if not Path(d).exists():
            continue
        free = shutil.disk_usage(d).free / 2**30
        rows.append((f"disk {d}", free >= MIN_FREE_GB, f"{free:.1f} GB free (min {MIN_FREE_GB})"))
    return rows


def _check_devanagari() -> tuple[bool, str]:
    sample = "डिजिलॉकर में मार्कशीट नहीं दिख रही — kya karu?"
    path = get_paths().cache / "_doctor_utf8.txt"
    path.write_text(sample, encoding="utf-8")
    ok = path.read_text(encoding="utf-8") == sample
    path.unlink()
    return ok, "round-trip ok" if ok else "round-trip FAILED"


def _check_forbidden() -> tuple[bool, str]:
    found = [p for p in FORBIDDEN_PACKAGES if importlib.util.find_spec(p) is not None]
    return not found, "none" if not found else f"installed: {found} (uninstall; spec §4.3)"


def run_doctor() -> int:
    rows: list[tuple[str, bool, str]] = []
    rows.append(("python 3.11", *_check_python()))
    rows.append(("UTF-8 mode", *_check_utf8()))
    rows.append(("CUDA", *_check_cuda()))
    rows.append(("Ollama", *_check_ollama()))
    rows.extend(_check_paths())
    rows.extend(_check_disk())
    rows.append(("Devanagari I/O", *_check_devanagari()))
    rows.append(("no ONNX Runtime", *_check_forbidden()))

    table = Table(title="ha doctor")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for name, ok, detail in rows:
        table.add_row(name, "[green]OK[/]" if ok else "[red]FAIL[/]", detail)
    console = Console()
    console.print(table)
    console.print(f"HA_UNSEAL={os.environ.get('HA_UNSEAL', '0')!r}")
    return 0 if all(ok for _, ok, _ in rows) else 1
