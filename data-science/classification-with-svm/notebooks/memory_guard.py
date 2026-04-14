# ===== memory_guard.py (single script) =====
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Optional


FREE_FRACTION_WARN_THRESHOLD = 0.30  # warn if free < 30% of total

TROUBLESHOOTING_TEXT = (
    "You can also refer to the <b>Troubleshooting</b> section of the "
    "<a href='https://github.com/HPInc/AI-Blueprints/blob/main/README.md#troubleshooting' "
    "target='_blank' rel='noopener noreferrer'>AI-Blueprints README</a> "
    "for guidance on resolving common blueprint issues.<br><br>"
    "If you need additional assistance, please feel free to open a new discussion in the "
    "<a href='https://github.com/HPInc/AI-Blueprints/discussions/categories/help' "
    "target='_blank' rel='noopener noreferrer'><b>Help</b></a> category of our repository."
)

WSL_NOTE_TEXT = (
    "<div style='margin-top:10px; font-size:13px; color:#9e9e9e;'>"
    "ℹ️ Values are measured from the Linux (WSL) environment and may differ from "
    "Windows Task Manager."
    "</div>"
)




@dataclass(frozen=True)
class MemoryStatus:
    free_gb: float
    total_gb: float
    available_gb: Optional[float] = None

    @property
    def effective_available_gb(self) -> float:
        return self.free_gb if self.available_gb is None else self.available_gb

    @property
    def used_gb(self) -> float:
        return round(max(self.total_gb - self.effective_available_gb, 0.0), 2)

    @property
    def used_fraction(self) -> float:
        return 0.0 if self.total_gb <= 0 else self.used_gb / self.total_gb

    # NEW: compatibility + explicit fractions
    @property
    def free_fraction(self) -> float:
        return 0.0 if self.total_gb <= 0 else self.free_gb / self.total_gb

    @property
    def available_fraction(self) -> float:
        return 0.0 if self.total_gb <= 0 else self.effective_available_gb / self.total_gb



def _run_powershell(cmd: str) -> str:
    import shutil

    ps = shutil.which("powershell.exe") or "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    return subprocess.check_output(
        [ps, "-NoProfile", "-Command", cmd],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _check_vram_nvidia_smi_used_total() -> Optional[MemoryStatus]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

        if not out:
            return None

        first = out.splitlines()[0]
        m = re.match(r"^\s*(\d+)\s*,\s*(\d+)\s*$", first)
        if not m:
            return None

        used_mb = int(m.group(1))
        total_mb = int(m.group(2))

        used_gb = _mb_to_gb(used_mb)
        total_gb = _mb_to_gb(total_mb)
        free_gb = round(max(total_gb - used_gb, 0.0), 2)

        return MemoryStatus(free_gb=free_gb, total_gb=total_gb)
    except Exception:
        return None


def _check_host_ram_windows() -> Optional[MemoryStatus]:
    try:
        # Values are in KB
        out = _run_powershell(
            "(Get-CimInstance Win32_OperatingSystem | "
            "Select-Object TotalVisibleMemorySize,FreePhysicalMemory | "
            "ConvertTo-Json -Compress)"
        )
        import json
        obj = json.loads(out)
        total_kb = int(obj["TotalVisibleMemorySize"])
        free_kb = int(obj["FreePhysicalMemory"])
        return MemoryStatus(
            free_gb=round((free_kb * 1024) / (1024**3), 2),
            total_gb=round((total_kb * 1024) / (1024**3), 2),
        )
    except Exception:
        return None


def _check_host_vram_windows() -> Optional[MemoryStatus]:
    try:
        out = _run_powershell(
            "$p=(Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue).Source; "
            "if(-not $p){$p=(Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source}; "
            "if(-not $p){''} else { & $p --query-gpu=memory.free,memory.total --format=csv,noheader,nounits }"
        )
        first = out.splitlines()[0].strip() if out else ""
        m = re.match(r"^\s*(\d+)\s*,\s*(\d+)\s*$", first)
        if not m:
            return None
        free_mb = int(m.group(1))
        total_mb = int(m.group(2))
        return MemoryStatus(free_gb=_mb_to_gb(free_mb), total_gb=_mb_to_gb(total_mb), available_gb=_mb_to_gb(free_mb),)
    except Exception:
        return None


def _pct(x: float) -> str:
    return f"{x:.1f}%"

def _bytes_to_gb(value: int) -> float:
    return round(value / (1024**3), 2)


def _mb_to_gb(value: int) -> float:
    return round(value / 1024.0, 2)


def check_ram() -> MemoryStatus:
    meminfo: dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            k, v = line.split(":", 1)
            meminfo[k.strip()] = int(v.strip().split()[0])  # kB

    total_kb = meminfo["MemTotal"]
    avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
    free_kb = meminfo.get("MemFree", 0)

    total_gb = round((total_kb * 1024) / (1024**3), 2)
    available_gb = round((avail_kb * 1024) / (1024**3), 2)
    free_gb = round((free_kb * 1024) / (1024**3), 2)

    return MemoryStatus(
        free_gb=free_gb,
        total_gb=total_gb,
        available_gb=available_gb,
    )





def _check_vram_torch() -> Optional[MemoryStatus]:
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return None
        free_b, total_b = torch.cuda.mem_get_info()
        return MemoryStatus(
            free_gb=_bytes_to_gb(int(free_b)),
            total_gb=_bytes_to_gb(int(total_b)),
            available_gb=_bytes_to_gb(int(free_b)), 
        )
    except Exception:
        return None


def _check_vram_pynvml() -> Optional[MemoryStatus]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return MemoryStatus(
            free_gb=_bytes_to_gb(int(mem.free)),
            total_gb=_bytes_to_gb(int(mem.total)),
            available_gb=_bytes_to_gb(int(mem.free)), 
        )
    except Exception:
        return None


def _check_vram_nvidia_smi() -> Optional[MemoryStatus]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

        if not out:
            return None

        first = out.splitlines()[0]
        m = re.match(r"^\s*(\d+)\s*,\s*(\d+)\s*$", first)
        if not m:
            return None

        free_mb = int(m.group(1))
        total_mb = int(m.group(2))
        return MemoryStatus(
            free_gb=_mb_to_gb(free_mb),
            total_gb=_mb_to_gb(total_mb),
        )
    except Exception:
        return None


def check_vram() -> Optional[MemoryStatus]:
    v = _check_vram_nvidia_smi_used_total()
    if v is not None:
        return v

    v = _check_vram_torch()
    if v is not None:
        return v

    v = _check_vram_pynvml()
    if v is not None:
        return v

    return _check_vram_nvidia_smi()




def _display_usage_pies(ram: MemoryStatus, vram: Optional[MemoryStatus]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from IPython.display import display  # type: ignore
    except Exception:
        return

    items = [("RAM", ram.available_gb, ram.total_gb)]  # CHANGED: was ram.free_gb

    if vram is not None:
        items.append(("VRAM", vram.free_gb, vram.total_gb))

    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1:
        axes = [axes]

    for ax, (name, free_gb, total_gb) in zip(axes, items):
        used_gb = max(total_gb - free_gb, 0.0)

        ax.pie(
            [used_gb, free_gb],
            labels=[
                f"Used ({used_gb:.2f} GB)",
                f"Available ({free_gb:.2f} GB)",
            ],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#d32f2f", "#2e7d32"],  # red = used, green = free
        )
        ax.set_title(f"{name} Usage (Total {total_gb:.2f} GB)")

    plt.tight_layout()
    display(fig)
    plt.close(fig)


def _render_usage_bars(
    ram: MemoryStatus,
    vram: Optional[MemoryStatus],
    min_total_ram_gb: float,
    min_total_vram_gb: float,
) -> None:
    try:
        from IPython.display import HTML, display  # type: ignore
    except Exception:
        return

    def _build_row(
        name: str,
        minimum_required_gb: float,
        used_gb: float,
        total_gb: float,
        free_gb: float,
    ) -> str:
        used_pct = 0.0 if total_gb <= 0 else round((used_gb / total_gb) * 100.0, 1)

        return (
            "<div style='margin-top:18px;'>"
            "  <div style='display:grid; grid-template-columns:140px 260px 1fr 160px; "
            "              gap:18px; align-items:center;'>"
            f"    <div style='font-size:16px; font-weight:700; color:#e8e8e8;'>{name}</div>"
            f"    <div style='font-size:14px; color:#d0d0d0;'>Minimum required: ≥ {minimum_required_gb:.1f} GB</div>"
            "    <div>"
            "      <div style='height:10px; width:100%; background:rgba(255,255,255,0.14); "
            "                  border-radius:999px; overflow:hidden;'>"
            f"        <div style='height:100%; width:{used_pct}%; background:#f1f1f1; "
            "                    border-radius:999px;'></div>"
            "      </div>"
            f"      <div style='margin-top:8px; font-size:13px; color:#e0e0e0;'>"
            f"        {used_gb:.2f} / {total_gb:.2f} GB used ({used_pct:.1f}%)"
            "      </div>"
            "    </div>"
            f"    <div style='font-size:14px; color:#f3f3f3; text-align:right;'>{free_gb:.2f} GB free</div>"
            "  </div>"
            "</div>"
        )

    ram_free_gb = ram.effective_available_gb
    ram_used_gb = max(ram.total_gb - ram_free_gb, 0.0)

    rows = [
        _build_row(
            name="RAM",
            minimum_required_gb=min_total_ram_gb,
            used_gb=ram_used_gb,
            total_gb=ram.total_gb,
            free_gb=ram_free_gb,
        )
    ]

    if vram is not None:
        vram_free_gb = vram.effective_available_gb
        vram_used_gb = max(vram.total_gb - vram_free_gb, 0.0)

        rows.append(
            _build_row(
                name="VRAM",
                minimum_required_gb=min_total_vram_gb,
                used_gb=vram_used_gb,
                total_gb=vram.total_gb,
                free_gb=vram_free_gb,
            )
        )

    html = (
        "<div style='margin-top:14px; padding:18px 20px; "
        "            background:#1f1f1f; border:1px solid rgba(255,255,255,0.06); "
        "            border-radius:14px; box-shadow:0 4px 14px rgba(0,0,0,0.22);'>"
        "  <div>"
        "    <div style='font-size:18px; font-weight:800; color:#f5f5f5;'>Current Resources</div>"
        "  </div>"
        f"  {''.join(rows)}"
        f"  {WSL_NOTE_TEXT}"
        "</div>"
    )

    display(HTML(html))


def _shutdown_kernel() -> None:
    from IPython import get_ipython  # type: ignore

    ip = get_ipython()
    if ip is not None and hasattr(ip, "kernel"):
        ip.kernel.do_shutdown(restart=False)
    # raise SystemExit


def run_memory_check_notebook(
    min_total_ram_gb: float,
    min_total_vram_gb: float,  # use 0.0 to disable VRAM checks
) -> None:
    try:
        from IPython.display import Markdown, display  # type: ignore
    except Exception:
        Markdown = None  # type: ignore
        display = None  # type: ignore

    ram = check_ram()
    vram = check_vram() if min_total_vram_gb > 0 else None

    # -------------------------
    # Evaluate "total not enough" (RED)
    # -------------------------
    total_fail_reasons: list[str] = []

    if min_total_ram_gb > 0 and ram.total_gb < min_total_ram_gb:
        total_fail_reasons.append(
            "<div style='margin-bottom:10px;'>"
            "<div style='font-weight:800; color:#d32f2f;'>❌ RAM hardware is not sufficient</div>"
            "<ul style='margin:6px 0 0 16px;'>"
            f"<li>Total RAM: <b>{ram.total_gb} GB</b></li>"
            f"<li>Required Total RAM: <b>≥ {min_total_ram_gb} GB</b></li>"
            f"<li>Free RAM: <b>{ram.available_gb} GB</b></li>"
            "</ul>"
            "</div>"
        )
    
    if min_total_vram_gb > 0:
        if vram is None:
            total_fail_reasons.append(
                "<div style='margin-bottom:10px;'>"
                "<div style='font-weight:800; color:#d32f2f;'>❌ VRAM hardware is not sufficient</div>"
                "<ul style='margin:6px 0 0 16px;'>"
                "<li>Total VRAM: <b style='color:#ffb300;'>Not detected</b></li>"
                f"<li>Required Total VRAM: <b>≥ {min_total_vram_gb} GB</b></li>"
                "</ul>"
                "</div>"
            )
        elif vram.total_gb < min_total_vram_gb:
            total_fail_reasons.append(
                "<div style='margin-bottom:10px;'>"
                "<div style='font-weight:800; color:#d32f2f;'>❌ VRAM hardware is not sufficient</div>"
                "<ul style='margin:6px 0 0 16px;'>"
                f"<li>Total VRAM: <b>{vram.total_gb} GB</b></li>"
                f"<li>Required Total VRAM: <b>≥ {min_total_vram_gb} GB</b></li>"
                f"<li>Free VRAM: <b>{vram.free_gb} GB</b></li>"
                "</ul>"
                "</div>"
            )


    if total_fail_reasons:
        md = (
            "<div style='border-left:6px solid #d32f2f; padding:12px;'>"
            "<h2 style='margin:0;'>🛑 System Resource Check Failed</h2>"
            "<div style='margin-top:10px;'>"
            + "\n\n".join(total_fail_reasons)
            + "</div>"
            "<div style='margin-top:12px;'>"
            + TROUBLESHOOTING_TEXT
            + "</div></div>"
        )
        if display and Markdown:
            display(Markdown(md))
        else:
            print("SYSTEM RESOURCE CHECK FAILED\n" + "\n\n".join(total_fail_reasons) + "\n" + TROUBLESHOOTING_TEXT)

        # Kernel shutdown only for total hardware insufficiency (RED)
        _shutdown_kernel()
        return

    # -------------------------
    # Evaluate "free may not be enough" (YELLOW)
    # -------------------------
    low_free_reasons: list[str] = []

    if ram.available_fraction < FREE_FRACTION_WARN_THRESHOLD:
        low_free_reasons.append(
            "<div style='margin-bottom:10px;'>"
            "<div style='font-weight:800; color:#f9a825;'>⚠️ Free RAM may not be sufficient</div>"
            "<ul style='margin:6px 0 0 16px;'>"
            f"<li>Free RAM: <b>{ram.available_gb} GB</b> "
            f"(<b>{round(ram.available_fraction  * 100, 1)}%</b>)</li>"
            f"<li>Total RAM: <b>{ram.total_gb} GB</b></li>"
            f"<li>Required Total RAM: <b>≥ {min_total_ram_gb} GB</b></li>"
            "</ul>"
            "</div>"
        )
    
    if min_total_vram_gb > 0 and vram is not None and vram.available_fraction < FREE_FRACTION_WARN_THRESHOLD:
        low_free_reasons.append(
            "<div style='margin-bottom:10px;'>"
            "<div style='font-weight:800; color:#f9a825;'>⚠️ Free VRAM may not be sufficient</div>"
            "<ul style='margin:6px 0 0 16px;'>"
            f"<li>Free VRAM: <b>{vram.free_gb} GB</b> "
            f"(<b>{round(vram.available_fraction  * 100, 1)}%</b>)</li>"
            f"<li>Total VRAM: <b>{vram.total_gb} GB</b></li>"
            f"<li>Required Total VRAM: <b>≥ {min_total_vram_gb} GB</b></li>"
            "</ul>"
            "</div>"
        )


    # -------------------------
    # Render YELLOW or GREEN
    # -------------------------
    vram_lines = []
    if min_total_vram_gb > 0:
        if vram is None:
            vram_lines.append(f"- Total VRAM: `Not detected` (Required: `>= {min_total_vram_gb} GB`)")
        else:
            vram_lines.append(f"- Free VRAM: `{vram.free_gb}/{vram.total_gb} GB` (Required Total: `>= {min_total_vram_gb} GB`)")

    ram_used_pct = ram.used_fraction * 100.0
    base_lines = [
        f"- <b>RAM</b>: Available  {ram.available_gb}/{ram.total_gb} GB | Used {_pct(ram_used_pct)} | Required Total ≥ {min_total_ram_gb} GB",
    ]
    
    if min_total_vram_gb > 0:
        if vram is None:
            base_lines.append(
                f"- <b>VRAM</b>: <span style='color:#ffb300;'>Not detected</span> | Required Total ≥ {min_total_vram_gb} GB"
            )
        else:
            vram_used_pct = (1.0 - vram.free_fraction) * 100.0
            base_lines.append(
                f"- <b>VRAM</b>: Free {vram.free_gb}/{vram.total_gb} GB | Used {_pct(vram_used_pct)} | Required Total ≥ {min_total_vram_gb} GB"
            )

    if low_free_reasons:
        md = (
            "<div style='border-left:6px solid #f9a825; padding:12px;'>"
            "<h2 style='margin:0;'>⚠️ System Resource Check Warning</h2>"
            "<div style='margin-top:10px;'>"
            + "\n\n".join(low_free_reasons)
            + "</div>"
            "<div style='margin-top:12px;'><b>Current Resources</b><br>"
            + "<br>".join(line.replace("- ", "") for line in base_lines)
            + WSL_NOTE_TEXT
            + "</div>"
            "<div style='margin-top:12px;'>"
            + TROUBLESHOOTING_TEXT
            + "</div></div>"
        )
        if display and Markdown:
            display(Markdown(md))
            _render_usage_bars(ram, vram, min_total_ram_gb, min_total_vram_gb)
        else:
            print("SYSTEM RESOURCE CHECK WARNING\n" + "\n\n".join(low_free_reasons) + "\n\n" + "\n".join(base_lines))
        return

    md = (
        "<div style='border-left:8px solid #2e7d32; background:#0b1f12; padding:14px 16px; "
        "border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,0.25);'>"
        "<div style='display:flex; align-items:center; gap:10px;'>"
        "<div style='font-size:22px; line-height:1;'>✅</div>"
        "<div>"
        "<div style='font-size:18px; font-weight:800; margin:0;'>System Resource Check Passed</div>"
        "<div style='opacity:0.85; margin-top:2px;'>All required thresholds are satisfied.</div>"
        "</div>"
        "</div>"
        "<div style='margin-top:12px; padding:10px 12px; background:rgba(255,255,255,0.06); "
        "border-radius:10px;'>"
        "<div style='font-weight:700; margin-bottom:6px;'>Current Resources</div>"
        "<div style='line-height:1.7;'>"
        + "<br>".join(line.replace("- ", "") for line in base_lines)
        + WSL_NOTE_TEXT
        + "</div>"
        "</div>"
        "</div>"
    )
    if display and Markdown:
        display(Markdown(md))
        _render_usage_bars(ram, vram, min_total_ram_gb, min_total_vram_gb)
    else:
        print("SYSTEM RESOURCE CHECK PASSED\n" + "\n".join(base_lines))
