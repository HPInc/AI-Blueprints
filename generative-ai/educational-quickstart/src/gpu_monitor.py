# ─────── Standard Library Imports ───────
import logging  # Built-in logging system
import time  # Timestamps for metric history
from datetime import datetime  # Human-readable timestamp format
from typing import List, Dict, Optional  # Type hints for function signatures

# ─────── Third-Party Imports ───────
import plotly.graph_objects as go  # Low-level Plotly figure construction
import plotly.express as px  # High-level Plotly shorthand charts
from plotly.subplots import make_subplots  # Multi-panel dashboard layout

# Set up module-level logger
logger = logging.getLogger(__name__)


def get_gpu_stats() -> dict:
    """
    Collect current GPU performance statistics from the system.

    What does this measure?
        - GPU Utilization (%): How busy the GPU's cores are (0% = idle, 100% = maxed out)
        - Memory Used (MB): How many megabytes of VRAM are currently occupied
        - Memory Total (MB): Total VRAM available on the GPU
        - Temperature (°C): Current GPU die temperature
        - Power Draw (W): How many watts the GPU is currently consuming

    Why monitor the GPU?
        Running AI models on a GPU can consume all available VRAM very quickly,
        causing out-of-memory errors. Monitoring helps you understand resource usage,
        optimize your code, and avoid crashes.

    Returns:
        A dictionary with keys: gpu_name, utilization, memory_used_mb,
        memory_total_mb, memory_percent, temperature, power_draw.
        Returns safe zero-defaults if no GPU is available.

    Learn more about CUDA and GPU monitoring:
        https://developer.nvidia.com/cuda-toolkit
        https://pytorch.org/docs/stable/notes/cuda.html
    """
    stats = {
        "gpu_name": "No GPU",
        "utilization": 0.0,
        "memory_used_mb": 0.0,
        "memory_total_mb": 0.0,
        "memory_percent": 0.0,
        "temperature": 0.0,
        "power_draw": 0.0,
        "tokens_per_second": 0.0,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }

    try:
        import torch  # PyTorch: used to query CUDA memory statistics

        if not torch.cuda.is_available():
            logger.warning("⚠️ No GPU detected — reporting all-zero stats.")
            return stats

        # Query memory from PyTorch (more reliable than GPUtil for CUDA allocations)
        memory_used = torch.cuda.memory_allocated(0) / (1024**2)  # Convert bytes → MB
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        gpu_name = torch.cuda.get_device_name(0)

        stats["gpu_name"] = gpu_name
        stats["memory_used_mb"] = round(memory_used, 1)
        stats["memory_total_mb"] = round(memory_total, 1)
        stats["memory_percent"] = (
            round((memory_used / memory_total) * 100, 1) if memory_total > 0 else 0.0
        )

        # Try to get utilization, temperature, and real memory via GPUtil (requires nvidia-smi)
        # GPUtil reads from nvidia-smi, so it captures VRAM used by ALL processes —
        # including LlamaCpp, which bypasses PyTorch's allocator entirely.
        try:
            import GPUtil  # Third-party library that wraps nvidia-smi output

            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use the first GPU
                stats["utilization"] = round(
                    gpu.load * 100, 1
                )  # GPUtil returns 0.0–1.0
                stats["temperature"] = round(gpu.temperature, 1)
                stats["power_draw"] = round(getattr(gpu, "powerDraw", 0) or 0, 1)
                # Override PyTorch allocator values with nvidia-smi values so that
                # non-PyTorch consumers (e.g. LlamaCpp) are included in the reading.
                if gpu.memoryTotal and gpu.memoryTotal > 0:
                    stats["memory_used_mb"] = round(gpu.memoryUsed, 1)
                    stats["memory_total_mb"] = round(gpu.memoryTotal, 1)
                    stats["memory_percent"] = round(
                        (gpu.memoryUsed / gpu.memoryTotal) * 100, 1
                    )
        except ImportError:
            logger.warning(
                "⚠️ GPUtil not installed — utilization/temperature unavailable."
            )
        except Exception as e:
            logger.warning(f"⚠️ GPUtil query failed: {e}")

    except Exception as e:
        logger.warning(f"⚠️ GPU stats collection failed: {e}")

    return stats


def create_gpu_dashboard(
    history: List[dict], inference_history: Optional[List[dict]] = None
) -> go.Figure:
    """
    Create a 5-panel interactive Plotly dashboard from GPU stat snapshots.

    Dashboard layout (3 rows × 2 columns):
        ┌──────────────────┬──────────────────┐
        │ GPU Utilization% │  Memory Usage MB  │
        │   (line chart)   │   (area chart)    │
        ├──────────────────┼──────────────────┤
        │  Temperature °C  │  Tokens / Second │
        │   (line chart)   │   (line chart)   │
        ├─────────────────────────────────────┤
        │   Current Stats (gauge — full row)  │
        └─────────────────────────────────────┘

    Why Plotly?
        Plotly creates interactive charts that you can zoom, hover, and explore
        directly inside Jupyter notebooks with `fig.show()`.

    Args:
        history: A list of stat dicts, each produced by get_gpu_stats().
                 Each dict must have: timestamp, utilization, memory_used_mb,
                 memory_total_mb, temperature.
        inference_history: An optional list of dicts produced by
                           GPUMonitor.log_inference(). Each dict must have
                           tokens_per_second and timestamp keys.

    Returns:
        A Plotly Figure object. Display it with fig.show() in a notebook cell.

    Learn more about Plotly:
        https://plotly.com/python/
        https://plotly.com/python/subplots/
    """
    # If no history was recorded yet, create a minimal placeholder entry
    if not history:
        history = [get_gpu_stats()]

    # Extract time-series data from the history list
    timestamps = [s.get("timestamp", "") for s in history]
    utilizations = [s.get("utilization", 0) for s in history]
    memory_used = [s.get("memory_used_mb", 0) for s in history]
    memory_total = [s.get("memory_total_mb", 0) for s in history]
    temperatures = [s.get("temperature", 0) for s in history]

    # Get the most recent snapshot for the gauge indicators
    latest = history[-1]
    gpu_name = latest.get("gpu_name", "GPU")

    # Create a 3×2 subplot grid — Row 3 is a full-width gauge spanning both columns
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "GPU Utilization (%)",
            "Memory Usage (MB)",
            "Temperature (°C)",
            "Tokens / Second",
            "",
            "",
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],  # Row 1: Utilization + Memory
            [{"type": "xy"}, {"type": "xy"}],  # Row 2: Temperature + Tokens/sec
            [
                {"type": "indicator", "colspan": 2},
                None,
            ],  # Row 3: Gauge spanning both cols
        ],
        row_heights=[0.33, 0.33, 0.34],
    )

    # ── Panel 1: GPU Utilization % ────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=utilizations,
            mode="lines+markers",
            name="Utilization %",
            line=dict(color="#4CAF50", width=2),  # Green line
            fill="tozeroy",
            fillcolor="rgba(76,175,80,0.15)",  # Light green fill under the line
        ),
        row=1,
        col=1,
    )

    # Add a visual warning zone at 80% utilization
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="orange",
        annotation_text="80% — High",
        row=1,
        col=1,
    )

    # ── Panel 2: Memory Usage (area chart showing used vs. total) ─────────────
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=memory_total,
            mode="lines",
            name="Total VRAM",
            line=dict(
                color="#9E9E9E", width=1, dash="dot"
            ),  # Grey dotted = capacity ceiling
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=memory_used,
            mode="lines+markers",
            name="Used VRAM",
            line=dict(color="#2196F3", width=2),  # Blue line = actual usage
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.15)",
        ),
        row=1,
        col=2,
    )

    # ── Panel 3: Temperature ──────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=temperatures,
            mode="lines+markers",
            name="Temp °C",
            line=dict(color="#FF5722", width=2),  # Orange-red line
            fill="tozeroy",
            fillcolor="rgba(255,87,34,0.15)",
        ),
        row=2,
        col=1,
    )
    # Add a 80°C warning threshold line — sustained temperatures above this
    # can throttle GPU performance and reduce hardware lifespan
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="red",
        annotation_text="80°C — Warning",
        row=2,
        col=1,
    )

    # ── Panel 4: Tokens / Second (inference throughput over time) ───────────
    tps_data = inference_history or []
    tps_values = [e.get("tokens_per_second", 0) for e in tps_data]
    tps_labels = [e.get("timestamp", f"#{i + 1}") for i, e in enumerate(tps_data)]
    fig.add_trace(
        go.Scatter(
            x=tps_labels if tps_values else ["(no data)"],
            y=tps_values if tps_values else [0],
            mode="lines+markers",
            name="Tokens/sec",
            line=dict(color="#9C27B0", width=2),  # Purple line
            fill="tozeroy",
            fillcolor="rgba(156,39,176,0.15)",
        ),
        row=2,
        col=2,
    )

    # ── Panel 5: Gauge Indicator (current utilization — spans full row 3) ────
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=latest.get("utilization", 0),
            title={
                "text": f"Current Stats — {gpu_name}<br>Utilization %",
                "font": {"size": 11},
            },
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4CAF50"},
                "steps": [
                    {"range": [0, 50], "color": "#e8f5e9"},  # Light green = low
                    {"range": [50, 80], "color": "#fff3e0"},  # Yellow = moderate
                    {"range": [80, 100], "color": "#ffebee"},  # Red = high
                ],
            },
            domain={"row": 0, "column": 0},
        ),
        row=3,
        col=1,
    )

    # ── Layout and Styling ────────────────────────────────────────────────────
    fig.update_layout(
        title={
            "text": "🖥️ GPU Monitoring Dashboard",
            "x": 0.5,  # Center the title
            "font": {"size": 18},
        },
        template="plotly_white",  # Clean white background — works well in both light/dark notebooks
        height=750,  # Taller to accommodate the 3-row layout
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )

    # Add axis labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Inference #", row=2, col=2)
    fig.update_yaxes(title_text="Utilization (%)", row=1, col=1)
    fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Tokens/sec", row=2, col=2)

    return fig


def log_gpu_metrics_to_mlflow(stats: dict) -> None:
    """
    Log current GPU statistics as metrics in the active MLflow run.

    What is MLflow metric logging?
        MLflow can track numeric metrics over time (like training loss, accuracy,
        or in this case GPU usage). You can view these metrics in the MLflow UI
        under the Experiments tab as interactive charts.

    Args:
        stats: A dictionary produced by get_gpu_stats()

    Learn more about MLflow metrics:
        https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metric
    """
    try:
        import mlflow  # MLflow experiment tracking library

        # Only log if there is an active MLflow run (avoids errors when no run is started)
        if mlflow.active_run():
            mlflow.log_metric("gpu_utilization", stats.get("utilization", 0))
            mlflow.log_metric("gpu_memory_used_mb", stats.get("memory_used_mb", 0))
            mlflow.log_metric("gpu_temperature", stats.get("temperature", 0))
            mlflow.log_metric("tokens_per_second", stats.get("tokens_per_second", 0))
    except ImportError:
        logger.warning("⚠️ MLflow not available — skipping metric logging.")
    except Exception as e:
        logger.warning(f"⚠️ MLflow metric logging failed: {e}")


class GPUMonitor:
    """
    Stateful GPU monitor that collects snapshots and inference timings over time.

    Usage pattern in notebooks:
        monitor = GPUMonitor()               # Create once, alongside model init
        # ... run model.predict() ...
        elapsed = time.time() - t0
        monitor.log_inference(               # Record tokens/sec after each inference
            num_tokens=len(answer) // 4,
            elapsed_seconds=elapsed,
        )
        monitor.display_dashboard()          # Show 5-panel dashboard at end

    Why "stateful"?
        Each `snapshot()` call appends to `self.history` (GPU stats over time).
        Each `log_inference()` call appends to `self._inference_history` (TPS over time).
        At the end of your notebook both histories are visualised together.
    """

    def __init__(self):
        """Initialize the monitor with empty history lists."""
        self.history: List[dict] = []  # One dict per snapshot() call
        self._inference_history: List[dict] = []  # One dict per log_inference() call

    def snapshot(self) -> dict:
        """
        Collect one GPU stat reading and add it to the history.

        Returns:
            The stats dict that was just collected (also stored in self.history)
        """
        stats = get_gpu_stats()  # Collect current readings
        self.history.append(stats)  # Remember this snapshot
        return stats

    def log_inference(self, num_tokens: int, elapsed_seconds: float) -> float:
        """
        Record one inference run's token throughput.

        Call this immediately after each model.predict() to accumulate
        tokens-per-second history shown in the dashboard's Tokens/sec panel.

        Args:
            num_tokens: Approximate number of output tokens. Use len(answer) // 4
                        as a rough 4-chars-per-token estimate for English text.
            elapsed_seconds: Wall-clock seconds the predict() call took.

        Returns:
            The computed tokens-per-second value (also stored in _inference_history).
        """
        tps = round(num_tokens / elapsed_seconds, 1) if elapsed_seconds > 0 else 0.0
        self._inference_history.append(
            {
                "tokens_per_second": tps,
                "num_tokens": num_tokens,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        )
        return tps

    def dashboard(self) -> go.Figure:
        """
        Build and return a 5-panel Plotly dashboard from all collected snapshots.

        Returns:
            A Plotly Figure — display with fig.show() in a notebook cell
        """
        return create_gpu_dashboard(self.history, self._inference_history)

    def log_to_mlflow(self) -> None:
        """Log the most recent snapshot to the active MLflow run (if one exists)."""
        if self.history:
            log_gpu_metrics_to_mlflow(self.history[-1])  # Only log the latest reading

    def display_dashboard(self) -> None:
        """
        Collect a GPU snapshot and display the 5-panel dashboard inline in the notebook.

        This is a convenience wrapper that combines snapshot() + dashboard() + fig.show()
        into a single one-liner call — the pattern used in all starter notebooks.
        """
        self.snapshot()  # Collect current GPU stats
        fig = self.dashboard()  # Build the Plotly figure
        fig.show()  # Display inline in the Jupyter cell output

    def summary(self) -> str:
        """
        Return a formatted text summary of the current GPU state.

        Returns:
            A multi-line string with key GPU stats, suitable for printing.
        """
        if not self.history:
            return "No GPU snapshots collected yet. Call monitor.snapshot() first."

        latest = self.history[-1]
        tps_line = ""
        if self._inference_history:
            latest_tps = self._inference_history[-1]["tokens_per_second"]
            avg_tps = sum(
                e["tokens_per_second"] for e in self._inference_history
            ) / len(self._inference_history)
            tps_line = (
                f"\n  Tokens/sec  : {latest_tps:.1f} "
                f"(avg {avg_tps:.1f} over {len(self._inference_history)} inferences)"
            )
        return (
            f"GPU: {latest.get('gpu_name', 'N/A')}\n"
            f"  Utilization : {latest.get('utilization', 0):.1f}%\n"
            f"  Memory Used : {latest.get('memory_used_mb', 0):.0f} MB "
            f"/ {latest.get('memory_total_mb', 0):.0f} MB "
            f"({latest.get('memory_percent', 0):.1f}%)\n"
            f"  Temperature : {latest.get('temperature', 0):.1f}°C"
            + tps_line
            + f"\n  Snapshot at : {latest.get('timestamp', 'N/A')}"
        )
