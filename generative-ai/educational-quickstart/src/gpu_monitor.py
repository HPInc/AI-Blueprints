# ─────── Standard Library Imports ───────
import logging              # Built-in logging system
import time                 # Timestamps for metric history
from datetime import datetime  # Human-readable timestamp format
from typing import List, Dict, Optional  # Type hints for function signatures

# ─────── Third-Party Imports ───────
import plotly.graph_objects as go      # Low-level Plotly figure construction
import plotly.express as px            # High-level Plotly shorthand charts
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
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }

    try:
        import torch  # PyTorch: used to query CUDA memory statistics

        if not torch.cuda.is_available():
            logger.warning("⚠️ No GPU detected — reporting all-zero stats.")
            return stats

        # Query memory from PyTorch (more reliable than GPUtil for CUDA allocations)
        memory_used = torch.cuda.memory_allocated(0) / (1024 ** 2)   # Convert bytes → MB
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        gpu_name = torch.cuda.get_device_name(0)

        stats["gpu_name"] = gpu_name
        stats["memory_used_mb"] = round(memory_used, 1)
        stats["memory_total_mb"] = round(memory_total, 1)
        stats["memory_percent"] = round((memory_used / memory_total) * 100, 1) if memory_total > 0 else 0.0

        # Try to get utilization and temperature via GPUtil (requires nvidia-smi)
        try:
            import GPUtil  # Third-party library that wraps nvidia-smi output
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use the first GPU
                stats["utilization"] = round(gpu.load * 100, 1)    # GPUtil returns 0.0–1.0
                stats["temperature"] = round(gpu.temperature, 1)
                stats["power_draw"] = round(getattr(gpu, "powerDraw", 0) or 0, 1)
        except ImportError:
            logger.warning("⚠️ GPUtil not installed — utilization/temperature unavailable.")
        except Exception as e:
            logger.warning(f"⚠️ GPUtil query failed: {e}")

    except Exception as e:
        logger.warning(f"⚠️ GPU stats collection failed: {e}")

    return stats


def create_gpu_dashboard(history: List[dict]) -> go.Figure:
    """
    Create a 4-panel interactive Plotly dashboard from a list of GPU stat snapshots.

    Dashboard layout (2 rows × 2 columns):
        ┌──────────────────┬──────────────────┐
        │ GPU Utilization% │  Memory Usage MB  │
        │   (line chart)   │   (area chart)    │
        ├──────────────────┼──────────────────┤
        │  Temperature °C  │  Current Stats   │
        │   (line chart)   │ (gauge meters)   │
        └──────────────────┴──────────────────┘

    Why Plotly?
        Plotly creates interactive charts that you can zoom, hover, and explore
        directly inside Jupyter notebooks with `fig.show()`.

    Args:
        history: A list of stat dicts, each produced by get_gpu_stats().
                 Each dict must have: timestamp, utilization, memory_used_mb,
                 memory_total_mb, temperature.

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
    timestamps    = [s.get("timestamp", "") for s in history]
    utilizations  = [s.get("utilization", 0) for s in history]
    memory_used   = [s.get("memory_used_mb", 0) for s in history]
    memory_total  = [s.get("memory_total_mb", 0) for s in history]
    temperatures  = [s.get("temperature", 0) for s in history]

    # Get the most recent snapshot for the gauge indicators
    latest = history[-1]
    gpu_name = latest.get("gpu_name", "GPU")

    # Create a 2×2 subplot grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "GPU Utilization (%)",
            "Memory Usage (MB)",
            "Temperature (°C)",
            f"Current Stats — {gpu_name}",
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],            # Row 1: line/area charts
            [{"type": "xy"}, {"type": "indicator"}],     # Row 2: line + gauge indicators
        ],
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
            fillcolor="rgba(76,175,80,0.15)",      # Light green fill under the line
        ),
        row=1, col=1,
    )

    # Add a visual warning zone at 80% utilization
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="orange",
        annotation_text="80% — High",
        row=1, col=1,
    )

    # ── Panel 2: Memory Usage (area chart showing used vs. total) ─────────────
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=memory_total,
            mode="lines",
            name="Total VRAM",
            line=dict(color="#9E9E9E", width=1, dash="dot"),  # Grey dotted = capacity ceiling
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=memory_used,
            mode="lines+markers",
            name="Used VRAM",
            line=dict(color="#2196F3", width=2),              # Blue line = actual usage
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.15)",
        ),
        row=1, col=2,
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
        row=2, col=1,
    )
    # Add a 80°C warning threshold line — sustained temperatures above this
    # can throttle GPU performance and reduce hardware lifespan
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="red",
        annotation_text="80°C — Warning",
        row=2, col=1,
    )

    # ── Panel 4: Gauge Indicators (current snapshot values) ──────────────────
    # Gauge 1 — Current Utilization
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=latest.get("utilization", 0),
            title={"text": "Utilization %", "font": {"size": 11}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4CAF50"},
                "steps": [
                    {"range": [0, 50],   "color": "#e8f5e9"},  # Light green = low
                    {"range": [50, 80],  "color": "#fff3e0"},  # Yellow = moderate
                    {"range": [80, 100], "color": "#ffebee"},  # Red = high
                ],
            },
            domain={"row": 0, "column": 0},
        ),
        row=2, col=2,
    )

    # ── Layout and Styling ────────────────────────────────────────────────────
    fig.update_layout(
        title={
            "text": "🖥️ GPU Monitoring Dashboard",
            "x": 0.5,  # Center the title
            "font": {"size": 18},
        },
        template="plotly_white",   # Clean white background — works well in both light/dark notebooks
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )

    # Add axis labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Utilization (%)", row=1, col=1)
    fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)

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
            mlflow.log_metric("gpu_utilization",    stats.get("utilization", 0))
            mlflow.log_metric("gpu_memory_used_mb", stats.get("memory_used_mb", 0))
            mlflow.log_metric("gpu_temperature",    stats.get("temperature", 0))
    except ImportError:
        logger.warning("⚠️ MLflow not available — skipping metric logging.")
    except Exception as e:
        logger.warning(f"⚠️ MLflow metric logging failed: {e}")


class GPUMonitor:
    """
    Stateful GPU monitor that collects snapshots over time and produces dashboards.

    Usage pattern in notebooks:
        monitor = GPUMonitor()
        # ... run some AI code ...
        monitor.snapshot()          # Collect a data point
        # ... run more AI code ...
        monitor.snapshot()          # Collect another data point
        fig = monitor.dashboard()   # Build the 4-panel dashboard
        fig.show()                  # Display it inline in the notebook

    Why "stateful"?
        Each `snapshot()` call appends to an internal list (`self.history`).
        At the end of your notebook you can see how GPU usage evolved over time.
    """

    def __init__(self):
        """Initialize the monitor with an empty history list."""
        self.history: List[dict] = []   # Will store one dict per snapshot

    def snapshot(self) -> dict:
        """
        Collect one GPU stat reading and add it to the history.

        Returns:
            The stats dict that was just collected (also stored in self.history)
        """
        stats = get_gpu_stats()        # Collect current readings
        self.history.append(stats)     # Remember this snapshot
        return stats

    def dashboard(self) -> go.Figure:
        """
        Build and return a 4-panel Plotly dashboard from all collected snapshots.

        Returns:
            A Plotly Figure — display with fig.show() in a notebook cell
        """
        return create_gpu_dashboard(self.history)

    def log_to_mlflow(self) -> None:
        """Log the most recent snapshot to the active MLflow run (if one exists)."""
        if self.history:
            log_gpu_metrics_to_mlflow(self.history[-1])  # Only log the latest reading

    def summary(self) -> str:
        """
        Return a formatted text summary of the current GPU state.

        Returns:
            A multi-line string with key GPU stats, suitable for printing.
        """
        if not self.history:
            return "No GPU snapshots collected yet. Call monitor.snapshot() first."

        latest = self.history[-1]
        return (
            f"GPU: {latest.get('gpu_name', 'N/A')}\n"
            f"  Utilization : {latest.get('utilization', 0):.1f}%\n"
            f"  Memory Used : {latest.get('memory_used_mb', 0):.0f} MB "
            f"/ {latest.get('memory_total_mb', 0):.0f} MB "
            f"({latest.get('memory_percent', 0):.1f}%)\n"
            f"  Temperature : {latest.get('temperature', 0):.1f}°C\n"
            f"  Snapshot at : {latest.get('timestamp', 'N/A')}"
        )
