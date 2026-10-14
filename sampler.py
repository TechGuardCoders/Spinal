"""Metrics sampler: polls vLLM /metrics in a background thread while the
tracer's requests are in flight, recording engine state over time."""
import threading
import time
import urllib.request


def fetch_gauges(base):
    """Pull the gauges we correlate with request traces."""
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=5) as r:
            text = r.read().decode()
    except Exception:
        return None
    out = {"running": 0.0, "waiting": 0.0, "kvCache": None}
    for line in text.split("\n"):
        if line.startswith("vllm:num_requests_running{"):
            try:
                out["running"] = float(line.rsplit("}", 1)[1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("vllm:num_requests_waiting{"):
            try:
                out["waiting"] = float(line.rsplit("}", 1)[1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("vllm:kv_cache_usage_perc{"):
            try:
                out["kvCache"] = float(line.rsplit("}", 1)[1])
            except (ValueError, IndexError):
                pass
    return out


def fetch_fleet_gauges(bases):
    """Poll /metrics on every node; aggregate across the fleet.

    Counters-style gauges (running, waiting) are summed; bounded gauges
    (kvCache) are merged as the fleet max. Per-node values are kept under
    "nodes" keyed by base URL so a single hot lane stays visible.
    """
    nodes = {}
    for base in bases:
        g = fetch_gauges(base)
        if g is not None:
            nodes[base] = g
    if not nodes:
        return None
    return {
        "running": sum(n["running"] for n in nodes.values()),
        "waiting": sum(n["waiting"] for n in nodes.values()),
        "kvCache": max((n["kvCache"] for n in nodes.values() if n["kvCache"] is not None), default=None),
        "nodes": nodes,
    }


class MetricsSampler:
    def __init__(self, base, interval=1.0):
        """base: a single base URL or a list of fleet node base URLs."""
        self.bases = [base] if isinstance(base, str) else list(base)
        self.base = self.bases[0]  # back-compat for callers reading .base
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            if len(self.bases) == 1:
                g = fetch_gauges(self.bases[0])
            else:
                g = fetch_fleet_gauges(self.bases)
            if g is not None:
                self.samples.append({"ts": time.time(), **g})
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
