"""Offline tests for Spinal logic - no network, no key."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workload import PROMPT_CLASSES, WEIGHTS, pick_prompt  # noqa: E402
from sampler import fetch_gauges  # noqa: E402

SAMPLE_METRICS = '''# HELP vllm:num_requests_running running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="glm-5.3-flash"} 2.0
# HELP vllm:num_requests_waiting waiting
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="glm-5.3-flash"} 1.0
# HELP vllm:kv_cache_usage_perc kv
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="glm-5.3-flash"} 0.37
'''


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def read(self):
        return SAMPLE_METRICS.encode()


def test_workload_covers_all_classes():
    rng = __import__("random").Random(3)
    classes = [pick_prompt(rng)[0] for _ in range(1500)]
    assert set(classes) == set(WEIGHTS.keys())


def test_workload_weights_approx():
    rng = __import__("random").Random(3)
    classes = [pick_prompt(rng)[0] for _ in range(1500)]
    from collections import Counter
    c = Counter(classes)
    for cls, w in WEIGHTS.items():
        assert abs(c[cls] / 1500 - w) < 0.07, f"{cls} off"


def test_fetch_gauges_parses():
    import unittest.mock as mock
    with mock.patch("urllib.request.urlopen", return_value=FakeResponse()):
        g = fetch_gauges("http://fake:8000")
    assert g["running"] == 2.0
    assert g["waiting"] == 1.0
    assert abs(g["kvCache"] - 0.37) < 1e-6


def test_fetch_gauges_fails_graceful():
    def boom(*a, **k):
        raise OSError("down")
    import unittest.mock as mock
    with mock.patch("urllib.request.urlopen", side_effect=boom):
        assert fetch_gauges("http://fake:8000") is None
