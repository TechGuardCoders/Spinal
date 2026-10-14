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


def _fake_response_for(text):
    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return text.encode()
    return R


def test_fetch_fleet_gauges_aggregates():
    import unittest.mock as mock
    from sampler import fetch_fleet_gauges

    def urlopen(req, timeout=5):
        url = req.full_url if hasattr(req, "full_url") else req
        if "182" in url:
            return _fake_response_for(SAMPLE_METRICS)()
        # second node: 1 running, 0 waiting, kv 0.5
        text = SAMPLE_METRICS.replace('} 1.0', '} 0.0').replace('} 2.0', '} 1.0').replace('} 0.37', '} 0.5')
        return _fake_response_for(text)()
    with mock.patch("urllib.request.urlopen", side_effect=urlopen):
        g = fetch_fleet_gauges(["http://182:8000", "http://157:8421"])
    assert g["running"] == 3.0
    assert g["waiting"] == 1.0
    assert abs(g["kvCache"] - 0.5) < 1e-6
    assert set(g["nodes"]) == {"http://182:8000", "http://157:8421"}


def test_fetch_fleet_gauges_skips_dead_nodes():
    import unittest.mock as mock
    from sampler import fetch_fleet_gauges

    calls = []
    def urlopen(req, timeout=5):
        url = req.full_url if hasattr(req, "full_url") else req
        calls.append(url)
        if "dead" in url:
            raise OSError("down")
        return _fake_response_for(SAMPLE_METRICS)()
    with mock.patch("urllib.request.urlopen", side_effect=urlopen):
        g = fetch_fleet_gauges(["http://dead:8422", "http://alive:8000"])
    assert g is not None
    assert set(g["nodes"]) == {"http://alive:8000"}
    assert g["running"] == 2.0


def test_fetch_fleet_gauges_all_down_is_none():
    import unittest.mock as mock
    from sampler import fetch_fleet_gauges

    def boom(*a, **k):
        raise OSError("down")
    with mock.patch("urllib.request.urlopen", side_effect=boom):
        assert fetch_fleet_gauges(["http://a:1", "http://b:2"]) is None


def test_sampler_accepts_node_list():
    from sampler import MetricsSampler
    s = MetricsSampler(["http://a:1", "http://b:2"])
    assert s.bases == ["http://a:1", "http://b:2"]
    assert s.base == "http://a:1"
    s1 = MetricsSampler("http://solo:1")
    assert s1.bases == ["http://solo:1"]
