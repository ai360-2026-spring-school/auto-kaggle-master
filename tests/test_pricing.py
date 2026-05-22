"""Unit tests for agent.pricing."""
from agent.pricing import cost_rub, get_rate


def test_known_model_qwen():
    # qwen3-235b-a22b-fp8: 500/500 ₽ per 1M tokens
    assert get_rate("qwen3-235b-a22b-fp8") is not None
    c = cost_rub("qwen3-235b-a22b-fp8", 1_000_000, 1_000_000)
    assert abs(c - 1000.0) < 1e-6


def test_known_model_deepseek_asymmetric():
    # deepseek-v32: 500 input / 800 output
    c = cost_rub("deepseek-v32", 1_000_000, 1_000_000)
    assert abs(c - 1300.0) < 1e-6


def test_known_model_gpt_oss_20b_cheapest():
    # gpt-oss-20b: 100/100
    c = cost_rub("gpt-oss-20b", 1_000_000, 1_000_000)
    assert abs(c - 200.0) < 1e-6


def test_yandexgpt_5_pro_vs_51():
    c5 = cost_rub("yandexgpt", 1_000_000, 1_000_000)         # 1200/1200
    c51 = cost_rub("yandexgpt-5.1", 1_000_000, 1_000_000)     # 800/800
    assert abs(c5 - 2400.0) < 1e-6
    assert abs(c51 - 1600.0) < 1e-6
    assert c51 < c5


def test_case_insensitive():
    a = cost_rub("Qwen3-235B-A22B-FP8", 100, 50)
    b = cost_rub("qwen3-235b-a22b-fp8", 100, 50)
    assert a == b


def test_strips_gpt_uri_prefix():
    a = cost_rub("gpt://b1g.../qwen3-235b-a22b-fp8/latest", 100_000, 50_000)
    b = cost_rub("qwen3-235b-a22b-fp8", 100_000, 50_000)
    assert a == b


def test_unknown_model_returns_none():
    assert cost_rub("totally-unknown-model", 1000, 1000) is None
    assert get_rate("nonexistent") is None


def test_zero_tokens():
    assert cost_rub("qwen3-235b-a22b-fp8", 0, 0) == 0.0


def test_partial_tokens():
    # 250k input + 100k output for qwen3 at 500/500 = (250+100)*500/1000 = 175
    c = cost_rub("qwen3-235b-a22b-fp8", 250_000, 100_000)
    assert abs(c - 175.0) < 1e-6
