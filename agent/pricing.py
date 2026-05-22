"""
Per-model pricing tables.

Mirrors the prices documented in `config.yaml` (synchronous mode, ₽ per 1M
tokens). Used by the backends to attach `cost_rub` to TOKEN_USAGE journal
events. Prices change — update both this file and `config.yaml` when they
do.

Keys are matched case-insensitively against the model name the backend
emits. For Yandex AI Studio short names (`qwen3-235b-a22b-fp8`,
`yandexgpt-5.1`, ...) we strip an optional `gpt://<folder>/.../latest`
wrapper before lookup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Rate:
    input_per_1m_rub: float
    output_per_1m_rub: float

    def cost_rub(self, in_toks: int, out_toks: int) -> float:
        return (in_toks * self.input_per_1m_rub
                + out_toks * self.output_per_1m_rub) / 1_000_000.0


# Yandex AI Studio — synchronous, ₽ / 1M tokens.
# Source: https://aistudio.yandex.ru/docs/ru/ai-studio/pricing
_YANDEX = {
    # gRPC native YandexGPT
    "yandexgpt":         Rate(1200, 1200),   # = YandexGPT 5 Pro
    "yandexgpt-5-pro":   Rate(1200, 1200),
    "yandexgpt-5.1":     Rate(800, 800),     # = YandexGPT 5.1 Pro
    "yandexgpt-lite":    Rate(200, 200),
    # OpenAI-compat open-weight catalog
    "qwen3-235b-a22b-fp8": Rate(500, 500),
    "deepseek-v32":        Rate(500, 800),
    "gpt-oss-120b":        Rate(300, 300),
    "gpt-oss-20b":         Rate(100, 100),
    "qwen3.6-35b":         Rate(200, 300),
}

# GigaChat — placeholder; fill in if/when official prices are known.
# Setting None disables cost calculation rather than reporting bogus zero.
_GIGACHAT: dict[str, Rate] = {
    # Example shape (rates are not authoritative — replace with current
    # contract numbers before relying on them):
    # "GigaChat-2-Max":   Rate(<input>, <output>),
}


_ALL = {**_YANDEX, **_GIGACHAT}


def _normalize(model: str) -> str:
    m = model.strip().lower()
    # Strip a Yandex URI prefix and trailing /latest /rc /etc.
    if m.startswith("gpt://"):
        # gpt://<folder>/<name>/<ver>  → <name>
        parts = m.split("/")
        if len(parts) >= 4:
            m = parts[3]
        elif len(parts) >= 3:
            m = parts[2]
    return m


def get_rate(model: str) -> Optional[Rate]:
    """Return Rate for a model name or None if unknown."""
    if not model:
        return None
    key = _normalize(model)
    if key in _ALL:
        return _ALL[key]
    # Tolerate trailing version suffixes ('/latest', etc.) on bare names
    base = key.split("/")[0]
    return _ALL.get(base)


def cost_rub(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """₽ cost for this completion, or None when pricing is unknown."""
    rate = get_rate(model)
    if rate is None:
        return None
    return rate.cost_rub(int(input_tokens or 0), int(output_tokens or 0))
