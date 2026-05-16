"""
Tiny helper for the cross-iteration EDA notebook.

`eda_notebook.md` accumulates everything the agent learned across iterations
(ydata-profiling seed + agent-authored insights via `add_insight`). It gets
fed back into every iteration's prompt so insight from iter 2 still informs
iter 9 — the equivalent of a long-running scientist's lab notebook.
"""
from __future__ import annotations

import time
from pathlib import Path


def init_notebook(path: Path, seed: str = "") -> None:
    """Create or overwrite the notebook with a seed digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# EDA notebook\n\n" + seed.rstrip() + "\n\n---\n## Agent observations\n"
    path.write_text(header, encoding="utf-8")


def append_insight(path: Path, text: str, iteration: int | None = None) -> None:
    """Append a single agent note. `iteration` is recorded if provided."""
    if not path.exists():
        init_notebook(path)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    tag = f"iter {iteration}" if iteration is not None else stamp
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n- [{tag}] {text.strip()}\n")


def read(path: Path, max_chars: int = 16000) -> str:
    if not path.exists():
        return ""
    txt = path.read_text(encoding="utf-8")
    if len(txt) > max_chars:
        # keep the seed (head) and the most recent notes (tail)
        head = txt[: max_chars // 2]
        tail = txt[-max_chars // 2 :]
        return head + "\n...\n[truncated middle of notebook]\n...\n" + tail
    return txt
