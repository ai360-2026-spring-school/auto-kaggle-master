"""
ydata-profiling integration.

Runs ONCE at the start of `ResearchLoop.run` (before iteration 0) on a sample
of the training set, distills the structured report into a compact text
digest, and writes it into `eda_notebook.md`. The agent reads it for free
on every iteration — the model never has to spend tool calls regenerating
the obvious schema/missing/correlation table.

Heavy import (`ydata_profiling`) is lazy and optional: if the package isn't
installed, we fall back to the harness's own lightweight `eda.profile` and
log a single line about it. The run keeps working.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from harness import eda as harness_eda

from .notebook import init_notebook

DEFAULT_SAMPLE_THRESHOLD = 200_000
DEFAULT_SAMPLE_SIZE = 100_000


def seed_eda_notebook(
    spec,
    workdir: Path,
    sample_threshold: int = DEFAULT_SAMPLE_THRESHOLD,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_state: int = 42,
) -> str:
    """Compute a profile of `spec.train`, distill, and write to eda_notebook.md.

    Returns the digest string (also written to the notebook).
    """
    notebook_path = workdir / "eda_notebook.md"
    df = spec.train
    if len(df) > sample_threshold:
        df = df.sample(n=sample_size, random_state=random_state)

    try:
        from ydata_profiling import ProfileReport
    except Exception as e:  # noqa: BLE001
        digest = _fallback_digest(df, spec, reason=f"ydata-profiling unavailable ({e!r})")
        init_notebook(notebook_path, seed=digest)
        return digest

    try:
        profile = ProfileReport(df, minimal=True, title="auto-profile",
                                progress_bar=False, explorative=False,
                                samples=None, correlations=None, interactions=None)
        # `.get_description()` returns a structured dict.
        report = profile.get_description()
        digest = _distill(report, target=spec.target_col, n_total=len(spec.train))
        try:
            (workdir / "auto_profile.json").write_text(
                profile.to_json(), encoding="utf-8")
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001
        digest = _fallback_digest(df, spec,
                                  reason=f"ydata-profiling failed ({e!r})")

    init_notebook(notebook_path, seed=digest)
    return digest


def _fallback_digest(df: pd.DataFrame, spec, reason: str) -> str:
    """Cheap digest using the harness's own eda toolkit. Used when ydata fails."""
    try:
        profile = harness_eda.profile(df, target=spec.target_col)
    except Exception as e:  # noqa: BLE001
        return f"# auto-EDA seed unavailable ({reason}; fallback also failed: {e!r})\n"
    lines = [f"## auto-EDA seed (fallback — {reason})",
             f"- rows={profile['n_rows']} cols={profile['n_cols']} "
             f"target={spec.target_col!r}"]
    high_missing, high_card = [], []
    for col, info in profile["columns"].items():
        if info.get("missing_frac", 0) > 0.1:
            high_missing.append((col, info["missing_frac"]))
        if info.get("high_cardinality"):
            high_card.append((col, info["n_unique"]))
    if high_missing:
        lines.append("- columns with >10% missing: "
                     + ", ".join(f"{c}({m:.0%})" for c, m in high_missing[:10]))
    if high_card:
        lines.append("- high-cardinality columns: "
                     + ", ".join(f"{c}({n})" for c, n in high_card[:10]))
    if "target" in profile:
        lines.append(f"- target stats: {profile['target']}")
    try:
        tr = harness_eda.target_relation(df, spec.target_col, top=15)
        lines.append("- top features by mutual information with target:")
        for name, val in tr["mutual_info"].head(15).items():
            lines.append(f"    {name}: {val:.4f}")
    except Exception:
        pass
    try:
        leaks = harness_eda.leakage_scan(df, spec.target_col)
        if leaks:
            lines.append("- POSSIBLE LEAKAGE FLAGS:")
            for f in leaks:
                lines.append(f"    {f}")
    except Exception:
        pass
    return "\n".join(lines)


def _distill(report: dict, target: str, n_total: int) -> str:
    """Compress a ydata-profiling description dict into a few hundred lines max."""
    # Accept both dict and the BaseDescription pydantic model
    if not isinstance(report, dict):
        try:
            report = report.dict()
        except Exception:
            report = dict(getattr(report, "__dict__", {}))

    lines = ["## auto-EDA seed (ydata-profiling, minimal mode)"]
    table = _get(report, "table") or {}
    n_rows = table.get("n", "?")
    n_cols = table.get("n_var", table.get("n_vars", "?"))
    n_dup = table.get("n_duplicates", 0)
    lines.append(f"- profiled rows={n_rows} cols={n_cols} duplicates={n_dup} "
                 f"(full train rows={n_total}, target={target!r})")

    alerts = _get(report, "alerts") or []
    if alerts:
        lines.append("### Alerts (ydata flags)")
        for a in alerts[:30]:
            lines.append(f"  - {_alert_str(a)}")

    vars_ = _get(report, "variables") or {}
    if vars_ and target in vars_:
        tv = vars_[target]
        kind = tv.get("type") or tv.get("kind") or "?"
        n_unique = tv.get("n_unique") or tv.get("nunique") or "?"
        nstats = []
        for k in ("mean", "min", "max", "std", "skewness", "kurtosis"):
            if k in tv:
                try:
                    nstats.append(f"{k}={float(tv[k]):.4g}")
                except Exception:
                    pass
        lines.append(f"### Target ({target!r})")
        lines.append(f"  - type={kind} n_unique={n_unique} " + " ".join(nstats))

    if vars_:
        # cheap per-column digest: just missing% and cardinality, the rest is
        # available to the agent via python_exec
        rows = []
        for name, v in vars_.items():
            miss = v.get("p_missing", v.get("missing", 0))
            try:
                miss_p = float(miss)
            except Exception:
                miss_p = 0.0
            nu = v.get("n_unique") or v.get("nunique") or 0
            rows.append((name, v.get("type", "?"), miss_p, nu))
        rows.sort(key=lambda r: r[2], reverse=True)
        miss_rows = [r for r in rows if r[2] > 0.0][:15]
        if miss_rows:
            lines.append("### Columns with missing values (top 15 by %)")
            for name, kind, miss, nu in miss_rows:
                lines.append(f"  - {name}: type={kind} missing={miss:.1%} n_unique={nu}")
        # high cardinality
        hc = sorted(rows, key=lambda r: r[3], reverse=True)[:10]
        lines.append("### Highest-cardinality columns")
        for name, kind, miss, nu in hc:
            lines.append(f"  - {name}: type={kind} n_unique={nu} missing={miss:.1%}")

    # correlations: ydata returns dict-of-dataframes under 'correlations'
    corrs = _get(report, "correlations") or {}
    if corrs and target:
        for cname in ("phi_k", "spearman", "pearson", "auto"):
            cmat = corrs.get(cname)
            if cmat is None:
                continue
            try:
                ser = cmat[target].drop(labels=[target], errors="ignore")
                ser = ser.abs().sort_values(ascending=False).head(15)
            except Exception:
                continue
            lines.append(f"### Top |{cname}| correlations with {target!r}")
            for k, v in ser.items():
                lines.append(f"  - {k}: {float(v):.4f}")
            break  # only the strongest available correlation type

    return "\n".join(lines)


def _get(report: dict, key: str) -> Any:
    if isinstance(report, dict):
        return report.get(key)
    return getattr(report, key, None)


def _alert_str(alert: Any) -> str:
    if isinstance(alert, str):
        return alert
    try:
        col = getattr(alert, "column_name", None) or alert.get("column_name", "?")
        atype = getattr(alert, "alert_type", None) or alert.get("alert_type", "?")
        if hasattr(atype, "name"):
            atype = atype.name
        return f"{atype} on {col}"
    except Exception:
        return repr(alert)[:200]
