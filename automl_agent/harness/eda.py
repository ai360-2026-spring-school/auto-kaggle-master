"""
EDA toolkit.

The agent does *real* EDA by calling these functions in its sandbox and
reading the results — it is not handed a static report and told to pretend.
`profile()` is the cheap first look; the agent then drills in with
`target_relation()`, `interaction_scan()`, `leakage_scan()` etc. on whatever
columns its hypotheses point to.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "integer" if pd.api.types.is_integer_dtype(s) else "float"
    return "categorical"


def profile(df: pd.DataFrame, target: str | None = None) -> dict:
    """Compact, machine-readable schema + quality snapshot."""
    out = {"n_rows": len(df), "n_cols": df.shape[1], "columns": {}}
    for c in df.columns:
        s = df[c]
        info = {
            "kind": _kind(s),
            "missing_frac": round(float(s.isna().mean()), 4),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if info["kind"] in ("integer", "float"):
            d = s.dropna()
            if len(d):
                info.update(
                    min=float(d.min()), max=float(d.max()),
                    mean=float(d.mean()), std=float(d.std() or 0),
                    skew=float(d.skew()) if len(d) > 2 else 0.0,
                    zeros_frac=round(float((d == 0).mean()), 4),
                )
        elif info["kind"] in ("categorical", "bool"):
            vc = s.value_counts(dropna=True).head(5)
            info["top_values"] = {str(k): int(v) for k, v in vc.items()}
            info["high_cardinality"] = info["n_unique"] > 0.5 * len(df)
        out["columns"][c] = info
    if target and target in df.columns:
        out["target"] = {
            "name": target,
            "kind": _kind(df[target]),
            "distribution": {
                str(k): int(v)
                for k, v in df[target].value_counts(dropna=False).head(10).items()
            } if df[target].nunique() <= 30 else {
                "min": float(df[target].min()),
                "max": float(df[target].max()),
                "mean": float(df[target].mean()),
                "skew": float(df[target].skew()),
            },
        }
    return out


def target_relation(df: pd.DataFrame, target: str, top: int = 25) -> pd.DataFrame:
    """Mutual information of each feature with the target (model-agnostic
    signal strength, captures non-linear relations a correlation misses)."""
    y = df[target]
    X = df.drop(columns=[target]).copy()
    discrete = []
    for i, c in enumerate(X.columns):
        if _kind(X[c]) in ("categorical", "bool", "integer"):
            X[c] = X[c].astype("category").cat.codes
            discrete.append(True)
        else:
            X[c] = X[c].fillna(X[c].median())
            discrete.append(False)
    is_clf = y.dtype == object or y.nunique() <= max(20, int(0.05 * len(y)))
    if is_clf:
        mi = mutual_info_classif(X, y.astype("category").cat.codes,
                                 discrete_features=discrete, random_state=0)
    else:
        mi = mutual_info_regression(X, y, discrete_features=discrete,
                                    random_state=0)
    return (pd.Series(mi, index=X.columns)
            .sort_values(ascending=False).head(top).to_frame("mutual_info"))


def leakage_scan(df: pd.DataFrame, target: str, thresh: float = 0.999) -> list:
    """Flag columns suspiciously predictive of the target (possible leaks)."""
    flags = []
    y = df[target]
    num = df.select_dtypes(include=np.number)
    if target in num and y.nunique() > 2:
        corr = num.corrwith(y).abs().drop(labels=[target], errors="ignore")
        flags += [f"{c}: |corr|={v:.4f}" for c, v in corr.items() if v > 0.98]
    for c in df.columns:
        if c == target:
            continue
        g = df.groupby(c, observed=True)[target].nunique()
        if len(g) > 1 and (g <= 1).mean() > thresh and df[c].nunique() > 5:
            flags.append(f"{c}: each value maps to a single target (likely id/leak)")
    return flags


def interaction_scan(df: pd.DataFrame, target: str, cols: list,
                      max_pairs: int = 8) -> pd.DataFrame:
    """Quick check whether pairwise products/ratios beat single features by MI."""
    rows = []
    nums = [c for c in cols if _kind(df[c]) in ("integer", "float")][:6]
    y = df[target]
    is_clf = y.dtype == object or y.nunique() <= 20
    yv = y.astype("category").cat.codes if is_clf else y
    fn = mutual_info_classif if is_clf else mutual_info_regression
    seen = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if seen >= max_pairs:
                break
            a, b = nums[i], nums[j]
            prod = (df[a].fillna(0) * df[b].fillna(0)).to_frame("x")
            ratio = (df[a] / df[b].replace(0, np.nan)).fillna(0).to_frame("x")
            mi_p = fn(prod, yv, random_state=0)[0]
            mi_r = fn(ratio, yv, random_state=0)[0]
            rows.append({"pair": f"{a}*{b}", "mi": float(mi_p)})
            rows.append({"pair": f"{a}/{b}", "mi": float(mi_r)})
            seen += 1
    return pd.DataFrame(rows).sort_values("mi", ascending=False)
