"""
Entrypoint.

    python run.py --task task.txt --data data.txt \\
        --train train.csv --test test.csv --out submission.csv

Runs the autonomous research loop, then takes the final incumbent solution,
fits it ONCE on the full training set (fit/transform contract guarantees no
leak even here), predicts the test set, applies postprocess, and writes
submission.csv. Defaults to the offline expert agent so it runs end-to-end
out of the box; set --backend gigachat or yandex (with the right env vars)
to run with a real LLM.
"""
from __future__ import annotations

import os
from pathlib import Path as _Path

# MUST be set BEFORE grpc / yandex-cloud-ml-sdk load any transitive gRPC.
# Yandex Cloud signs its TLS certs with the Russian Trusted Root CA, which
# is NOT in Mozilla's bundle (= certifi). We ship a combined bundle
# (certifi + Yandex root CA) at the repo root as `yandex_combined_ca.pem`
# and point gRPC at it. Env var is read once at C-library init.
if "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH" not in os.environ:
    _combined_ca = _Path(__file__).resolve().parent / "yandex_combined_ca.pem"
    if _combined_ca.exists():
        os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = str(_combined_ca)
    else:
        try:
            import certifi as _certifi
            os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = _certifi.where()
        except Exception:
            pass

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import HarnessConfig, load_solution  # noqa: E402
from agent import ResearchLoop, OfflineExpertAgent, make_agent  # noqa: E402


def _resolve_backend(name: str) -> str:
    """`auto` picks the best available backend based on env vars."""
    if name != "auto":
        return name
    if os.environ.get("GIGACHAT_CREDENTIALS"):
        return "gigachat"
    if os.environ.get("YANDEX_API_KEY") and os.environ.get("YANDEX_FOLDER_ID"):
        return "yandex"
    return "offline"


def _fit_full_and_predict(inc_path, spec, cfg):
    """Refit the winning solution on ALL training rows and predict test."""
    from harness.model import LockedModel
    from harness.cv import _encode_target

    sol = load_solution(inc_path)
    y_raw = spec.train[spec.target_col]
    X = spec.train.drop(columns=[spec.target_col])
    sol.fit(X, y_raw, spec)
    f_tr = sol.transform(X)
    f_te = sol.transform(spec.test)

    # small internal holdout just for CatBoost early stopping
    n = len(f_tr)
    rng = np.random.RandomState(cfg.random_seed)
    idx = rng.permutation(n)
    cut = int(n * 0.9)
    tr, va = idx[:cut], idx[cut:]
    y_enc = _encode_target(y_raw, spec)

    model = LockedModel(spec.problem_type, spec.n_classes, cfg.catboost_params)
    model.fit(f_tr.iloc[tr], y_enc[tr], f_tr.iloc[va], y_enc[va])
    raw = model.predict(f_te)
    return sol.postprocess(raw, spec.test)


def _format_submission(pred, spec) -> pd.DataFrame:
    id_col = spec.id_col or "id"
    ids = (spec.test[spec.id_col] if spec.id_col in spec.test.columns
           else pd.RangeIndex(len(spec.test)))
    tgt = spec.target_col
    if spec.problem_type == "binary":
        out = pd.DataFrame({id_col: ids, tgt: np.asarray(pred).ravel()})
    elif spec.problem_type == "multiclass":
        cls = np.array(spec.class_labels)
        out = pd.DataFrame({id_col: ids, tgt: cls[np.argmax(pred, axis=1)]})
    else:
        out = pd.DataFrame({id_col: ids, tgt: np.asarray(pred).ravel()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, nargs="+",
                    help="Path(s) to task description text file(s). Multiple "
                         "files are concatenated with blank lines.")
    ap.add_argument("--data", required=True, nargs="+",
                    help="Path(s) to data description file(s). Concatenated.")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--backend",
                    choices=["auto", "gigachat", "yandex", "yandex-openai",
                             "offline"],
                    default="auto")
    ap.add_argument("--model", default=None,
                    help="Model name override (e.g. qwen3-235b-a22b-fp8, "
                         "deepseek-v3.2, gpt-oss-120b for yandex-openai; "
                         "yandexgpt-5-pro for yandex; GigaChat-2-Max for gigachat).")
    ap.add_argument("--workdir", default="runs/latest")
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--cv-folds", type=int, default=None)
    ap.add_argument("--max-seconds-per-eval", type=float, default=None)
    ap.add_argument("--max-seconds-total", type=float, default=None)
    ap.add_argument("--max-tool-calls", type=int, default=None,
                    help="Tool calls per iteration cap for ReAct backends.")
    args = ap.parse_args()

    task = "\n\n".join(Path(p).read_text() for p in args.task)
    data = "\n\n".join(Path(p).read_text() for p in args.data)
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    cfg = HarnessConfig()
    cfg.paths.workdir = args.workdir
    if args.max_iters is not None:
        cfg.budget.max_iterations = args.max_iters
    if args.cv_folds is not None:
        cfg.budget.cv_folds = args.cv_folds
    if args.max_seconds_per_eval is not None:
        cfg.budget.max_seconds_per_eval = args.max_seconds_per_eval
    if args.max_seconds_total is not None:
        cfg.budget.max_seconds_total = args.max_seconds_total

    backend = _resolve_backend(args.backend)
    agent_kwargs = {}
    if args.max_tool_calls is not None and backend in ("gigachat", "yandex",
                                                       "yandex-openai"):
        agent_kwargs["max_tool_calls"] = args.max_tool_calls
    if args.model is not None and backend in ("gigachat", "yandex",
                                              "yandex-openai"):
        agent_kwargs["model"] = args.model
    if backend == "yandex-openai":
        # Avast/corporate TLS interception breaks strict OpenSSL on Windows.
        agent_kwargs.setdefault("verify_ssl", False)
    agent = make_agent(backend, **agent_kwargs)
    use_tool_prompt = backend in ("gigachat", "yandex", "yandex-openai")
    print(f"agent backend: {type(agent).__name__}")

    program_md = str(Path(__file__).resolve().parent / "program.md")
    loop = ResearchLoop(
        cfg, agent, program_md,
        use_tool_prompt=use_tool_prompt,
        max_tool_calls=args.max_tool_calls or 15,
    )
    spec, metric, incumbent, inc_path = loop.run(task, data, train, test)

    print(f"\nBest CV {metric.name}: {incumbent.score:.6f} "
          f"(+/- {incumbent.score_std:.6f}) "
          f"over {len(incumbent.fold_scores)} folds")

    pred = _fit_full_and_predict(inc_path, spec, cfg)
    sub = _format_submission(pred, spec)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    sub.to_csv(Path(args.workdir) / Path(args.out).name, index=False)
    print(f"submission written: {out_path}  shape={sub.shape}")
    print(sub.head().to_string(index=False))


if __name__ == "__main__":
    main()
