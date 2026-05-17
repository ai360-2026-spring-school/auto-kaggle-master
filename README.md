# Autonomous tabular-ML competition agent

An agentic pipeline that takes `(task description, data description, train, test)`
and autonomously produces `submission.csv`. The ML model is **frozen**; the
agent's entire job is data — feature engineering before the model and prediction
shaping after it. Inside each research iteration, a tool-using LLM (GigaChat,
YandexGPT, or one of the open-weight models Yandex AI Studio serves via its
OpenAI-compatible endpoint — Qwen3-235B, DeepSeek-V3.2, GPT-OSS-120B/20B)
drives a multi-turn **ReAct loop** with a live Python sandbox over the actual
data: it runs EDA, inspects out-of-fold residuals, forms a hypothesis, and
only then commits a new `solution.py` for the harness to CV-evaluate.

The design is a direct port of [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
to tabular ML:

| autoresearch | this repo |
|---|---|
| `prepare.py` — locked infra, owns the metric (`val_bpb`) | `harness/` — locked, owns CV + the competition metric |
| `train.py` — the one file the agent rewrites | `solution.py` — the one file the agent rewrites |
| `program.md` — human-edited research-org instructions | `program.md` — same role, edited by you |
| fixed 5-min budget → comparable runs | fixed CatBoost + fixed CV protocol → comparable runs |
| modify → train 5 min → keep/discard → repeat | propose → leak-safe CV → keep/discard → repeat |

## The core problem and the core idea

You want the agent to be able to do *anything* to the data before the model and
*anything* to the predictions after. Taken literally that destroys the only
thing that makes autonomy work: a trustworthy score. A free-form script that
mutates the whole dataframe will, sooner or later, leak the target or
contaminate train with test statistics, and then the CV number — the agent's
only feedback signal — becomes a lie that the loop will happily optimize.

The resolution is a **contract** instead of a script ([harness/contract.py](harness/contract.py)).
The agent expresses its data work as a stateful estimator:

```python
class Solution(BaseSolution):
    def fit(self, train_df, y, spec): ...        # learn stats from train rows
    def transform(self, df): ...                 # apply; never sees y
    def postprocess(self, raw_pred, df): ...     # shape predictions; never sees y
```

The **harness**, not the agent, decides *when* these run. Inside CV it calls
`fit()` on the training folds only, then `transform()` on the held-out fold and
the test set. Every statistic the agent invents — target encodings, scalers,
imputers, frequency maps, aggregates — is therefore learned without ever
touching a held-out row. The agent keeps total creative freedom over *what* the
features are, but it is **structurally incapable of leaking**, so the CV number
stays honest.

## Components

```
harness/            LOCKED. Agent imports only BaseSolution.
  config.py          budgets + the frozen CatBoost params
  data.py            target/id detection, problem-type framing
  metrics.py         metric registry + auto-selection from task text
  model.py           the frozen CatBoost black box (GPU auto-detected)
  contract.py        fit/transform/postprocess + structural leak guards
  cv.py              leakage-safe CV — the single ground-truth number
  eda.py             real EDA tools the agent calls for evidence
  sandbox.py         static screening + isolated import of solution.py

agent/
  llm.py                  Agent protocol + Proposal + OfflineExpertAgent
                          (deterministic curriculum, no API key required)
  loop.py                 ResearchLoop — baseline → iterate → journal → repeat
  prompts.py              system / tool-aware / iteration prompt assembly
  exec_sandbox.py         Sandbox — persistent Python REPL the agent uses for EDA
  tools.py                ToolSpec registry: python_exec, read_incumbent,
                          read_journal, add_insight, submit_solution
  react.py                ReActDriver — backend-agnostic multi-turn tool loop
  gigachat_agent.py       GigaChat backend (langchain-gigachat + retry/timeout)
  yandex_agent.py         Yandex AI Studio gRPC backend (YandexGPT family)
  yandex_openai_agent.py  Yandex AI Studio OpenAI-compatible HTTP backend
                          (Qwen3-235B, DeepSeek-V3.2, GPT-OSS-120B/20B, ...)
  auto_profile.py         one-shot ydata-profiling seed → eda_notebook.md
  notebook.py             cross-iteration memory (eda_notebook.md helper)

solution_template.py the baseline the agent starts from and rewrites
program.md           human-authored strategy (edit this, not the harness)
run.py               orchestrate; refit winner on full train; write submission
tests/               pytest units (sandbox AST/timeout, tools dispatch, ReAct)
competitions/        real competition inputs (overview/data/train/test)
runs/                per-run workdir: journal.jsonl, eda_notebook.md, solution.py
```

## How a run unfolds

1. **Auto-EDA seed.** If `ydata-profiling` is installed, the very first thing
   `ResearchLoop.run` does is generate a minimal-mode profile of the (sampled)
   training data and distill its alerts, top correlations, missing/cardinality
   columns into `runs/<workdir>/eda_notebook.md`. Without the package it falls
   back to `harness.eda.profile` + `leakage_scan`. Either way, iteration 0
   starts with a real data picture, not a blank slate.
2. **Baseline.** `solution_template.py` is scored by leak-safe CV → incumbent.
3. **For each iteration:**
   - A fresh `Sandbox` is built around `(spec, train, test, oof, feature_importance,
     incumbent_source)` and exposed to the agent through a `ToolContext`.
   - The agent backend runs through `ReActDriver`. Internally:
     - `python_exec(code)` — persistent Python REPL preloaded with `train`,
       `test`, `spec`, `oof`, `feature_importance`, `pd`, `np`, `scipy`,
       `sklearn`, plus the harness `eda` module. 60-second wall-clock cap
       per call, AST-screened for forbidden imports (catboost / network /
       file IO / os / sys).
     - `read_incumbent()` — current `solution.py` source.
     - `read_journal(last_n=5)` — recent `RESULT`/`ACCEPT`/`EVAL_ERROR`
       events (`TOOL_*` noise filtered out).
     - `add_insight(text)` — append a note to `eda_notebook.md` so later
       iterations inherit the finding.
     - `submit_solution(code, hypothesis, expected_effect)` — terminal tool;
       validates the code (`static_check` + presence of `class Solution`),
       optionally strips markdown fences or decodes literal `\n` escapes
       that some providers emit, then closes the ReAct loop.
   - The harness statically screens the candidate, imports it in isolation,
     and runs the **same** CV protocol that produced the baseline.
   - A candidate replaces the incumbent only if it beats it by
     `min_improvement_sigmas × fold-std` — a noise-aware margin so the loop
     chases real signal, not CV variance.
4. **Logging.** Every event — `START`, `AUTO_PROFILE`, `BASELINE`, `TOOL_CALL`,
   `TOOL_RESULT`, `PROPOSE`, `RESULT`, `ACCEPT`, `EVAL_ERROR`, `AGENT_ERROR`,
   `REACT_TIMEOUT`, `DONE` — is appended to `runs/<workdir>/journal.jsonl`.
   Leave the loop running overnight; read the log in the morning.
5. **Submission.** `run.py` refits the final incumbent once on the full
   training set (90/10 internal split for CatBoost early stop only) and
   writes `submission.csv`.

## Why CatBoost is the frozen model

It is a strong, low-variance default on heterogeneous tabular data, handles
raw categoricals and missing values natively, and has built-in early stopping —
so a single untuned config performs acceptably across classification and
regression without the agent needing to touch modeling. The agent's leverage
is deliberately confined to data, which is where most competition signal
actually lives.

`LockedModel` ([harness/model.py](harness/model.py)) auto-detects a CUDA GPU
once per process and runs CatBoost on it when present, falling back to CPU
otherwise. Force CPU with `AUTOML_AGENT_CATBOOST_GPU=0`.

## Run it

```bash
pip install -e ".[full]"   # base deps + all optional backends + ydata-profiling
```

Optional dependency groups (pick what you need):

- `[gigachat]` — GigaChat backend (`GIGACHAT_CREDENTIALS`, optional
  `GIGACHAT_SCOPE`).
- `[yandex]` — YandexGPT gRPC backend (`YANDEX_FOLDER_ID`, `YANDEX_API_KEY`).
- `[yandex-openai]` — Yandex AI Studio OpenAI-compatible HTTP backend for
  open-weight models (Qwen3-235B, DeepSeek-V3.2, GPT-OSS-120B/20B). Same
  env vars as `[yandex]`.
- `[profiling]` — ydata-profiling auto-EDA seed.
- `[full]` — all of the above.

### Backends

| `--backend` | LLM | Tool use | Requires |
|---|---|---|---|
| `auto` | first available below | — | — |
| `gigachat` | GigaChat-2-Max | yes (ReAct) | `GIGACHAT_CREDENTIALS` |
| `yandex` | YandexGPT family via gRPC SDK (`--model yandexgpt`, `yandexgpt-5-pro`, `yandexgpt-5.1`, `yandexgpt-lite`, ...) | yes (ReAct) | `YANDEX_FOLDER_ID` + `YANDEX_API_KEY` |
| `yandex-openai` | Open-weight models via Yandex's OpenAI-compatible HTTP endpoint (`--model qwen3-235b-a22b-fp8`, `deepseek-v32`, `gpt-oss-120b`, `gpt-oss-20b`) | yes (ReAct) | `YANDEX_FOLDER_ID` + `YANDEX_API_KEY` |
| `offline` | none — fixed curriculum | — | nothing |

Backend choice notes:

- **GigaChat** has a built-in retry layer (4 attempts, exponential backoff
  2/4/8/16s) over transient `httpx` errors and a 180-second read timeout —
  GigaChat occasionally drops the first request after a long CatBoost CV
  pause (idle TLS reset).
- **Yandex gRPC** (`yandex`): the SDK requires a CA bundle that includes the
  Russian Trusted Root CA (and any corporate SSL inspection is the typical Windows-dev case).
  `run.py` points `GRPC_DEFAULT_SSL_ROOTS_FILE_PATH` at `./yandex_combined_ca.pem` if that
  file exists, otherwise at certifi's bundle. Build the combined bundle once
  with `cat $(python -c "import certifi; print(certifi.where())") corporate_root.pem > yandex_combined_ca.pem`.
- **Yandex OpenAI-compat** (`yandex-openai`): a subset of the AI Studio
  catalog is exposed only over the OpenAI-compatible HTTP endpoint
  (`https://llm.api.cloud.yandex.net/v1`), NOT the gRPC SDK. This backend
  uses the official `openai` Python SDK with custom auth headers (`Authorization: Api-Key …`).
- **Yandex models that map by short name**: `yandexgpt` and `yandexgpt-5-pro`
  both resolve to YandexGPT 5 Pro; `yandexgpt-5.1` is YandexGPT 5.1 Pro (the
  only 5.1-tier variant in the catalog at writing); other 5.1 short names
  return NOT_FOUND.

### Sample commands

```bash
# F1 Pit Next Lap (binary classification, ROC AUC) — GigaChat
python run.py \
  --task  competitions/competition-2/overview.txt \
  --data  competitions/competition-2/data.txt \
  --train competitions/competition-2/train.csv \
  --test  competitions/competition-2/test.csv \
  --out   competitions/competition-2/submission.csv \
  --backend gigachat --max-iters 8 --max-tool-calls 12 \
  --workdir runs/comp2

# Same task, YandexGPT 5.1 Pro via gRPC
python run.py \
  --task  competitions/competition-2/overview.txt \
  --data  competitions/competition-2/data.txt \
  --train competitions/competition-2/train.csv \
  --test  competitions/competition-2/test.csv \
  --out   competitions/competition-2/submission_yagpt.csv \
  --backend yandex --model yandexgpt-5.1 \
  --max-iters 8 --workdir runs/comp2-yagpt

# Same task, Qwen3-235B via OpenAI-compatible HTTP
python run.py \
  --task  competitions/competition-2/overview.txt \
  --data  competitions/competition-2/data.txt \
  --train competitions/competition-2/train.csv \
  --test  competitions/competition-2/test.csv \
  --out   competitions/competition-2/submission_qwen.csv \
  --backend yandex-openai --model qwen3-235b-a22b-fp8 \
  --max-iters 8 --workdir runs/comp2-qwen

# Flood prediction (regression, R²) — overnight run
python run.py \
  --task  competitions/competition-1/overview.txt \
  --data  competitions/competition-1/data.txt \
  --train competitions/competition-1/train.csv \
  --test  competitions/competition-1/test.csv \
  --out   competitions/competition-1/submission.csv \
  --backend yandex-openai --model deepseek-v32 \
  --max-iters 12 --max-seconds-total 21600 --max-seconds-per-eval 1800

# Offline (no LLM): walks the deterministic curriculum in agent/llm.py
python run.py \
  --task competitions/competition-2/overview.txt \
  --data competitions/competition-2/data.txt \
  --train competitions/competition-2/train.csv \
  --test  competitions/competition-2/test.csv \
  --backend offline --max-iters 5 --workdir runs/comp2-offline
```

### Quick demos (fast iteration)

For development, subsample any competition to a smaller demo:

```bash
python -c "import pandas as pd, pathlib as p; \
  src=p.Path('competitions/competition-2'); out=p.Path('competitions/competition-2-demo'); \
  out.mkdir(exist_ok=True); \
  pd.read_csv(src/'train.csv').head(50_000).to_csv(out/'train.csv', index=False); \
  pd.read_csv(src/'test.csv').head(20_000).to_csv(out/'test.csv', index=False); \
  import shutil; \
  [shutil.copy(src/f, out/f) for f in ('overview.txt','data.txt','sample_submission.csv')]"
```

Then run against `competitions/competition-2-demo/`. With CatBoost on GPU,
baseline CV on 50k rows finishes in ~7 minutes versus ~17 minutes on the
full 440k.

### CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--task PATH [PATH ...]` | required | Task description files (concatenated). |
| `--data PATH [PATH ...]` | required | Data description files (concatenated). |
| `--train CSV` | required | |
| `--test CSV` | required | |
| `--out FILE` | `submission.csv` | Where the final submission lands. |
| `--backend` | `auto` | One of `auto/gigachat/yandex/yandex-openai/offline`. |
| `--model` | (backend default) | Per-backend model name override (e.g. `qwen3-235b-a22b-fp8`, `deepseek-v32`, `gpt-oss-120b`, `yandexgpt-5.1`, `GigaChat-2-Max`). |
| `--workdir DIR` | `runs/latest` | Per-run journal + incumbent + notebook. |
| `--max-iters` | `25` (from `harness/config.py`) | Agent edit/evaluate cycles. |
| `--cv-folds` | `5` | StratifiedKFold/KFold splits. |
| `--max-seconds-per-eval` | `900` | Wall-clock cap per CV run. |
| `--max-seconds-total` | `3600` | Overall loop budget. |
| `--max-tool-calls` | `15` | ReAct tool calls per iteration. |

### Auto-EDA seed

With `ydata-profiling` installed, every run starts by writing a compact
markdown digest (alerts, top correlations with target, missing/cardinality
columns) into `runs/<workdir>/eda_notebook.md`. The full structured report
is also saved as `auto_profile.json`. The notebook is appended to by
`add_insight` throughout the run and read into every iteration's prompt,
so insight from iteration 2 still informs iteration 9.

## Anatomy of `runs/<workdir>/`

After a run completes, the workdir contains:

- `journal.jsonl` — every event in append-only JSON (BASELINE / PROPOSE /
  TOOL_CALL / TOOL_RESULT / RESULT / ACCEPT / EVAL_ERROR / DONE / ...).
  This is the primary debugging artifact.
- `eda_notebook.md` — auto-EDA seed + every `add_insight` call.
- `auto_profile.json` — full ydata-profiling structured report (if installed).
- `incumbent.py` — the best `solution.py` so far.
- `solution.py` — the last candidate the agent submitted (may differ from
  incumbent if it was rejected).
- `submission.csv` — final test predictions.

## Tests

```bash
pytest tests/
```

39 unit tests cover the sandbox (AST blocks, timeout, persistent namespace,
DataFrame auto-summary), tool dispatch (schema shape, fence stripping,
literal-escape decode that preserves em-dashes, `submit_solution`
validation), and the ReAct driver (success path, code-block fallback,
tool-budget exhaustion, hard failure without solution).

## Adding a new LLM backend

1. Create `agent/<provider>_agent.py` with an adapter `_<Provider>Backend`
   that turns our internal `Message` list into a provider-native API call
   and parses the response into `AssistantMessage`. ~50 lines.
2. In `<Provider>Agent.propose`, build a `ReActDriver` with that backend,
   the tool registry, and the `ToolContext` passed in. Everything else —
   multi-turn dispatch, journaling, hard caps — is shared.
3. Register the backend in `agent/__init__.py:make_agent` and `run.py`'s
   `--backend` choices.
4. Add the SDK as an optional dependency group in `pyproject.toml`.

See `agent/gigachat_agent.py`, `agent/yandex_agent.py`, and
`agent/yandex_openai_agent.py` for working references covering three quite
different SDK shapes (langchain wrapping, native gRPC with custom
ToolCallList objects, and OpenAI-compat HTTP).

## Limits and natural extensions

- The sandbox is a guardrail (static AST screen + import isolation), not a
  security boundary. For untrusted code run the process in a
  container/seccomp jail — autoresearch's "disable all permissions" is
  likewise enforced at the sandbox layer, not the prompt.
- CV is the metric; a held-out gap vs CV reflects genuine distribution
  shift, not leakage (leakage would show CV ≫ test, e.g. 0.99 vs 0.81).
- Natural next steps: time-aware / grouped CV when the task implies it
  (`fit_groups` hook into [harness/cv.py:_make_folds](harness/cv.py)),
  an adversarial-validation feature stability check, OOF-prediction
  post-processing fit on a nested split (`fit_postprocess` hook), and
  a containerized exec sandbox for untrusted runs.
