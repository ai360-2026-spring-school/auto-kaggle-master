# Autonomous tabular-ML competition agent

An agentic pipeline that takes `(task description, data description, train,
test)` and autonomously produces `submission.csv`. The ML model is **frozen**;
the agent's entire job is data — feature engineering before the model and
prediction shaping after it.

The design is a direct port of the [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
philosophy to tabular ML:

| autoresearch | this repo |
|---|---|
| `prepare.py` — locked infra, owns the metric (`val_bpb`) | `harness/` — locked, owns CV + the competition metric |
| `train.py` — the one file the agent rewrites | `solution.py` — the one file the agent rewrites |
| `program.md` — human-edited research-org instructions | `program.md` — same role, edited by you |
| fixed 5-min budget → comparable runs | fixed CatBoost + fixed CV protocol → comparable runs |
| modify → train 5 min → keep/discard → repeat | propose → leak-safe CV → keep/discard → repeat |

## The core problem and the core idea

You asked for the agent to be able to do *anything* to the data before the
model and *anything* to the predictions after. Taken literally that destroys
the only thing that makes autonomy work: a trustworthy score. A free-form
script that mutates the whole dataframe will, sooner or later, leak the target
or contaminate train with test statistics, and then the CV number — the agent's
only feedback signal — becomes a lie that the loop will happily optimize.

The resolution is a **contract** instead of a script (`harness/contract.py`).
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
stays honest. This is the whole trick.

## Components

```
harness/            LOCKED. Agent imports only BaseSolution.
  config.py          budgets + the frozen CatBoost params
  data.py            target/id detection, problem-type framing
  metrics.py         metric registry + auto-selection from task text
  model.py           the frozen CatBoost black box
  contract.py        fit/transform/postprocess + structural leak guards
  cv.py              leakage-safe CV — the single ground-truth number
  eda.py             real EDA tools the agent calls for evidence
  sandbox.py         static screening + isolated import of solution.py
agent/
  llm.py             AnthropicAgent (real) + OfflineExpertAgent (offline)
  prompts.py         system/iteration prompt assembly
  loop.py            propose → evaluate → keep/discard → journal → repeat
solution_template.py the baseline the agent starts from and rewrites
program.md           human-authored strategy (edit this, not the harness)
run.py               orchestrate; refit winner on full train; write submission
examples/make_demo.py synthetic competition for an end-to-end smoke test
```

## How the loop works

1. The baseline `solution_template.py` is scored by leak-safe CV → incumbent.
2. Each iteration the agent receives: problem framing, the metric and its
   direction, the incumbent source + score, the experiment journal, and the
   EDA notes gathered so far. It (optionally) calls EDA tools, forms one
   hypothesis, and emits a complete new `solution.py`.
3. The harness statically screens it (no `catboost`/network/file-IO/`eval`),
   imports it, and runs the **same** CV protocol.
4. A candidate replaces the incumbent only if it beats it by
   `min_improvement_sigmas × fold-std` — a noise-aware margin so the loop
   chases real signal, not CV variance.
5. Everything is appended to `journal.jsonl`. Leave it running; read the log
   later, exactly like autoresearch's overnight runs.
6. `run.py` refits the final incumbent once on the full training set and writes
   `submission.csv`.

## Why CatBoost is the frozen model

It is a strong, low-variance default on heterogeneous tabular data, handles raw
categoricals and missing values natively, and has built-in early stopping — so
a single untuned config performs acceptably across classification and
regression without the agent needing to touch modeling. The agent's leverage is
deliberately confined to data, which is where most competition signal actually
lives.

## Run it

```bash
pip install -e ".[full]"     # base deps + all optional backends + ydata-profiling
cd automl_agent
python examples/make_demo.py
python run.py \
  --task  examples/demo/task.txt \
  --data  examples/demo/data.txt \
  --train examples/demo/train.csv \
  --test  examples/demo/test.csv \
  --out   submission.csv \
  --max-iters 8
```

### Backends

| `--backend` | LLM | Tool use | Requires |
|---|---|---|---|
| `auto` | first available below | — | — |
| `gigachat` | GigaChat-2-Max | yes (ReAct) | `GIGACHAT_CREDENTIALS` env var |
| `yandex` | YandexGPT | yes (ReAct) | `YANDEX_FOLDER_ID` + `YANDEX_API_KEY` |
| `anthropic` | Claude (single-shot) | no | `ANTHROPIC_API_KEY` |
| `offline` | none — fixed curriculum | — | nothing |

ReAct backends (`gigachat`, `yandex`) drive a multi-turn python sandbox per
iteration: the model can call `python_exec`, `read_incumbent`, `read_journal`,
`add_insight`, and finally `submit_solution` once it has a hypothesis. The
sandbox preloads `train`, `test`, `spec`, `oof`, `feature_importance`, plus
the harness `eda` module and `pd`/`np`/`scipy`/`sklearn`.

Without any LLM credentials, the deterministic `OfflineExpertAgent` walks a
curriculum of standard tabular techniques — the whole system runs end-to-end
with zero external dependencies and serves as a competent baseline.

### Auto-EDA seed

If `ydata-profiling` is installed, the first thing every run does is generate
a minimal-mode profile of the (sampled) training data and distill its alerts,
correlations, and per-column stats into `runs/<workdir>/eda_notebook.md`.
That notebook is fed into every iteration's prompt, so iteration 0 starts
with a rich data picture instead of a blank slate. Without the package the
loop falls back to the lighter `harness.eda.profile`.

### Sample real-competition commands

```bash
# Flood prediction (regression, R²)
python run.py \
  --task  ../competitions/competition-1/overview.txt \
          ../competitions/competition-1/data.txt \
  --train ../competitions/competition-1/train.csv \
  --test  ../competitions/competition-1/test.csv \
  --out   ../competitions/competition-1/submission.csv \
  --backend gigachat --max-iters 12 \
  --max-seconds-total 21600 --max-seconds-per-eval 1800

# F1 Pit Next Lap (binary classification, ROC AUC)
python run.py \
  --task  ../competitions/competition-2/overview.txt \
          ../competitions/competition-2/data.txt \
  --train ../competitions/competition-2/train.csv \
  --test  ../competitions/competition-2/test.csv \
  --out   ../competitions/competition-2/submission.csv \
  --backend yandex --max-iters 10
```

## Limits and natural extensions

- The sandbox is a guardrail (static AST screen + import isolation), not a
  security boundary. For untrusted code run the process in a
  container/seccomp jail — autoresearch's "disable all permissions" is likewise
  enforced at the sandbox layer, not the prompt.
- CV is the metric; a held-out gap vs CV reflects genuine distribution shift,
  not leakage (leakage would show CV ≫ test, e.g. 0.99 vs 0.81).
- Natural next steps: time-aware / grouped CV when the task implies it, an
  adversarial-validation feature stability check, OOF-prediction post-processing
  fit on a nested split, and giving the LLM agent a live code-execution tool so
  it runs EDA itself mid-turn rather than reading a digest.
```
