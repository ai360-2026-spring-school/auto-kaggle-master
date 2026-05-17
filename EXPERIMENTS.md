# Experiments: 5 LLMs on the same demo task

The point of this writeup is **how the models behave inside the ReAct loop**,
not "who wins the leaderboard". Every run was on a 5 000-row subset of one
binary-classification task — far too noisy to draw competitive conclusions
from. CV scores appear only as side-evidence for behavior; the headline
findings are about tool-use, exploration patterns, code quality, and
failure modes.

## Setup

- **Task**: `competitions/competition-2-demo` — F1 Pit Next Lap, binary
  classification (target `PitNextLap`), ROC AUC.
- **Data**: first 5 000 rows of train.csv, first 2 000 of test.csv. Picked
  via `df.head()` (not random) on purpose — this leaks unseen categorical
  levels into test, which is a useful stress test for whether the agents
  reach for `LabelEncoder` blind.
- **Pipeline budget**: `--max-iters 6 --max-tool-calls 12` for every model;
  sequential runs (parallel runs OOM'd a 16 GB host).
- **Hardware**: CatBoost on RTX 4060 GPU.
- **Baseline**: `solution_template.py` (drop `id`, raw features → CatBoost).
  Baseline CV ≈ 0.918 ± 0.009 — std ≈ 0.009 makes the `0.15σ ≈ 0.0013`
  margin tight for a 5k subset; **all five models stayed within noise**.
  That's the only numeric fact worth remembering across the board.

## Behavioral fingerprints

These are the patterns that actually transfer to larger datasets.

| Trait | YaGPT 5.1 Pro | GigaChat-2-Max | Qwen3-235B | DeepSeek-V3.2 | GPT-OSS-120B |
|---|---|---|---|---|---|
| Transport | gRPC | langchain HTTP | OpenAI-compat HTTP | OpenAI-compat HTTP | OpenAI-compat HTTP |
| `python_exec` calls per iter | 0 | 1-3 | **6-10 (parallel)** | 6-10 | 4-7 |
| Reads `read_journal` before deciding | no | sometimes | every iter | **every iter** | every iter |
| Multiple tool calls in one turn (parallel) | n/a | no | **yes** | sometimes | sometimes |
| Cites concrete EDA findings in hypothesis | no | no | yes ("rare drivers MSC/D223") | **yes ("zero-degradation = stint start")** | yes ("skewed numerics") |
| Quantitative expected-effect prediction | no | no | no | **yes ("≈+0.002 AUC")** | no |
| Repeats a failed hypothesis verbatim | no | yes (×3) | yes (×4) | **no** | yes (×3) |
| Builds incrementally on incumbent | partially | no | replaces | **yes — extends each iter** | partially |
| Code-quality (EVAL_ERROR rate, 6 iters) | 4/6 | 3/6 | 3/6 | **1/6** | 1-2/6 |
| Wraps assistant text in protocol tokens (e.g. `[TOOL_CALL_START]`) | **yes (leaked)** | no | no | no | no |
| Embeds unicode quirks (em-dashes, non-breaking hyphens) | yes | no | no | no | **yes (broke our stdout)** |

## What each model actually did, one paragraph each

### YandexGPT 5.1 Pro — `yandex` backend

Never called `python_exec` once — wrote candidate solutions directly into
`submit_solution` based purely on the prompt context. About half the
iterations produced a valid CV-able solution; the other half tripped on
classic bugs (`fit`-on-train-`transform`-on-test with `LabelEncoder` on
unseen labels like 'D109'/'D384', missing-helper-function errors,
referencing derived columns before defining them). Quirk: leaks internal
protocol tokens (`[TOOL_CALL_START]submit_solution`) into the assistant
text, suggesting fine-tuning artifacts in the response formatting. Also
emits em-dashes in docstrings (which caught our Windows-cp1251 printer
until we made `_log` UTF-8-safe).

### GigaChat-2-Max — `gigachat` backend

Uses tools, but lightly: typically one `python_exec` per iter (almost
always `eda.leakage_scan`), one `add_insight`, then `submit_solution`.
Fixates on "drop the id column" and proposes the same hypothesis in 3 of 6
iters — even though the baseline already drops id and the journal shows
the previous iteration was rejected with this same hypothesis. Cheapest
of the three "real" tool users by a wide margin (~12k input tokens per
iter vs Qwen/DeepSeek's 80–125k). Best fit for a fast, low-cost screen.

### Qwen3-235B-A22B-FP8 — `yandex-openai` backend

Single most distinctive trait: **dispatches 3-7 tool calls in a single
turn** — true parallel tool use, which our `ReActDriver` happily fan-dispatches
sequentially. Reads incumbent and journal aggressively. Hypothesis quality
is high (cites specific high-residual drivers like "MSC" and "D223" by
name from EDA). But: **fixates on smoothed target encoding of Driver in 5
of 6 iterations**, and never iterates away from it after the harness
rejects it repeatedly. Most expensive in tokens per iter; arguably the
most "agentic-feeling" but the least adaptive.

### DeepSeek-V3.2 — `yandex-openai` backend

Closest to what the system was designed for. Reads `read_incumbent` and
`read_journal` every iter, proposes a **different hypothesis each time**,
and crucially **builds on top of the prior iteration** rather than
restarting. Hypothesis vocabulary is concrete and quantitative:
"binary zero-indicators for LapTime_Delta, Cumulative_Degradation,
Position_Change, PitStop will capture threshold effects, +0.002 AUC".
Only 1 of 6 iterations failed CV (a single TypeError); the rest produced
valid solutions, mostly within margin. Highest token cost (~125k/iter)
but uses them on EDA rather than on retrying.

### GPT-OSS-120B — `yandex-openai` backend

Schizophrenic — alternates between conservative "frequency encode + log
transform" (which gets close to baseline) and "smoothed target encoding +
log + interactions" (which destroys CV). Not consistent about which it
picks; ignores its own previous failure. Has a habit of putting
`[TOOL_CALL_START]` markers AND em-dashes / non-breaking hyphens in
assistant text — uncovered our cp1251-print bug that earlier models had
luck not triggering.

## What the loop revealed

Independent of any one model, the experiment surfaced a few patterns in
how off-the-shelf LLMs operate inside an autoresearch-style ReAct loop on
tabular ML:

1. **Strong models still don't avoid contract violations.** All five
   models, including the largest ones, produced at least one
   `KeyError('Column not found: PitNextLap')` — they tried to access the
   target column inside `transform()`, which the harness guarantees is
   gone. The `fit()/transform()` split is described in both the system
   prompt and `program.md`, but the constraint is "deep" enough that
   models slip on first contact.

2. **CatBoost-native categorical handling is competitive with anything
   the agents propose.** Every backend that called `python_exec`
   eventually proposed "target encode Driver" — none of them noticed that
   CatBoost was already exploiting Driver via per-fold ordered target
   stats. When agent-side TE *replaced* the raw column, OOF AUC dropped
   3–7 points. When it *augmented* the raw column, it neither helped nor
   hurt much. The right answer ("don't touch the raw column") was
   discovered exactly once across 30 iterations (DeepSeek iter 0).

3. **The `0.15σ` margin rule did its job.** Four of five models came within
   0.0005 of baseline at least once (within noise). The harness rejected
   all of them. If we had used `> baseline` instead of `> baseline +
   0.15σ`, we would have accepted at least one noise-driven "win" per
   run — exactly the failure mode the autoresearch reference design
   warns against. The rejection traces in `journal.jsonl` confirm the
   safety rail held under realistic LLM-noise pressure.

4. **Repetition-blindness is the dominant failure mode of weaker models.**
   GigaChat-2-Max, GPT-OSS-120B, Qwen3-235B all repeated a verbatim
   hypothesis after seeing it rejected in the journal. The
   strategic-event-filtered journal we added to `prompts.py` (recently)
   makes prior PROPOSE/RESULT entries visible, but **reading them is not
   the same as updating on them**.

5. **Code-quality dominates score under the contract.** The single
   strongest predictor of which iterations make it to CV-evaluation
   (rather than dying in `fit/transform` with a Python exception) is the
   model's ability to write correct, harness-aware code on the first
   try. By that metric DeepSeek-V3.2 wins, by a lot, with 5/6 clean
   iterations.

6. **Provider quirks bite even after the SDK adapters work.** We hit at
   least four distinct provider-specific quirks during these runs that
   were *not* about model quality:
   - GigaChat literal-`\n` escape sequences in tool args (fixed in
     `tools.py:_decode_literal_escapes`)
   - GigaChat unclosed leading markdown fences (fixed in `_LEADING_FENCE_RE`)
   - Yandex gRPC `Avast`-CA TLS strict-validation failure (fixed by
     shipping a combined CA bundle)
   - Yandex gRPC `role="tool"` rejection + native `ToolCallList` echo
     requirement (fixed in `_to_sdk` and `Message.raw_tool_calls`)
   - Yandex OpenAI-compat `Authorization: Api-Key` non-Bearer auth (fixed
     in `_YandexOpenAIBackend`)
   - YaGPT 5.1 + GPT-OSS em-dashes / non-breaking hyphens breaking
     Windows cp1251 console print (fixed in `loop.py:_log`)

   Each fix is a 10-line patch but the cumulative engineering tax to make
   "an OpenAI-compat ReAct loop" actually work across five providers is
   non-trivial.

## What to actually pick

For this stack on classic-tabular tasks:

- **DeepSeek-V3.2** — the only model that consistently behaves like a real
  data scientist inside the loop: reads journal, proposes incrementally,
  quantitative predictions, low code-error rate. Expensive in tokens
  but cheap in iterations.
- **GigaChat-2-Max** — by far the cheapest tool-using backend; reasonable
  for fast smoke-tests or for stacked runs where you want N candidate
  hypotheses generated cheaply.
- **YandexGPT 5.1 Pro** — surprisingly competent for a model that
  doesn't actually call any EDA tools. Useful as a "blind oracle" that
  writes a candidate from prompt context alone — fast, but uneven.
- **Qwen3-235B** — most fun to watch (parallel tool dispatch) but most
  prone to a stuck-loop failure mode. Worth using on tasks where the
  EDA-heavy upfront is valuable (e.g. larger datasets where one wants to
  hammer ydata-profiling slices), less useful when iteration depth matters.
- **GPT-OSS-120B** — middle ground; nothing wrong with it, nothing it
  uniquely does better than DeepSeek.

The take-away the experiment leaves us with isn't "model X is best on
this dataset" — that would be reading too much into 5 000 rows. It's that
**the difference between models lies almost entirely in their tool-use
discipline and incremental-thinking habit**, not in their tabular ML
knowledge. All five produced reasonable data-scientist hypotheses. Only
one consistently built each iteration on the previous one.

## Reproduce

```bash
# Build the 5k demo subset (once)
python -c "import pandas as pd, shutil, pathlib as p; \
  s=p.Path('competitions/competition-2'); d=p.Path('competitions/competition-2-demo'); \
  d.mkdir(exist_ok=True); \
  pd.read_csv(s/'train.csv').head(5000).to_csv(d/'train.csv', index=False); \
  pd.read_csv(s/'test.csv').head(2000).to_csv(d/'test.csv', index=False); \
  [shutil.copy(s/f, d/f) for f in ('overview.txt','data.txt','sample_submission.csv')]"

export YANDEX_API_KEY=...   # or set $env: on PowerShell
export YANDEX_FOLDER_ID=...
export GIGACHAT_CREDENTIALS=...

# YandexGPT 5.1 Pro (gRPC)
python run.py --backend yandex --model yandexgpt-5.1 \
  --task competitions/competition-2-demo/overview.txt \
  --data competitions/competition-2-demo/data.txt \
  --train competitions/competition-2-demo/train.csv \
  --test  competitions/competition-2-demo/test.csv \
  --max-iters 6 --max-tool-calls 12 --workdir runs/yagpt

# Open-weight models via OpenAI-compat HTTP
python run.py --backend yandex-openai --model qwen3-235b-a22b-fp8 \
  ... --workdir runs/qwen
python run.py --backend yandex-openai --model deepseek-v32 \
  ... --workdir runs/deepseek
python run.py --backend yandex-openai --model gpt-oss-120b \
  ... --workdir runs/gptoss

# GigaChat-2-Max
python run.py --backend gigachat --workdir runs/gigachat ...
```

Every run leaves a full event journal in `runs/<workdir>/journal.jsonl`
and the auto-EDA seed + agent insights in `runs/<workdir>/eda_notebook.md`.
All the behavioral evidence above came from a one-pass scan of those
files; nothing in this writeup is hand-edited or paraphrased.
