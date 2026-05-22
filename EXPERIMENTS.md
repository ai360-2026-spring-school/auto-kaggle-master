# Experiment: GPT-OSS-120B on competition-2 (full, 440k rows), 10-hour ReAct run

## TL;DR

GPT-OSS-120B (served via Yandex AI Studio's OpenAI-compatible HTTP endpoint)
ran inside our autoresearch-style ReAct loop on the full F1 Pit Next Lap
competition for almost exactly the 10-hour wall-clock budget set in
`config.yaml`. It completed **92 iterations**, accepted **4 improvements**,
and lifted the held-out CV AUC from baseline **0.946133** to **0.946791**
(+0.000658 absolute, +0.07%). Cost: **~9.86M tokens, ≈2 958 ₽**.

The improvements are real (each ACCEPT cleared the 0.1σ margin rule), but
the loop spent **the vast majority of those 10 hours rediscovering the
same handful of hypotheses** rather than exploring new ones. The agent's
data-analysis habit was sharp; its strategy-update habit was not.

## What the run looked like

```
event               count
TOKEN_USAGE           899
TOOL_CALL             868
TOOL_RESULT           868
PROPOSE                92
RESULT                 92
ACCEPT                  4
EVAL_ERROR             10      (~11% of iterations: bad agent code)
REACT_TIMEOUT           1      (iter 36 hit the 900s ReAct wallclock)
```

**ACCEPT events** (every step the harness took as a real improvement):

| iter | time elapsed | score | Δ vs baseline | n features | hypothesis |
|---:|---:|---:|---:|---:|---|
| baseline | 0:00 | 0.946133 | — | 14 | drop id, raw features → CatBoost |
| 0 | ~10 min | 0.946235 | +0.000102 | 24 | frequency encodings for Driver/Compound/Race + zero-indicator flags for sparse columns + log1p of skewed numerics |
| 8 | ~1 h | 0.946571 | +0.000438 | 29 | smoothed target encoding (TE) for **Year, Stint, Driver, Race, Compound** (the trick was including Year/Stint, not just the categoricals) |
| 11 | ~1.5 h | 0.946678 | +0.000545 | 31 | sin/cos cyclic encoding of `RaceProgress` (captures periodic race-stage effect) |
| 59 | ~6.5 h | **0.946791** | **+0.000658** | 32 | `Stint_te × RaceProgress` interaction — "how a driver's typical pit propensity for a given stint changes as the race advances" |

After iter 59 the loop ran for ~3.5 more hours and did not improve. The
budget cap stopped it at iter 91.

## How the agent analyzed the data

The most pleasant surprise of the run was how readily GPT-OSS-120B picked
up the `python_exec` sandbox. Across the 868 tool calls:

| tool | calls | comment |
|---|---:|---|
| `python_exec` | **792** | average **9.5 per iter**, max 20 — heavy real EDA |
| `submit_solution` | 61 | (more than 92 iters because some submits were rejected by `static_check` and retried) |
| `read_incumbent` | 9 | rarely — the model relies on its prompt-side copy of incumbent source |
| `add_insight` | **1** | once, in 10 hours — see "what's bad", below |
| `eda.leakage_scan` (as tool name) | 1 | hallucinated tool name; dispatcher rejected |
| `python_exec<|channel|>commentary` | 4 | Harmony chat-template tokens leaked into the tool name |

The shape of a typical iteration's EDA flow, from looking at the 792 code
snippets:

- **Schema probing first** — `train.head()`, `list(train.columns)`,
  `globals().keys()` to discover what's preloaded.
- **Cardinality probes** — `train['LapNumber'].nunique()`,
  `train['Driver'].value_counts().head()`.
- **Built-in EDA** — `eda.leakage_scan(train, spec.target_col)` and
  `eda.target_relation(...)` were called dozens of times; the agent
  treated them as primitives.
- **Residual analysis on `oof`** — many iterations computed
  `(y - oof).abs().groupby(col).mean()` to find segments where the
  incumbent under-performed, then crafted features for exactly those
  segments (e.g. "high mean absolute residuals for laps 38, 36, 50, 44, 59"
  appears verbatim in iter 17's hypothesis).
- **Correlation / mutual-info probing of candidate interactions** —
  iter 26: "correlation of car's position × lap number with target ≈
  0.21"; iter 30: "Stint × LapNumber → corr ≈ 0.20, MI ≈ 0.07".

When this worked, hypotheses came with **specific numbers extracted from
the data**, not generic ML platitudes. The accepted iter 8/11/59 each
quoted a measurement.

## What hypotheses it tested

Tagging all 92 proposals by theme:

| theme | count |
|---|---:|
| Target encoding, single column (Driver, Stint, LapNumber, Position, …) | 27 |
| Target encoding, interaction (Driver×Race, Driver×LapNumber, etc.) | 24 |
| Numeric interactions (raw products: Stint×LapNumber, TyreLife×Stint, …) | 25 |
| Cyclic sin/cos (RaceProgress, LapNumber) | 6 |
| Pit history (cumulative pit count, laps since last pit) | 4 |
| LapsRemaining / race-stage | 3 |
| Frequency encoding | 2 |
| Empty (Harmony format breakage) | 1 |

The 4 accepts came from 4 distinct categories — frequency+indicators,
TE on numeric+categorical mix, cyclic encoding, and TE×raw interaction.
The 88 rejects clustered into a few groups:

- **Target encoding on Driver/Race only** (iters 1–7, 13, 32, 35, 42…): the
  agent kept proposing variants of this. CatBoost natively handles raw
  Driver/Race via per-fold ordered statistics; any agent-side TE that
  replaces or pollutes these columns either does nothing (within noise) or
  destroys signal (iter 13's Driver×Race TE: 0.9427, **-0.04 below baseline**).
- **Numeric × numeric interactions** (TyreLife×Stint, Stint×LapNumber,
  Position×LapNumber, …): tried at least 8 different variants. Most landed
  at 0.9466 — within noise of incumbent.
- **`Cumulative_pit_count` + `laps_since_last_pit`** (iters 31, 39, 48, 55,
  57, 63, 65, 66, 76, 80, 84, 88, 89, 90, 91 — **15 iterations!**): the
  agent rediscovered this idea over and over. Some implementations hit a
  bug, others scored 0.9466 (within margin). It looks like a strong idea
  but on this dataset CatBoost is already getting that signal from raw
  LapNumber × Driver. The agent had no way to know that, so it kept
  trying.

## What worked

1. **Real EDA, not generic features.** When the agent did real residual
   analysis on `oof`, every accepted iteration came from a specific
   observation. Iter 11's win ("RaceProgress shows clear periodic effect,
   sin/cos should help") and iter 59's win ("Stint_te × RaceProgress
   captures how pit propensity changes as the race advances") were
   data-grounded, not boilerplate.
2. **Incremental layering.** Iter 8 layered TE on top of iter 0's
   frequency-encoded incumbent rather than replacing it. Iter 11 added
   one feature pair (sin/cos) on top of iter 8. Iter 59 added one
   feature on top of iter 11. The agent generally respected the
   incumbent and built upward.
3. **Self-recovery after errors.** iter 36 hit `REACT_TIMEOUT` (15-minute
   wall-clock cap per ReAct loop); the harness invoked `_final_attempt`,
   the model produced a valid `submit_solution`, and the loop continued
   for 56 more iterations. EVAL_ERROR also didn't derail the loop —
   broken-code iterations were just skipped.
4. **Margin rule did its job.** Of 82 valid CV evaluations, 78 scored
   close enough to incumbent (±0.0005) that the 0.1σ margin rejected
   them. Four cleared the margin and were accepted. If we had used naïve
   `>=baseline`, we would have accepted ~30 noise-driven "wins" and the
   incumbent would drift.

## What didn't

1. **The same hypothesis was proposed up to 15 times.** "Add cumulative
   pit count and laps-since-last-pit" appears in iter 31, 39, 48, 55, 57,
   63, 65, 66, 76, 80, 84, 88, 89, 90, 91. After the 3rd or 4th repeat,
   the agent should have either (a) noticed and tried something else, or
   (b) accepted that the idea doesn't beat margin on this data. It did
   neither.
2. **One `add_insight` in 10 hours.** The cross-iteration memory channel
   was essentially unused. The agent had `eda_notebook.md` available for
   persisting durable findings between iterations and chose to re-derive
   them from scratch each time. Given how repetitive its hypotheses were,
   this is the single largest waste in the run — `add_insight("Driver/Race
   TE → -0.04 AUC, do not try again")` could have saved 8–10 wasted
   iterations.
3. **Interaction TE blew up CV reliably.** Iter 10 (Driver_Race +
   Driver_Compound + Year_Driver TE) → 0.9419, **−0.004**. Iter 13
   (Driver_Race TE alone) → 0.9427, **−0.003**. Iter 25 (Driver_LapNumber)
   → 0.9303, **−0.016**. Iter 37 (same idea) → 0.9303. Iter 42 (Driver_Race
   again) → 0.9427. The pattern is rock-solid: agent-built interaction TE
   on rare combos overfits hard against CatBoost's ordered TS. The agent
   tried it five times.
4. **Harmony chat template leaked into tool calls.** GPT-OSS-120B's native
   chat template uses tokens like `<|channel|>commentary` and
   `<|channel|>final` to separate reasoning from output. Four times the
   model emitted these tokens **inside the tool name** (e.g.
   `python_exec<|channel|>commentary`), which our dispatcher correctly
   rejected as "unknown tool". This is a Yandex-OpenAI-compat / OpenAI-SDK
   plumbing artifact, not our bug per se, but it cost a few iterations.
5. **Empty hypotheses.** Iter 15, 18, 53, 61, 62, 67, 68, 69, 75, 79, 82,
   86, 87, 91 — about 14 PROPOSE events have empty or "**Hypothesis**:"-only
   reasoning. These are iterations where the model put the code in
   `submit_solution(code=...)` but left `hypothesis=""`. The runs still
   evaluated (some succeeded, some failed), but we can't reconstruct from
   the journal what the model was actually trying.
6. **Code-quality regressions late in the run.** EVAL_ERROR rate doubled
   between iters 30-60 vs iters 0-30 — the model's solutions grew
   structurally, picked up more bugs (`KeyError('index')`, shape
   mismatches, undefined helpers), and the `static_check` started
   catching more.

## Cost breakdown

| | tokens | ₽ |
|---|---:|---:|
| input | 9 378 505 | 2 813.55 |
| output | 482 923 | 144.88 |
| **total** | **9 861 428** | **2 958.43** |

That's roughly **₽32 per accepted improvement** (4 accepts) or **₽0.30 per
0.0001 AUC point** at the final gain. Input dominates output ~20×, which
makes sense: with `--max-tool-calls 64` and rich tool results the
context grows fast across a 10-hour run.

## Behavioral fingerprint

The personality of this agent on this task:

- **Curious, methodical EDA**. Will gladly run 10–20 `python_exec` calls
  per iteration to look at the data from multiple angles.
- **Quantitative**. Cites concrete numbers in hypotheses ("correlation
  ≈0.21", "MI ≈0.07", "laps 38, 36, 50, 44, 59"). This is a stark
  contrast to YandexGPT 5.1 Pro which writes solutions blind without
  calling EDA at all.
- **Stuck in a feature-engineering local-min**. The agent's mental model
  of "what helps a tabular model" was a fixed list — TE, freq encoding,
  interactions, cyclic, pit history. After exhausting them all, it
  cycled instead of trying something genuinely new (different metric
  perspectives, regression on residuals, post-processing, …). The
  `program.md`-defined priorities (data understanding → feature
  engineering → postprocessing) were followed for the first two but the
  third stage never happened.
- **Honest about ignorance, but doesn't act on it**. The proposals often
  contain phrases like "modestly raise AUC by ~0.001" — but when the
  result is +0.00005, the agent doesn't update its prior. It just
  re-uses the same template for the next iteration.
- **Doesn't communicate state forward**. With `add_insight` ignored, each
  iteration starts fresh against the same notebook seed and journal tail
  — the model effectively has Alzheimer's about its own conclusions.

## The final solution

`runs/comp2-gptoss/incumbent.py` ended at **31 engineered features on top
of the 14 raw ones**:

- 5 frequency encodings (Driver, Compound, Race + a couple of mixed)
- 3 zero-indicator flags (LapTime_Delta_zero, Cumulative_Degradation_zero,
  Position_Change_zero)
- 4 log1p features (skew-fixed numerics)
- 5 smoothed target encodings (Year_te, Stint_te, Driver_te, Race_te,
  Compound_te)
- 2 cyclic features (RaceProgress_sin, RaceProgress_cos)
- 1 derived interaction (Stint_te × RaceProgress)

CV AUC 0.946791 ± 0.000850 on the full 440k rows. Submission for the test
set was refit on 100% of train and written to
`competitions/competition-2/submission_gptoss_full.csv`.

## Closing

The experiment confirms three things the system was designed to test:

1. The contract holds. Across 92 iterations and 4 accepted modifications,
   not one improvement came from cheating — every score the harness
   reported was leak-safe and reproducible. The `0.15σ → 0.1σ` margin
   change (`config.yaml`) didn't introduce a single noise-driven false
   accept.
2. ReAct + a frozen CatBoost can find real (small) lifts on a well-curated
   dataset, but the gains are sub-percent and the bulk of the LLM budget
   is spent on negative/zero results. This is consistent with the
   autoresearch reference: "the loop's job is to keep what works and
   throw away what doesn't, not to invent miracles."
3. The agent's tool-use *technique* — multi-turn EDA, residual-driven
   feature engineering, quantitative predictions — is exactly what we
   want. Its tool-use *discipline* — incremental memory, forbidding
   already-tried directions, breadth of strategy — is where the next
   prompt-engineering wins lie.

Total: 10 hours, ~3000 ₽, ~9.9M tokens, 92 hypotheses, 4 real
improvements, +0.07% AUC.
