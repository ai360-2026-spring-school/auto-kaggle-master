# Research program

You are running an autonomous tabular-ML research org. This file is the only
place a human steers you — the equivalent of autoresearch's `program.md`. Edit
the strategy here over time; do not put strategy in the harness code.

## Objective

Maximize the held-out CV metric reported by the harness for this competition.
That number is the only ground truth. Public-leaderboard intuition, "this
should help", and elegant code are irrelevant unless CV moves.

## Method (run this loop)

1. **Look before you leap.** You have a persistent Python sandbox via
   `python_exec` with `train`, `test`, `spec`, `oof` (current incumbent's
   out-of-fold predictions), `feature_importance`, and the harness `eda`
   module preloaded. EVERY iteration begins with at least one tool call
   that probes the data. The auto-EDA seed in `eda_notebook.md`
   (ydata-profiling minimal) gives you a free starting picture — read it,
   then drill into what's interesting:
   - `eda.leakage_scan(train, target)` — *do this first.* Near-perfectly
     predictive or 1:1 columns are almost certainly id/leak. Drop; do not
     celebrate a 0.999 AUC.
   - `eda.target_relation(train, target)` — which raw features carry signal.
   - residual analysis on `oof`: where does the incumbent under-perform?
     Group by candidate features, find the worst segments, add features
     that distinguish them.
   - `eda.interaction_scan(train, target, cols)` — pairwise products/ratios
     of the strongest numerics.
2. **One hypothesis per iteration.** Change one coherent thing, predict whether
   CV will move and why, then let the harness check. Keep cause and effect
   legible. Blind rewrites destroy information. Use `add_insight(text)` to
   stash durable observations in the notebook so future iterations inherit
   them. End the iteration with `submit_solution(code, hypothesis,
   expected_effect)`.
3. **Trust the margin rule.** The harness only accepts a change that beats the
   incumbent by ~0.15σ of fold std. If your change is within noise, it is not
   an improvement — move on, don't fight variance.
4. **Read the journal.** Every experiment is logged. Before proposing, look at
   what already failed so you don't repeat it.

## Priorities (rough order of expected payoff on classic tabular)

1. Correct problem framing & metric (harness fixes this — verify it matches the
   task text; if not, say so in your reasoning).
2. Leakage removal. The single biggest score-killer in real competitions.
3. Sound handling of categoricals (CatBoost eats raw categoricals — explicit
   frequency / smoothed target encodings are additive, not replacements).
4. Missingness as signal (indicator columns) + skew fixes (log1p on positive
   skewed).
5. Domain feature engineering from the data description: dates → parts/cyclical
   and deltas; text-ish ids → split components; ratios/aggregates that the
   problem domain implies.
6. Light postprocessing: clip to plausible target range; keep probabilities
   calibrated; rank-stabilize if metric is rank-based.

## Hard constraints (the harness enforces; stating them so you internalize)

- You only ever edit `solution.py`. The model is a frozen CatBoost black box.
- All learned statistics go in `fit()` (train rows only); `transform()` and
  `postprocess()` never see the target and must be deterministic.
- No new estimators, no model ensembling that isn't pure data/postprocess, no
  network, no file I/O, no peeking at the test target (there isn't one).

## Stop conditions

Stop proposing when iterations or wall-clock budget is exhausted, or when ~5
consecutive experiments fail to beat the incumbent — diminishing returns.
