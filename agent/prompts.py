"""
Prompt construction.

`program.md` is the human-edited "research org" instruction (the autoresearch
analogue). The agent system prompt is assembled from it plus the immutable
rules of the contract. The per-iteration user prompt feeds the agent the live
state: the journal so far, the EDA findings it has gathered, the incumbent
score, and the incumbent's source — exactly the loop "check if it improved,
keep or discard, repeat".
"""
from __future__ import annotations

import json
from pathlib import Path

CONTRACT_RULES = """\
HARD RULES (the harness enforces these; violating them fails the experiment):
- You write exactly one file: solution.py, defining `class Solution(BaseSolution)`.
- Implement fit(self, train_df, y, spec), transform(self, df), and optionally
  postprocess(self, raw_pred, df).
- fit() may look at y. transform() and postprocess() MUST NOT use y, and must
  be deterministic given fit() state. Learn ALL statistics (encodings, scalers,
  imputers, frequency maps, target stats) in fit(); apply them in transform().
- transform() returns ONLY features: never the target, never the id column.
- You may NOT import catboost/xgboost/lightgbm or change the model. The model
  is a fixed black box. Your only levers are the features and the postprocess.
- Allowed libs: numpy, pandas, scikit-learn, scipy. No network, no file I/O.

WHY: the harness fits your Solution on training folds only and applies it to
held-out folds, so well-formed code physically cannot leak. The single CV
number it returns is ground truth — optimize that.
"""


def build_system_prompt(program_md: str) -> str:
    return (
        "You are an autonomous ML research agent competing on a tabular "
        "data competition. You iterate scientifically: form a hypothesis from "
        "evidence, implement it, let the harness measure it, keep it only if "
        "the CV score improves, then repeat.\n\n"
        + CONTRACT_RULES
        + "\n\n=== RESEARCH PROGRAM (human-authored) ===\n"
        + program_md
    )


TOOL_USAGE_BLOCK = """
=== EXECUTION ENVIRONMENT (multi-turn tool use) ===
You have a persistent Python sandbox and several tools. EVERY iteration:

  1. **READ THE INCUMBENT FIRST.** The current `solution.py` is the BEST result
     so far — it has been accepted because it beat every prior candidate by a
     noise-aware margin. Your job is to BUILD ON TOP OF IT, not replace it.
     The incumbent's full source is in `incumbent_source` (preloaded in the
     sandbox) and also returned by `read_incumbent()`. Inspect it, understand
     every feature it builds, every encoding it learns. Your new solution.py
     MUST preserve all of that and ADD a focused improvement. Discarding the
     incumbent's features will almost always lose CV ground that was already
     won — even a strictly-better hypothesis usually fails CV if you delete
     the existing engineered features that the harness incumbent score reflects.

  2. Look at the data with python_exec. Useful preloaded names:
     `train`, `test` (pandas DataFrames, copies — mutate freely),
     `spec` (.target_col/.id_col/.problem_type), `oof` (out-of-fold preds
     of the current incumbent; np.ndarray or None), `feature_importance`
     (pd.Series of CatBoost importances on the incumbent or None),
     `incumbent_source` (str), plus the harness `eda` module
     (eda.profile / eda.leakage_scan / eda.target_relation /
     eda.interaction_scan; also eda.ydata_profile if installed). Variables
     persist across calls in this iteration. Use `oof` for residual analysis
     — find segments where the incumbent under-performs, those are where
     a NEW feature might pay off.

  3. Decide ONE concrete hypothesis grounded in what you observed AND in
     what the incumbent is already doing. Avoid duplicating what the
     incumbent already captures. Avoid blanket rewrites. One legible change
     per iteration so cause-and-effect on CV stays interpretable.

  4. Persist non-obvious findings via add_insight(text) so future iterations
     can read them from eda_notebook.md. Use it for facts that survive
     iterations (e.g. "Driver has 480 levels and high target MI"), not
     fleeting thoughts.

  5. Read read_journal for what already failed before repeating it. If a
     prior PROPOSE matches yours, do something different.

  6. End the iteration by calling submit_solution(code, hypothesis,
     expected_effect) EXACTLY ONCE. The `code` MUST contain everything the
     incumbent had PLUS your focused new change. Sanity-check: open
     `incumbent_source` in your head and verify your submitted code retains
     every engineered feature, encoding, and postprocess step it had.

Tool calls per iteration are capped (typically <= 15). Use them for evidence
gathering, not micro-tuning of the same idea.
""".strip()


def build_tool_system_prompt(program_md: str) -> str:
    """System prompt for tool-using ReAct backends (GigaChat, Yandex, ...)."""
    return build_system_prompt(program_md) + "\n\n" + TOOL_USAGE_BLOCK


_STRATEGIC_EVENTS = {"BASELINE", "PROPOSE", "RESULT", "ACCEPT", "EVAL_ERROR",
                     "AGENT_ERROR", "REACT_TIMEOUT"}


def _strategic_journal(journal_tail: list[dict], last_n: int = 32) -> list[dict]:
    """Filter the journal to STRATEGIC events only and keep the last `last_n`.

    Without this filter, TOOL_CALL/TOOL_RESULT/TOKEN_USAGE noise from a single
    iteration's ReAct loop (10–15 entries each) pushes PROPOSE/RESULT from
    prior iterations out of any reasonable sliding window — making the agent
    effectively amnesiac about what it already tried.
    """
    strategic = [r for r in journal_tail if r.get("event") in _STRATEGIC_EVENTS]
    return strategic[-last_n:]


def build_iteration_prompt(
    spec_summary: str,
    task_description: str,
    data_description: str,
    metric_name: str,
    greater_is_better: bool,
    incumbent_source: str,
    incumbent_score: float | None,
    journal_tail: list[dict],
    eda_notes: str,
) -> str:
    direction = "HIGHER is better" if greater_is_better else "LOWER is better"
    inc = f"{incumbent_score:.6f}" if incumbent_score is not None else "none yet"
    journal = json.dumps(_strategic_journal(journal_tail), indent=2,
                          default=str)
    return f"""\
COMPETITION
  {spec_summary}
  metric: {metric_name} ({direction})
  best CV so far: {inc}

TASK DESCRIPTION
{task_description}

DATA DESCRIPTION
{data_description}

EDA NOTEBOOK (auto-profile + insights from prior iterations)
{eda_notes or "(empty)"}

EXPERIMENT JOURNAL (strategic events only — every prior PROPOSE/RESULT/ACCEPT is here; DO NOT repeat a hypothesis already tried)
{journal}

CURRENT INCUMBENT solution.py  <-- COPY THIS BLOCK AND ADD TO IT.
```python
{incumbent_source}
```

Decide the single most promising next change. If you are a tool-using backend,
ALWAYS run at least one python_exec for evidence (check oof residuals, run
eda.leakage_scan, probe a candidate feature with eda.target_relation, etc.)
BEFORE writing code, then call submit_solution. If you are not a tool-using
backend, briefly state your hypothesis and output the COMPLETE new
solution.py in a python code block.

IMPORTANT: Your submitted solution.py MUST start from the incumbent above and
EXTEND it (preserve every engineered feature/encoding it adds, then layer
your new change on top). Replacing the incumbent with a fresh from-scratch
solution will almost certainly lose CV ground it already won. One coherent
well-motivated change per iteration — never blind rewrites.
"""


def load_program_md(path: str | Path) -> str:
    p = Path(path)
    return p.read_text() if p.exists() else "(no program.md found)"
