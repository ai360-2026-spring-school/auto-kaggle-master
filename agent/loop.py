"""
The autonomous research loop.

Directly mirrors autoresearch's overnight loop: the agent modifies the one
editable artifact, the harness measures it under a fixed protocol, the result
is kept iff it beats the incumbent by a noise-aware margin, everything is
written to a journal, repeat until the budget runs out. You can leave it
running and read the journal in the morning.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from harness import (HarnessConfig, build_problem_spec, select_metric,
                     evaluate_solution, is_improvement, load_solution)

from .auto_profile import seed_eda_notebook
from .exec_sandbox import Sandbox
from .llm import Agent, Proposal
from .notebook import read as read_notebook
from .tools import ToolContext
from . import prompts


class ResearchLoop:
    def __init__(self, cfg: HarnessConfig, agent: Agent, program_md_path: str,
                 use_tool_prompt: bool = True, max_tool_calls: int = 15):
        self.cfg = cfg
        self.agent = agent
        self.program_md = prompts.load_program_md(program_md_path)
        self.work = Path(cfg.paths.workdir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.work / cfg.paths.journal_file
        self.notebook_path = self.work / "eda_notebook.md"
        self.fi_notes = ""
        # ReAct backends want the tool-augmented system prompt; the offline
        # curriculum and the single-shot Anthropic backend ignore it but it
        # is harmless to include.
        self.use_tool_prompt = use_tool_prompt
        self.max_tool_calls = max_tool_calls

    # -- journal ----------------------------------------------------------- #
    def _log(self, record: dict) -> None:
        record["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.journal_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        tag = record.get("event", "")
        extra = ""
        if "score" in record:
            extra = f" score={record['score']:.6f}" if record["score"] == record["score"] else ""
        print(f"[{record['t']}] {tag}{extra} {record.get('msg','')}")

    def _read_journal(self) -> list[dict]:
        if not self.journal_path.exists():
            return []
        return [json.loads(l) for l in self.journal_path.read_text().splitlines() if l]

    # -- main -------------------------------------------------------------- #
    def run(self, task_description, data_description, train, test):
        # fresh journal per run (the workdir is the experiment's lab notebook)
        if self.journal_path.exists():
            self.journal_path.unlink()
        spec = build_problem_spec(task_description, data_description, train, test)
        metric = select_metric(task_description, spec.problem_type)
        self._log({"event": "START", "msg": spec.summary(),
                   "metric": metric.name})

        # Seed EDA notebook with ydata-profiling (or fallback) — one shot.
        try:
            digest = seed_eda_notebook(spec, self.work)
            self._log({"event": "AUTO_PROFILE", "msg": f"len={len(digest)}"})
        except Exception as e:  # noqa: BLE001
            self._log({"event": "AUTO_PROFILE_ERROR", "msg": repr(e)})

        # incumbent := baseline solution_template.py
        # NOTE: write/read explicitly as UTF-8 so non-ASCII chars the LLM may
        # emit (em-dashes, Cyrillic, etc.) do not break Python's source
        # compilation on Windows where the default cp1252 encoding triggers
        # SyntaxError(unicode error) when importlib re-parses the file.
        inc_path = self.work / self.cfg.paths.incumbent_file
        inc_path.write_text(
            Path(__file__).resolve().parents[1].joinpath(
                "solution_template.py").read_text(encoding="utf-8"),
            encoding="utf-8")
        incumbent = evaluate_solution(load_solution(inc_path), spec, metric,
                                      self.cfg)
        self._log({"event": "BASELINE", "ok": incumbent.ok,
                   "score": incumbent.score, "std": incumbent.score_std,
                   "n_features": incumbent.n_features, "msg": incumbent.error})
        if not incumbent.ok:
            raise RuntimeError(f"Baseline failed: {incumbent.error}")
        self._note_top_features(incumbent)

        sys_prompt = (prompts.build_tool_system_prompt(self.program_md)
                      if self.use_tool_prompt
                      else prompts.build_system_prompt(self.program_md))
        t_start = time.time()

        for it in range(self.cfg.budget.max_iterations):
            if time.time() - t_start > self.cfg.budget.max_seconds_total:
                self._log({"event": "STOP", "msg": "total time budget reached"})
                break

            # Build per-iteration sandbox + tool context. ReAct backends
            # use these; non-tool backends ignore the `context` kwarg.
            sandbox = Sandbox(
                spec=spec, train=spec.train, test=spec.test,
                oof=incumbent.oof,
                incumbent_source=inc_path.read_text(),
                feature_importance=incumbent.feature_importance,
                workdir=self.work,
            )
            tool_ctx = ToolContext(
                sandbox=sandbox, workdir=self.work,
                incumbent_path=inc_path, journal_path=self.journal_path,
                notebook_path=self.notebook_path,
                incumbent_score=incumbent.score, metric_name=metric.name,
                on_event=self._log, iteration=it,
            )

            notebook_text = read_notebook(self.notebook_path)
            eda_notes = (notebook_text + "\n\n" + self.fi_notes).strip()
            it_prompt = prompts.build_iteration_prompt(
                spec.summary(), spec.task_description, spec.data_description,
                metric.name, metric.greater_is_better,
                inc_path.read_text(), incumbent.score,
                self._read_journal(), eda_notes,
            )
            try:
                prop: Proposal = self.agent.propose(sys_prompt, it_prompt, it,
                                                    context=tool_ctx)
            except Exception as e:  # noqa: BLE001
                self._log({"event": "AGENT_ERROR", "iter": it,
                           "msg": repr(e)})
                sandbox.close()
                continue
            finally:
                # Sandbox is intentionally short-lived per iteration.
                pass
            sandbox.close()

            cand_path = self.work / self.cfg.paths.solution_file
            cand_path.write_text(prop.solution_source, encoding="utf-8")
            self._log({"event": "PROPOSE", "iter": it,
                       "msg": prop.reasoning[:300]})

            try:
                cand_sol = load_solution(cand_path)
                result = evaluate_solution(cand_sol, spec, metric, self.cfg)
            except Exception as e:  # noqa: BLE001
                self._log({"event": "EVAL_ERROR", "iter": it,
                           "msg": f"{e!r}\n{traceback.format_exc()[-600:]}"})
                continue

            accepted = is_improvement(result, incumbent, metric, self.cfg)
            self._log({
                "event": "RESULT", "iter": it, "ok": result.ok,
                "score": result.score, "std": result.score_std,
                "folds": result.fold_scores, "n_features": result.n_features,
                "seconds": round(result.seconds, 1),
                "accepted": accepted, "msg": result.error,
            })

            if accepted:
                inc_path.write_text(prop.solution_source, encoding="utf-8")
                incumbent = result
                self._note_top_features(result)
                self._log({"event": "ACCEPT", "iter": it,
                           "score": incumbent.score,
                           "msg": "new incumbent"})

        self._log({"event": "DONE", "score": incumbent.score,
                   "msg": "final incumbent selected"})
        return spec, metric, incumbent, inc_path

    def _note_top_features(self, result):
        if result.feature_importance is not None:
            top = result.feature_importance.head(15)
            self.fi_notes = (
                "Top features by CatBoost importance on current incumbent:\n"
                + "\n".join(f"  {k}: {v:.2f}" for k, v in top.items())
            )
