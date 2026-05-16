"""Unit tests for agent.exec_sandbox.Sandbox."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agent.exec_sandbox import Sandbox, static_screen


# -- fixtures --------------------------------------------------------------- #

@pytest.fixture
def spec():
    train = pd.DataFrame({"id": range(20), "x": np.arange(20.0),
                          "y": np.arange(20) * 0.5 + 1.0})
    test = pd.DataFrame({"id": range(20, 30), "x": np.arange(20, 30.0)})
    return SimpleNamespace(
        train=train, test=test,
        target_col="y", id_col="id",
        problem_type="regression", n_classes=1, class_labels=None,
    )


@pytest.fixture
def sb(tmp_path, spec):
    s = Sandbox(spec=spec, train=spec.train, test=spec.test,
                oof=np.zeros(20), incumbent_source="class Solution: pass",
                feature_importance=pd.Series({"x": 1.0}),
                workdir=tmp_path, timeout_sec=3)
    yield s
    s.close()


# -- static screen ---------------------------------------------------------- #

def test_static_screen_blocks_catboost():
    issues = static_screen("import catboost\nx = 1")
    assert any("catboost" in i for i in issues)


def test_static_screen_blocks_open():
    issues = static_screen("open('foo.txt')")
    assert any("open" in i for i in issues)


def test_static_screen_blocks_socket():
    issues = static_screen("import socket")
    assert any("socket" in i for i in issues)


def test_static_screen_blocks_os():
    issues = static_screen("import os")
    assert any("os" in i for i in issues)


def test_static_screen_allows_pandas():
    assert static_screen("import pandas as pd\npd.DataFrame()") == []


# -- run() basics ----------------------------------------------------------- #

def test_run_stdout_capture(sb):
    r = sb.run("print('hello')")
    assert r.ok
    assert "hello" in r.stdout


def test_run_last_expression_value(sb):
    r = sb.run("a = 2 + 3\na")
    assert r.ok
    assert r.value_repr.strip() == "5"


def test_run_dataframe_summary(sb):
    r = sb.run("train")
    assert r.ok
    assert "DataFrame" in r.value_repr
    assert "shape=" in r.value_repr
    assert "head(10)" in r.value_repr


def test_run_namespace_persists(sb):
    sb.run("my_var = 42")
    r = sb.run("my_var * 2")
    assert r.ok
    assert "84" in r.value_repr


def test_run_train_is_mutable_copy(sb):
    """User mutations to `train` must not leak back into spec.train."""
    sb.run("train.loc[0, 'x'] = -999")
    assert sb.spec.train.loc[0, "x"] == 0.0


def test_run_eda_module_available(sb):
    r = sb.run("eda.profile(train, 'y')")
    assert r.ok
    assert "n_rows" in r.value_repr or "columns" in r.value_repr


def test_run_oof_available(sb):
    r = sb.run("oof.shape")
    assert r.ok
    assert "(20,)" in r.value_repr


def test_run_syntax_error_returns_error(sb):
    r = sb.run("def (((")
    assert not r.ok
    assert "rejected" in r.error and "SyntaxError" in r.error


def test_run_runtime_error_captured(sb):
    r = sb.run("1 / 0")
    assert not r.ok
    assert "ZeroDivisionError" in r.error


def test_run_rejects_forbidden(sb):
    r = sb.run("import catboost")
    assert not r.ok
    assert "catboost" in r.error


def test_as_text_packs_everything(sb):
    r = sb.run("print('A'); 7")
    txt = r.as_text()
    assert "stdout" in txt
    assert "value" in txt
    assert "A" in txt
    assert "7" in txt
