# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The search engine end to end: a toy kernel space searched with explicit
Params and a CallableBackend, the winner recorded in the tuning DB and
served back; transfer warm starts; the search_configs hook; the CLI; and
the torch-free import."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from flydsl.tuning import (
    Budget,
    CallableBackend,
    CompileOutcome,
    Param,
    SearchSpace,
    TunedRecord,
    TuningDB,
    WarmStart,
    problem_key,
    record_result,
    search_configs,
    tune,
)
from flydsl.tuning.__main__ import main as cli_main
from flydsl.tuning.synthetic import gemm_like

TILE = 128


def _toy_space() -> SearchSpace:
    return SearchSpace(
        (
            Param.ordinal("unroll", (1, 2, 4, 8)),
            Param.boolean("prefetch"),
            Param.ordinal("prefetch_depth", (1, 2, 3), gate="prefetch"),
            Param.ordinal("waves_per_eu", (1, 2, 3, 4)),
        ),
        # unroll 8 x tile 128 breaks the register budget
        is_valid=lambda c: c.get("unroll", 1) * TILE <= 512,
        name="searchtoy",
    )


def _truth(c) -> float:
    t = 10.0 / c.get("unroll", 1) ** 0.5
    if c.get("prefetch"):
        t *= {1: 0.97, 2: 0.9, 3: 0.95}[c.get("prefetch_depth", 1)]
    t *= {1: 1.1, 2: 1.0, 3: 0.96, 4: 1.05}[c.get("waves_per_eu", 2)]
    return t


def _backend() -> CallableBackend:
    return CallableBackend(
        compile=lambda cfg, key: CompileOutcome(True, artifact=dict(cfg)),
        check=lambda cfg, key, art: (True, "ok"),
        measure=lambda cfg, key, art, fid: [_truth(art)] * fid.reps,
    )


def _best_truth(space: SearchSpace) -> float:
    return min(_truth(space.decode(g)) for g in space.enumerate() if space.valid(g))


def test_toy_space_refuses_invalid_configs_with_a_reason():
    space = _toy_space()
    g = space.encode({"unroll": 8})
    assert space.key(g) is None and not space.valid(g)
    keys = {space.key(g) for g in space.enumerate() if space.valid(g)}
    assert None not in keys and len(keys) >= 3 * 4


def test_tune_records_and_db_serves_the_pick(tmp_path):
    space = _toy_space()
    db = TuningDB(tmp_path / "db.json")
    problem = problem_key({"m": 1024, "dtype": "bf16"})
    res = tune(
        space,
        _backend(),
        budget=Budget(evaluations=200),
        strategy="exhaustive",
        baseline=space.encode({}),
        trace=tmp_path / "t.jsonl",
    )
    assert res.best_value == pytest.approx(_best_truth(space))
    rec = record_result(
        res,
        kernel="searchtoy",
        arch="gfx950",
        problem=problem,
        config_key=res.best_key,
        knobs=res.best_config,
        db=db,
        features={"m": 1024},
        trace=str(tmp_path / "t.jsonl"),
    )
    assert rec is not None
    # confirmed against the baseline: paired speedup and p-value
    c = res.confirmed
    default = c["baseline"]
    assert default in c["values"] and c["winner"] != default
    assert rec.speedup == pytest.approx(c["values"][default] / c["values"][c["winner"]])
    assert rec.speedup > 1.0 and rec.p_value < 0.05 and rec.baseline_key == default
    assert rec.toolchain and rec.machine is not None
    again = TuningDB(tmp_path / "db.json")
    assert again.get("searchtoy", "gfx950", problem) == rec
    assert again.serve("searchtoy", "gfx950", problem) == rec
    assert again.serve("searchtoy", "gfx942", problem) is None
    assert again.serve("searchtoy", "gfx950", problem_key({"m": 1})) is None


def test_spilling_baseline_is_still_the_confirmation_reference():
    """The shipped default is the reference even when it spills: the spill gate
    keeps it out of the search, not out of the paired confirmation."""
    space = _toy_space()
    shipped = space.decode(space.encode({}))
    backend = CallableBackend(
        compile=lambda cfg, key: CompileOutcome(
            True, artifact=dict(cfg), resources={"spill_vgpr": 8.0 if dict(cfg) == shipped else 0.0}
        ),
        check=lambda cfg, key, art: (True, "ok"),
        measure=lambda cfg, key, art, fid: [_truth(art)] * fid.reps,
    )
    res = tune(
        space,
        backend,
        budget=Budget(evaluations=200),
        strategy="exhaustive",
        baseline=space.encode({}),
        confirm=2,
        confirm_rounds=3,
    )
    c = res.confirmed
    assert c["baseline"] in c["values"], "the spilling baseline was left out of the confirmation"
    assert c["speedup"] > 1.0 and c["winner"] != c["baseline"]
    assert not any(o.status == "ok" and o.key == c["baseline"] for o in res.observations if o.full)


def test_unconfirmed_pick_is_stored_but_never_served():
    space = _toy_space()
    res = tune(
        space,
        _backend(),
        budget=Budget(evaluations=200),
        strategy="exhaustive",
        confirm=0,
    )
    db = TuningDB(None)
    rec = record_result(
        res,
        kernel="searchtoy",
        arch="gfx950",
        problem="p",
        config_key=res.best_key,
        knobs=res.best_config,
        db=db,
    )
    assert rec is not None and rec.speedup is None
    assert db.get("searchtoy", "gfx950", "p") == rec
    assert db.serve("searchtoy", "gfx950", "p") is None


def test_transfer_warm_starts_from_neighbors_and_other_archs(tmp_path):
    db = TuningDB(tmp_path / "db.json")

    def rec(problem, arch, m, unroll, value):
        return TunedRecord(
            kernel="f",
            arch=arch,
            problem=problem,
            config_key=f"k{problem}{arch}",
            knobs={"unroll": unroll},
            value=value,
            features={"m": m},
        )

    db.put(rec("p1", "gfx950", 1024, 2, 1.0))
    db.put(rec("p2", "gfx950", 8192, 4, 1.0))
    db.put(rec("p3", "gfx942", 2048, 8, 1.0))
    near = db.neighbors("f", "gfx950", {"m": 2048}, k=2)
    assert [r.problem for _d, r in near] == ["p1", "p2"]
    warm = db.warm_starts("f", "gfx950", "p3", {"m": 2048}, k=1)
    assert [w.source for w in warm] == ["neighbor:p1", "arch:gfx942"]
    # a slower record does not replace a faster one; the same config refreshes
    slower = replace(rec("p1", "gfx950", 1024, 1, 2.0), config_key="kz")
    assert not db.put(slower)
    assert db.put(rec("p1", "gfx950", 1024, 2, 1.5))
    assert db.get("f", "gfx950", "p1").value == 1.5


def test_tuned_record_validation():
    base = dict(kernel="f", arch="a", problem="p", config_key="1", knobs={})
    with pytest.raises(ValueError, match="finite"):
        TunedRecord(value=math.inf, **base)
    with pytest.raises(ValueError, match="finite"):
        TunedRecord(value=0.0, **base)
    with pytest.raises(TypeError, match="JSON scalar"):
        TunedRecord(value=1.0, **{**base, "knobs": {"u": [1]}})


def test_problem_key_is_order_independent_and_distinguishes_problems():
    assert problem_key({"m": 64, "n": 8}) == problem_key({"n": 8, "m": 64})
    assert problem_key({"m": 64}) != problem_key({"m": 65})


# ------------------------------------------------------------ hooks
def test_search_configs_and_sweep_hook():
    space, f = gemm_like()
    gs = [g for g in space.enumerate() if space.valid(g)][:300]
    configs = [space.decode(g) for g in gs]
    idx, res = search_configs(
        configs,
        measure=lambda c, fid: f(c),
        check=lambda c: True,
        strategy="de",
        budget=Budget(evaluations=40),
        confirm=0,
    )
    assert idx is not None and res.evaluations <= 40
    assert f(configs[idx]) == res.best_value


# --------------------------------------------------------------- CLI
def test_cli_demo_replay_and_show(tmp_path, capsys):
    assert cli_main(["strategies"]) == 0
    assert "portfolio" in capsys.readouterr().out
    assert (
        cli_main(
            [
                "demo",
                "--objective",
                "sphere",
                "--strategy",
                "de",
                "--budget",
                "16",
                "--seeds",
                "1",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["runs"][0]["evaluations"] <= 16
    csv_path = tmp_path / "sweep.csv"
    rows = ["M,tile,pipe,tflops"]
    for m in (128, 256):
        for tile in (32, 64, 128, 256):
            for pipe in ("a", "b"):
                rows.append(f"{m},{tile},{pipe},{tile / (1 + (pipe == 'b')) + m / 100}")
    csv_path.write_text("\n".join(rows) + "\n")
    out_json = tmp_path / "r.json"
    assert (
        cli_main(
            [
                "replay",
                "--csv",
                str(csv_path),
                "--config-cols",
                "tile,pipe",
                "--value-col",
                "tflops",
                "--higher-is-better",
                "--problem-cols",
                "M",
                "--strategies",
                "de,random,exhaustive",
                "--budgets",
                "8",
                "--seeds",
                "1",
                "--min-configs",
                "4",
                "--out",
                str(out_json),
            ]
        )
        == 0
    )
    rep = json.loads(capsys.readouterr().out)
    assert rep["problems"] == ["128", "256"]
    assert rep["summary"]["8"]["exhaustive"]["worst"] == 1.0
    runs = json.loads(out_json.read_text())["runs"]
    assert all(0 < r["best_seen"] <= 1.0 for r in runs)
    # show
    db = TuningDB(tmp_path / "db.json")
    db.put(
        TunedRecord(
            kernel="f",
            arch="a",
            problem="p",
            config_key="k",
            knobs={"u": 1},
            value=1.0,
        )
    )
    space, f = gemm_like()
    from flydsl.tuning import SyntheticBackend

    tune(
        space,
        SyntheticBackend(f),
        strategy="random",
        budget=Budget(evaluations=4),
        trace=tmp_path / "t.jsonl",
        confirm=0,
    )
    assert (
        cli_main(
            [
                "show",
                "--db",
                str(tmp_path / "db.json"),
                "--trace",
                str(tmp_path / "t.jsonl"),
            ]
        )
        == 0
    )
    shown = capsys.readouterr().out
    assert '"config_key": "k"' in shown and '"observations": 4' in shown


def test_cli_tune_with_factory(tmp_path, capsys, monkeypatch):
    mod = tmp_path / "searchtoy_factory.py"
    mod.write_text(
        "from flydsl.tuning import SyntheticBackend\n"
        "from flydsl.tuning.synthetic import sphere\n"
        "def make(args):\n"
        "    assert args.factory_args == {'shape': '64x64', 'dtype': 'f16'}\n"
        "    space, f = sphere(levels=5, dims=2)\n"
        "    def record(res):\n"
        "        return dict(kernel='f', arch='gfx950', problem='p',\n"
        "                    config_key=res.best_key[:8], knobs=res.best_config)\n"
        "    return {'space': space, 'backend': SyntheticBackend(f), 'record': record}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    db = tmp_path / "db.json"
    assert (
        cli_main(
            [
                "tune",
                "--factory",
                "searchtoy_factory:make",
                "--strategy",
                "de",
                "--budget",
                "12",
                "--db",
                str(db),
                "--confirm",
                "0",
                "--arg",
                "shape=64x64",
                "dtype=f16",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert len(TuningDB(db)) == 1


def test_warm_start_json_round_trip():
    w = WarmStart({"a": 1, "b": "x"}, "selector")
    assert WarmStart.from_json(json.loads(json.dumps(w.to_json()))) == w


def test_engine_imports_without_torch():
    code = "import sys; import flydsl.tuning as s; " "assert 'torch' not in sys.modules, 'torch imported'; print('ok')"
    root = Path(__file__).resolve().parents[3] / "python"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(root)},
    )
    assert out.returncode == 0, out.stderr
