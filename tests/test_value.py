"""python -m pytest tests -q   (runs the whole Part 1 pipeline on a small fake season)"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def test_parse_play_by_play():
    from src.fetch_runners import parse
    pbp = {"allPlays": [{"about": {"atBatIndex": 4},
                         "playEvents": [{"index": 0, "isPitch": True}, {"index": 1, "isPitch": False},
                                        {"index": 2, "isPitch": True}],
                         "runners": [{"movement": {"start": "1B", "end": "3B", "isOut": False},
                                      "details": {"playIndex": 2, "runner": {"id": 7}, "eventType": "single"}},
                                     {"movement": {"start": None, "end": "1B", "isOut": False},
                                      "details": {"playIndex": 2, "runner": {"id": 8}, "eventType": "single"}}]}]}
    rows = parse(pbp, 99)
    assert len(rows) == 2
    assert rows[0]["at_bat_number"] == 5 and rows[0]["bip_index"] == 2 and rows[0]["end"] == "3B"


def test_starts_finds_dh_and_skips_pitcher_batting():
    from src.starts import starts
    rows, ab = [], 0
    for half, (bat_base, fld_base) in {"Top": (200, 100), "Bot": (100, 200)}.items():
        lineup = [bat_base + p for p in range(2, 10)] + [bat_base + (10 if half == "Top" else 50)]  # home: pitcher bats
        for b in lineup:
            ab += 1
            rows.append({"game_pk": 1, "inning": 1, "inning_topbot": half, "at_bat_number": ab, "batter": b,
                         "pitcher": fld_base + 50, "home_team": "H", "away_team": "A",
                         **{f"fielder_{p}": fld_base + p for p in range(2, 10)}})
    s = starts(pd.DataFrame(rows))
    assert set(s.loc[s["pos"] == 10, "player"]) == {210}       # away DH found, home pitcher-batting has no DH
    assert (s["pos"].between(2, 9)).sum() == 16


@pytest.fixture(scope="module")
def sim_root(tmp_path_factory):
    from tests import sim
    root = tmp_path_factory.mktemp("bw")
    sim.write(root)
    env = {**os.environ, "BW_ROOT": str(root), "PYTHONPATH": str(REPO)}
    r = subprocess.run([sys.executable, "-m", "src.value_mix"], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return root


def test_linear_weights_are_sane(sim_root):
    lw = pd.read_csv(sim_root / "reports" / "linear_weights.csv").iloc[0]
    assert lw["out"] < 0 < lw["walk"] < lw["single"] < lw["double"] < lw["home_run"]


def test_summary_has_every_position_and_valid_shares(sim_root):
    s = pd.read_csv(sim_root / "reports" / "value_mix_summary.csv")
    assert list(s["pos"]) == ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
    assert ((s["off_share"] >= 0) & (s["off_share"] <= 1)).all()
    assert s.loc[s["pos"] == "DH", "off_share"].iloc[0] == 1.0
    assert (sim_root / "reports" / "value_mix.png").exists()


def test_players_centred_within_season_position(sim_root):
    p = pd.read_csv(sim_root / "reports" / "value_mix_players.csv")
    m = p.groupby(["season", "pos"])[["off_c", "def_c"]].mean().abs()
    assert (m < 1e-9).all().all()
