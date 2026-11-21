"""Defensive value per player-season at one position, in runs above average (IQ_Savant models, ABS excluded).

  C            framing runs (count-valued, data/league/catching) + blocking runs + throwing runs
  1B/2B/3B/SS  RS Field + double-play runs
  LF/CF/RF     RS Range + RS Arm
Each comes from data/league/fielding_<year>.csv.gz -- one row per player / game / position.
"""
from __future__ import annotations

import pandas as pd

from src.config import INFIELD, LEAGUE, OUTFIELD


def season_defense(year: int) -> pd.DataFrame:
    f = pd.read_csv(LEAGUE / f"fielding_{year}.csv.gz")
    f = f[f["pos"].between(2, 9)].copy()
    num = lambda c: pd.to_numeric(f[c], errors="coerce").fillna(0.0) if c in f else 0.0
    f["def_runs"] = 0.0
    inf, of, c = f["pos"].isin(INFIELD), f["pos"].isin(OUTFIELD), f["pos"] == 2
    f.loc[inf, "def_runs"] = (num("runs") + num("dp_runs"))[inf]
    f.loc[of, "def_runs"] = (num("runs") + num("arm_runs"))[of]
    f.loc[c, "def_runs"] = (num("c_runs") + num("t_runs"))[c]
    out = f.groupby(["player", "pos"]).agg(outs=("outs", "sum"), def_runs=("def_runs", "sum")).reset_index()
    fr_p = LEAGUE / f"catching_{year}.csv.gz"
    if fr_p.exists():
        fr = pd.read_csv(fr_p).groupby("catcher")["runs"].sum().rename("framing")
        out = out.join(fr, on="player")
        is_c = out["pos"] == 2
        out.loc[is_c, "def_runs"] += out.loc[is_c, "framing"].fillna(0.0)
        out = out.drop(columns="framing")
    out["inn"] = out["outs"] / 3
    out.insert(0, "season", year)
    return out
