"""Defensive value per GAME from IQ_Savant's models (ABS excluded), in runs above average at the position.

  C            framing runs (count-valued, data/league/catching) + blocking runs + throwing runs
  1B/2B/3B/SS  RS Field + double-play runs
  LF/CF/RF     RS Range + RS Arm
Source: data/league/fielding_<year>.csv.gz (one row per player / game / position) + catching_<year>.csv.gz.
These give the game-to-game SHAPE; src.war rescales each player-season so its total matches Baseball-Reference.
"""
from __future__ import annotations

import pandas as pd

from src.config import INFIELD, LEAGUE, OUTFIELD


def game_defense(year: int) -> pd.DataFrame:
    f = pd.read_csv(LEAGUE / f"fielding_{year}.csv.gz")
    f = f[f["pos"].between(2, 9)].copy()
    num = lambda c: pd.to_numeric(f[c], errors="coerce").fillna(0.0) if c in f else 0.0
    f["def_iq"] = 0.0
    inf, of, c = f["pos"].isin(INFIELD), f["pos"].isin(OUTFIELD), f["pos"] == 2
    f.loc[inf, "def_iq"] = (num("runs") + num("dp_runs"))[inf]
    f.loc[of, "def_iq"] = (num("runs") + num("arm_runs"))[of]
    f.loc[c, "def_iq"] = (num("c_runs") + num("t_runs"))[c]
    f["game_pk"] = pd.to_numeric(f["game_pk"], errors="coerce").astype("int64")
    out = f.groupby(["player", "game_pk", "pos"], as_index=False).agg(
        game_date=("game_date", "first"), outs=("outs", "sum"), def_iq=("def_iq", "sum"))
    fr_p = LEAGUE / f"catching_{year}.csv.gz"
    if fr_p.exists():
        fr = pd.read_csv(fr_p)
        fr["game_pk"] = pd.to_numeric(fr["game_pk"], errors="coerce").astype("int64")
        fr = fr.groupby(["catcher", "game_pk"])["runs"].sum().rename("framing")
        out = out.join(fr, on=["player", "game_pk"])
        is_c = out["pos"] == 2
        out.loc[is_c, "def_iq"] += out.loc[is_c, "framing"].fillna(0.0)
        out = out.drop(columns="framing")
    out.insert(0, "season", year)
    return out


def season_defense(year: int) -> pd.DataFrame:
    g = game_defense(year)
    s = g.groupby(["player", "pos"]).agg(outs=("outs", "sum"), def_runs=("def_iq", "sum")).reset_index()
    s["inn"] = s["outs"] / 3
    s.insert(0, "season", year)
    return s
