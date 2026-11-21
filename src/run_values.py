"""Run values built from the season's own plays: RE24 -> linear weights per event, plus park factors.

RE24         average runs from a base/out state to the end of the half-inning (state at the first pitch of each PA).
Linear wts   for every PA: runs on the play + RE(state the next PA starts in, 0 if the inning ended) - RE(state at
             its last pitch). Averaged by result: 1B, 2B, 3B, HR, BB, HBP and OUT (every other PA that counts in
             wOBA -- strikeouts, outs in play, errors, fielder's choices, sac flies). Season-specific, so BB / HBP
             carry their real value each year. IBB and sac bunts are left out, as in wOBA.
Park factor  runs per game in a team's home games / in its road games, pooled over every study season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.statcast import HITS


def re24(pa: pd.DataFrame) -> dict:
    to_end = pa["half_end"] - pa["start_score"]
    return to_end.groupby([pa["b0"].astype(int), pa["o0"].astype(int)]).mean().to_dict()


def re_of(re: dict, bases, outs) -> np.ndarray:
    b = np.asarray(bases, dtype=float)
    o = np.asarray(outs, dtype=float)
    out = np.zeros(len(b))
    ok = ~np.isnan(b) & ~np.isnan(o) & (o < 3)
    out[ok] = [re.get((int(x), int(y)), 0.0) for x, y in zip(b[ok], o[ok])]
    return out


def play_value(pa: pd.DataFrame, re: dict) -> pd.Series:
    """Run value of each PA's result (from the state at its last pitch)."""
    before = re_of(re, pa["bases"], pa["outs"])
    after = re_of(re, pa["next_bases"], pa["next_outs"])          # NaN next state (inning over) -> 0
    return pd.Series(pa["runs_on_play"].to_numpy() + after - before, index=pa.index)


def lw_class(pa: pd.DataFrame) -> pd.Series:
    ev = pa["events"].astype(str)
    cls = pd.Series("out", index=pa.index)
    for e in ["single", "double", "triple", "home_run", "walk", "hit_by_pitch"]:
        cls[ev == e] = e
    return cls.where(pa["woba_denom"].fillna(0) > 0)                # not a wOBA PA (IBB, sac bunt, CS...) -> NaN


def linear_weights(pa: pd.DataFrame, re: dict) -> dict:
    rv, cls = play_value(pa, re), lw_class(pa)
    return rv.groupby(cls).mean().to_dict()


def park_factors(games: pd.DataFrame) -> dict:
    """games: one row per game with home_team, away_team, total runs. -> {team: home R/G / road R/G}."""
    home = games.groupby("home_team")["runs"].mean()
    road = games.groupby("away_team")["runs"].mean()
    pf = (home / road).dropna()
    return pf.to_dict()


def game_runs(pa: pd.DataFrame) -> pd.DataFrame:
    """Total runs per game from the PA table (final score of each side)."""
    side = pa.groupby(["game_pk", "batting_team"])["score_post"].max().groupby(level=0).sum()
    teams = pa.drop_duplicates("game_pk").set_index("game_pk")[["home_team", "away_team"]]
    return teams.join(side.rename("runs")).reset_index()
