"""Load a season of saved Statcast pitches (regular season only) and turn it into plate appearances."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DERIVED, RAW

COLS = ["game_pk", "game_date", "game_type", "at_bat_number", "pitch_number", "inning", "inning_topbot",
        "batter", "pitcher", "events", "bb_type", "hit_location", "on_1b", "on_2b", "on_3b", "outs_when_up",
        "bat_score", "post_bat_score", "home_team", "away_team", "woba_denom",
        "post_home_score", "post_away_score"] + [f"fielder_{i}" for i in range(2, 10)]

HITS = {"single", "double", "triple", "home_run"}
LW_CLASSES = ["single", "double", "triple", "home_run", "walk", "hit_by_pitch", "out"]


def load(season: int) -> pd.DataFrame:
    path = RAW / f"statcast_{season}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- data/raw should link to IQ_Savant's data/raw "
                                f"(or run IQ_Savant's pipeline.build_league_reference {season})")
    df = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
    if "game_type" in df:
        df = df[df["game_type"].astype(str) == "R"]
    for c in ["game_pk", "at_bat_number", "pitch_number", "inning", "outs_when_up", "bat_score",
              "post_bat_score", "woba_denom", "post_home_score", "post_away_score", "hit_location",
              "batter", "pitcher", "on_1b", "on_2b", "on_3b"] + [f"fielder_{i}" for i in range(2, 10)]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["events"] = df["events"].where(df["events"].notna(), None)
    return df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def base_code(df: pd.DataFrame) -> pd.Series:
    """1 = runner on 1st, 2 = on 2nd, 4 = on 3rd (summed)."""
    return (df["on_1b"].notna().astype(int) + 2 * df["on_2b"].notna().astype(int)
            + 4 * df["on_3b"].notna().astype(int))


def plate_appearances(df: pd.DataFrame) -> pd.DataFrame:
    """One row per at-bat number: state at its first pitch, state at its last pitch, the result, and the state the
    NEXT at-bat in the same half-inning starts in (none = the inning ended)."""
    half = ["game_pk", "inning", "inning_topbot"]
    g = df.groupby(["game_pk", "at_bat_number"], sort=False)
    first, last = g.head(1).set_index(["game_pk", "at_bat_number"]), g.tail(1).set_index(["game_pk", "at_bat_number"])
    pa = pd.DataFrame({
        "inning": last["inning"], "inning_topbot": last["inning_topbot"],
        "batter": last["batter"], "pitcher": last["pitcher"], "events": last["events"],
        "bb_type": last["bb_type"], "hit_location": last["hit_location"], "woba_denom": last["woba_denom"],
        "home_team": last["home_team"], "away_team": last["away_team"],
        "b0": base_code(first.reset_index()).to_numpy(), "o0": first["outs_when_up"],
        "bases": base_code(last.reset_index()).to_numpy(), "outs": last["outs_when_up"],
        "score_pre": last["bat_score"], "score_post": last["post_bat_score"],
        "start_score": first["bat_score"],
        "on_1b": last["on_1b"], "on_2b": last["on_2b"], "on_3b": last["on_3b"],
    }).reset_index()
    pa["half_end"] = pa.groupby(half)["score_post"].transform("max")
    nxt = pa.groupby(half)
    pa["next_bases"] = nxt["b0"].shift(-1)
    pa["next_outs"] = nxt["o0"].shift(-1)
    pa["runs_on_play"] = (pa["score_post"] - pa["score_pre"]).clip(lower=0)
    pa["batting_team"] = np.where(pa["inning_topbot"].astype(str) == "Top", pa["away_team"], pa["home_team"])
    return pa


def prepare(season: int, force: bool = False) -> None:
    """Read the big pitch file ONCE per season and save the two small tables everything else uses:
    plate appearances (data/derived/pa_<year>) and starts by position (data/derived/starts_<year>)."""
    from src.starts import starts
    DERIVED.mkdir(parents=True, exist_ok=True)
    pa_p, st_p = DERIVED / f"pa_{season}.csv.gz", DERIVED / f"starts_{season}.csv.gz"
    if pa_p.exists() and st_p.exists() and not force:
        return
    print(f"[{season}] reading Statcast pitches...", flush=True)
    df = load(season)
    plate_appearances(df).to_csv(pa_p, index=False)
    starts(df).to_csv(st_p, index=False)


def cached_pa(season: int) -> pd.DataFrame:
    prepare(season)
    return pd.read_csv(DERIVED / f"pa_{season}.csv.gz", low_memory=False)


def cached_starts(season: int) -> pd.DataFrame:
    prepare(season)
    return pd.read_csv(DERIVED / f"starts_{season}.csv.gz")
