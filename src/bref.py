"""Baseball-Reference WAR (bWAR) per player-season -- the season-level anchor for game-by-game WAR.

    python -m src.bref        # (re)download and cache data/raw/bref_war_bat.csv.gz

From B-Ref's daily WAR file (via pybaseball.bwar_bat), team stints summed into one row per player-season:
  off_runs   runs_above_avg_off   batting + baserunning + double plays (no position, no replacement)
  def_runs   runs_defense          fielding + catcher defense (DRS-based), no position
  pos_runs   runs_position         positional adjustment
  rep_runs   runs_replacement      replacement level
  WAR, WAR_off, WAR_def, WAA, runs_above_avg
Runs per win for each season = median of runs_above_avg / WAA across players (B-Ref's own conversion).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FIRST_SEASON, RAW

CACHE = RAW / "bref_war_bat.csv.gz"
KEEP = ["mlb_ID", "name_common", "year_ID", "stint_ID", "team_ID", "pitcher", "G", "PA", "Inn", "runs_bat",
        "runs_br", "runs_dp", "runs_field", "runs_catcher", "runs_defense", "runs_position", "runs_replacement",
        "runs_above_avg", "runs_above_avg_off", "runs_above_avg_def", "WAA", "WAR", "WAR_off", "WAR_def"]


def download() -> pd.DataFrame:
    from pybaseball import bwar_bat
    d = bwar_bat(return_all=True)
    d = d[pd.to_numeric(d["year_ID"], errors="coerce") >= FIRST_SEASON]
    d = d[[c for c in KEEP if c in d.columns]]
    RAW.mkdir(parents=True, exist_ok=True)
    d.to_csv(CACHE, index=False)
    print(f"saved {len(d):,} B-Ref batting rows ({int(d['year_ID'].min())}-{int(d['year_ID'].max())}) -> {CACHE}")
    return d


def load(refresh: bool = False) -> pd.DataFrame:
    d = download() if refresh or not CACHE.exists() else pd.read_csv(CACHE)
    for c in d.columns:
        if c not in ("name_common", "team_ID", "pitcher"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["mlb_ID"].notna()]
    num = [c for c in KEEP if c in d.columns and c not in ("mlb_ID", "name_common", "year_ID", "stint_ID",
                                                            "team_ID", "pitcher")]
    s = d.groupby(["mlb_ID", "year_ID"]).agg({**{c: "sum" for c in num}, "name_common": "first"}).reset_index()
    if "runs_above_avg_off" not in s:
        s["runs_above_avg_off"] = s[["runs_bat", "runs_br", "runs_dp"]].sum(axis=1)
    s = s.rename(columns={"mlb_ID": "player", "year_ID": "season", "name_common": "name",
                          "runs_above_avg_off": "off_runs", "runs_defense": "def_runs",
                          "runs_position": "pos_runs", "runs_replacement": "rep_runs"})
    s["player"] = s["player"].astype("int64")
    s["season"] = s["season"].astype(int)
    return s


def runs_per_win(s: pd.DataFrame) -> dict:
    ok = s["WAA"].abs() > 0.5
    r = (s.loc[ok, "runs_above_avg"] / s.loc[ok, "WAA"])
    return r.groupby(s.loc[ok, "season"]).median().to_dict()


if __name__ == "__main__":
    s = load(refresh=True)
    print(f"{len(s):,} player-seasons; runs per win: "
          + ", ".join(f"{y}: {v:.2f}" for y, v in sorted(runs_per_win(s).items())))
