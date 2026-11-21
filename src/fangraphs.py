"""FanGraphs WAR (fWAR) per player-season -- a season-level anchor for game-by-game WAR, read from files saved in
the browser (FanGraphs blocks scripts).

Save one file per season into data/raw/ as fg_<year>.json, from this link (change both 2025s):
  https://www.fangraphs.com/api/leaders/major-league/data?pos=all&stats=bat&lg=all&qual=0&type=8&season=2025&season1=2025&ind=0&month=0&pageitems=3000&pagenum=1

Standard columns (same as src.bref):
  off_runs   Offense            batting + baserunning
  def_runs   RAR - Offense - Positional - League - Replacement
             = fielding (Statcast-based) + catcher framing, however FanGraphs splits them
  pos_runs   Positional
  rep_runs   Replacement + wLeague (league adjustment, spread by PA like replacement)
  WAR, RAR, name
Runs per win per season = median RAR / WAR (FanGraphs' own conversion).

    python -m src.fangraphs     # check the saved files: seasons, rows, and a catcher sanity table
"""
from __future__ import annotations

import json

import pandas as pd

from src.config import RAW

COLS = ["xMLBAMID", "PlayerName", "Season", "PA", "G", "Batting", "BaseRunning", "Fielding", "CFraming",
        "Positional", "wLeague", "Replacement", "Offense", "Defense", "RAR", "WAR", "position"]


def files() -> dict[int, object]:
    out = {}
    for p in sorted(RAW.glob("fg_*.json")):
        try:
            out[int(p.stem.split("_")[1])] = p
        except ValueError:
            continue
    return out


def load() -> pd.DataFrame:
    fs = files()
    if not fs:
        raise FileNotFoundError("No data/raw/fg_<year>.json files -- see src/fangraphs.py for the link to save")
    frames = []
    for year, p in fs.items():
        j = json.load(open(p))
        rows = j.get("data", j) if isinstance(j, dict) else j
        d = pd.DataFrame(rows)
        d = d[[c for c in COLS if c in d.columns]].copy()
        if "Season" not in d:
            d["Season"] = year
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    for c in d.columns:
        if c not in ("PlayerName", "position"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["xMLBAMID"].notna()]
    # one row per player-season (if a file ever has per-team rows too, keep the biggest = the season total)
    d = d.sort_values("PA", ascending=False).drop_duplicates(["xMLBAMID", "Season"])
    z = lambda c: d[c].fillna(0.0) if c in d else 0.0
    out = pd.DataFrame({
        "player": d["xMLBAMID"].astype("int64"), "season": d["Season"].astype(int), "name": d["PlayerName"],
        "off_runs": z("Offense"), "pos_runs": z("Positional"), "rep_runs": z("Replacement") + z("wLeague"),
        "WAR": d["WAR"], "RAR": d["RAR"], "fielding": z("Fielding"), "framing": z("CFraming"),
        "fg_position": d.get("position")})
    out["def_runs"] = d["RAR"] - out["off_runs"] - out["pos_runs"] - out["rep_runs"]
    return out.reset_index(drop=True)


def runs_per_win(s: pd.DataFrame) -> dict:
    ok = s["WAR"].abs() > 0.5
    return (s.loc[ok, "RAR"] / s.loc[ok, "WAR"]).groupby(s.loc[ok, "season"]).median().to_dict()


if __name__ == "__main__":
    s = load()
    print("seasons:", sorted(s["season"].unique()), "| player-seasons:", len(s))
    print("runs per win:", {k: round(v, 2) for k, v in sorted(runs_per_win(s).items())})
    c = s[s["fg_position"].astype(str).eq("C")].sort_values("def_runs")
    print("\nCatchers -- defense = everything defensive in fWAR; check it includes framing:")
    print(c[["season", "name", "fielding", "framing", "def_runs", "WAR"]].head(5).round(1).to_string(index=False))
    print(c[["season", "name", "fielding", "framing", "def_runs", "WAR"]].tail(5).round(1).to_string(index=False))
