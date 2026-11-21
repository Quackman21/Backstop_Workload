"""Game-by-game workload log for every position player, with framing joined on for catchers.

    python -m src.build_workload              # every season from 2021 on
    python -m src.build_workload 2024 2025    # just these seasons

Input   data/league/fielding_<year>.csv.gz   (one row per player / game / position, from IQ Savant)
        data/league/catching_<year>.csv.gz   (framing runs per catcher / game / zone region)
Output  data/workload/workload_<year>.csv.gz (one row per player / game)

Workload columns
  pos_main        position with the most outs that game (2 = C, 3 = 1B ... 9 = RF)
  outs, inn       defensive outs / innings that game (all positions)
  outs_c          outs caught that game
  game_n          player's nth game of the season (games in the field only)
  days_rest       days since his previous game (0 = doubleheader, NaN = first game)
  streak          consecutive calendar days played, counting this one
  g_7d, g_14d     games played in the last 7 / 14 days, counting this one
  inn_7d, inn_14d innings in the last 7 / 14 days
  inn_cum, inn_c_cum  season innings so far (all / at catcher), counting this game
Catcher performance
  frame_n, frame_runs, frame_strikes, frame_exp   framing pitches, runs, called strikes, expected strikes

Note: only games he played in the FIELD show up -- DH-only days aren't in the fielding files yet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEAGUE = ROOT / "data" / "league"
OUT = ROOT / "data" / "workload"
FIRST_SEASON = 2021  # catcher stances changed; earlier seasons are left out of the study


def seasons() -> list[int]:
    return sorted(y for y in (int(p.name[9:13]) for p in LEAGUE.glob("fielding_*.csv.gz")) if y >= FIRST_SEASON)


def player_games(year: int) -> pd.DataFrame:
    f = pd.read_csv(LEAGUE / f"fielding_{year}.csv.gz", usecols=["game_date", "game_pk", "player", "pos", "outs"])
    f["game_date"] = pd.to_datetime(f["game_date"])
    f["game_pk"] = f["game_pk"].astype("int64")
    f["outs_c"] = f["outs"].where(f["pos"] == 2, 0)
    main = (f.sort_values("outs", ascending=False)
             .drop_duplicates(["player", "game_pk"])[["player", "game_pk", "pos"]]
             .rename(columns={"pos": "pos_main"}))
    g = (f.groupby(["player", "game_pk", "game_date"], as_index=False)[["outs", "outs_c"]].sum()
          .merge(main, on=["player", "game_pk"]))
    return g


def add_workload(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values(["player", "game_date", "game_pk"]).reset_index(drop=True)
    g["inn"] = g["outs"] / 3
    by = g.groupby("player", sort=False)
    g["game_n"] = by.cumcount() + 1
    g["days_rest"] = by["game_date"].diff().dt.days
    g["inn_cum"] = by["inn"].cumsum()
    g["inn_c_cum"] = by["outs_c"].cumsum() / 3

    # consecutive calendar days played (a doubleheader doesn't break or add to it)
    new_run = g["days_rest"].isna() | (g["days_rest"] > 1)
    run_id = new_run.cumsum()
    day_n = g.groupby(run_id)["game_date"].transform(lambda d: (d - d.iloc[0]).dt.days + 1)
    g["streak"] = day_n

    # rolling 7 / 14 day windows (time-based, counting this game)
    for days in (7, 14):
        roll = (g.set_index("game_date").groupby("player", sort=False)["inn"]
                 .rolling(f"{days}D"))
        g[f"g_{days}d"] = roll.count().to_numpy()
        g[f"inn_{days}d"] = roll.sum().to_numpy()
    return g


def framing(year: int) -> pd.DataFrame:
    p = LEAGUE / f"catching_{year}.csv.gz"
    if not p.exists():
        return pd.DataFrame(columns=["player", "game_pk"])
    c = pd.read_csv(p)
    c["game_pk"] = c["game_pk"].astype("int64")
    return (c.groupby(["catcher", "game_pk"], as_index=False)
             .agg(frame_n=("n", "sum"), frame_runs=("runs", "sum"),
                  frame_strikes=("strikes", "sum"), frame_exp=("exp", "sum"))
             .rename(columns={"catcher": "player"}))


def build(year: int) -> pd.DataFrame:
    g = add_workload(player_games(year))
    g = g.merge(framing(year), on=["player", "game_pk"], how="left")
    g.insert(0, "season", year)
    return g


def main(argv: list[str]) -> None:
    years = [int(a) for a in argv] or seasons()
    if len(years) == 2 and years[1] - years[0] > 1:
        years = list(range(years[0], years[1] + 1))
    OUT.mkdir(parents=True, exist_ok=True)
    for y in years:
        w = build(y)
        w.to_csv(OUT / f"workload_{y}.csv.gz", index=False)
        c = w[w["pos_main"] == 2]
        print(f"{y}: {len(w):,} player-games, {w['player'].nunique()} players, "
              f"{c['player'].nunique()} catchers, max C streak {int(c['streak'].max()) if len(c) else 0} days")


if __name__ == "__main__":
    main(sys.argv[1:])
