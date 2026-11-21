"""Game-by-game WAR for every position player, anchored to Baseball-Reference season totals.

    python -m src.war            # writes data/derived/game_war_<year>.csv.gz + reports/war_check.csv

For each player-game:
  off   our offense that game (batting + baserunning, from every PA / runner event)
        + (B-Ref season offense - our season offense) x his share of the season's PAs
  def   our IQ defense that game (framing / blocking / throwing, fielding, DP, arm)
        + (B-Ref season defense - our season defense) x his share of the season's defensive innings
  pos   B-Ref season positional adjustment x share of innings in the field (share of games for DH-only players)
  rep   B-Ref season replacement runs x share of PAs
  WAR = (off + def + pos + rep) / that season's runs per win (B-Ref's).
So every player-season sums EXACTLY to B-Ref's runs, while the ups and downs from game to game come from our
event-level data. Players B-Ref doesn't have (very rare) keep our numbers and are flagged bref_match = False.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import bref, config
from src.config import DERIVED, REPORTS
from src.defense import game_defense
from src.offense import by_game, offense_events
from src.run_values import game_runs, linear_weights, park_factors, re24
from src.statcast import cached_pa


def _share(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.where(den > 0)).fillna(0.0)


def build_season(year: int, pf: dict, b: pd.DataFrame, rpw: dict) -> pd.DataFrame:
    pa = cached_pa(year)
    re = re24(pa)
    lw = linear_weights(pa, re)
    off = by_game(offense_events(year, pa, re, lw, pf))
    dfn = game_defense(year)
    d_game = dfn.groupby(["player", "game_pk"]).agg(outs=("outs", "sum"), def_iq=("def_iq", "sum"),
                                                     game_date=("game_date", "first")).reset_index()
    main_pos = dfn.sort_values("outs", ascending=False).drop_duplicates(["player", "game_pk"])[
        ["player", "game_pk", "pos"]]
    g = off.merge(d_game, on=["player", "game_pk"], how="outer").merge(main_pos, on=["player", "game_pk"], how="left")
    dates = pa.drop_duplicates("game_pk").set_index("game_pk")
    if "game_date" in pa:
        g["game_date"] = g["game_date"].fillna(g["game_pk"].map(dates["game_date"]))
    for c in ["bat", "wsb", "ubr", "wgdp", "bsr", "off_runs", "pa", "outs", "def_iq"]:
        g[c] = g[c].fillna(0.0)
    g["pos"] = g["pos"].fillna(10).astype(int)                 # no fielding that game -> DH / PH
    g = g.rename(columns={"off_runs": "off_ours"})

    by = g.groupby("player")
    g["pa_season"], g["outs_season"] = by["pa"].transform("sum"), by["outs"].transform("sum")
    g["n_games"] = by["game_pk"].transform("size")
    g["off_season_ours"], g["def_season_ours"] = by["off_ours"].transform("sum"), by["def_iq"].transform("sum")

    bs = b[b["season"] == year].set_index("player")
    g["bref_match"] = g["player"].isin(bs.index)
    for c in ["off_runs", "def_runs", "pos_runs", "rep_runs", "WAR", "name"]:
        g[f"bref_{c}"] = g["player"].map(bs[c]) if c in bs else np.nan
    m = g["bref_match"]
    pa_sh, out_sh = _share(g["pa"], g["pa_season"]), _share(g["outs"], g["outs_season"])
    game_sh = 1.0 / g["n_games"]
    field_sh = np.where(g["outs_season"] > 0, out_sh, game_sh)
    g["off"] = g["off_ours"] + np.where(m, (g["bref_off_runs"].fillna(0) - g["off_season_ours"]) * pa_sh, 0.0)
    g["def"] = g["def_iq"] + np.where(m, (g["bref_def_runs"].fillna(0) - g["def_season_ours"]) * field_sh, 0.0)
    g["pos_adj"] = np.where(m, g["bref_pos_runs"].fillna(0) * field_sh, 0.0)
    g["rep"] = np.where(m, g["bref_rep_runs"].fillna(0) * pa_sh, 0.0)
    r = rpw.get(year, 10.0)
    g["runs_per_win"] = r
    g["war_off"], g["war_def"] = g["off"] / r, g["def"] / r
    g["war_pos"], g["war_rep"] = g["pos_adj"] / r, g["rep"] / r
    g["war"] = g["war_off"] + g["war_def"] + g["war_pos"] + g["war_rep"]
    g.insert(0, "season", year)
    keep = ["season", "game_date", "game_pk", "player", "bref_name", "pos", "pa", "outs",
            "bat", "wsb", "ubr", "wgdp", "off_ours", "def_iq", "off", "def", "pos_adj", "rep",
            "war_off", "war_def", "war_pos", "war_rep", "war", "runs_per_win", "bref_match", "bref_WAR"]
    return g[keep].rename(columns={"bref_name": "name"})


def load_all(force: bool = False) -> dict[int, pd.DataFrame]:
    years = config.seasons()
    if not years:
        raise SystemExit("No Statcast seasons found in data/raw -- is the data/raw link to IQ_Savant there?")
    DERIVED.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for y in years:
        p = DERIVED / f"game_war_{y}.csv.gz"
        if p.exists() and not force:
            out[y] = pd.read_csv(p)
        else:
            todo.append(y)
    if todo:
        b = bref.load()
        rpw = bref.runs_per_win(b)
        pf = park_factors(pd.concat([game_runs(cached_pa(y)) for y in years]))
        pd.Series(pf).sort_values().rename("park_factor").to_csv(REPORTS / "park_factors.csv")
        for y in todo:
            print(f"[{y}] game-by-game WAR...", flush=True)
            out[y] = build_season(y, pf, b, rpw)
            out[y].to_csv(DERIVED / f"game_war_{y}.csv.gz", index=False)
    return out


def check(games: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for y, g in games.items():
        s = g.groupby("player").agg(war=("war", "sum"), bref=("bref_WAR", "first"), pa=("pa", "sum"),
                                    match=("bref_match", "first"))
        s = s[s["match"] & (s["pa"] >= 100)]
        rows.append({"season": y, "players_100pa": len(s), "corr_with_bWAR": s["war"].corr(s["bref"]),
                     "mean_abs_diff_WAR": (s["war"] - s["bref"]).abs().mean(),
                     "unmatched_player_games": int((~g["bref_match"]).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    games = load_all(force=True)
    c = check(games)
    c.to_csv(REPORTS / "war_check.csv", index=False)
    print(c.round(3).to_string(index=False))
    print("\nGame-by-game WAR saved to data/derived/game_war_<year>.csv.gz; check in reports/war_check.csv")


if __name__ == "__main__":
    main()
