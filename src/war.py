"""Game-by-game WAR for every position player, anchored to FanGraphs (fWAR) or Baseball-Reference (bWAR) seasons.

    python -m src.war                  # fWAR if every season has a data/raw/fg_<year>.json, else bWAR
    python -m src.war --anchor bref    # force one anchor (fg / bref)
Writes data/derived/game_war_<anchor>_<year>.csv.gz + reports/war_check_<anchor>.csv

For each player-game:
  off   our offense that game (batting + baserunning, from every PA / runner event)
        + (the anchor season offense - our season offense) x his share of the season's PAs
  def   our IQ defense that game (framing / blocking / throwing, fielding, DP, arm)
        + (the anchor season defense - our season defense) x his share of the season's defensive innings
  pos   the anchor season positional adjustment x share of innings in the field (share of games for DH-only players)
  rep   the anchor season replacement runs x share of PAs
  WAR = (off + def + pos + rep) / that season's runs per win (the anchor's), plus a small conversion term spread by PA so
  the season sums exactly to the anchor's WAR (it converts runs to wins with each player's own run environment).
So every player-season sums EXACTLY to the anchor's runs, while the ups and downs from game to game come from our
event-level data. Players the anchor doesn't have (very rare) keep our numbers and are flagged anchor_match = False.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import bref, config, fangraphs
from src.config import DERIVED, REPORTS
from src.defense import game_defense
from src.offense import by_game, offense_events
from src.run_values import game_runs, linear_weights, park_factors, re24
from src.statcast import cached_pa


def _share(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.where(den > 0)).fillna(0.0)


ANCHORS = {"fg": fangraphs, "bref": bref}


def default_anchor() -> str:
    have = set(fangraphs.files())
    return "fg" if have and set(config.seasons()) <= have else "bref"


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
    g["anchor_match"] = g["player"].isin(bs.index)
    for c in ["off_runs", "def_runs", "pos_runs", "rep_runs", "WAR", "name"]:
        g[f"a_{c}"] = g["player"].map(bs[c]) if c in bs else np.nan
    m = g["anchor_match"]
    pa_sh, out_sh = _share(g["pa"], g["pa_season"]), _share(g["outs"], g["outs_season"])
    game_sh = 1.0 / g["n_games"]
    field_sh = np.where(g["outs_season"] > 0, out_sh, game_sh)
    g["off"] = g["off_ours"] + np.where(m, (g["a_off_runs"].fillna(0) - g["off_season_ours"]) * pa_sh, 0.0)
    g["def"] = g["def_iq"] + np.where(m, (g["a_def_runs"].fillna(0) - g["def_season_ours"]) * field_sh, 0.0)
    g["pos_adj"] = np.where(m, g["a_pos_runs"].fillna(0) * field_sh, 0.0)
    g["rep"] = np.where(m, g["a_rep_runs"].fillna(0) * pa_sh, 0.0)
    r = rpw.get(year, 10.0)
    g["runs_per_win"] = r
    g["war_off"], g["war_def"] = g["off"] / r, g["def"] / r
    g["war_pos"], g["war_rep"] = g["pos_adj"] / r, g["rep"] / r
    g["war_linear"] = g["war_off"] + g["war_def"] + g["war_pos"] + g["war_rep"]
    # the anchor turns runs into wins with each player's own run environment, so its WAR differs a little from
    # runs / league runs-per-win. Spread that leftover by PA share so every season sums EXACTLY to bWAR.
    left = g["player"].map(bs["WAR"]) - g.groupby("player")["war_linear"].transform("sum") if "WAR" in bs else 0.0
    g["war_conv"] = np.where(m, pd.Series(left, index=g.index).fillna(0.0) * np.where(g["pa_season"] > 0, pa_sh, game_sh), 0.0)
    g["war"] = g["war_linear"] + g["war_conv"]
    g.insert(0, "season", year)
    keep = ["season", "game_date", "game_pk", "player", "a_name", "pos", "pa", "outs",
            "bat", "wsb", "ubr", "wgdp", "off_ours", "def_iq", "off", "def", "pos_adj", "rep",
            "war_off", "war_def", "war_pos", "war_rep", "war_conv", "war_linear", "war", "runs_per_win", "anchor_match", "a_WAR"]
    return g[keep].rename(columns={"a_name": "name"})


def load_all(anchor: str | None = None, force: bool = False) -> dict[int, pd.DataFrame]:
    anchor = anchor or default_anchor()
    years = config.seasons()
    if not years:
        raise SystemExit("No Statcast seasons found in data/raw -- is the data/raw link to IQ_Savant there?")
    DERIVED.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for y in years:
        p = DERIVED / f"game_war_{anchor}_{y}.csv.gz"
        if p.exists() and not force:
            out[y] = pd.read_csv(p)
        else:
            todo.append(y)
    if todo:
        src = ANCHORS[anchor]
        b = src.load()
        missing = [y for y in todo if y not in set(b["season"])]
        if missing:
            raise SystemExit(f"{anchor} anchor has no data for {missing} -- "
                             + ("save data/raw/fg_<year>.json for those seasons (link in src/fangraphs.py)"
                                if anchor == "fg" else "run  python -m src.bref"))
        rpw = src.runs_per_win(b)
        pf = park_factors(pd.concat([game_runs(cached_pa(y)) for y in years]))
        pd.Series(pf).sort_values().rename("park_factor").to_csv(REPORTS / "park_factors.csv")
        for y in todo:
            print(f"[{y}] game-by-game WAR ({anchor} anchor)...", flush=True)
            out[y] = build_season(y, pf, b, rpw)
            out[y]["anchor"] = anchor
            out[y].to_csv(DERIVED / f"game_war_{anchor}_{y}.csv.gz", index=False)
    return out


def check(games: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for y, g in games.items():
        s = g.groupby("player").agg(war=("war_linear", "sum"), anc=("a_WAR", "first"), pa=("pa", "sum"),
                                    match=("anchor_match", "first"))
        s = s[s["match"] & (s["pa"] >= 100)]
        rows.append({"season": y, "players_100pa": len(s), "corr_runs_based_vs_anchor_WAR": s["war"].corr(s["anc"]),
                     "mean_abs_diff_before_conv": (s["war"] - s["anc"]).abs().mean(),
                     "max_abs_diff_after_conv": (g.groupby("player")["war"].sum().reindex(s.index) - s["anc"]).abs().max(),
                     "unmatched_player_games": int((~g["anchor_match"]).sum())})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    import sys
    argv = sys.argv[1:] if argv is None else argv
    anchor = argv[argv.index("--anchor") + 1] if "--anchor" in argv else default_anchor()
    REPORTS.mkdir(parents=True, exist_ok=True)
    games = load_all(anchor, force=True)
    c = check(games)
    c.to_csv(REPORTS / f"war_check_{anchor}.csv", index=False)
    print(c.round(3).to_string(index=False))
    print(f"\nGame-by-game WAR ({anchor}) saved to data/derived/game_war_{anchor}_<year>.csv.gz")


if __name__ == "__main__":
    main()
