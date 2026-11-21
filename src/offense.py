"""Offensive value in runs above average -- batting + baserunning -- kept EVENT BY EVENT so it can be summed per game
or per season.

Batting       each wOBA PA: linear weight of the result - league average per PA, park-adjusted in that game's park
              (- (park factor - 1) x league runs per PA).                              (= park-adjusted wRAA)
Baserunning   wSB   each steal / caught stealing (pickoff-caught-stealing counts as CS) valued from RE24; each time
                    he reaches first (1B / BB / HBP) is charged the league's steal runs per time on first.
              UBR   each runner on a single, double or out in play: run value of where he ended up - the league-average
                    result for the same play type, base, outs and field (LF / CF / RF / infield).
              wGDP  each PA with a runner on 1st and < 2 outs: (DP or not - league DP rate) x the extra runs a DP costs.
Every function returns rows of (player, game_pk, <component>) runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RAW
from src.run_values import lw_class, play_value, re_of

BASE_BIT = {"1B": 1, "2B": 2, "3B": 4}
SB_TYPES = {"stolen_base_2b": "sb", "stolen_base_3b": "sb", "stolen_base_home": "sb",
            "caught_stealing_2b": "cs", "caught_stealing_3b": "cs", "caught_stealing_home": "cs",
            "pickoff_caught_stealing_2b": "cs", "pickoff_caught_stealing_3b": "cs",
            "pickoff_caught_stealing_home": "cs"}
COMPONENTS = ["bat", "wsb", "ubr", "wgdp"]


# ---------------------------------------------------------------- batting
def batting(pa: pd.DataFrame, lw: dict, pf: dict) -> pd.DataFrame:
    cls = lw_class(pa)
    w = pa[cls.notna()].copy()
    w["lw"] = cls[cls.notna()].map(lw).astype(float)
    tot_runs = pa.groupby(["game_pk", "batting_team"])["score_post"].max().sum()
    runs_per_pa = tot_runs / len(w)
    park = -(w["home_team"].map(pf).fillna(1.0) - 1.0) * runs_per_pa
    return pd.DataFrame({"player": w["batter"], "game_pk": w["game_pk"], "bat": w["lw"] - w["lw"].mean() + park,
                         "pa": 1})


# ---------------------------------------------------------------- baserunning: steals
def runner_rows(year: int) -> pd.DataFrame:
    p = RAW / f"runners_{year}.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run  python -m src.fetch_runners --season {year}")
    r = pd.read_csv(p)
    for c in ["game_pk", "at_bat_number", "play_index", "bip_index", "runner"]:
        r[c] = pd.to_numeric(r[c], errors="coerce")
    return r


def steals(pa: pd.DataFrame, runners: pd.DataFrame, re: dict) -> pd.DataFrame:
    ev = runners[runners["event_type"].isin(SB_TYPES)].copy()
    ev["kind"] = ev["event_type"].map(SB_TYPES)
    ev = ev.drop_duplicates(["game_pk", "at_bat_number", "play_index", "runner"])
    st = pa.set_index(["game_pk", "at_bat_number"])[["b0", "o0"]]
    ev = ev.join(st, on=["game_pk", "at_bat_number"])
    bit = ev["start"].map(BASE_BIT)
    ok = bit.notna() & ev["b0"].notna() & ((ev["b0"].fillna(0).astype(int) & bit.fillna(0).astype(int)) > 0)
    v = ev[ok].copy()
    b = v["b0"].astype(int)
    sbit = v["start"].map(BASE_BIT).astype(int)
    ebit = v["end"].map(BASE_BIT).fillna(0).astype(int)
    after_b = np.where(v["kind"] == "sb", (b - sbit) | ebit, b - sbit)
    after_o = v["o0"] + (v["kind"] == "cs")
    runs = ((v["kind"] == "sb") & (v["end"] == "score")).astype(float)
    v["rv"] = runs + re_of(re, after_b, after_o) - re_of(re, b, v["o0"])
    value = v.groupby("kind")["rv"].mean().to_dict()
    ev["wsb"] = ev["kind"].map(value)
    on1 = pa[pa["events"].isin(["single", "walk", "hit_by_pitch"])]
    lg_rate = ev["wsb"].sum() / max(len(on1), 1)
    credit = ev[["runner", "game_pk", "wsb"]].rename(columns={"runner": "player"})
    charge = pd.DataFrame({"player": on1["batter"], "game_pk": on1["game_pk"], "wsb": -lg_rate})
    return pd.concat([credit, charge], ignore_index=True)


# ---------------------------------------------------------------- baserunning: extra bases (UBR)
def _bip_group(pa: pd.DataFrame) -> pd.Series:
    ev, bb = pa["events"].astype(str), pa["bb_type"].astype(str)
    g = pd.Series(None, index=pa.index, dtype=object)
    g[ev == "single"] = "single"
    g[ev == "double"] = "double"
    air = bb.isin(["fly_ball", "line_drive", "popup"])
    g[ev.isin(["field_out", "sac_fly"]) & air] = "air_out"
    g[(ev == "field_out") & (bb == "ground_ball")] = "ground_out"
    return g


def _field(hl: pd.Series) -> pd.Series:
    h = pd.to_numeric(hl, errors="coerce")
    return pd.Series(np.select([h == 7, h == 8, h == 9, h.between(1, 6)], ["LF", "CF", "RF", "IF"], "NA"),
                     index=hl.index)


def ubr(pa: pd.DataFrame, runners: pd.DataFrame, re: dict) -> pd.DataFrame:
    p = pa.assign(group=_bip_group(pa), field=_field(pa["hit_location"]))
    p = p[p["group"].notna() & (p["bases"] > 0)]
    mv = runners[(runners["play_index"] == runners["bip_index"]) & runners["start"].isin(BASE_BIT)]
    agg = mv.groupby(["game_pk", "at_bat_number", "runner"], sort=False).agg(
        start=("start", "first"), end=("end", "last"), is_out=("is_out", "max")).reset_index()
    agg["result"] = np.where(agg["is_out"].astype(bool), "out", agg["end"].fillna("out"))
    d = agg.merge(p[["game_pk", "at_bat_number", "group", "field", "outs"]], on=["game_pk", "at_bat_number"])
    if d.empty:
        return pd.DataFrame(columns=["player", "game_pk", "ubr"])
    batter_bit = d["group"].map({"single": 1, "double": 2}).fillna(0).astype(int)
    batter_out = d["group"].isin(["air_out", "ground_out"]).astype(int)
    results = ["1B", "2B", "3B", "score", "out"]

    def value(res: str) -> np.ndarray:
        outs = d["outs"].to_numpy() + batter_out.to_numpy() + (1 if res == "out" else 0)
        bases = batter_bit.to_numpy() | BASE_BIT.get(res, 0)
        return (1.0 if res == "score" else 0.0) + re_of(re, bases, outs)

    vals = {r: value(r) for r in results}
    d["v_actual"] = np.select([d["result"] == r for r in results], [vals[r] for r in results], np.nan)
    for keys, tag in ((["group", "start", "outs", "field"], "f"), (["group", "start", "outs"], "t")):
        freq = pd.crosstab([d[k] for k in keys], d["result"], normalize="index")
        for r in results:
            col = freq[r] if r in freq else pd.Series(0.0, index=freq.index)
            d = d.join(col.rename(f"p_{tag}_{r}"), on=keys)
        d = d.join(d.groupby(keys).size().rename(f"n_{tag}"), on=keys)
    use_full = d["n_f"] >= 20
    exp = np.zeros(len(d))
    for r in results:
        exp += np.nan_to_num(np.where(use_full, d[f"p_f_{r}"], d[f"p_t_{r}"])) * vals[r]
    return pd.DataFrame({"player": d["runner"], "game_pk": d["game_pk"], "ubr": d["v_actual"] - exp})


# ---------------------------------------------------------------- baserunning: double plays
def gdp(pa: pd.DataFrame, re: dict) -> pd.DataFrame:
    opp = pa[(pa["woba_denom"].fillna(0) > 0) & ((pa["bases"].fillna(0).astype(int) & 1) > 0) & (pa["outs"] < 2)]
    is_gdp = opp["events"].astype(str).eq("grounded_into_double_play")
    rv = play_value(opp, re)
    other_out = lw_class(opp).eq("out") & ~is_gdp
    cost = rv[is_gdp].mean() - rv[other_out].mean()
    return pd.DataFrame({"player": opp["batter"], "game_pk": opp["game_pk"],
                         "wgdp": (is_gdp.astype(float) - is_gdp.mean()) * cost})


# ---------------------------------------------------------------- all together
def offense_events(year: int, pa: pd.DataFrame, re: dict, lw: dict, pf: dict) -> pd.DataFrame:
    runners = runner_rows(year)
    parts = [batting(pa, lw, pf), steals(pa, runners, re), ubr(pa, runners, re), gdp(pa, re)]
    ev = pd.concat(parts, ignore_index=True)
    for c in COMPONENTS + ["pa"]:
        ev[c] = pd.to_numeric(ev[c], errors="coerce").fillna(0.0) if c in ev else 0.0
    ev = ev[ev["player"].notna()]
    ev["player"] = ev["player"].astype("int64")
    ev["game_pk"] = ev["game_pk"].astype("int64")
    return ev


def by_game(ev: pd.DataFrame) -> pd.DataFrame:
    g = ev.groupby(["player", "game_pk"])[COMPONENTS + ["pa"]].sum()
    g["bsr"] = g["wsb"] + g["ubr"] + g["wgdp"]
    g["off_runs"] = g["bat"] + g["bsr"]
    return g.reset_index()


def season_offense(year: int, pa: pd.DataFrame, re: dict, lw: dict, pf: dict) -> pd.DataFrame:
    g = by_game(offense_events(year, pa, re, lw, pf))
    s = g.groupby("player")[COMPONENTS + ["pa", "bsr", "off_runs"]].sum().rename(columns={"bat": "bat_runs"})
    s.insert(0, "season", year)
    return s.reset_index()
