"""Offensive value per player-season, in runs above average: batting + baserunning.

Batting       sum over his wOBA PAs of (linear weight of the result - league average per PA), park-adjusted PA by
              PA: - (park factor of that game's park - 1) x league runs per PA.   (= park-adjusted wRAA)
Baserunning   wSB   steals / caught stealing (pickoff-caught-stealing counts as CS), valued from RE24, minus the
                    league's steal runs per time on first (1B + BB + HBP).
              UBR   extra bases on singles, doubles and outs in play: for every runner on base, the run value of
                    where he ended up minus the league-average result for the same play type, base, outs and
                    field (LF / CF / RF / infield).
              wGDP  double plays hit into vs the league rate, on PAs with a runner on 1st and < 2 outs.
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


# ---------------------------------------------------------------- batting
def batting(pa: pd.DataFrame, lw: dict, pf: dict) -> pd.DataFrame:
    cls = lw_class(pa)
    w = pa[cls.notna()].copy()
    w["lw"] = cls[cls.notna()].map(lw).astype(float)
    lg = w["lw"].mean()
    tot_runs = pa.groupby(["game_pk", "batting_team"])["score_post"].max().sum()
    runs_per_pa = tot_runs / len(w)
    w["park"] = -(w["home_team"].map(pf).fillna(1.0) - 1.0) * runs_per_pa
    w["bat"] = w["lw"] - lg + w["park"]
    out = w.groupby("batter").agg(pa=("bat", "size"), bat_runs=("bat", "sum"), park_runs=("park", "sum"))
    out.index.name = "player"
    return out


# ---------------------------------------------------------------- baserunning: steals
def _runner_rows(year: int) -> pd.DataFrame:
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
    # value each kind from RE24, using the state at the start of the PA it happened in (single-runner moves only)
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
    ev["rv"] = ev["kind"].map(value)
    sb_runs = ev.groupby("runner")["rv"].sum()
    # opportunities: times on first via 1B / BB / HBP
    on1 = pa[pa["events"].isin(["single", "walk", "hit_by_pitch"])].groupby("batter").size()
    lg_rate = ev["rv"].sum() / on1.sum()
    players = sb_runs.index.union(on1.index)
    w = sb_runs.reindex(players, fill_value=0) - lg_rate * on1.reindex(players, fill_value=0)
    return pd.DataFrame({"wsb": w, "sb_value": value.get("sb", np.nan), "cs_value": value.get("cs", np.nan)})


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


def ubr(pa: pd.DataFrame, runners: pd.DataFrame, re: dict) -> pd.Series:
    p = pa.assign(group=_bip_group(pa), field=_field(pa["hit_location"]))
    p = p[p["group"].notna() & (p["bases"] > 0)]
    mv = runners[(runners["play_index"] == runners["bip_index"]) & runners["start"].isin(BASE_BIT)]
    # one row per runner per PA: first start base, last end, out if any movement says out
    agg = mv.groupby(["game_pk", "at_bat_number", "runner"], sort=False).agg(
        start=("start", "first"), end=("end", "last"), is_out=("is_out", "max")).reset_index()
    agg["result"] = np.where(agg["is_out"], "out", agg["end"].fillna("out"))
    d = agg.merge(p[["game_pk", "at_bat_number", "group", "field", "outs", "batting_team"]],
                  on=["game_pk", "at_bat_number"])
    if d.empty:
        return pd.Series(dtype=float)
    batter_bit = d["group"].map({"single": 1, "double": 2}).fillna(0).astype(int)
    batter_out = d["group"].isin(["air_out", "ground_out"]).astype(int)
    results = ["1B", "2B", "3B", "score", "out"]

    def value(res: str) -> np.ndarray:
        rbit = BASE_BIT.get(res, 0)
        outs = d["outs"].to_numpy() + batter_out.to_numpy() + (1 if res == "out" else 0)
        bases = batter_bit.to_numpy() | rbit
        return (1.0 if res == "score" else 0.0) + re_of(re, bases, outs)

    vals = {r: value(r) for r in results}
    d["v_actual"] = np.select([d["result"] == r for r in results], [vals[r] for r in results], np.nan)
    # league-average result for the same play type / start base / outs / field (backs off to no field if thin)
    keys_full, keys_thin = ["group", "start", "outs", "field"], ["group", "start", "outs"]
    for keys, tag in ((keys_full, "f"), (keys_thin, "t")):
        freq = pd.crosstab([d[k] for k in keys], d["result"], normalize="index")
        n = d.groupby(keys).size().rename(f"n_{tag}")
        for r in results:
            col = freq[r] if r in freq else pd.Series(0.0, index=freq.index)
            d = d.join(col.rename(f"p_{tag}_{r}"), on=keys)
        d = d.join(n, on=keys)
    use_full = d["n_f"] >= 20
    exp = np.zeros(len(d))
    for r in results:
        p_r = np.where(use_full, d[f"p_f_{r}"], d[f"p_t_{r}"])
        exp += np.nan_to_num(p_r) * vals[r]
    d["ubr"] = d["v_actual"] - exp
    return d.groupby("runner")["ubr"].sum()


# ---------------------------------------------------------------- baserunning: double plays
def gdp(pa: pd.DataFrame, re: dict) -> pd.Series:
    opp = pa[(pa["woba_denom"].fillna(0) > 0) & ((pa["bases"].fillna(0).astype(int) & 1) > 0) & (pa["outs"] < 2)]
    is_gdp = opp["events"].astype(str).eq("grounded_into_double_play")
    rv = play_value(opp, re)
    other_out = lw_class(opp).eq("out") & ~is_gdp
    cost = rv[is_gdp].mean() - rv[other_out].mean()           # extra runs lost by a DP instead of an ordinary out
    lg = is_gdp.mean()
    by = opp.assign(g=is_gdp).groupby("batter")["g"].agg(["sum", "size"])
    return (by["sum"] - lg * by["size"]) * cost


# ---------------------------------------------------------------- all together
def season_offense(year: int, pa: pd.DataFrame, re: dict, lw: dict, pf: dict) -> pd.DataFrame:
    runners = _runner_rows(year)
    bat = batting(pa, lw, pf)
    sb = steals(pa, runners, re)["wsb"]
    ub = ubr(pa, runners, re)
    gd = gdp(pa, re)
    out = bat.join(sb.rename("wsb"), how="left").join(ub.rename("ubr"), how="left").join(gd.rename("wgdp"), how="left")
    out[["wsb", "ubr", "wgdp"]] = out[["wsb", "ubr", "wgdp"]].fillna(0.0)
    out["bsr"] = out["wsb"] + out["ubr"] + out["wgdp"]
    out["off_runs"] = out["bat_runs"] + out["bsr"]
    out.insert(0, "season", year)
    return out.reset_index()
