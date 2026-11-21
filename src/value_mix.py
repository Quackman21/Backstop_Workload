"""Part 1 -- how each position gets its value: share of the differences between starters that come from offense
vs defense, 2021 -> latest season pooled.

    python -m src.value_mix            # builds everything it needs, then the chart + tables in reports/

Steps per season: Statcast -> PAs + starts -> RE24 / linear weights -> offense runs (batting + baserunning) ->
defense runs (IQ models) -> qualified starters (81+ starts at one position) -> rates per 150 games ->
centred on that season's position average. Then pooled, per position:
  SD offense, SD defense, their correlation, and offense share = var_off / (var_off + var_def). The overlap
  (covariance) is reported as a correlation beside it rather than folded in -- splitting it evenly can push a
  share past 100% when offense and defense pull in opposite directions.
  and a 95% interval on that share from resampling PLAYERS (a starter's six seasons move together).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.config import BOOTSTRAP, DERIVED, MIN_STARTS, POS, REPORTS, SCALE_GAMES, SEED
from src.defense import season_defense
from src.offense import season_offense
from src.run_values import game_runs, linear_weights, park_factors, re24
from src.statcast import cached_pa, cached_starts


def qualified(starts: pd.DataFrame, year: int) -> pd.DataFrame:
    n = starts.groupby(["player", "pos"]).size().rename("starts").reset_index()
    top = n.sort_values("starts", ascending=False).drop_duplicates("player")
    q = top[top["starts"] >= MIN_STARTS].copy()
    q.insert(0, "season", year)
    return q


def build_season(year: int, pf: dict) -> tuple[pd.DataFrame, dict]:
    pa = cached_pa(year)
    re = re24(pa)
    lw = linear_weights(pa, re)
    off = season_offense(year, pa, re, lw, pf)
    dfn = season_defense(year)
    q = qualified(cached_starts(year), year)
    pa_per_game = len(pa[pa["woba_denom"].fillna(0) > 0]) / (pa["game_pk"].nunique() * 2 * 9)
    d = q.merge(off.drop(columns="season"), on="player", how="left")
    d = d.merge(dfn.drop(columns="season"), on=["player", "pos"], how="left")
    d["off_rate"] = d["off_runs"] / d["pa"] * pa_per_game * SCALE_GAMES
    d["def_rate"] = np.where(d["pos"] == 10, 0.0, d["def_runs"] / d["inn"] * 9 * SCALE_GAMES)
    d["pa_per_game"] = pa_per_game
    return d, {"season": year, **{k: round(v, 4) for k, v in lw.items()}, "pa_per_game": round(pa_per_game, 3)}


def split(d: pd.DataFrame) -> dict:
    o, f = d["off_c"].to_numpy(), d["def_c"].to_numpy()
    vo, vf = o.var(ddof=1), f.var(ddof=1)
    cov = np.cov(o, f, ddof=1)[0, 1] if len(d) > 2 else 0.0
    tot = vo + vf + 2 * cov
    return {"sd_off": np.sqrt(vo), "sd_def": np.sqrt(vf), "cov": cov, "corr": cov / np.sqrt(vo * vf) if vf > 0 else 0.0,
            "sd_total": np.sqrt(tot), "off_share": vo / (vo + vf) if vo + vf > 0 else np.nan}


def summarize(pool: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for pos, name in POS.items():
        d = pool[pool["pos"] == pos]
        if len(d) < 5:
            continue
        s = split(d)
        players = d["player"].unique()
        by = {p: g for p, g in d.groupby("player")}
        boots = []
        for _ in range(BOOTSTRAP):
            pick = rng.choice(players, size=len(players), replace=True)
            boots.append(split(pd.concat([by[p] for p in pick]))["off_share"])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        rows.append({"pos": name, "player_seasons": len(d), "players": len(players), **s,
                     "def_share": 1 - s["off_share"], "off_share_lo": lo, "off_share_hi": hi})
    return pd.DataFrame(rows)


def chart(summary: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OFF, DEF = "#2a78d6", "#eb6834"            # categorical slots 1-2 (validated: CVD dE 24.7, contrast >= 3:1)
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    x = np.arange(len(summary))
    w, gap = 0.62, 0.006
    for i, r in summary.reset_index(drop=True).iterrows():
        dsh, osh = r["def_share"], r["off_share"]
        ax.bar(i, dsh - gap, w, bottom=0, color=DEF, zorder=2)
        ax.bar(i, osh - gap, w, bottom=dsh + gap, color=OFF, zorder=2)
        # interval on the offense/defense dividing line
        lo, hi = 1 - r["off_share_hi"], 1 - r["off_share_lo"]
        ax.errorbar(i + w * 0.36, dsh, yerr=[[dsh - lo], [hi - dsh]], fmt="none", ecolor=INK, elinewidth=1.4, capsize=6, zorder=4)
        if dsh <= 0.09 and r["sd_def"] > 0:            # small defense slice: label it just above the bar's base
            ax.text(i + w / 2 + 0.02, dsh, f"Def {dsh:.0%}\nSD {r['sd_def']:.1f}", ha="left", va="center",
                    color=INK2, fontsize=7.5, zorder=5)
        if dsh > 0.09:
            ax.text(i, dsh / 2, f"Defense\nSD {r['sd_def']:.1f}\n{dsh:.0%}", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold", zorder=5)
        if osh > 0.09:
            ax.text(i, dsh + osh / 2, f"Offense\nSD {r['sd_off']:.1f}\n{osh:.0%}", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold", zorder=5)
    ax.set_xticks(x, [f"{p}\nn = {n}\n" + (f"r = {c:+.2f}" if p != "DH" else "r = n/a") for p, n, c in
                      zip(summary["pos"], summary["player_seasons"], summary["corr"])],
                  fontsize=10, color=INK)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"], color=INK2, fontsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    seasons = sorted(set(summary.attrs.get("seasons", [])))
    span = f"{seasons[0]}-{seasons[-1]}" if seasons else ""
    fig.text(0.06, 0.955, f"What separates starters at each position: offense or defense? ({span})",
             fontsize=15, fontweight="bold", color=INK)
    fig.text(0.06, 0.885, "Share of the spread in value between starters (81+ starts at the position). SD = standard "
             "deviation in runs per 150 games.\nBlack bars: 95% interval on the split. n = player-seasons.",
             fontsize=10, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=OFF), plt.Rectangle((0, 0), 1, 1, color=DEF)]
    ax.legend(handles, ["Offense (batting + baserunning)", "Defense (IQ fielding / catching, no ABS)"],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=10)
    fig.text(0.06, 0.02, "Share = each side's variance / (offense variance + defense variance). r = correlation between "
             "a starter's offense and defense. Each season is centred on its own position average before pooling.",
             fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.15)
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def main() -> None:
    years = config.seasons()
    if not years:
        raise SystemExit("No Statcast seasons found in data/raw -- is the data/raw link to IQ_Savant there?")
    print(f"Seasons: {years}")
    pas = {y: cached_pa(y) for y in years}
    pf = park_factors(pd.concat([game_runs(p) for p in pas.values()]))
    seasons, lws = [], []
    for y in years:
        print(f"[{y}] offense + defense...", flush=True)
        d, lw = build_season(y, pf)
        seasons.append(d)
        lws.append(lw)
    pool = pd.concat(seasons, ignore_index=True)
    pool = pool[pool["off_rate"].notna() & pool["def_rate"].notna()].copy()
    grp = pool.groupby(["season", "pos"])
    pool["off_c"] = pool["off_rate"] - grp["off_rate"].transform("mean")
    pool["def_c"] = pool["def_rate"] - grp["def_rate"].transform("mean")
    DERIVED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pool.assign(pos_name=pool["pos"].map(POS)).to_csv(REPORTS / "value_mix_players.csv", index=False)
    summary = summarize(pool)
    summary.attrs["seasons"] = years
    summary.to_csv(REPORTS / "value_mix_summary.csv", index=False)
    pd.Series(pf).sort_values().rename("park_factor").to_csv(REPORTS / "park_factors.csv")
    pd.DataFrame(lws).to_csv(REPORTS / "linear_weights.csv", index=False)
    chart(summary, REPORTS / "value_mix.png")
    show = summary.copy()
    for c in ["off_share", "def_share", "off_share_lo", "off_share_hi"]:
        show[c] = (show[c] * 100).round(0)
    for c in ["sd_off", "sd_def", "sd_total", "cov", "corr"]:
        show[c] = show[c].round(2)
    print(show[["pos", "player_seasons", "players", "sd_off", "sd_def", "corr", "off_share", "off_share_lo",
                "off_share_hi"]].to_string(index=False))
    print(f"\nSaved reports/value_mix.png, value_mix_summary.csv, value_mix_players.csv, park_factors.csv, linear_weights.csv")


if __name__ == "__main__":
    main()
