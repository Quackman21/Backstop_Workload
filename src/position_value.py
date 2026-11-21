"""Run value by position: the average QUALIFIED starter, projected to 162 games, split into offense and defense.
FanGraphs only (via the browser-saved fg_<year>.json files) -- no IQ models.

    python -m src.position_value

Qualified   81+ starts at one position in the season (starts from Statcast, pulled with pybaseball);
            players qualifying nowhere are left out.
Offense     FanGraphs Batting + Base Running (runs above average)
Defense     FanGraphs Fielding (Statcast-based; catchers include framing) + Positional adjustment
            = FanGraphs "Def". Fielding and positional are also shown separately.
Per 162     pooled over 2021 -> latest: sum of runs / sum of games played x 162, so everyday players count by
            how much they actually played. League adjustment and replacement level are left out (same for all).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, fangraphs
from src.config import POS, REPORTS
from src.statcast import cached_starts
from src.value_mix import qualified

PARTS = ["batting", "baserunning", "fielding", "positional"]


def table() -> tuple[pd.DataFrame, pd.DataFrame]:
    fg = fangraphs.load().rename(columns={"def_runs": "fielding_all", "pos_runs": "positional"})
    fg["fielding"] = fg["fielding_all"]                    # everything defensive except positional (incl. framing)
    years = [y for y in config.seasons() if y in set(fg["season"])]
    rows = []
    for y in years:
        q = qualified(cached_starts(y), y)
        rows.append(q.merge(fg[fg["season"] == y].drop(columns="season"), on="player", how="inner"))
    p = pd.concat(rows, ignore_index=True)
    p["offense"] = p["batting"] + p["baserunning"]
    p["defense"] = p["fielding"] + p["positional"]
    out = []
    for pos, name in POS.items():
        d = p[p["pos"] == pos]
        if d.empty:
            continue
        g = d["G"].sum()
        r = {"pos": name, "player_seasons": len(d), "players": d["player"].nunique(), "avg_games": d["G"].mean()}
        for c in PARTS + ["offense", "defense"]:
            r[c] = d[c].sum() / g * 162
        r["total"] = r["offense"] + r["defense"]
        r["war_162"] = d["WAR"].sum() / g * 162
        out.append(r)
    t = pd.DataFrame(out)
    t.attrs["seasons"] = years
    return t, p


def chart(t: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OFF, DEF = "#2a78d6", "#eb6834"
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    w = 0.6
    lim = max(10, np.abs(t[["offense", "defense"]]).to_numpy().max() * 1.0,
              (t[["offense", "defense"]].clip(lower=0).sum(axis=1)).max(),
              (-t[["offense", "defense"]].clip(upper=0).sum(axis=1)).max()) * 1.25
    for i, r in t.reset_index(drop=True).iterrows():
        up, down = 0.0, 0.0
        for val, col, lbl in ((r["defense"], DEF, "Def"), (r["offense"], OFF, "Off")):
            base = up if val >= 0 else down
            ax.bar(i, val, w, bottom=base, color=col, zorder=2, edgecolor=SURF, linewidth=1.5)
            if abs(val) >= lim * 0.08:
                ax.text(i, base + val / 2, f"{lbl}\n{val:+.1f}", ha="center", va="center", color="white",
                        fontsize=8.5, fontweight="bold", zorder=5)
            if val >= 0:
                up += val
            else:
                down += val
        ax.plot(i, r["total"], "o", ms=9, color=INK, zorder=6, markeredgecolor=SURF, markeredgewidth=2)
        ax.text(i + w / 2 + 0.04, r["total"], f"{r['total']:+.1f}", ha="left", va="center", color=INK,
                fontsize=9.5, fontweight="bold", zorder=6)
    ax.axhline(0, color=INK2, lw=1, zorder=3)
    ax.set_xticks(range(len(t)), [f"{p}\nn = {n}\n{w:.1f} WAR" for p, n, w in
                                  zip(t["pos"], t["player_seasons"], t["war_162"])], fontsize=10, color=INK)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("Runs above average per 162 games", color=INK2, fontsize=10)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.tick_params(length=0, colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ys = t.attrs.get("seasons", [])
    span = f"{ys[0]}-{ys[-1]}" if ys else ""
    fig.text(0.07, 0.955, f"Run value of the average qualified starter, per 162 games ({span})",
             fontsize=15, fontweight="bold", color=INK)
    fig.text(0.07, 0.905, "FanGraphs. Offense = batting + baserunning. Defense = fielding (incl. catcher framing) "
             "+ positional adjustment.\nQualified = 81+ starts at the position. Dot = total runs above average; "
             "WAR per 162 under each position.", fontsize=10, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=OFF), plt.Rectangle((0, 0), 1, 1, color=DEF),
               plt.Line2D([], [], marker="o", color=INK, ls="", ms=8)]
    ax.legend(handles, ["Offense", "Defense", "Total"], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
              frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.14)
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    t, p = table()
    t.to_csv(REPORTS / "position_value_162.csv", index=False)
    p.assign(pos_name=p["pos"].map(POS)).to_csv(REPORTS / "position_value_players.csv", index=False)
    chart(t, REPORTS / "position_value_162.png")
    show = t[["pos", "player_seasons", "avg_games", "batting", "baserunning", "offense", "fielding", "positional",
              "defense", "total", "war_162"]].round(1)
    print(f"Average qualified starter per 162 games, FanGraphs, {t.attrs['seasons'][0]}-{t.attrs['seasons'][-1]} "
          "(runs above average)")
    print(show.to_string(index=False))
    print("\nSaved reports/position_value_162.png, position_value_162.csv, position_value_players.csv")


if __name__ == "__main__":
    main()
