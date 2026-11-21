"""Which defense numbers measure skill better -- IQ models, Baseball-Reference (DRS) or FanGraphs (Statcast + framing)?

    python -m src.compare_defense

Per position, for qualified starters (81+ starts):
  sd_*     spread of season defense, runs per 150 games
  r_iq_*   how much the anchor agrees with IQ on the same player-season
  yoy_*    year-to-year correlation for the same player at the same position (next season).
                     Real skill repeats; noise doesn't. The more repeatable metric is capturing more skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import POS, REPORTS, SCALE_GAMES
from src.value_mix import qualified
from src.statcast import cached_starts
from src.war import load_all


def main() -> None:
    from src import fangraphs
    from src.config import seasons
    sources = ["bref"] + (["fg"] if set(seasons()) <= set(fangraphs.files()) else [])
    per = {a: load_all(a) for a in sources}
    rows = []
    for y in per["bref"]:
        base = per["bref"][y].groupby("player").agg(outs=("outs", "sum"), iq=("def_iq", "sum"))
        for a in sources:
            base[a] = per[a][y].groupby("player")["def"].sum()
        q = qualified(cached_starts(y), y).join(base, on="player")
        q = q[(q["pos"] != 10) & (q["outs"] > 0)]
        for c in ["iq"] + sources:
            q[c] = q[c] / (q["outs"] / 3) * 9 * SCALE_GAMES
            q[c] -= q.groupby("pos")[c].transform("mean")
        rows.append(q)
    d = pd.concat(rows, ignore_index=True)
    cols = ["iq"] + sources
    nxt = d[["season", "player", "pos"] + cols].copy()
    nxt["season"] -= 1
    pair = d.merge(nxt, on=["season", "player", "pos"], suffixes=("", "_next"))
    out = []
    for pos, name in POS.items():
        a, p = d[d["pos"] == pos], pair[pair["pos"] == pos]
        if len(a) < 5:
            continue
        r = {"pos": name, "n": len(a), "yoy_pairs": len(p)}
        for c in cols:
            r[f"sd_{c}"] = a[c].std()
        for c in sources:
            r[f"r_iq_{c}"] = a["iq"].corr(a[c])
        for c in cols:
            r[f"yoy_{c}"] = p[c].corr(p[f"{c}_next"])
        out.append(r)
    t = pd.DataFrame(out)
    t.to_csv(REPORTS / "defense_compare.csv", index=False)
    print(t.round(2).to_string(index=False))
    print("\nyoy = same player, same position, next season. Higher = more real skill, less noise.")


if __name__ == "__main__":
    main()
