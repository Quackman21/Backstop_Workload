"""Which defense numbers measure skill better -- IQ models or Baseball-Reference (DRS)?

    python -m src.compare_defense

Per position, for qualified starters (81+ starts):
  sd_iq / sd_bref    spread of season defense, runs per 150 games
  corr               how much the two agree on the same player-season
  yoy_iq / yoy_bref  year-to-year correlation for the same player at the same position (next season).
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
    games = load_all()
    rows = []
    for y, g in games.items():
        s = g.groupby("player").agg(outs=("outs", "sum"), iq=("def_iq", "sum"), bref=("def", "sum"))
        q = qualified(cached_starts(y), y).join(s, on="player")
        q = q[(q["pos"] != 10) & (q["outs"] > 0)]
        for c in ("iq", "bref"):
            q[c] = q[c] / (q["outs"] / 3) * 9 * SCALE_GAMES
            q[c] -= q.groupby("pos")[c].transform("mean")
        rows.append(q)
    d = pd.concat(rows, ignore_index=True)
    nxt = d[["season", "player", "pos", "iq", "bref"]].copy()
    nxt["season"] -= 1
    pair = d.merge(nxt, on=["season", "player", "pos"], suffixes=("", "_next"))
    out = []
    for pos, name in POS.items():
        a, p = d[d["pos"] == pos], pair[pair["pos"] == pos]
        if len(a) < 5:
            continue
        out.append({"pos": name, "n": len(a), "sd_iq": a["iq"].std(), "sd_bref": a["bref"].std(),
                    "corr": a["iq"].corr(a["bref"]), "yoy_pairs": len(p),
                    "yoy_iq": p["iq"].corr(p["iq_next"]), "yoy_bref": p["bref"].corr(p["bref_next"])})
    t = pd.DataFrame(out)
    t.to_csv(REPORTS / "defense_compare.csv", index=False)
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
