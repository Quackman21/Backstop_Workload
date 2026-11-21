"""Who STARTED at each position in every game, straight from Statcast.

Fielders   fielder_2..fielder_9 on the team's first defensive pitch of the game.
DH         the one batter among his team's first nine batters who wasn't in the field on that pitch and isn't the
           starting pitcher (2021 NL parks had no DH -- the pitcher batted -- so no DH start is counted there).
"""
from __future__ import annotations

import pandas as pd

FIELDERS = [f"fielder_{i}" for i in range(2, 10)]


def starts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pk, half), g in df.groupby(["game_pk", "inning_topbot"], sort=False):
        half = str(half)
        side_field = "home" if half == "Top" else "away"           # top of an inning: home team in the field
        first_def = g[g["inning"] == g["inning"].min()].iloc[0]
        team_field = first_def["home_team"] if side_field == "home" else first_def["away_team"]
        fielders = {}
        for i, c in enumerate(FIELDERS, start=2):
            v = first_def[c]
            if pd.notna(v):
                fielders[int(v)] = i
                rows.append((pk, team_field, int(v), i))
        sp = first_def["pitcher"]
        # the batting team in this half is the OTHER side; its lineup is checked against ITS fielders below
        g_bat = g.drop_duplicates("batter").sort_values("at_bat_number").head(9)
        rows.append((pk, "__BAT__" + half, g_bat["batter"].astype("int64").tolist(), None))
        rows.append((pk, "__SP__" + half, int(sp) if pd.notna(sp) else None, None))
        rows.append((pk, "__FLD__" + half, list(fielders), None))
    # second pass: DH = batter in the first nine who isn't one of his own team's fielders / starting pitcher
    out, meta = [], {}
    for pk, team, player, pos in rows:
        if isinstance(team, str) and team.startswith("__"):
            meta[(pk, team)] = player
        else:
            out.append({"game_pk": pk, "team": team, "player": player, "pos": pos})
    games = df.drop_duplicates("game_pk").set_index("game_pk")[["home_team", "away_team"]]
    for pk, (home, away) in games.iterrows():
        for bat_half, own_field_half, team in (("Top", "Bot", away), ("Bot", "Top", home)):
            nine = meta.get((pk, "__BAT__" + bat_half)) or []
            own_fielders = set(meta.get((pk, "__FLD__" + own_field_half)) or [])
            own_sp = meta.get((pk, "__SP__" + own_field_half))
            if len(nine) < 9 or not own_fielders:
                continue
            dh = [b for b in nine if b not in own_fielders and b != own_sp]
            if len(dh) == 1:
                out.append({"game_pk": pk, "team": team, "player": dh[0], "pos": 10})
    return pd.DataFrame(out, columns=["game_pk", "team", "player", "pos"])
