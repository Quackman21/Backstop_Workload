"""Tiny fake season (Statcast pitches + play-by-play runner moves + fielding / framing files) for tests.

Not baseball-accurate -- just the same columns and shapes as the real files, with known structure:
each team has a fixed lineup (8 fielders + DH), one catcher who starts every game.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TEAMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
OUTCOMES = ["field_out", "strikeout", "single", "double", "home_run", "walk", "grounded_into_double_play"]
PROBS = [0.46, 0.22, 0.15, 0.05, 0.03, 0.08, 0.01]


def roster(t: int) -> dict:
    base = 100000 + t * 100
    return {"fielders": {p: base + p for p in range(2, 10)}, "dh": base + 10, "sp": base + 50}


def season(year: int, games_per_pair: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed + year)
    rows, runners = [], []
    pk = year * 10000
    for gi in range(games_per_pair):
        for hi, h in enumerate(TEAMS):
            for ai, a in enumerate(TEAMS):
                if h == a:
                    continue
                pk += 1
                ab = 0
                score = {"Top": 0, "Bot": 0}
                ptr = {"Top": 0, "Bot": 0}
                for inning in range(1, 10):
                    for half in ("Top", "Bot"):
                        bat_t, fld_t = (ai, hi) if half == "Top" else (hi, ai)
                        bat, fld = roster(bat_t), roster(fld_t)
                        lineup = [bat["fielders"][p] for p in range(2, 10)] + [bat["dh"]]
                        outs, bases = 0, {1: None, 2: None, 3: None}
                        while outs < 3:
                            ab += 1
                            batter = lineup[ptr[half] % 9]
                            ptr[half] += 1
                            ev = rng.choice(OUTCOMES, p=PROBS)
                            if ev == "grounded_into_double_play" and (bases[1] is None or outs == 2):
                                ev = "field_out"
                            pre = score[half]
                            row = {"game_pk": pk, "game_date": f"{year}-06-01", "game_type": "R", "at_bat_number": ab,
                                   "pitch_number": 1, "inning": inning, "inning_topbot": half, "batter": batter,
                                   "pitcher": fld["sp"], "events": ev, "home_team": h, "away_team": a,
                                   "bb_type": "ground_ball" if ev in ("grounded_into_double_play",) else
                                   ("fly_ball" if ev in ("home_run", "double") else
                                    ("line_drive" if ev == "single" else ("ground_ball" if rng.random() < .5 else "fly_ball"))
                                    if ev != "strikeout" and ev != "walk" else None),
                                   "hit_location": int(rng.integers(1, 10)) if ev not in ("strikeout", "walk") else None,
                                   "on_1b": bases[1], "on_2b": bases[2], "on_3b": bases[3], "outs_when_up": outs,
                                   "bat_score": pre, "woba_denom": 1,
                                   **{f"fielder_{p}": fld["fielders"][p] for p in range(2, 10)}}
                            mv = []                          # (runner, start, end, out)
                            # a steal attempt sometimes, before the result (same PA, earlier play index)
                            if bases[1] is not None and bases[2] is None and outs < 2 and rng.random() < 0.08:
                                r = bases[1]
                                ok = rng.random() < 0.75
                                runners.append({"game_pk": pk, "at_bat_number": ab, "play_index": 0, "bip_index": 1,
                                                "runner": r, "start": "1B", "end": "2B" if ok else None,
                                                "is_out": not ok,
                                                "event_type": "stolen_base_2b" if ok else "caught_stealing_2b",
                                                "reason": None})
                                bases[1] = None
                                if ok:
                                    bases[2] = r
                                else:
                                    outs += 1
                                row.update({"on_1b": bases[1], "on_2b": bases[2], "outs_when_up": outs})
                                if outs >= 3:
                                    row["events"] = "caught_stealing_2b"
                                    row["woba_denom"] = 0
                                    row["post_bat_score"] = score[half]
                                    rows.append(row)
                                    ab -= 0
                                    break
                            runs = 0
                            if ev in ("field_out", "strikeout"):
                                outs += 1
                                if ev == "field_out" and outs < 3 and bases[3] is not None and row["bb_type"] == "fly_ball" \
                                        and rng.random() < 0.6:
                                    mv.append((bases[3], "3B", "score", False)); runs += 1; bases[3] = None
                            elif ev == "grounded_into_double_play":
                                mv.append((bases[1], "1B", None, True)); bases[1] = None; outs += 2
                            elif ev == "walk":
                                if bases[1] is not None:
                                    if bases[2] is not None:
                                        if bases[3] is not None:
                                            mv.append((bases[3], "3B", "score", False)); runs += 1
                                        mv.append((bases[2], "2B", "3B", False)); bases[3] = bases[2]
                                    mv.append((bases[1], "1B", "2B", False)); bases[2] = bases[1]
                                bases[1] = batter
                            else:
                                adv = {"single": 1, "double": 2, "home_run": 4}[ev]
                                for b in (3, 2, 1):
                                    r = bases[b]
                                    if r is None:
                                        continue
                                    extra = 1 if (ev == "single" and b == 1 and rng.random() < (0.45 if r % 3 == 0 else 0.15)) else 0
                                    if ev == "single" and b == 1 and rng.random() < 0.03:
                                        mv.append((r, "1B", None, True)); bases[b] = None; outs += 1; continue
                                    nb = b + adv + extra
                                    name = {1: "1B", 2: "2B", 3: "3B"}
                                    if nb >= 4:
                                        mv.append((r, name[b], "score", False)); runs += 1
                                    else:
                                        mv.append((r, name[b], name[nb], False)); bases[nb] = r
                                    bases[b] = None
                                if adv == 4:
                                    runs += 1
                                else:
                                    bases[adv] = batter
                            score[half] += runs
                            row["post_bat_score"] = score[half]
                            rows.append(row)
                            for r, s, e, o in mv:
                                runners.append({"game_pk": pk, "at_bat_number": ab, "play_index": 1, "bip_index": 1,
                                                "runner": r, "start": s, "end": e, "is_out": o,
                                                "event_type": ev, "reason": None})
                            if outs >= 3:
                                break
                for r in rows:
                    if r["game_pk"] == pk:
                        r["post_home_score"], r["post_away_score"] = score["Bot"], score["Top"]
    sc = pd.DataFrame(rows)
    ru = pd.DataFrame(runners)
    # fielding + framing files (per player / game / position), with one strong-defense catcher (team 0)
    games = sc[["game_pk", "game_date", "home_team", "away_team"]].drop_duplicates()
    frows, crows = [], []
    for _, g in games.iterrows():
        for t in (g.home_team, g.away_team):
            ti = TEAMS.index(t)
            for p, pid in roster(ti)["fielders"].items():
                runs = rng.normal(0.08 if (ti == 0 and p in (2, 6)) else 0.0, 0.15)
                frows.append({"game_date": g.game_date, "game_pk": float(g.game_pk), "player": pid, "pos": p,
                              "outs": 27.0, "runs": runs if p != 2 else 0.0, "dp_runs": 0.0, "arm_runs": 0.0,
                              "c_runs": rng.normal(0, .03) if p == 2 else 0.0,
                              "t_runs": rng.normal(0, .03) if p == 2 else 0.0})
            crows.append({"game_date": g.game_date, "game_pk": float(g.game_pk), "catcher": roster(ti)["fielders"][2],
                          "region": "IN", "n": 60, "strikes": 20.0, "exp": 19.5,
                          "runs": rng.normal(0.15 if ti == 0 else 0.0, 0.2), "runs_sit": 0.0})
    return sc, ru, pd.DataFrame(frows), pd.DataFrame(crows)


def write(root, years=(2021, 2022)) -> None:
    from pathlib import Path
    root = Path(root)
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "league").mkdir(parents=True, exist_ok=True)
    for y in years:
        sc, ru, fl, ca = season(y)
        sc.to_csv(root / "data" / "raw" / f"statcast_{y}.csv.gz", index=False)
        ru.to_csv(root / "data" / "raw" / f"runners_{y}.csv.gz", index=False)
        fl.to_csv(root / "data" / "league" / f"fielding_{y}.csv.gz", index=False)
        ca.to_csv(root / "data" / "league" / f"catching_{y}.csv.gz", index=False)
    # a fake Baseball-Reference WAR file for every rostered hitter
    rng = np.random.default_rng(7)
    rows = []
    for y in years:
        for t in range(len(TEAMS)):
            r = roster(t)
            for pos, pid in list(r["fielders"].items()) + [(10, r["dh"])]:
                off, dfn = rng.normal(0, 15), (rng.normal(0, 8) if pos != 10 else 0.0)
                posr = {2: 9, 3: -9, 4: 2, 5: 2, 6: 6, 7: -5, 8: 2, 9: -5, 10: -12}[pos]
                raa = off + dfn + posr
                rows.append({"mlb_ID": pid, "name_common": f"P{pid}", "year_ID": y, "stint_ID": 1, "team_ID": TEAMS[t],
                             "pitcher": "N", "G": 84, "PA": 350, "Inn": 700, "runs_bat": off, "runs_br": 0, "runs_dp": 0,
                             "runs_field": dfn, "runs_catcher": 0, "runs_defense": dfn, "runs_position": posr,
                             "runs_replacement": 11, "runs_above_avg": raa, "runs_above_avg_off": off,
                             "runs_above_avg_def": dfn + posr, "WAA": raa / 10, "WAR": (raa + 11) / 10,
                             "WAR_off": 0, "WAR_def": 0})
    pd.DataFrame(rows).to_csv(root / "data" / "raw" / "bref_war_bat.csv.gz", index=False)
