"""Every runner movement from MLB's free play-by-play (one call per game) -- for baserunning value.

    python -m src.fetch_runners                 # every study season, skips seasons already saved
    python -m src.fetch_runners --season 2026   # one season (re-downloads it)

Saved to data/raw/runners_<year>.csv.gz, one row per runner movement:
  game_pk, at_bat_number (Statcast numbering), play_index (which event in the PA it happened on),
  bip_index (the PA's last pitch -- movements on it are the result of the PA itself),
  runner, start, end ('1B' / '2B' / '3B' / 'score', blank = out), is_out, event_type, reason
Six games at a time; a full season takes roughly 10-20 minutes.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.config import FIRST_SEASON, RAW

WORKERS = 6


def parse(pbp: dict, game_pk: int) -> list[dict]:
    rows = []
    for play in pbp.get("allPlays", []):
        ab = int(play.get("about", {}).get("atBatIndex", -1)) + 1          # Statcast at_bat_number is 1-based
        pitches = [e.get("index") for e in play.get("playEvents", []) if e.get("isPitch")]
        bip = max(pitches) if pitches else None
        for rn in play.get("runners", []):
            det, mv = rn.get("details", {}), rn.get("movement", {})
            rows.append({"game_pk": game_pk, "at_bat_number": ab, "play_index": det.get("playIndex"),
                         "bip_index": bip, "runner": (det.get("runner") or {}).get("id"),
                         "start": mv.get("start"), "end": mv.get("end"), "is_out": bool(mv.get("isOut")),
                         "event_type": det.get("eventType"), "reason": det.get("movementReason")})
    return rows


def season(year: int) -> pd.DataFrame:
    import requests
    from core import mlb_api
    games = sorted({g["gamePk"] for g in mlb_api.get_season_games(year)
                    if g.get("gameType") == "R" and g.get("status", {}).get("abstractGameState") == "Final"})
    print(f"[{year}] play-by-play for {len(games):,} games...", flush=True)
    rows, failed, t0 = [], [], time.time()
    sess = requests.Session()

    def one(pk):
        for attempt in range(4):
            try:
                r = sess.get(f"https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay", timeout=30)
                r.raise_for_status()
                return parse(r.json(), pk)
            except Exception:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(pk)

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(one, pk): pk for pk in games}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows += f.result()
            except Exception:
                failed.append(futs[f])
            if i % 250 == 0:
                print(f"    {i:,}/{len(games):,} games ({time.time() - t0:.0f}s)", flush=True)
    if failed:
        print(f"    {len(failed)} games failed after retries -- run this season again to fill them in")
    df = pd.DataFrame(rows)
    RAW.mkdir(parents=True, exist_ok=True)
    if not failed:
        df.to_csv(RAW / f"runners_{year}.csv.gz", index=False)
        print(f"[{year}] saved {len(df):,} runner movements in {time.time() - t0:.0f}s")
    return df


def main(argv: list[str]) -> None:
    from src.config import seasons
    if "--season" in argv:
        season(int(argv[argv.index("--season") + 1]))
        return
    for y in seasons() or range(FIRST_SEASON, 2027):
        if (RAW / f"runners_{y}.csv.gz").exists():
            print(f"[{y}] already saved")
            continue
        season(y)


if __name__ == "__main__":
    main(sys.argv[1:])
