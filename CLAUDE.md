# Backstop Workload — context for Claude

Read this first, then `STUDY_DESIGN.md` (the full method). This file records **decisions already made with the
project owner (Ivan Quackenbush, GitHub Quackman21)** so they aren't re-litigated, plus where things stand.

## What the study is
How catcher (and position-player) workload changes performance and value over a season. Two parts:
1. **Value (in progress):** how each position — catchers above all — generates value, split into offense and defense.
2. **Workload (not designed yet):** game-by-game value vs. workload (consecutive days caught, rest, pitches received,
   day-after-night, heat, travel…). Design it *with* the owner before building — he wants to discuss first.

## Rules the owner set (follow these)
- **Data sources: FanGraphs + pybaseball only.** IQ_Savant's own models (framing, fielding, xwOBA, etc.) are a
  **reference only** — they must never drive a result. They may appear only as a clearly labelled comparison.
- **Seasons: 2021 → current.** Catcher stances changed; earlier seasons are excluded (`FIRST_SEASON` in `src/config.py`).
- **Qualified = 81+ starts at one position in a season** (starts from Statcast first pitch / first nine batters;
  DH detected from the lineup). Assigned to the position with the most starts. Applies to every position.
- **Groups kept separate:** C, 1B, 2B, 3B, SS, LF, CF, RF, DH.
- **No ABS challenges** in defense (only 2026 has them).
- **Pool 2021-2026** as an average; centre each season on its own position average when comparing spreads.
- **Value per 162 games** for the position comparison.
- Discuss before building anything new; the owner likes to agree the definition first.

## Decisions and why
- **FanGraphs (fWAR) is the anchor, not Baseball-Reference.** B-Ref's WAR leaves out catcher framing, so its catcher
  defense had zero year-to-year repeatability (0.00) vs FanGraphs 0.68. FanGraphs' Fielding for catchers already
  includes framing (`CFraming` is a subset of `Fielding`). B-Ref code (`src/bref.py`) is kept only for comparison.
- **FanGraphs blocks scripts** (pybaseball's `batting_stats` returns 0 rows; the JSON API returns 403). The season files
  were saved from a browser: `data/raw/fg_<year>.json` (link in `src/fangraphs.py`). If they need refreshing, save them
  in a browser again (one combined download is easiest — browsers block multiple automatic downloads).
- **Defense = RAR − Offense − Positional − League − Replacement** (= Fielding incl. framing). Offense = Batting + Base Running.
- **"How positions get value"** is shown two ways:
  - `src/position_value.py` — average qualified starter per 162 games, offense vs defense (defense = fielding + positional). **Owner's main view.**
  - `src/value_mix.py` — share of the *spread between starters* from offense vs defense (variance shares, correlation
    shown separately because splitting covariance evenly can exceed 100%), with SDs and player-bootstrap 95% intervals.

## Results so far (FanGraphs, 2021-2026, qualified starters)
Share of the spread between starters that comes from defense: **C 36%** (27-46%), LF 24%, 2B 20%, SS 15%, 3B 13%,
RF 13%, CF 12%, 1B 9%, DH 0%. Catcher defense SD 13.7 runs/150 g vs 5-8 elsewhere — framing drives it.
Catcher samples: 128 player-seasons, 60 catchers. `position_value` (per-162 offense/defense) has been built; the owner
is running it — check `reports/position_value_162.csv` once pushed.
Side note (reference only): IQ fielding agrees with FanGraphs at C (r 0.92), poorly at SS (0.35) and 2B (0.44).

## Open items
1. **Game-by-game defense without IQ models.** `src/war.py` builds game-level WAR summing exactly to fWAR, but spreads
   each season's defense across games using IQ per-game values — that conflicts with the "reference only" rule.
   Options to put to the owner: spread evenly by innings, or build per-game catcher measures from Statcast pitches
   (pybaseball). Game-level offense uses our RE24/linear weights from Statcast + MLB play-by-play for baserunning.
2. **Keep the FanGraphs files in git?** They can't be re-downloaded by script; proposed moving them to a committed
   `data/fangraphs/` folder. Not decided.
3. Catcher deep-dive (batting, baserunning, framing, blocking, throwing per 162 — which separates catchers most).
4. Part 2 workload design.

## How to run (owner's Mac: ~/Dev/Backstop_Workload, `.venv`, Python 3.10)
    python -m src.fetch_runners      # MLB play-by-play runner moves (once)
    python -m src.bref               # B-Ref WAR (comparison only)
    python -m src.fangraphs          # check fg_<year>.json files
    python -m src.war                # game-by-game WAR (fg anchor)
    python -m src.position_value     # value per 162 by position (opens the chart)
    python -m src.value_mix          # share of spread chart
    python -m src.compare_defense    # IQ vs B-Ref vs FanGraphs defense (reference)
    python -m pytest tests -q        # runs everything on a simulated season

`data/raw` is a symlink to `~/Dev/IQ_Savant/data/raw` (Statcast pitch files from IQ_Savant's pipeline). `data/raw` and
`data/derived` are not in git. Tests need no real data (`tests/sim.py`, `BW_ROOT` env var).

## Owner's machine quirks
- Old git: no `git init -b`; large pushes need `git config http.postBuffer 524288000`.
- Homebrew can't install `gh` quickly (macOS 13 builds from source) — use plain git.
- Give terminal commands as one pasteable block; he runs them and pastes the output back.
