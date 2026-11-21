# Backstop Workload

How catcher and position player workload changes performance and value over a season.

- `core/`, `models/`, `pipeline/`: ported from IQ_Savant (MLB API, Savant pulls, weather, framing / blocking / throwing, IQ xwOBA, IQ Swing Decisions)
- `STUDY_DESIGN.md`: what the study measures and how (read this first)
- `src/war.py`: game-by-game WAR for every position player, season totals anchored to Baseball-Reference
- `src/value_mix.py`: Part 1, how each position gets its value (offense vs defense), chart in `reports/`
- `src/build_workload.py`: game-by-game workload log (`python -m src.build_workload`)
- `data/league/`: per-game fielding + framing files, 2015+
- `data/raw/`: pitch-level Statcast (not in git)

Setup:

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python -m src.build_workload
