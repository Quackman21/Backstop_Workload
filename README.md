# Backstop Workload

How catcher and position player workload changes performance and value over a season.

- `core/`, `models/`, `pipeline/`: ported from IQ_Savant (MLB API, Savant pulls, weather, framing / blocking / throwing, IQ xwOBA, IQ Swing Decisions)
- `src/build_workload.py`: game-by-game workload log (`python -m src.build_workload`)
- `data/league/`: per-game fielding + framing files, 2015+
- `data/raw/`: pitch-level Statcast (not in git)

Setup:

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python -m src.build_workload
