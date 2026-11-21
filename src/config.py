"""Study-wide settings. Change a rule here and every step picks it up."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("BW_ROOT") or Path(__file__).resolve().parents[1])   # BW_ROOT: tests only
RAW = ROOT / "data" / "raw"            # pitch-level Statcast + play-by-play (linked to IQ_Savant, never pushed)
LEAGUE = ROOT / "data" / "league"      # per-game fielding / framing files from IQ_Savant
DERIVED = ROOT / "data" / "derived"    # tables this study builds (rebuildable, not pushed)
REPORTS = ROOT / "reports"             # charts + tables (pushed)

FIRST_SEASON = 2021                    # catcher stances changed; earlier seasons are left out of the study

# Qualified = started at least half the season's games at one position (2..9 = fielding positions, 10 = DH)
MIN_STARTS = 81
POS = {2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF", 10: "DH"}
INFIELD = {3, 4, 5, 6}
OUTFIELD = {7, 8, 9}

# Rates are scaled to 150 full games: defense per 1,350 innings, offense per 150 games of PA
SCALE_GAMES = 150

BOOTSTRAP = 2000                       # resamples of PLAYERS (not seasons) for the error bars
SEED = 21


def seasons() -> list[int]:
    """Every study season with a saved Statcast file."""
    have = sorted(int(p.name[9:13]) for p in RAW.glob("statcast_*.csv.gz"))
    return [y for y in have if y >= FIRST_SEASON]
