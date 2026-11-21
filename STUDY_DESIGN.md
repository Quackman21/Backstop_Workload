# Backstop Workload — Study Design

**Seasons:** 2021 to current. Catcher stances changed, so earlier seasons are left out.

## Part 1: How each position gets its value

**Question.** Among starters at each position, how much of the difference between a good one and a bad one comes from offense and how much from defense?

### Who counts (qualified)
- A player qualifies at a position if he started **81 or more games there in the season**.
- If he reaches 81 at more than one position, he is assigned to the position where he started most.
- Starts come straight from Statcast:
  - Fielders are the players in the field on their team's first defensive pitch.
  - The DH is the batter in his team's first nine who wasn't in the field and isn't the starting pitcher. In 2021 NL parks the pitcher batted, so no DH start counts there.
- Groups are kept separate: C, 1B, 2B, 3B, SS, LF, CF, RF, DH.

### Offense (runs above average, batting + baserunning)
- **Batting.** For each plate appearance: the season's linear weight for the result, minus the league average per PA, park-adjusted using that game's park. This is equivalent to park-adjusted wRAA.
  - Linear weights come from the season's own **RE24**: runs on the play plus the run expectancy after it, minus the run expectancy before it, averaged by result (1B, 2B, 3B, HR, BB, HBP, out).
  - BB and HBP carry their real value each year.
  - IBB and sac bunts are left out, as in wOBA.
  - Park factor: runs per game in a team's home games divided by runs per game in its road games, pooled over 2021 onward.
- **Baserunning** comes from MLB play-by-play runner movements:
  - **wSB:** steals and caught stealing (pickoff-caught-stealing counts as CS), valued from RE24, minus the league's steal runs per time on first base.
  - **UBR:** extra bases on singles, doubles and outs in play. For each runner, the run value of where he ended up, minus the league-average result for the same play type, starting base, outs and field (LF / CF / RF / infield).
  - **wGDP:** double plays hit into compared with the league rate, on PAs with a runner on first and fewer than 2 outs.
- Results, not expected stats. Value means what actually happened.

### Defense (runs above average at the position, IQ_Savant models)
- **C:** framing runs + blocking runs + throwing runs. **ABS challenges are left out**, because the other seasons don't have them.
- **1B / 2B / 3B / SS:** fielding runs + double-play runs.
- **LF / CF / RF:** range runs + arm runs.
- **DH:** 0.

### Putting offense and defense on the same scale
- Both are scaled to **150 full games**:
  - Defense per 1,350 innings.
  - Offense per 150 games' worth of PAs, using the league PAs per lineup spot per game.
- Each season is **centered on its own position average** before pooling, so swings in league scoring between seasons don't count as differences between players.

### Measure
For each position, over the pooled player-seasons:
- **SD of offense and SD of defense,** in runs per 150 games.
- **Offense share = offense variance / (offense variance + defense variance).** Defense share is the remainder.
- **Correlation (r)** between a starter's offense and defense, reported separately. Splitting their overlap evenly between the two shares can push a share past 100% when offense and defense pull in opposite directions.
- **95% interval on the split,** from resampling **players** 2,000 times. A starter's seasons move together, so they aren't treated as independent.

### Output
`reports/value_mix.png` has one 100% bar per position, defense and offense stacked and labeled with each side's SD and share, with the error bar on the dividing line. Tables are in `reports/value_mix_summary.csv` and `value_mix_players.csv`, with checks in `linear_weights.csv` and `park_factors.csv`.

### Known limits
- Catchers in an even platoon (both under 81 starts) are left out, so the catcher group leans toward true starters.
- Defense comes from IQ models. Savant and FanGraphs are for checking only.
- Observed seasons include luck as well as skill on both sides. That's right for "value produced," but it isn't a talent split.
- Baserunning outs on plays the UBR contexts don't cover (fielder's choices, force plays) aren't credited to the runner.

### How to run (Mac, in the repo with the .venv active)
    python -m src.fetch_runners      # once: play-by-play runner moves, ~10-20 min per season
    python -m src.value_mix          # everything else, then the chart

## Part 2: Workload (not designed yet)
