"""
roster_projection.py
Bye-week and injury-adjusted roster projections for the post-draft summary.

Season projections are totals; this module converts them to a per-game rate and
walks the season week by week, asking what the optimal startable lineup is worth
once byes (certain absences) and per-position injury risk (probabilistic absences)
are taken into account. Bench players slide up to fill holes, so the gap between
the ideal lineup and the adjusted number is what the bench is actually worth.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


GAMES_PER_SEASON = 17
REGULAR_SEASON_WEEKS = 18       # Week 18 exists, but every team has one bye in 1-18

# Baseline probability that a player misses any given game, by position. Rough
# league-wide games-missed rates - RBs take the most contact, K/DEF the least.
POSITION_MISS_RATE = {
    'QB': 0.07, 'RB': 0.14, 'WR': 0.11, 'TE': 0.12, 'K': 0.02, 'DEF': 0.00
}
DEFAULT_MISS_RATE = 0.10

# Positions eligible for the FLEX slot
FLEX_ELIGIBLE = ('RB', 'WR', 'TE')


@dataclass
class TeamProjection:
    team_id: int
    ceiling: float              # ideal lineup, nobody ever misses a game
    no_bench_floor: float       # that same fixed lineup, holes score zero
    team_projection: float      # full roster, bench fills in - the headline number
    bench_insurance: float      # team_projection - no_bench_floor
    absence_cost: float         # ceiling - team_projection
    lineup: pd.DataFrame        # slot, name, position, team, bye_week, projected_points
    weekly: pd.DataFrame        # week, expected_points, on_bye, expected_holes
    per_player: pd.DataFrame    # name, position, ..., exp_games_started, exp_points


def _miss_rate(position: str) -> float:
    return POSITION_MISS_RATE.get(position, DEFAULT_MISS_RATE)


def availability(position: str, bye_week: Optional[float], week: int) -> float:
    """Probability a player is available in a given week. A bye is a certain absence."""
    if bye_week is not None and pd.notna(bye_week) and int(bye_week) == week:
        return 0.0
    return 1.0 - _miss_rate(position)


def greedy_fill(players: List[Tuple[float, str]], roster_slots: Dict[str, int]) -> float:
    """
    Fills the lineup from the highest scorer down, assigning each player to their own
    position slot if one is open, else FLEX, else the bench. This greedy order is
    exactly optimal for a slot structure whose FLEX is a superset of the dedicated
    positions - verified against exhaustive enumeration in this module's __main__.

    `players` is a list of (points, position); returns the total lineup points.
    """
    used = {pos: 0 for pos in roster_slots}
    flex_used = 0
    flex_slots = roster_slots.get('FLEX', 0)
    total = 0.0

    for points, position in sorted(players, key=lambda p: -p[0]):
        if used.get(position, 0) < roster_slots.get(position, 0):
            used[position] = used.get(position, 0) + 1
            total += points
        elif position in FLEX_ELIGIBLE and flex_used < flex_slots:
            flex_used += 1
            total += points

    return total


def ideal_lineup(roster_df: pd.DataFrame, roster_slots: Dict[str, int]) -> pd.DataFrame:
    """Returns the optimal starting lineup by season points, assuming full availability."""
    if roster_df.empty:
        return pd.DataFrame(columns=['slot', 'name', 'position', 'team', 'bye_week', 'projected_points'])

    used = {pos: 0 for pos in roster_slots}
    flex_used = 0
    flex_slots = roster_slots.get('FLEX', 0)
    rows = []

    for _, player in roster_df.sort_values('projected_points', ascending=False).iterrows():
        position = player['position']
        slot = None

        if used.get(position, 0) < roster_slots.get(position, 0):
            used[position] = used.get(position, 0) + 1
            slot = f"{position}{used[position]}" if roster_slots.get(position, 0) > 1 else position
        elif position in FLEX_ELIGIBLE and flex_used < flex_slots:
            flex_used += 1
            slot = "FLEX"

        if slot:
            rows.append({
                'slot': slot,
                'name': player.get('name', 'Unknown'),
                'position': position,
                'team': player.get('team', 'FA'),
                'bye_week': player.get('bye_week'),
                'projected_points': player['projected_points'],
            })

    return pd.DataFrame(rows)


def _expected_single_slot(candidates: List[Tuple[float, float]]) -> Tuple[float, Dict[int, float]]:
    """
    Expected points from an independent single slot (QB, K, DEF): the best available
    candidate starts. `candidates` is (points, availability) sorted by points desc.

    Returns (expected_points, {candidate_index: probability_of_starting}).
    """
    expected = 0.0
    start_probs = {}
    prob_all_above_out = 1.0

    for idx, (points, avail) in enumerate(candidates):
        p_starts = avail * prob_all_above_out
        expected += points * p_starts
        start_probs[idx] = p_starts
        prob_all_above_out *= (1.0 - avail)

    return expected, start_probs


def _expected_flex_group(
    candidates: List[Tuple[float, float, str]],
    roster_slots: Dict[str, int]
) -> Tuple[float, Dict[int, float]]:
    """
    Expected points from the coupled RB/WR/TE/FLEX group, computed exactly.

    Walks candidates in descending points order carrying a probability distribution
    over the lineup fill state (rb_used, wr_used, te_used, flex_used). Because the
    greedy descending assignment is optimal, the state fully determines where the
    next player lands, so this is an exact expectation rather than an approximation.

    `candidates` is (points, availability, position) sorted by points desc.
    Returns (expected_points, {candidate_index: probability_of_starting}).
    """
    rb_max = roster_slots.get('RB', 0)
    wr_max = roster_slots.get('WR', 0)
    te_max = roster_slots.get('TE', 0)
    flex_max = roster_slots.get('FLEX', 0)
    limits = {'RB': rb_max, 'WR': wr_max, 'TE': te_max}

    # state -> probability. State is (rb_used, wr_used, te_used, flex_used).
    dist = {(0, 0, 0, 0): 1.0}
    expected = 0.0
    start_probs = {}

    for idx, (points, avail, position) in enumerate(candidates):
        next_dist = {}
        p_starts = 0.0

        for state, prob in dist.items():
            # Player unavailable: state is untouched.
            if avail < 1.0:
                next_dist[state] = next_dist.get(state, 0.0) + prob * (1.0 - avail)
            if avail <= 0.0:
                continue

            rb_used, wr_used, te_used, flex_used = state
            used = {'RB': rb_used, 'WR': wr_used, 'TE': te_used}
            new_state = state

            if used.get(position, 0) < limits.get(position, 0):
                bumped = dict(used)
                bumped[position] += 1
                new_state = (bumped['RB'], bumped['WR'], bumped['TE'], flex_used)
                p_starts += prob * avail
            elif flex_used < flex_max:
                new_state = (rb_used, wr_used, te_used, flex_used + 1)
                p_starts += prob * avail
            # else: benched, state unchanged

            next_dist[new_state] = next_dist.get(new_state, 0.0) + prob * avail

        dist = next_dist
        expected += points * p_starts
        start_probs[idx] = p_starts

    return expected, start_probs


def expected_week_points(
    roster_df: pd.DataFrame,
    week: int,
    roster_slots: Dict[str, int]
) -> Tuple[float, pd.Series]:
    """
    Exact expected starting-lineup points for one week, given each player's
    availability that week. Returns (expected_points, per_player_start_probability).
    """
    start_probs = pd.Series(0.0, index=roster_df.index)
    if roster_df.empty:
        return 0.0, start_probs

    rates = roster_df.apply(
        lambda r: availability(r['position'], r.get('bye_week'), week), axis=1
    )
    per_game = roster_df['projected_points'] / GAMES_PER_SEASON
    total = 0.0

    # Coupled RB/WR/TE/FLEX group
    flex_mask = roster_df['position'].isin(FLEX_ELIGIBLE)
    flex_idx = roster_df[flex_mask].index
    if len(flex_idx) > 0:
        ordered = sorted(flex_idx, key=lambda i: -per_game[i])
        candidates = [(per_game[i], rates[i], roster_df.at[i, 'position']) for i in ordered]
        expected, probs = _expected_flex_group(candidates, roster_slots)
        total += expected
        for pos_in_list, i in enumerate(ordered):
            start_probs[i] = probs[pos_in_list]

    # Independent single-slot positions
    for position in ('QB', 'K', 'DEF'):
        if roster_slots.get(position, 0) <= 0:
            continue
        pos_idx = roster_df[roster_df['position'] == position].index
        if len(pos_idx) == 0:
            continue

        ordered = sorted(pos_idx, key=lambda i: -per_game[i])
        candidates = [(per_game[i], rates[i]) for i in ordered]
        expected, probs = _expected_single_slot(candidates)
        total += expected
        for pos_in_list, i in enumerate(ordered):
            start_probs[i] = probs[pos_in_list]

    return total, start_probs


def project_team(
    roster_df: pd.DataFrame,
    roster_slots: Dict[str, int],
    weeks: List[int],
    team_id: int = 0
) -> TeamProjection:
    """
    Computes the four headline projections for one roster across the given weeks.

    Requires columns: position, projected_points (season total). Optional: name,
    team, bye_week.
    """
    lineup = ideal_lineup(roster_df, roster_slots)
    n_weeks = len(weeks)

    if roster_df.empty or n_weeks == 0:
        empty_weekly = pd.DataFrame(columns=['week', 'expected_points', 'on_bye', 'expected_holes'])
        return TeamProjection(team_id, 0.0, 0.0, 0.0, 0.0, 0.0, lineup, empty_weekly, roster_df)

    roster_df = roster_df.reset_index(drop=True)
    per_game = roster_df['projected_points'] / GAMES_PER_SEASON
    total_slots = sum(roster_slots.get(s, 0) for s in ('QB', 'RB', 'WR', 'TE', 'FLEX', 'K', 'DEF'))

    # Ceiling: the ideal lineup, played every week with nobody ever missing.
    ceiling = (lineup['projected_points'].sum() / GAMES_PER_SEASON) * n_weeks if not lineup.empty else 0.0

    # No-bench floor: that same fixed lineup, but absences leave holes scoring zero.
    starter_keys = set(zip(lineup['name'], lineup['position'])) if not lineup.empty else set()
    is_starter = roster_df.apply(
        lambda r: (r.get('name', 'Unknown'), r['position']) in starter_keys, axis=1
    )

    team_total = 0.0
    floor_total = 0.0
    weekly_rows = []
    cumulative_starts = pd.Series(0.0, index=roster_df.index)
    cumulative_points = pd.Series(0.0, index=roster_df.index)

    for week in weeks:
        week_points, start_probs = expected_week_points(roster_df, week, roster_slots)
        team_total += week_points
        cumulative_starts += start_probs
        cumulative_points += start_probs * per_game

        floor_total += sum(
            per_game[i] * availability(roster_df.at[i, 'position'], roster_df.at[i, 'bye_week'] if 'bye_week' in roster_df.columns else None, week)
            for i in roster_df.index if is_starter[i]
        )

        on_bye = [
            roster_df.at[i, 'name'] if 'name' in roster_df.columns else str(i)
            for i in roster_df.index
            if 'bye_week' in roster_df.columns
            and pd.notna(roster_df.at[i, 'bye_week'])
            and int(roster_df.at[i, 'bye_week']) == week
        ]

        weekly_rows.append({
            'week': week,
            'expected_points': week_points,
            'on_bye': ", ".join(on_bye) if on_bye else "-",
            'expected_holes': max(0.0, total_slots - start_probs.sum()),
        })

    per_player = roster_df.copy()
    per_player['exp_games_started'] = cumulative_starts
    per_player['exp_points'] = cumulative_points
    per_player['role'] = np.where(is_starter, 'Starter', 'Insurance')
    per_player = per_player.sort_values('exp_points', ascending=False)

    return TeamProjection(
        team_id=team_id,
        ceiling=ceiling,
        no_bench_floor=floor_total,
        team_projection=team_total,
        bench_insurance=team_total - floor_total,
        absence_cost=ceiling - team_total,
        lineup=lineup,
        weekly=pd.DataFrame(weekly_rows),
        per_player=per_player,
    )


def build_roster_df(draft, engine, team_id: int) -> pd.DataFrame:
    """
    Assembles a projection-ready roster frame for one fantasy team by joining that
    team's picks against the player pool (which carries position, adp and bye_week),
    then attaching season projections via the shared valuation path.
    """
    picks = draft.get_team_roster(team_id)
    if not picks:
        return pd.DataFrame(columns=['player_id', 'name', 'position', 'team', 'bye_week', 'projected_points'])

    pick_ids = [p.player_id for p in picks]
    roster = draft.players_df[draft.players_df['player_id'].isin(pick_ids)].copy()

    if 'bye_week' not in roster.columns:
        roster['bye_week'] = np.nan

    roster = engine.add_projected_points(roster)
    return roster.reset_index(drop=True)


def project_all_teams(draft, engine, roster_slots: Dict[str, int], weeks: List[int]) -> List[TeamProjection]:
    """Projects every fantasy team in the league, sorted best to worst."""
    projections = [
        project_team(build_roster_df(draft, engine, team_id), roster_slots, weeks, team_id=team_id)
        for team_id in range(1, draft.teams + 1)
    ]
    return sorted(projections, key=lambda p: p.team_projection, reverse=True)


if __name__ == "__main__":
    import itertools

    print("--- Running Roster Projection Verification ---")
    SLOTS = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'K': 1, 'DEF': 1, 'BENCH': 6}

    # 1. Greedy lineup fill must match exhaustive enumeration.
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 9))
        players = [(round(float(rng.uniform(0, 100)), 2), str(rng.choice(['QB', 'RB', 'WR', 'TE'])))
                   for _ in range(n)]

        best = 0.0
        for k in range(min(n, 7) + 1):
            for combo in itertools.combinations(range(n), k):
                for perm in itertools.permutations(combo):
                    used = {p: 0 for p in SLOTS}
                    flex, total, ok = 0, 0.0, True
                    for i in perm:
                        pts, pos = players[i]
                        if used.get(pos, 0) < SLOTS.get(pos, 0):
                            used[pos] += 1; total += pts
                        elif pos in FLEX_ELIGIBLE and flex < SLOTS['FLEX']:
                            flex += 1; total += pts
                        else:
                            ok = False; break
                    if ok:
                        best = max(best, total)
        assert abs(greedy_fill(players, SLOTS) - best) < 1e-9, f"Greedy not optimal: {players}"
    print("[ok] Greedy lineup fill is exactly optimal (300 random rosters)")

    # 2. The DP must equal brute-force enumeration over all availability outcomes.
    roster = pd.DataFrame([
        {'name': 'RB A', 'position': 'RB', 'team': 'KC',  'bye_week': 5,  'projected_points': 280.0},
        {'name': 'RB B', 'position': 'RB', 'team': 'BUF', 'bye_week': 7,  'projected_points': 210.0},
        {'name': 'RB C', 'position': 'RB', 'team': 'DAL', 'bye_week': 14, 'projected_points': 150.0},
        {'name': 'WR A', 'position': 'WR', 'team': 'CIN', 'bye_week': 6,  'projected_points': 260.0},
        {'name': 'WR B', 'position': 'WR', 'team': 'SF',  'bye_week': 8,  'projected_points': 195.0},
        {'name': 'TE A', 'position': 'TE', 'team': 'DET', 'bye_week': 6,  'projected_points': 170.0},
        {'name': 'TE B', 'position': 'TE', 'team': 'PHI', 'bye_week': 10, 'projected_points': 95.0},
        {'name': 'QB A', 'position': 'QB', 'team': 'BAL', 'bye_week': 13, 'projected_points': 330.0},
        {'name': 'QB B', 'position': 'QB', 'team': 'GB',  'bye_week': 11, 'projected_points': 240.0},
    ])

    for week in (1, 5, 6, 13):
        dp_value, _ = expected_week_points(roster, week, SLOTS)

        brute = 0.0
        rates = [availability(r['position'], r['bye_week'], week) for _, r in roster.iterrows()]
        per_game = [r['projected_points'] / GAMES_PER_SEASON for _, r in roster.iterrows()]
        for outcome in itertools.product([0, 1], repeat=len(roster)):
            prob = 1.0
            for i, up in enumerate(outcome):
                prob *= rates[i] if up else (1.0 - rates[i])
            if prob == 0.0:
                continue
            active = [(per_game[i], roster.at[i, 'position']) for i, up in enumerate(outcome) if up]
            brute += prob * greedy_fill(active, SLOTS)

        assert abs(dp_value - brute) < 1e-9, f"Week {week}: DP {dp_value} != brute {brute}"
    print("[ok] DP matches exhaustive enumeration over all 2^9 availability outcomes")

    # 3. Perfect availability collapses the team projection onto the ceiling.
    healthy = roster.copy()
    healthy['bye_week'] = np.nan
    saved_rates = dict(POSITION_MISS_RATE)
    POSITION_MISS_RATE.update({k: 0.0 for k in POSITION_MISS_RATE})
    proj = project_team(healthy, SLOTS, list(range(1, 18)))
    assert abs(proj.team_projection - proj.ceiling) < 1e-9, "No absences should equal ceiling"
    assert abs(proj.bench_insurance) < 1e-9, "No absences means the bench adds nothing"
    POSITION_MISS_RATE.update(saved_rates)
    print("[ok] Zero absence risk collapses team projection onto ceiling")

    # 4. Ordering, and the bench genuinely covering a bye.
    weeks = list(range(1, 18))
    full = project_team(roster, SLOTS, weeks)
    assert full.no_bench_floor < full.team_projection < full.ceiling, (
        f"Expected floor < team < ceiling, got {full.no_bench_floor:.1f} "
        f"< {full.team_projection:.1f} < {full.ceiling:.1f}"
    )
    assert full.bench_insurance > 0, "A real bench must add value"

    starters_only = roster[roster['name'].isin(['RB A', 'RB B', 'WR A', 'WR B', 'TE A', 'QB A'])]
    thin = project_team(starters_only, SLOTS, weeks)
    handcuff = pd.concat([starters_only, pd.DataFrame([
        {'name': 'RB D', 'position': 'RB', 'team': 'NYJ', 'bye_week': 13, 'projected_points': 200.0}
    ])], ignore_index=True)
    deep = project_team(handcuff, SLOTS, weeks)
    assert deep.team_projection > thin.team_projection, "Adding a strong backup must raise the projection"
    print("[ok] floor < team < ceiling, and bench depth strictly increases the projection")

    print(f"\nSample roster over {len(weeks)} weeks:")
    print(f"  Ceiling          {full.ceiling:8.1f}")
    print(f"  Team Projection  {full.team_projection:8.1f}")
    print(f"  No-Bench Floor   {full.no_bench_floor:8.1f}")
    print(f"  Bench Insurance  {full.bench_insurance:8.1f}")
    print("\nAll assertions passed successfully!")
