import numpy as np
import pandas as pd
import scipy.stats as stats
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import roster_projection

@dataclass
class LeagueSettings:
    format: str = "ppr"             # 'ppr', 'half-ppr', 'standard'
    teams: int = 12
    pass_td_pts: float = 4.0
    roster_slots: Dict[str, int] = field(default_factory=lambda: {
        'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'K': 1, 'DEF': 1, 'BENCH': 6
    })

    @property
    def total_rounds(self) -> int:
        """Calculates total draft rounds based on total roster slots."""
        return sum(self.roster_slots.values())


@dataclass
class Recommendation:
    player_id: str
    name: str
    position: str
    team: str
    adp: float
    projected_points: float
    roster_weight: float
    survival_prob_next_pick: float
    vona_score: float
    
    def __repr__(self) -> str:
        return (f"<{self.position} {self.name} | VONA: {self.vona_score:.1f} | "
                f"Proj: {self.projected_points:.1f} | Wt: {self.roster_weight:.2f} | "
                f"Surv: {self.survival_prob_next_pick:.1%}>")


class ValuationEngine:
    # Two consecutively value-ranked players at a position are considered
    # separate tiers once their ADPs are at least this many pooled standard
    # deviations apart - i.e. the market (not just this model) treats them as
    # drafted in statistically distinct windows. See _find_tier_boundary for
    # why ADP+stdev replaced a VONA-gap-magnitude heuristic here.
    TIER_ADP_Z_THRESHOLD = 1.75
    # Floor/fallback for a player's effective ADP stdev when missing or zero,
    # matching the fallback _compute_survival_and_baseline uses for the same
    # purpose (a fixed floor for very early picks, scaling with ADP later).
    TIER_ADP_STDEV_FLOOR = 1.5
    TIER_ADP_STDEV_ADP_FRACTION = 0.12
    # Only look for tiers within the top N positive-VONA players per position -
    # tier structure far down the board isn't actionable this pick.
    TIER_MAX_CANDIDATES = 15

    # Streaming-position (DEF/K) gates, expressed as "rounds remaining" rather
    # than an absolute round number so they scale with however many rounds
    # this league's draft actually has, instead of assuming a ~15-round
    # league. A draft shorter than the old hardcoded thresholds (e.g. 10
    # rounds) would otherwise never open the gate at all and leave DEF/K
    # slots empty. Values match the previous hardcoded rounds (11, 13)
    # exactly at the default 15-round length.
    #
    # Opponent demand for DEF/K is hard-floored to zero outside this many
    # rounds remaining, regardless of roster need - real drafters don't reach
    # for a streaming position just because the slot is open. Matches
    # draft_manager.DraftSession.BOT_STREAMING_LAST_ROUNDS, which the bot
    # simulator itself uses to decide when to start considering DEF/K.
    OPPONENT_STREAMING_LAST_ROUNDS = 3
    # The user's own DEF/K weight is heavily suppressed (but not hard-zeroed -
    # a great value that falls this far is still worth a small nod) outside
    # this many rounds remaining, then opens up fully.
    USER_STREAMING_LAST_ROUNDS = 5

    # For the user's own QB/RB/WR/TE backup weighting, only this many top
    # candidates per position get the roster_projection-based marginal
    # team-value calculation (see _flex_marginal_weight and
    # _single_slot_marginal_weight). Players outside this window fall back
    # to the flat depth-chart heuristic - they're too far down the board to
    # matter for this pick.
    FLEX_WEIGHT_MAX_CANDIDATES = 15

    # Bye-week collision nudge for QB/RB/WR/TE: multiplier applied once 2 (or
    # 3+) roster players already share a candidate's bye week. A single
    # existing same-week player is normal roster construction and isn't
    # penalized - this only discourages stacking a real cluster.
    BYE_COLLISION_MULTIPLIERS = {2: 0.9, 3: 0.75}

    def __init__(self, settings: LeagueSettings):
        self.settings = settings

    def _project_baseline_points(self, position: str, adp: float) -> float:
        """Calculates expected points using an exponential decay model."""
        params = {
            'QB': (250, 0.012, 150),
            'RB': (280, 0.015, 50),
            'WR': (270, 0.013, 60),
            'TE': (200, 0.018, 40),
            'K':  (70, 0.010, 90),
            'DEF': (70, 0.010, 90)
        }
        
        alpha, beta, gamma = params.get(position, params['WR'])
        points = alpha * np.exp(-beta * adp) + gamma
        
        if self.settings.format == 'standard' and position in ['RB', 'WR', 'TE']:
            points *= 0.80
        elif self.settings.format == 'half-ppr' and position in ['RB', 'WR', 'TE']:
            points *= 0.90
            
        return points

    def add_projected_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds a 'projected_points' season total column if one is not already present."""
        if df.empty or 'projected_points' in df.columns:
            return df

        df = df.copy()
        df['projected_points'] = df.apply(
            lambda r: self._project_baseline_points(r['position'], r['adp']), axis=1
        )
        return df

    def _calculate_opponent_need_factors(
        self,
        teams_picking: Optional[List[int]],
        all_roster_counts: Optional[Dict[int, Dict[str, int]]],
        current_pick: int
    ) -> Dict[str, float]:
        """
        Estimates how eager the opponents picking in the window are to draft
        each position right now, by applying the same per-team roster-need
        logic used for the user's own recommendations (get_roster_need_multiplier)
        to every opponent roster, then averaging. A team missing a starter is a
        likely taker; a team that already has one is not; backup depth decays
        the same way it does for the user's own picks.

        Averaging this across the teams actually picking in the window (rather
        than a fixed league-wide proportion) lets survival estimates react to
        what THIS draft's opponents have rostered so far - e.g. a QB lasting
        longer than its ADP once most starting QB slots are already filled, or
        an RB getting scooped up early because several teams in the window
        still lack their second starter.

        Streaming positions (DEF/K) are hard-floored to zero opponent demand
        outside OPPONENT_STREAMING_LAST_ROUNDS (counted from the end of the
        draft) regardless of roster need - get_roster_need_multiplier's own
        early-round allowance (a small weight, tuned for the user's personal
        recommendations) isn't a strong enough guardrail on its own to keep
        the model from recommending a reach on a defense just because the
        slot happens to be open.
        """
        round_num = (current_pick - 1) // self.settings.teams + 1
        rounds_remaining = self.settings.total_rounds - round_num + 1
        positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']

        if not teams_picking or not all_roster_counts:
            # No roster data available - fall back to neutral (pure ADP-implied) demand.
            need_factors = {p: 1.0 for p in positions}
        else:
            need_factors = {
                p: float(np.mean([
                    self.get_roster_need_multiplier(p, all_roster_counts.get(t, {}), current_pick)
                    for t in teams_picking
                ]))
                for p in positions
            }

        if rounds_remaining > self.OPPONENT_STREAMING_LAST_ROUNDS:
            need_factors['DEF'] = 0.0
            need_factors['K'] = 0.0

        return need_factors

    def get_roster_need_multiplier(
        self,
        position: str,
        current_roster: Dict[str, int],
        current_pick: int,
        projected_points: Optional[float] = None,
        current_roster_df: Optional[pd.DataFrame] = None,
        _flex_baseline: Optional[float] = None,
        bye_week: Optional[float] = None,
    ) -> float:
        """
        Applies positional decay and marginal bench utility multipliers.

        For RB/WR/TE backups, when `projected_points` and `current_roster_df`
        are supplied (the user's own drafted roster, with points), the
        backup-depth weight is replaced by an actual marginal team-value
        calculation - see _flex_marginal_weight. TE shares the FLEX slot with
        RB/WR just like they share it with each other, so it gets the same
        treatment once its own starter slot (TE1) is filled.

        QB2+ gets the analogous treatment via _single_slot_marginal_weight:
        since QB has no FLEX outlet, a backup QB only ever pays off through
        bye/injury bench insurance, which needs the probabilistic weekly
        model (not the deterministic ideal-lineup snapshot _flex_marginal_weight
        uses) to see any value at all - see that method's docstring.

        Callers that only have roster counts (e.g. the opponent-need
        modeling in _calculate_opponent_need_factors) fall back to the flat
        heuristics below, scaled by rounds remaining in this league's actual
        draft length rather than a hardcoded round number.
        """
        count = current_roster.get(position, 0)
        starter_limit = self.settings.roster_slots.get(position, 0)
        round_num = (current_pick - 1) // self.settings.teams + 1
        rounds_remaining = self.settings.total_rounds - round_num + 1

        # 1. Streaming Positions (DEF, K)
        if position in ['DEF', 'K']:
            if count >= starter_limit or starter_limit == 0:
                return 0.0
            return 0.05 if rounds_remaining > self.USER_STREAMING_LAST_ROUNDS else 1.0

        # 2. Open Starter Slots
        if count < starter_limit:
            return 1.15 if position in ['QB', 'TE'] else 1.0

        # 3. QB Backups (single-starter, not FLEX-eligible)
        if position == 'QB':
            if count != starter_limit:  # QB3+: not worth modeling precisely
                return 0.05
            if current_roster_df is not None and projected_points is not None:
                return self._single_slot_marginal_weight(
                    'QB', projected_points, bye_week, current_roster_df
                )
            # Fallback heuristic when full roster context isn't available
            return 0.15 if rounds_remaining > self.USER_STREAMING_LAST_ROUNDS else 0.35

        # 4. FLEX-Eligible Backups (RB, WR, TE) - all share the one FLEX slot
        if position in ['RB', 'WR', 'TE']:
            if current_roster_df is not None and projected_points is not None:
                return self._flex_marginal_weight(
                    position, projected_points, current_roster_df, baseline=_flex_baseline
                )
            # Fallback heuristic when full roster context isn't available
            if position == 'TE':
                return 0.20 if round_num <= 10 else 0.40
            backup_depth = count - starter_limit + 1
            if backup_depth == 1:   # RB3/WR3
                return 0.85
            elif backup_depth == 2: # RB4/WR4
                return 0.65
            else:                   # RB5+/WR5+
                return 0.45

        return 0.05

    def _compute_flex_baseline(self, current_roster_df: pd.DataFrame) -> float:
        """Current team's ideal-lineup season points, before adding any candidate."""
        players = list(zip(current_roster_df['projected_points'], current_roster_df['position']))
        return roster_projection.greedy_fill(players, self.settings.roster_slots)

    def _flex_marginal_weight(
        self,
        position: str,
        projected_points: float,
        current_roster_df: pd.DataFrame,
        baseline: Optional[float] = None,
    ) -> float:
        """
        Bounded multiplier representing what fraction of this player's raw
        points would actually crack the user's ideal starting lineup, using
        the same greedy starter/FLEX assignment as the post-draft summary
        (roster_projection.greedy_fill/ideal_lineup) - exact for a FLEX that's
        a superset of the dedicated slots, per that module's own verification.

        This replaces a flat per-position depth-chart heuristic with the real
        thing: RB and WR backups both compete for the same single FLEX slot,
        so a roster already stacked at one position sees a shrinking marginal
        value for more of that position - without needing to hardcode it.
        Bounded to the same range as the rest of get_roster_need_multiplier so
        it scales the underlying VONA score rather than overriding it; a deep
        reach still only gets a small, market-anchored VONA to scale.
        """
        if baseline is None:
            baseline = self._compute_flex_baseline(current_roster_df)

        players = list(zip(current_roster_df['projected_points'], current_roster_df['position']))
        players.append((projected_points, position))
        with_candidate = roster_projection.greedy_fill(players, self.settings.roster_slots)

        marginal_fraction = (with_candidate - baseline) / max(projected_points, 1e-6)
        return float(np.clip(marginal_fraction, 0.05, 1.15))

    def _single_slot_marginal_weight(
        self,
        position: str,
        projected_points: float,
        bye_week: Optional[float],
        current_roster_df: pd.DataFrame,
    ) -> float:
        """
        Bounded multiplier for single-starter, non-FLEX-eligible backups
        (QB2+): the fraction of this candidate's raw points that would
        actually convert into extra realized season production for the
        user's team.

        Unlike _flex_marginal_weight's deterministic "does it crack today's
        ideal lineup" snapshot, a backup QB never cracks that snapshot unless
        he's outright better than the starter on paper - all of his real
        value is bye/injury bench insurance, which only shows up
        probabilistically across the season. So this runs the same per-week
        "best-available-starts" cascade roster_projection.project_team uses
        (see roster_projection._expected_single_slot), scoped to just this
        one position - cheap, since QB/K/DEF aren't part of the exponential
        FLEX state DP, just an O(candidates) sort per week.
        """
        existing = current_roster_df[current_roster_df['position'] == position]
        rows = [
            (float(r['projected_points']) / roster_projection.GAMES_PER_SEASON, r.get('bye_week'))
            for _, r in existing.iterrows()
        ]
        candidate_row = (projected_points / roster_projection.GAMES_PER_SEASON, bye_week)

        def season_total(week_rows):
            total = 0.0
            for week in range(1, roster_projection.REGULAR_SEASON_WEEKS + 1):
                rated = sorted(
                    ((pts, roster_projection.availability(position, bye, week)) for pts, bye in week_rows),
                    key=lambda x: -x[0]
                )
                expected = 0.0
                prob_all_above_out = 1.0
                for pts, avail in rated:
                    p_starts = avail * prob_all_above_out
                    expected += pts * p_starts
                    prob_all_above_out *= (1.0 - avail)
                total += expected
            return total

        baseline = season_total(rows)
        with_candidate = season_total(rows + [candidate_row])

        marginal_fraction = (with_candidate - baseline) / max(projected_points, 1e-6)
        return float(np.clip(marginal_fraction, 0.05, 1.15))

    def _bye_week_conflicts(self, df: pd.DataFrame, current_roster_df: Optional[pd.DataFrame]) -> pd.Series:
        """
        Per-row count of already-rostered players sharing that row's bye week,
        scoped to who actually matters for that row's slot.

        A candidate that would fill an open starter slot (QB1/TE1, or RB/WR
        1-3 once the shared FLEX slot is counted in) is checked against the
        roster's other starters only - a stacked bye among backups doesn't
        cost a started lineup anything. A candidate that would instead be a
        backup (QB2+, TE2+, RB4+/WR4+) is checked only against its own
        position's starter(s) - the only failure mode a backup actually
        guards against is its starting counterpart being out that week, not
        an unrelated backup at another position also being on bye.
        """
        if (
            current_roster_df is None or current_roster_df.empty
            or 'bye_week' not in current_roster_df.columns or 'bye_week' not in df.columns
            or 'position' not in current_roster_df.columns or 'position' not in df.columns
        ):
            return pd.Series(0, index=df.index)

        starters_by_pos: Dict[str, pd.DataFrame] = {}
        starter_limit_by_pos: Dict[str, int] = {}
        counts_by_pos: Dict[str, int] = {}
        for pos in ('QB', 'RB', 'WR', 'TE'):
            pos_df = current_roster_df[current_roster_df['position'] == pos]
            starter_limit = self.settings.roster_slots.get(pos, 0)
            if pos in ('RB', 'WR'):
                starter_limit += self.settings.roster_slots.get('FLEX', 0)
            starters_by_pos[pos] = pos_df.iloc[:starter_limit]
            starter_limit_by_pos[pos] = starter_limit
            counts_by_pos[pos] = len(pos_df)

        all_starters = pd.concat(starters_by_pos.values()) if starters_by_pos else current_roster_df.iloc[0:0]
        all_starter_bye_counts = all_starters['bye_week'].dropna().value_counts()

        def _conflicts(row) -> int:
            position = row['position']
            bye = row.get('bye_week')
            if position not in starters_by_pos or pd.isna(bye):
                return 0
            if counts_by_pos[position] < starter_limit_by_pos[position]:
                # This candidate would itself be a starter - compare against
                # the roster's other starters across all positions.
                return int(all_starter_bye_counts.get(bye, 0))
            # This candidate would be a backup - compare only against its
            # own position's starter(s).
            own_starter_byes = starters_by_pos[position]['bye_week']
            return int((own_starter_byes == bye).sum())

        return df.apply(_conflicts, axis=1).astype(int)

    def _bye_week_multiplier(self, position: str, conflicts: int) -> float:
        """
        Mild markdown once a candidate's bye week would stack with players
        already on the roster - QB/RB/WR/TE only, since these are the
        positions where losing multiple starters the same week actually
        costs you points (DEF/K are single, late-round, and already handled
        by the streaming-position logic above).
        """
        if position not in ('QB', 'RB', 'WR', 'TE'):
            return 1.0
        for threshold in sorted(self.BYE_COLLISION_MULTIPLIERS.keys(), reverse=True):
            if conflicts >= threshold:
                return self.BYE_COLLISION_MULTIPLIERS[threshold]
        return 1.0

    def _resolve_target_pick(self, current_pick: int, next_pick: Optional[int]) -> Tuple[int, int]:
        """
        Resolves the horizon to evaluate scarcity against. Normally just
        `next_pick`, but if it's only 1-2 picks away (the user is at a snake
        turn), that near pick offers almost no real choice - the board barely
        moves - so this looks past it to the pick after instead.

        Returns (target_pick, m), where m is the number of picks in the
        window [current_pick, target_pick) that opponents get to draft from.
        """
        if next_pick and next_pick > current_pick:
            pick_distance = next_pick - current_pick
            if pick_distance <= 2:
                target_pick = current_pick + (2 * self.settings.teams - 1)
            else:
                target_pick = next_pick
            m = target_pick - current_pick
        else:
            target_pick = current_pick
            m = 0
        return target_pick, m

    def _compute_survival_and_baseline(
        self,
        df: pd.DataFrame,
        current_pick: int,
        target_pick: int,
        m: int,
        all_team_roster_counts: Optional[Dict[int, Dict[str, int]]],
        user_team_id: Optional[int],
        team_at_pick_fn: Optional[Callable[[int], int]],
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Core scarcity model shared by the single-horizon VONA pipeline
        (compute_player_evaluations) and the multi-pick positional runway
        outlook (compute_positional_runway): for a given target_pick horizon,
        estimates each player's survival probability and, per position, the
        expected points of whichever player turns out to be the best one
        still available by then (a probabilistic waterfall over survival).

        Same ADP-normal-CDF + opponent-need queue-decay model either way -
        only the horizon (target_pick, m) changes between calls, so this
        never mutates `df`; callers assign survival_prob_next_pick themselves.
        """
        teams_picking_in_window: List[int] = []
        if team_at_pick_fn is not None and user_team_id is not None and target_pick > current_pick:
            teams_picking_in_window = [
                team for pick_num in range(current_pick, target_pick)
                if (team := team_at_pick_fn(pick_num)) != user_team_id
            ]

        sigmas = np.where(
            (df['stdev'].notna()) & (df['stdev'] > 0),
            df['stdev'],
            np.maximum(1.5, 0.12 * df['adp'])
        )

        # Cumulative probability of being drafted by pick c and pick k
        p_taken_k = stats.norm.cdf((target_pick - df['adp']) / sigmas)
        p_taken_c = stats.norm.cdf((current_pick - df['adp']) / sigmas)

        # Expected draft volume per player in window [current_pick, target_pick]
        work = df.assign(
            _expected_adp_picks=np.clip(p_taken_k - p_taken_c, 0.0, 1.0),
            _base_p_avail=np.clip(1.0 - p_taken_k, 0.0, 1.0),
        ).sort_values(by=['position', 'adp'])

        need_factors = self._calculate_opponent_need_factors(
            teams_picking_in_window, all_roster_counts=all_team_roster_counts, current_pick=current_pick
        )

        survival_series = pd.Series(index=work.index, dtype=float)

        for pos in work['position'].unique():
            pos_mask = work['position'] == pos
            pos_df = work[pos_mask]

            # Sum of ADP draft volume expected at this position in the window
            mu_adp = pos_df['_expected_adp_picks'].sum()
            gamma = need_factors.get(pos, 1.0)
            mu_flow = mu_adp * gamma

            if m <= 0 or mu_flow <= 0.001:
                # If opponents have 0 need or window is 0, players survive at base or 100%
                survival_series[pos_mask] = 1.0 if gamma == 0.0 else pos_df['_base_p_avail']
                continue

            # When opponent need is elevated (gamma > 1) or many players'
            # ADP windows overlap, mu_flow can exceed m, driving this
            # binomial-style variance negative. Floor it at 0 before sqrt -
            # np.sqrt(negative) returns NaN with a RuntimeWarning, and while
            # the outer max(1.0, nan) currently happens to swallow that NaN
            # (nan comparisons are always False, so max keeps its first
            # argument), that's an accident of argument order, not something
            # to depend on.
            variance = max(0.0, mu_flow * (1.0 - mu_flow / max(m, 1)))
            sigma_flow = max(1.0, np.sqrt(variance))
            ranks = np.arange(1, len(pos_df) + 1)

            # Queue decay factor: how likely positional demand reaches rank r
            z = (mu_flow - ranks + 0.5) / sigma_flow
            p_queue_reach = stats.norm.cdf(z)

            # Normalize queue reach relative to rank 1
            norm_factor = max(stats.norm.cdf((mu_flow - 1 + 0.5) / sigma_flow), 0.01)
            queue_decay = np.clip(p_queue_reach / norm_factor, 0.0, 1.0)

            # Modulate: A player is taken if their ADP allows it AND positional flow reaches them
            p_taken_dynamic = (1.0 - pos_df['_base_p_avail']) * queue_decay
            survival_series[pos_mask] = np.clip(1.0 - p_taken_dynamic, 0.0, 1.0)

        # Re-align to df's original row order/index for the caller
        survival_series = survival_series.reindex(df.index)

        # Probabilistic Waterfall Baseline: expected points of whichever
        # player ends up being the best one still available at this horizon.
        expected_points_at_pos: Dict[str, float] = {}
        for pos in df['position'].unique():
            pos_idx = df.index[df['position'] == pos]
            pos_points = df.loc[pos_idx, 'projected_points'].sort_values(ascending=False)

            expected_points = 0.0
            prob_best_remaining = 1.0
            for idx in pos_points.index:
                p_avail = survival_series.loc[idx]
                prob_is_best = p_avail * prob_best_remaining
                expected_points += prob_is_best * pos_points.loc[idx]
                prob_best_remaining *= (1.0 - p_avail)

            expected_points_at_pos[pos] = expected_points

        return survival_series, expected_points_at_pos

    def compute_player_evaluations(
        self,
        available_df: pd.DataFrame,
        current_roster_counts: Dict[str, int],
        current_pick: int,
        next_pick: Optional[int],
        all_team_roster_counts: Optional[Dict[int, Dict[str, int]]] = None,
        user_team_id: Optional[int] = None,
        team_at_pick_fn: Optional[Callable[[int], int]] = None,
        current_roster_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Computes dynamic VONA baseline, ADP-anchored survival probabilities,
        and utility scaling for all available players.

        `all_team_roster_counts`, `user_team_id`, and `team_at_pick_fn`
        (typically DraftSession.get_all_teams_roster_counts /
        DraftSession.get_team_at_pick) are optional but let the survival model
        weight ADP-implied draft volume by each opponent's actual roster need
        in the pick window - see _calculate_opponent_need_factors. Without
        them, survival falls back to pure ADP with no opponent-need modulation.

        `current_roster_df` (typically roster_projection.build_roster_df for
        user_team_id) is the user's own drafted players with projected_points
        attached. When supplied, roster_weight for the top candidates at each
        of QB/RB/WR/TE is computed from actual marginal team value (see
        _flex_marginal_weight and _single_slot_marginal_weight) rather than a
        flat depth-chart heuristic, and every QB/RB/WR/TE candidate gets a
        mild vona_score markdown if their bye week would stack with players
        already on the roster (see _bye_week_multiplier).
        """
        if available_df.empty:
            return available_df

        df = available_df.copy()

        # 1. Projected Points
        df = self.add_projected_points(df)


        # 2. Roster Weight Multiplier
        flex_baseline = None
        marginal_weight_eligible_ids = set()
        if current_roster_df is not None:
            flex_baseline = self._compute_flex_baseline(current_roster_df)
            for pos in ('QB', 'RB', 'WR', 'TE'):
                top_candidates = df[df['position'] == pos].nlargest(
                    self.FLEX_WEIGHT_MAX_CANDIDATES, 'projected_points'
                )
                marginal_weight_eligible_ids.update(top_candidates.index)

        def _row_roster_weight(row):
            if row.name in marginal_weight_eligible_ids:
                return self.get_roster_need_multiplier(
                    row['position'], current_roster_counts, current_pick,
                    projected_points=row['projected_points'],
                    current_roster_df=current_roster_df,
                    _flex_baseline=flex_baseline,
                    bye_week=row.get('bye_week'),
                )
            return self.get_roster_need_multiplier(row['position'], current_roster_counts, current_pick)

        df['roster_weight'] = df.apply(_row_roster_weight, axis=1)

        # 2b. Bye-Week Collision Nudge - a mild markdown for skill-position
        # players whose bye week already stacks up on the user's own roster,
        # so recommendations steer (without hard-blocking) away from a lineup
        # that loses several starters the same week with no flex cover.
        df['bye_week_conflicts'] = self._bye_week_conflicts(df, current_roster_df)
        df['bye_multiplier'] = df.apply(
            lambda row: self._bye_week_multiplier(row['position'], row['bye_week_conflicts']), axis=1
        )

        # 3. Target Pick & Pick Horizon, and the scarcity model at that horizon
        target_pick, m = self._resolve_target_pick(current_pick, next_pick)
        survival_series, expected_points_at_pos = self._compute_survival_and_baseline(
            df, current_pick, target_pick, m, all_team_roster_counts, user_team_id, team_at_pick_fn
        )
        df['survival_prob_next_pick'] = survival_series

        # 7. Final VONA Calculation
        def calculate_vona(row):
            expected_baseline = expected_points_at_pos.get(row['position'], 0.0)
            vona_raw = row['projected_points'] - expected_baseline
            return max(0.0, vona_raw) * row['roster_weight'] * row.get('bye_multiplier', 1.0)

        df['vona_score'] = df.apply(calculate_vona, axis=1).fillna(0.0)

        # Ties (most commonly a VONA of 0.0, once a position's baseline
        # exceeds every remaining player) fall back to ADP ascending, so
        # equally-valued players still appear in a sensible order instead of
        # whatever order sort_values' default unstable quicksort leaves them.
        return df.sort_values(
            ['vona_score', 'adp'], ascending=[False, True]
        ).reset_index(drop=True)

    def compute_positional_runway(
        self,
        available_df: pd.DataFrame,
        current_pick: int,
        upcoming_picks: List[int],
        all_team_roster_counts: Optional[Dict[int, Dict[str, int]]] = None,
        user_team_id: Optional[int] = None,
        team_at_pick_fn: Optional[Callable[[int], int]] = None,
    ) -> pd.DataFrame:
        """
        Projects, for each position, how the board's best-remaining value
        decays across the user's next several picks (`upcoming_picks`, e.g.
        from DraftSession.get_upcoming_user_picks) - reusing the same
        ADP-survival + opponent-need scarcity model that drives VONA for a
        single pick, just evaluated at multiple future horizons instead of
        one.

        This answers a different question than VONA or the tier-break alert:
        not "is this player good right now" but "which position can I safely
        defer, and which one won't still be here in two picks" - letting a
        drafter sequence positions across rounds instead of only reacting to
        whatever the board looks like this instant. A position with a short
        runway (steep early decay) is worth reaching a *round* earlier than
        VONA alone would suggest; a position with a long runway is safe to
        punt on now in favor of whatever's most urgent.

        Returns one row per (position, horizon) for horizon 1..len(upcoming_picks)
        (1-indexed, matching upcoming_picks order):
          - pick_number: the overall pick this horizon represents
          - expected_best_points: expected projected_points of whichever
            player turns out to be the best one still available at that
            position by then (probabilistic waterfall over survival)
          - decay_pct: fraction below the position's current best-available
            projected_points (0 = no drop yet, 1 = fully dried up)
        """
        positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
        if available_df.empty or not upcoming_picks:
            return pd.DataFrame(columns=['position', 'horizon', 'pick_number', 'expected_best_points', 'decay_pct'])

        df = self.add_projected_points(available_df.copy())
        present_positions = [p for p in positions if (df['position'] == p).any()]

        # What's actually on the board right now, for each position - the
        # reference point every future horizon's decay is measured against.
        baseline_best = {
            pos: float(df.loc[df['position'] == pos, 'projected_points'].max())
            for pos in present_positions
        }

        rows = []
        for h, pick_num in enumerate(upcoming_picks, start=1):
            if pick_num <= current_pick:
                continue
            m = pick_num - current_pick
            _, expected_points_at_pos = self._compute_survival_and_baseline(
                df, current_pick, pick_num, m, all_team_roster_counts, user_team_id, team_at_pick_fn
            )
            for pos in present_positions:
                expected_best = expected_points_at_pos.get(pos, 0.0)
                best_now = baseline_best.get(pos, 0.0)
                decay_pct = 0.0 if best_now <= 0 else float(np.clip(1.0 - (expected_best / best_now), 0.0, 1.0))
                rows.append({
                    'position': pos,
                    'horizon': h,
                    'pick_number': pick_num,
                    'expected_best_points': expected_best,
                    'decay_pct': decay_pct,
                })

        return pd.DataFrame(rows)

    def _effective_adp_stdev(self, adp: float, stdev: Optional[float]) -> float:
        """Same fallback _compute_survival_and_baseline uses: trust a real stdev, else scale with ADP."""
        if stdev is not None and not np.isnan(stdev) and stdev > 0:
            return float(stdev)
        return float(max(self.TIER_ADP_STDEV_FLOOR, self.TIER_ADP_STDEV_ADP_FRACTION * adp))

    def _find_tier_boundary(self, pos_df: pd.DataFrame) -> Optional[int]:
        """
        Given a position's candidates already ranked by descending VONA (best
        value first), returns the 1-indexed size of the top tier - how many
        players sit before the market draws a real line - or None if no
        statistically meaningful break exists among them.

        Tiers are detected from each pair's ADP gap relative to their pooled
        ADP stdev, not from the magnitude of the VONA gap between them. A
        VONA-gap threshold has to compare against some "typical" gap for the
        position - usually the median gap across the whole candidate list -
        but that list almost always contains more than one real tier
        boundary, so a later, unrelated break (e.g. a bunched RB3-7 group's
        own internal jitter) can inflate the median enough to mask an earlier
        break that's just as real, like a clear RB1/RB2 tier ahead of a
        tightly-packed group. ADP + stdev sidesteps that: it only asks
        whether the market itself is drafting these two players in
        statistically distinct windows, independent of how VONA happens to
        be distributed elsewhere in the list.
        """
        if len(pos_df) < 2 or pos_df['vona_score'].iloc[0] <= 0:
            return None

        adps = pos_df['adp'].to_numpy()
        stdevs = pos_df['stdev'].to_numpy() if 'stdev' in pos_df.columns else np.full(len(pos_df), np.nan)

        for i in range(len(pos_df) - 1):
            pooled_sigma = np.sqrt(
                self._effective_adp_stdev(adps[i], stdevs[i]) ** 2
                + self._effective_adp_stdev(adps[i + 1], stdevs[i + 1]) ** 2
            )
            z = (adps[i + 1] - adps[i]) / pooled_sigma
            if z >= self.TIER_ADP_Z_THRESHOLD:
                return i + 1

        return None

    def identify_tier_gaps(self, evaluated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects, per position, whether there's a real "tier break" among the
        top available players - a handful clearly ahead of the pack, after
        which VONA drops sharply - versus values that are just evenly spread
        out (large VONA but no urgency, since similar value likely survives
        to your next pick).

        This is deliberately kept separate from vona_score itself: VONA
        measures how good a player is relative to the expected replacement:
        tier urgency measures how much you stand to lose by not acting on
        that value *right now*, which is a distinct question. Conflating the
        two would make vona_score harder to reason about everywhere else it's
        used (bench weighting, recommendations).

        Adds four columns:
          - tier_rank: 1-indexed rank within a detected top tier (0 if none)
          - is_last_in_tier: True for the worst player still inside the tier
          - tier_gap_value: VONA points dropped from this tier to the next
          - tier_exhaust_prob: P(every player in the tier is drafted by
            someone else before your next pick), i.e. the tier disappears
          - tier_urgency: tier_gap_value * tier_exhaust_prob - the expected
            VONA points lost if you wait one more pick to address this need
        """
        result = evaluated_df.copy()
        result['tier_rank'] = 0
        result['is_last_in_tier'] = False
        result['tier_gap_value'] = 0.0
        result['tier_exhaust_prob'] = 0.0
        result['tier_urgency'] = 0.0

        if evaluated_df.empty:
            return result

        for pos in evaluated_df['position'].unique():
            pos_df = evaluated_df[(evaluated_df['position'] == pos) & (evaluated_df['vona_score'] > 0)]
            pos_df = pos_df.sort_values('vona_score', ascending=False).head(self.TIER_MAX_CANDIDATES)
            if len(pos_df) < 2:
                continue

            tier_size = self._find_tier_boundary(pos_df)
            if tier_size is None:
                continue

            tier_df = pos_df.iloc[:tier_size]
            next_tier_df = pos_df.iloc[tier_size:]
            if next_tier_df.empty:
                continue  # No visible next tier to compare against - nothing to warn about.

            gap_value = float(tier_df['vona_score'].iloc[-1] - next_tier_df['vona_score'].iloc[0])
            if gap_value <= 0:
                continue

            survival = tier_df['survival_prob_next_pick'].fillna(0.0).clip(0, 1)
            prob_tier_exhausted = float(np.prod(1.0 - survival))
            urgency = gap_value * prob_tier_exhausted

            for rank, idx in enumerate(tier_df.index, start=1):
                result.loc[idx, 'tier_rank'] = rank
                result.loc[idx, 'is_last_in_tier'] = (rank == tier_size)
                result.loc[idx, 'tier_gap_value'] = gap_value
                result.loc[idx, 'tier_exhaust_prob'] = prob_tier_exhausted
                result.loc[idx, 'tier_urgency'] = urgency

        return result

    def get_recommendations(self, evaluated_df: pd.DataFrame, top_n: int = 10) -> Dict[str, List[Recommendation]]:
        """Extracts top 'n' recommendations globally and by position."""
        if evaluated_df.empty:
            return {'overall': [], 'by_position': {}}
            
        def row_to_rec(row) -> Recommendation:
            return Recommendation(
                player_id=str(row['player_id']),
                name=row['name'],
                position=row['position'],
                team=row['team'],
                adp=row['adp'],
                projected_points=row['projected_points'],
                roster_weight=row['roster_weight'],
                survival_prob_next_pick=row['survival_prob_next_pick'],
                vona_score=row['vona_score']
            )

        recs = {'overall': [], 'by_position': {}}
        
        # Filter for positive VONA utility
        valid_overall = evaluated_df[evaluated_df['vona_score'] > 0.0]
        if valid_overall.empty:
            valid_overall = evaluated_df
            
        for _, row in valid_overall.head(top_n).iterrows():
            recs['overall'].append(row_to_rec(row))
            
        positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
        for pos in positions:
            pos_df = evaluated_df[evaluated_df['position'] == pos].head(top_n)
            recs['by_position'][pos] = [row_to_rec(row) for _, row in pos_df.iterrows()]

        return recs


if __name__ == "__main__":
    import warnings

    print("--- Running Valuation Verification ---")

    # mu_flow > m regression check: a short window (m) with many players
    # clustered at the front of a position's ADP and elevated opponent need
    # (gamma > 1, via an open QB starter slot for every opposing team) used
    # to drive the binomial-style variance in _compute_survival_and_baseline
    # negative, so np.sqrt(negative) raised a RuntimeWarning ("invalid value
    # encountered in sqrt"). Asserts that case is now silent and NaN-free.
    settings = LeagueSettings(teams=12)
    engine = ValuationEngine(settings)

    df = pd.DataFrame([
        {'player_id': f'qb{i}', 'name': f'QB {i}', 'position': 'QB', 'team': 'AAA',
         'adp': 10.0 + i * 0.5, 'stdev': 1.0}
        for i in range(10)
    ])
    df = engine.add_projected_points(df)

    current_pick, target_pick = 10, 12  # m = 2, deliberately short window
    m = target_pick - current_pick
    all_team_roster_counts = {t: {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'K': 0, 'DEF': 0} for t in range(12)}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        survival_series, expected_points_at_pos = engine._compute_survival_and_baseline(
            df, current_pick, target_pick, m,
            all_team_roster_counts, user_team_id=0,
            team_at_pick_fn=lambda pick_num: 1 + (pick_num % 11),
        )
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]

    assert not runtime_warnings, f"Unexpected RuntimeWarning(s): {[str(w.message) for w in runtime_warnings]}"
    assert not survival_series.isna().any(), "survival_series contains NaN when mu_flow > m"
    assert not any(np.isnan(v) for v in expected_points_at_pos.values()), "expected_points_at_pos contains NaN"
    print("[ok] mu_flow > m no longer raises a sqrt RuntimeWarning or produces NaN")

    print("\nAll assertions passed successfully!")