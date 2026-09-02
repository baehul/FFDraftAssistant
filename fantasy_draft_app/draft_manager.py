"""
draft_manager.py
Dynamic Fantasy Football Draft Assistant - Draft State Machine & Bot Simulator
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class PickRecord:
    overall_pick: int       # 1 to N*R
    round: int              # 1 to R
    pick_in_round: int      # 1 to N
    team_id: int            # 1 to N
    player_id: str          # Unique string identifier
    player_name: str        # Full name
    position: str           # QB, RB, WR, TE, K, DEF
    team: str               # NFL Team abbreviation
    bye_week: Optional[int] = None


class DraftSession:
    # Bot realism gates for backup/streaming positions, expressed as "rounds
    # remaining" rather than an absolute round number, so they scale with
    # however many rounds this particular draft actually has instead of
    # assuming a ~15-round league. Values are picked to match the previous
    # hardcoded rounds (10, 13) exactly at the default 15-round length -
    # see ValuationEngine's matching STREAMING_LAST_ROUNDS/
    # OPPONENT_STREAMING_LAST_ROUNDS, which this mirrors for consistency
    # between how bots draft and how the recommender expects them to.
    BOT_BACKUP_QB_TE_LAST_ROUNDS = 6
    BOT_STREAMING_LAST_ROUNDS = 3

    def __init__(
        self, 
        players_df: pd.DataFrame, 
        user_team_id: int = 1, 
        teams: int = 12, 
        rounds: int = 15,
        format: str = "ppr",
        random_seed: Optional[int] = None
    ):
        """
        Initializes the draft session with available players and league settings.
        """
        self.players_df = players_df.copy()
        
        # Standardize name column
        if 'name' not in self.players_df.columns and 'player_name' in self.players_df.columns:
            self.players_df = self.players_df.rename(columns={'player_name': 'name'})
        elif 'player_name' not in self.players_df.columns and 'name' in self.players_df.columns:
            self.players_df['player_name'] = self.players_df['name']
            
        # Ensure ADP column exists
        if 'adp' not in self.players_df.columns:
            self.players_df['adp'] = range(1, len(self.players_df) + 1)

        self.players_df.sort_values(by="adp", inplace=True)
            
        self.user_team_id = user_team_id
        self.teams = teams
        self.rounds = rounds
        self.format = format
        self.total_picks = self.teams * self.rounds
        
        if random_seed is not None:
            np.random.seed(random_seed)

        self.current_pick = 1
        self.drafted_picks: List[PickRecord] = []
        self.undone_picks: List[PickRecord] = []  # Stack for redo functionality
        
        # Fast lookup set for drafted players
        self._drafted_player_ids = set()

    @property
    def is_complete(self) -> bool:
        """Returns True if all picks across all rounds have been made."""
        return self.current_pick > self.total_picks

    @property
    def is_user_turn(self) -> bool:
        """Returns True if the current pick on the clock belongs to the user."""
        if self.is_complete:
            return False
        return self._get_team_at_pick(self.current_pick) == self.user_team_id

    def get_team_at_pick(self, pick_number: int) -> int:
        """Public accessor: which team is on the clock at a given overall pick number."""
        return self._get_team_at_pick(pick_number)

    def _get_team_at_pick(self, pick_number: int) -> int:
        """Computes 1-indexed team ID on the clock using snake draft alternating direction."""
        round_num = (pick_number - 1) // self.teams + 1
        pick_in_round = (pick_number - 1) % self.teams + 1
        
        if round_num % 2 != 0:
            # Odd rounds: 1 to N
            return pick_in_round
        else:
            # Even rounds: N to 1
            return self.teams - pick_in_round + 1

    def _get_round_and_pick(self, pick_number: int) -> Tuple[int, int]:
        """Calculates (round, pick_in_round) tuple from an overall pick number."""
        round_num = (pick_number - 1) // self.teams + 1
        pick_in_round = (pick_number - 1) % self.teams + 1
        return round_num, pick_in_round

    def _get_next_user_pick(self) -> Optional[int]:
        """
        Calculates the exact next overall pick number belonging to user_team_id
        that is strictly > self.current_pick. Returns None if user has no remaining picks.
        """
        picks = self.get_upcoming_user_picks(self.user_team_id, self.current_pick, n=1)
        return picks[0] if picks else None

    def get_upcoming_user_picks(self, user_team_id: int, current_pick: int, n: int = 4) -> List[int]:
        """
        Returns up to `n` future overall pick numbers belonging to user_team_id,
        strictly greater than current_pick, in ascending order. Powers the
        multi-pick positional runway outlook (ValuationEngine.compute_positional_runway) -
        reasoning about scarcity across several of the user's own picks, not
        just the very next one.
        """
        picks: List[int] = []
        for pick_num in range(current_pick + 1, self.total_picks + 1):
            if self._get_team_at_pick(pick_num) == user_team_id:
                picks.append(pick_num)
                if len(picks) == n:
                    break
        return picks

    def get_current_state(self) -> dict:
        """
        Returns a standardized dictionary summarizing current draft status.
        """
        if self.is_complete:
            last_pick = self.total_picks
            round_num, pick_in_round = self._get_round_and_pick(last_pick)
            return {
                "overall_pick": self.current_pick,
                "round": round_num,
                "current_pick": pick_in_round,
                "team_on_clock": self._get_team_at_pick(last_pick),
                "is_user_turn": False,
                "user_next_pick": None,
                "picks_until_user": 0,
                "is_complete": True
            }

        round_num, pick_in_round = self._get_round_and_pick(self.current_pick)
        on_the_clock = self._get_team_at_pick(self.current_pick)
        next_user_pick = self._get_next_user_pick()
        
        # 0 when it is the user's turn, otherwise pick distance
        if self.is_user_turn:
            picks_until = 0
        else:
            picks_until = (next_user_pick - self.current_pick) if next_user_pick is not None else 0

        return {
            "overall_pick": self.current_pick,
            "round": round_num,
            "current_pick": pick_in_round,
            "team_on_clock": on_the_clock,
            "is_user_turn": self.is_user_turn,
            "user_next_pick": next_user_pick,
            "picks_until_user": picks_until,
            "is_complete": False
        }

    def get_team_roster(self, team_id: int) -> List[PickRecord]:
        """Returns list of picks made by a specific team."""
        return [p for p in self.drafted_picks if p.team_id == team_id]

    def get_all_teams_rosters(self) -> Dict[int, List[PickRecord]]:
        """Returns a mapping of team_id -> list of drafted PickRecords."""
        rosters: Dict[int, List[PickRecord]] = {i: [] for i in range(1, self.teams + 1)}
        for pick in self.drafted_picks:
            rosters[pick.team_id].append(pick)
        return rosters

    def get_draft_board_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame structured as:
        - Rows: Rounds (1 to R)
        - Columns: Teams (1 to N)
        - Cells: Player Name + Position string (or empty if unpicked)
        """
        board = np.full((self.rounds, self.teams), "", dtype=object)
        for pick in self.drafted_picks:
            r_idx = pick.round - 1
            t_idx = pick.team_id - 1
            board[r_idx, t_idx] = f"{pick.player_name} ({pick.position})"

        df = pd.DataFrame(
            board,
            index=[f"Round {i}" for i in range(1, self.rounds + 1)],
            columns=[f"Team {i}" for i in range(1, self.teams + 1)]
        )
        return df

    def get_team_roster_counts(self, team_id: int) -> Dict[str, int]:
        """Returns dict of positional counts for a team (e.g., {'QB': 1, 'RB': 2, ...})."""
        roster = self.get_team_roster(team_id)
        counts: Dict[str, int] = {}
        for p in roster:
            counts[p.position] = counts.get(p.position, 0) + 1
        return counts

    def get_all_teams_roster_counts(self) -> Dict[int, Dict[str, int]]:
        """Returns positional counts for every team: team_id -> {position: count}."""
        return {t: self.get_team_roster_counts(t) for t in range(1, self.teams + 1)}

    def get_available_players(self) -> pd.DataFrame:
        """Returns DataFrame of currently available (undrafted) players."""
        return self.players_df[~self.players_df['player_id'].isin(self._drafted_player_ids)]

    def make_pick(self, player_id: str) -> PickRecord:
        """Drafts player_id for the current on-the-clock team and advances draft by 1."""
        if self.is_complete:
            raise ValueError("The draft is already complete.")
            
        player_id = str(player_id)
        if player_id in self._drafted_player_ids:
            raise ValueError(f"Player {player_id} has already been drafted.")

        player_row = self.players_df[self.players_df['player_id'] == player_id]
        if player_row.empty:
            raise ValueError(f"Player ID {player_id} not found in player pool.")
            
        player_data = player_row.iloc[0]
        round_num, pick_in_round = self._get_round_and_pick(self.current_pick)
        team_id = self._get_team_at_pick(self.current_pick)
        player_name = player_data.get('player_name', player_data.get('name', 'Unknown'))

        bye_week = player_data.get('bye_week')
        bye_week = int(bye_week) if pd.notna(bye_week) else None

        record = PickRecord(
            overall_pick=self.current_pick,
            round=round_num,
            pick_in_round=pick_in_round,
            team_id=team_id,
            player_id=player_id,
            player_name=player_name,
            position=player_data['position'],
            team=player_data.get('team', 'FA'),
            bye_week=bye_week
        )

        # Update draft state
        self.drafted_picks.append(record)
        self._drafted_player_ids.add(player_id)
        self.current_pick += 1
        self.undone_picks.clear()  # Clear redo stack on a new pick branch

        return record

    def _ensure_player_known(self, player_id: str, metadata: Dict, pick_no: int) -> None:
        """
        Appends a synthetic row for a player make_pick doesn't know about -
        e.g. a deep sleeper an opponent drafted on Sleeper that our ADP
        source never covered (the ADP feed only reaches a few hundred
        ranked players, not Sleeper's full ~4000-player pool). ADP is
        anchored to the pick number it was actually taken at, since we have
        no real ranking for this player; projected_points/bye_week are left
        unset so valuation falls back to the ADP-curve estimate.
        """
        if (self.players_df['player_id'] == player_id).any():
            return
        name = f"{metadata.get('first_name', '')} {metadata.get('last_name', '')}".strip() or player_id
        new_row = {
            'player_id': player_id,
            'name': name,
            'player_name': name,
            'position': metadata.get('position', 'FA'),
            'team': metadata.get('team', 'FA'),
            'adp': float(pick_no),
        }
        self.players_df = pd.concat([self.players_df, pd.DataFrame([new_row])], ignore_index=True)

    def sync_from_sleeper_picks(self, sleeper_picks: List[Dict]) -> List[PickRecord]:
        """
        Applies any picks from a live Sleeper draft not yet reflected
        locally. Sleeper's pick_no already matches this session's
        overall_pick numbering 1:1 (both count from 1 in draft order), and
        Sleeper's player_id is the same id space this app already uses
        everywhere (players_df is built from data_pipeline.fetch_sleeper_players),
        so unlike the ADP/projections pipelines, no name-matching is needed -
        picks are just applied in order.

        `sleeper_picks` should be sorted by pick_no ascending
        (fetch_sleeper_draft_picks already returns them that way).
        Already-applied picks are skipped, so this is safe to call
        repeatedly with the full picks list on every poll.
        """
        applied: List[PickRecord] = []
        for pick in sleeper_picks:
            pick_no = pick.get('pick_no')
            if pick_no is None or pick_no <= len(self.drafted_picks):
                continue  # already applied

            player_id = pick.get('player_id')
            if not player_id:
                break  # not finalized on Sleeper's side yet - stop here, retry next poll

            if pick_no != self.current_pick:
                raise ValueError(
                    f"Sleeper sync out of order: expected pick {self.current_pick}, got {pick_no}. "
                    "The local draft may have diverged (a manual pick or undo?) - reset to resync."
                )

            player_id = str(player_id)
            self._ensure_player_known(player_id, pick.get('metadata') or {}, pick_no)
            applied.append(self.make_pick(player_id))

        return applied

    def undo_last_pick(self) -> Optional[PickRecord]:
        """Reverts the last pick made and restores player to the available pool."""
        if not self.drafted_picks:
            return None
        
        last_pick = self.drafted_picks.pop()
        self._drafted_player_ids.remove(last_pick.player_id)
        self.current_pick -= 1
        self.undone_picks.append(last_pick)
        
        return last_pick

    def redo_pick(self) -> Optional[PickRecord]:
        """Re-applies an undone pick if no new branching pick was made."""
        if not self.undone_picks:
            return None
            
        next_pick = self.undone_picks.pop()
        self.drafted_picks.append(next_pick)
        self._drafted_player_ids.add(next_pick.player_id)
        self.current_pick += 1
        
        return next_pick

    def simulate_bot_pick(self, temperature: float = 3.0) -> PickRecord:
        """Calculates and executes a realistic pick for the current non-user team."""
        if self.is_complete:
            raise ValueError("Draft complete, cannot simulate pick.")
        if self.is_user_turn:
            raise ValueError("Cannot simulate bot pick during the user's turn.")

        team_id = self._get_team_at_pick(self.current_pick)
        round_num, _ = self._get_round_and_pick(self.current_pick)
        rounds_remaining = self.rounds - round_num + 1
        roster_counts = self.get_team_roster_counts(team_id)

        # Consider top 15 available players by ADP
        candidates = self.get_available_players().sort_values(by="adp").head(15)
        if candidates.empty:
            raise ValueError("No available players left to draft.")

        candidate_ids = candidates['player_id'].tolist()
        probabilities = []

        for rank_1_idx, (_, row) in enumerate(candidates.iterrows(), start=1):
            adp_score = np.exp(-rank_1_idx / temperature)
            pos = row['position']
            multiplier = 1.0
            
            # Realistic positional constraints
            if pos == "QB":
                if roster_counts.get("QB", 0) >= 2:
                    multiplier = 0.0
                elif roster_counts.get("QB", 0) >= 1 and rounds_remaining > self.BOT_BACKUP_QB_TE_LAST_ROUNDS:
                    multiplier = 0.0
            elif pos == "TE":
                if roster_counts.get("TE", 0) >= 2:
                    multiplier = 0.0
                elif roster_counts.get("TE", 0) >= 1 and rounds_remaining > self.BOT_BACKUP_QB_TE_LAST_ROUNDS:
                    multiplier = 0.0
            elif pos in ["K", "DEF"]:
                if rounds_remaining > self.BOT_STREAMING_LAST_ROUNDS:
                    multiplier = 0.0
                elif roster_counts.get(pos, 0) >= 1:
                    multiplier = 0.0
            elif pos in ["RB", "WR"]:
                if round_num <= 5:
                    multiplier = 1.2
                if roster_counts.get(pos, 0) >= 6:
                    multiplier = 0.5
            
            probabilities.append(adp_score * multiplier)

        sum_probs = sum(probabilities)
        if sum_probs == 0:
            probabilities = [1.0 if idx == 0 else 0.0 for idx in range(len(candidate_ids))]
        else:
            probabilities = [p / sum_probs for p in probabilities]

        chosen_id = np.random.choice(candidate_ids, p=probabilities)
        return self.make_pick(chosen_id)

    def simulate_until_user_turn(self, temperature: float = 3.0) -> List[PickRecord]:
        """Simulates all bot picks sequentially until is_user_turn or is_complete is True."""
        simulated = []
        while not self.is_user_turn and not self.is_complete:
            pick = self.simulate_bot_pick(temperature=temperature)
            simulated.append(pick)
        return simulated

    def get_effective_lookahead_pick(self, user_team_id: int, current_pick: int) -> int:
        """
        Returns target pick number for VONA replacement calculations:
        - If gap between current_pick and next_pick is <= 2 (user at the turn), looks ahead
          to the subsequent cycle's pick.
        - Otherwise returns standard next pick.
        """
        upcoming_picks = self.get_upcoming_user_picks(user_team_id, current_pick, n=2)

        if not upcoming_picks:
            return self.total_picks + 1
            
        next_pick = upcoming_picks[0]
        gap = next_pick - current_pick
        
        if gap <= 2:
            return upcoming_picks[1] if len(upcoming_picks) > 1 else (self.total_picks + 1)
        
        return next_pick


if __name__ == "__main__":
    # Unit verification against acceptance criteria
    print("--- Running Acceptance Verification ---")
    positions = (["RB"] * 20 + ["WR"] * 25 + ["QB"] * 10 + ["TE"] * 5) * 3
    mock_df = pd.DataFrame([
        {"player_id": str(i), "player_name": f"Player {i}", "position": positions[i % len(positions)], "team": "NFL", "adp": float(i)}
        for i in range(1, 201)
    ])

    # Case A: 12-team draft, User at Team 1
    session_1 = DraftSession(players_df=mock_df, user_team_id=1, teams=12, rounds=15, random_seed=42)
    s1 = session_1.get_current_state()
    assert s1["is_user_turn"] is True and s1["user_next_pick"] == 24 and s1["picks_until_user"] == 0, f"Failed Case A1: {s1}"

    session_1.make_pick("1")
    s2 = session_1.get_current_state()
    assert s2["is_user_turn"] is False and s2["team_on_clock"] == 2 and s2["user_next_pick"] == 24 and s2["picks_until_user"] == 22, f"Failed Case A2: {s2}"

    # Case B: 12-team draft, User at Team 12
    session_12 = DraftSession(players_df=mock_df, user_team_id=12, teams=12, rounds=15, random_seed=42)
    s12_1 = session_12.get_current_state()
    assert s12_1["is_user_turn"] is False and s12_1["user_next_pick"] == 12 and s12_1["picks_until_user"] == 11, f"Failed Case B1: {s12_1}"

    session_12.simulate_until_user_turn()
    s12_12 = session_12.get_current_state()
    assert s12_12["is_user_turn"] is True and s12_12["user_next_pick"] == 13 and s12_12["picks_until_user"] == 0, f"Failed Case B2: {s12_12}"

    # Multi-team & Draft Board sanity check
    all_rosters = session_12.get_all_teams_rosters()
    assert len(all_rosters) == 12
    board_df = session_12.get_draft_board_df()
    assert board_df.shape == (15, 12)

    print("All assertions passed successfully!")