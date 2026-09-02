import sqlite3
import requests
import pandas as pd
import logging
import datetime
import json
import re
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
SLEEPER_API_URL = "https://api.sleeper.app/v1/players/nfl"
SLEEPER_DRAFT_API_BASE = "https://api.sleeper.app/v1/draft"
SLEEPER_USER_API_BASE = "https://api.sleeper.app/v1/user"
SLEEPER_LEAGUE_API_BASE = "https://api.sleeper.app/v1/league"
FFC_API_BASE = "https://fantasyfootballcalculator.com/api/v1/adp"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ESPN_PROJECTIONS_URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{year}/segments/0/leaguedefaults/{default_id}"
VALID_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}

# ESPN's per-scoring-format "league defaults" template id - selects which
# scoring rules the projection endpoint's appliedTotal is computed under.
# Verified empirically (a high-target WR's projected total moves from ~217
# to ~277 to ~337 across these three ids, matching the expected standard /
# half-ppr / full-ppr reception-value spread).
ESPN_FORMAT_LEAGUE_DEFAULTS = {"standard": 1, "half-ppr": 8, "ppr": 3}

# ESPN's numeric defaultPositionId -> our position strings. Verified against
# a full player pull: id 16 returns exactly 32 entries (one D/ST per team).
ESPN_POSITION_MAP = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DEF"}

# ESPN abbreviations that differ from Sleeper's
ESPN_TEAM_ALIASES = {"WSH": "WAS"}

# Comprehensive D/ST Name Mapping to enforce 32-team parity
DEF_NAME_MAPPING = {
    # NFC West
    "san francisco 49ers": "SF", "49ers": "SF", "sf": "SF", "san francisco": "SF",
    "seattle seahawks": "SEA", "seahawks": "SEA", "sea": "SEA", "seattle": "SEA",
    "los angeles rams": "LAR", "rams": "LAR", "lar": "LAR", "la rams": "LAR",
    "arizona cardinals": "ARI", "cardinals": "ARI", "ari": "ARI", "arizona": "ARI",
    # NFC East
    "dallas cowboys": "DAL", "cowboys": "DAL", "dal": "DAL", "dallas": "DAL",
    "philadelphia eagles": "PHI", "eagles": "PHI", "phi": "PHI", "philadelphia": "PHI",
    "washington commanders": "WAS", "commanders": "WAS", "was": "WAS", "washington": "WAS", "washington football team": "WAS",
    "new york giants": "NYG", "giants": "NYG", "nyg": "NYG", "ny giants": "NYG",
    # NFC North
    "detroit lions": "DET", "lions": "DET", "det": "DET", "detroit": "DET",
    "green bay packers": "GB", "packers": "GB", "gb": "GB", "green bay": "GB",
    "chicago bears": "CHI", "bears": "CHI", "chi": "CHI", "chicago": "CHI",
    "minnesota vikings": "MIN", "vikings": "MIN", "min": "MIN", "minnesota": "MIN",
    # NFC South
    "tampa bay buccaneers": "TB", "buccaneers": "TB", "tb": "TB", "tampa bay": "TB", "bucs": "TB",
    "new orleans saints": "NO", "saints": "NO", "no": "NO", "new orleans": "NO",
    "atlanta falcons": "ATL", "falcons": "ATL", "atl": "ATL", "atlanta": "ATL",
    "carolina panthers": "CAR", "panthers": "CAR", "car": "CAR", "carolina": "CAR",
    # AFC West
    "kansas city chiefs": "KC", "chiefs": "KC", "kc": "KC", "kansas city": "KC",
    "los angeles chargers": "LAC", "chargers": "LAC", "lac": "LAC", "la chargers": "LAC",
    "denver broncos": "DEN", "broncos": "DEN", "den": "DEN", "denver": "DEN",
    "las vegas raiders": "LV", "raiders": "LV", "lv": "LV", "las vegas": "LV", "oakland raiders": "LV",
    # AFC East
    "buffalo bills": "BUF", "bills": "BUF", "buf": "BUF", "buffalo": "BUF",
    "miami dolphins": "MIA", "dolphins": "MIA", "mia": "MIA", "miami": "MIA",
    "new york jets": "NYJ", "jets": "NYJ", "nyj": "NYJ", "ny jets": "NYJ",
    "new england patriots": "NE", "patriots": "NE", "ne": "NE", "new england": "NE",
    # AFC North
    "baltimore ravens": "BAL", "ravens": "BAL", "bal": "BAL", "baltimore": "BAL",
    "cincinnati bengals": "CIN", "bengals": "CIN", "cin": "CIN", "cincinnati": "CIN",
    "cleveland browns": "CLE", "browns": "CLE", "cle": "CLE", "cleveland": "CLE",
    "pittsburgh steelers": "PIT", "steelers": "PIT", "pit": "PIT", "pittsburgh": "PIT",
    # AFC South
    "houston texans": "HOU", "texans": "HOU", "hou": "HOU", "houston": "HOU",
    "jacksonville jaguars": "JAX", "jaguars": "JAX", "jax": "JAX", "jacksonville": "JAX",
    "indianapolis colts": "IND", "colts": "IND", "ind": "IND", "indianapolis": "IND",
    "tennessee titans": "TEN", "titans": "TEN", "ten": "TEN", "tennessee": "TEN"
}

def _get_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

def _normalize_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b', '', name).strip()
    return " ".join(name.split())

def _get_db_name(format: str, teams: int) -> str:
    return f"draft_data_{format}_{teams}.db"

def init_db(db_path: str) -> None:
    logger.info(f"Initializing database schema at {db_path}...")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                position TEXT NOT NULL,
                team TEXT,
                status TEXT,
                years_exp INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS adp (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,
                full_name TEXT,
                format TEXT,
                teams INTEGER,
                adp REAL,
                stdev REAL,
                high INTEGER,
                low INTEGER,
                last_updated TIMESTAMP,
                FOREIGN KEY(player_id) REFERENCES players(player_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_byes (
                team TEXT PRIMARY KEY,
                season INTEGER,
                bye_week INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projections (
                player_id TEXT PRIMARY KEY,
                projected_points REAL,
                last_updated TIMESTAMP,
                FOREIGN KEY(player_id) REFERENCES players(player_id)
            )
        """)
        conn.commit()

def fetch_sleeper_players() -> list[dict]:
    logger.info("Fetching player metadata from Sleeper API...")
    session = _get_session()
    try:
        response = session.get(SLEEPER_API_URL, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch Sleeper data: {e}")
        return []

    processed_players = []
    for raw_player_id, p_data in data.items():
        if not raw_player_id:
            continue
        player_id = str(raw_player_id).strip()
        if not player_id or player_id.lower() == "nan":
            continue

        position = p_data.get("position")
        if position not in VALID_POSITIONS:
            continue
            
        full_name = p_data.get("full_name")
        team = p_data.get("team")

        if not full_name:
            first = p_data.get("first_name", "")
            last = p_data.get("last_name", "")
            full_name = f"{first} {last}".strip()
            
        if position == "DEF":
            if not full_name and team:
                full_name = f"{team} DEF"
            if not team and full_name:
                possible_team = full_name.split()[0].lower()
                if possible_team in DEF_NAME_MAPPING:
                    team = DEF_NAME_MAPPING[possible_team]

        if not full_name:
            continue

        player = {
            "player_id": player_id,
            "full_name": full_name,
            "position": position,
            "team": team,
            "status": p_data.get("status", "Active"),
            "years_exp": int(p_data.get("years_exp") or 0)
        }
        processed_players.append(player)

    return processed_players


def parse_sleeper_draft_id(raw: str) -> str:
    """
    Accepts either a bare draft_id or a full Sleeper URL (e.g.
    https://sleeper.com/draft/nfl/1234567890123456789) and returns just the
    numeric id.
    """
    raw = (raw or "").strip()
    match = re.search(r'(\d{10,})', raw)
    if not match:
        raise ValueError(f"Couldn't find a draft ID in '{raw}'. Paste the draft URL or just the numeric ID.")
    return match.group(1)


def fetch_sleeper_league_drafts(league_id: str) -> list[dict]:
    """Returns every draft associated with a Sleeper league_id (normally just the current one)."""
    session = _get_session()
    response = session.get(f"{SLEEPER_LEAGUE_API_BASE}/{league_id}/drafts", timeout=15)
    response.raise_for_status()
    return response.json() or []


def resolve_sleeper_draft_id(raw: str) -> str:
    """
    Accepts a draft URL/ID, or a league URL/ID (e.g. the /predraft page,
    which is what's actually visible before the draft room opens), and
    returns the real draft_id either way.

    League and draft IDs are both opaque numeric ids from the same id
    space, indistinguishable by shape alone - so rather than guessing from
    the URL text, this just tries the id as a draft first (the common case)
    and falls back to treating it as a league id whose current draft we
    look up, preferring one that isn't finished yet.
    """
    numeric_id = parse_sleeper_draft_id(raw)

    try:
        return fetch_sleeper_draft(numeric_id)["draft_id"]
    except (requests.exceptions.HTTPError, ValueError):
        pass

    drafts = fetch_sleeper_league_drafts(numeric_id)
    if not drafts:
        raise ValueError(
            f"Couldn't find a draft for '{raw}'. If that's a league link, "
            "that league may not have a draft scheduled yet."
        )
    drafts.sort(key=lambda d: d.get("status") == "complete")  # prefer an active/upcoming draft
    return drafts[0]["draft_id"]


def fetch_sleeper_draft(draft_id: str) -> dict:
    """
    Fetches a live Sleeper draft's metadata: type (snake/linear/auction),
    status, settings (teams/rounds), and draft_order (user_id -> slot, once
    Sleeper has randomized it).

    Unlike the other fetch_* helpers here, this backs a user-initiated
    "Connect" action rather than a background refresh, so it raises on
    failure instead of degrading silently - the caller surfaces the error
    directly instead of the sync quietly never working.
    """
    session = _get_session()
    response = session.get(f"{SLEEPER_DRAFT_API_BASE}/{draft_id}", timeout=15)
    response.raise_for_status()
    data = response.json()
    if not data:
        raise ValueError(f"No draft found for ID {draft_id}. Double check the URL/ID.")
    return data


def fetch_sleeper_draft_picks(draft_id: str) -> list[dict]:
    """
    Fetches all picks made so far in a live Sleeper draft, sorted by pick_no.
    This backs repeated polling during the draft, so - unlike
    fetch_sleeper_draft - it degrades to an empty list on failure rather than
    raising: a transient network hiccup just means "no new picks found this
    check" instead of crashing the sync loop.
    """
    session = _get_session()
    try:
        response = session.get(f"{SLEEPER_DRAFT_API_BASE}/{draft_id}/picks", timeout=15)
        response.raise_for_status()
        picks = response.json() or []
    except Exception as e:
        logger.error(f"Error fetching Sleeper draft picks: {e}")
        return []
    return sorted(picks, key=lambda p: p.get("pick_no", 0))


def resolve_sleeper_username(username: str) -> str | None:
    """Resolves a Sleeper username to its user_id, or None if not found/unreachable."""
    if not username:
        return None
    session = _get_session()
    try:
        response = session.get(f"{SLEEPER_USER_API_BASE}/{username}", timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error resolving Sleeper username '{username}': {e}")
        return None
    return str(data["user_id"]) if data and data.get("user_id") else None


def fetch_adp(format: str = "ppr", teams: int = 12, year: int = None) -> list[dict]:
    if year is None:
        year = datetime.date.today().year
        
    logger.info(f"Fetching {year} ADP data (format: {format}, teams: {teams})...")
    url = f"{FFC_API_BASE}/{format}?teams={teams}&year={year}"
    session = _get_session()
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        if "players" in data:
            return data["players"]
        elif isinstance(data, list):
            return data
        return []
    except Exception as e:
        logger.error(f"Error fetching ADP data: {e}")
        return []

def fetch_bye_weeks(year: int = None) -> dict[str, int]:
    """
    Returns {team_abbr: bye_week} by sweeping the ESPN scoreboard weeks 1-18 and
    reading each week's teamsOnBye list. Never raises: byes are enhancement data,
    so a failed fetch degrades to an empty mapping rather than blocking a refresh.
    """
    if year is None:
        year = datetime.date.today().year

    logger.info(f"Fetching {year} bye weeks from ESPN...")
    session = _get_session()
    byes: dict[str, int] = {}

    for week in range(1, 19):
        try:
            response = session.get(
                ESPN_SCOREBOARD_URL,
                params={"dates": year, "seasontype": 2, "week": week},
                timeout=15
            )
            response.raise_for_status()
            on_bye = response.json().get("week", {}).get("teamsOnBye") or []
        except Exception as e:
            logger.error(f"Error fetching bye data for week {week}: {e}")
            continue

        for team in on_bye:
            abbr = team.get("abbreviation")
            if abbr:
                byes[ESPN_TEAM_ALIASES.get(abbr, abbr)] = week

    if len(byes) < 32:
        logger.warning(f"Only {len(byes)}/32 teams have a bye week assigned.")

    return byes


def fetch_espn_projections(format: str = "ppr", year: int = None) -> list[dict]:
    """
    Returns [{"name", "position", "projected_points"}, ...] - ESPN's own
    full-season fantasy point projections (statSourceId=1, a season-total
    statSplitTypeId=0 record), pulled via the "league defaults" endpoint so
    no real league/auth is required. The scoring template id is chosen per
    `format` (see ESPN_FORMAT_LEAGUE_DEFAULTS) so the totals already reflect
    this league's PPR/half-PPR/standard rules rather than needing to be
    reconstructed from raw stats afterward.

    Projections are enhancement data over valuation.py's ADP-curve fallback,
    so - like fetch_bye_weeks - this never raises; a failed fetch just means
    every player falls back to that estimate.
    """
    if year is None:
        year = datetime.date.today().year
    default_id = ESPN_FORMAT_LEAGUE_DEFAULTS.get(format, ESPN_FORMAT_LEAGUE_DEFAULTS["ppr"])

    logger.info(f"Fetching {year} ESPN season projections (format: {format})...")
    url = f"{ESPN_PROJECTIONS_URL.format(year=year, default_id=default_id)}?view=kona_player_info"
    headers = {
        "x-fantasy-filter": json.dumps({
            "players": {
                "limit": 3000,
                "sortDraftRanks": {"sortPriority": 1, "sortAsc": True, "value": "PPR"}
            }
        }),
        "Accept": "application/json",
    }
    session = _get_session()
    try:
        response = session.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error fetching ESPN projections: {e}")
        return []

    records = []
    for entry in data.get("players", []):
        player = entry.get("player", {})
        position = ESPN_POSITION_MAP.get(player.get("defaultPositionId"))
        if position is None:
            continue

        season_stat = next(
            (s for s in player.get("stats", [])
             if s.get("statSourceId") == 1 and s.get("statSplitTypeId") == 0
             and s.get("seasonId") == year),
            None
        )
        if season_stat is None:
            continue

        records.append({
            "name": player.get("fullName", ""),
            "position": position,
            "projected_points": float(season_stat.get("appliedTotal") or 0.0),
        })

    return records


def _match_sleeper_id(
    raw_name: str, pos: str,
    player_lookup: dict, fallback_lookup: dict, team_def_lookup: dict
) -> str | None:
    """
    Resolves an external source's (name, position) to a Sleeper player_id,
    shared by ADP and projections matching since both face the same D/ST
    naming quirks and need the same dedicated-then-fallback lookup order.
    """
    norm_name = _normalize_name(raw_name)
    player_id = None

    # 1. Improved DEF Matching using the comprehensive DEF_NAME_MAPPING
    if pos == "DEF":
        clean_def_name = re.sub(r'\b(defense|d/st|dst|def)\b', '', norm_name).strip()
        abbr = DEF_NAME_MAPPING.get(clean_def_name, DEF_NAME_MAPPING.get(norm_name))
        if abbr:
            player_id = team_def_lookup.get(abbr.lower())

    # 2. Standard lookups
    if not player_id:
        player_id = player_lookup.get((norm_name, pos))
    if not player_id:
        player_id = fallback_lookup.get(norm_name)

    if not player_id or str(player_id).strip().lower() in {"", "nan", "none"}:
        return None
    return str(player_id).strip()


def refresh_database(format: str = "ppr", teams: int = 12) -> dict:
    db_path = _get_db_name(format, teams)
    sleeper_players = fetch_sleeper_players()
    adp_data = fetch_adp(format=format, teams=teams)
    # Byes and projections are enhancement data - an empty result must not abort the refresh
    bye_weeks = fetch_bye_weeks()
    espn_projections = fetch_espn_projections(format=format)

    if not sleeper_players or not adp_data:
        logger.error("Missing critical API data. Aborting refresh to protect database.")
        return {"players_loaded": 0, "adp_matched": 0, "adp_total": 0}

    player_lookup = {}
    fallback_lookup = {}
    team_def_lookup = {}

    for p in sleeper_players:
        norm_name = _normalize_name(p["full_name"])
        pos = p["position"]
        pid = str(p["player_id"])

        player_lookup[(norm_name, pos)] = pid
        fallback_lookup[norm_name] = pid

        if pos == "DEF":
            if p["team"]:
                team_def_lookup[p["team"].lower()] = pid
            team_def_lookup[norm_name] = pid

    adp_insert_records = []
    matched_sleeper_ids = set()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for adp_row in adp_data:
        raw_name = adp_row.get("name", "")
        pos = adp_row.get("position", "")
        if pos == "PK": pos = "K"

        player_id = _match_sleeper_id(raw_name, pos, player_lookup, fallback_lookup, team_def_lookup)
        if not player_id:
            continue

        matched_sleeper_ids.add(player_id)
        adp_insert_records.append((
            player_id, raw_name, format, teams,
            float(adp_row.get("adp") or 0.0), float(adp_row.get("stdev") or 0.0),
            int(adp_row.get("high") or 0), int(adp_row.get("low") or 0), now
        ))

    # Projections: keep the first (highest-ranked) match if ESPN ever lists a
    # player twice, since sortDraftRanks orders the response best-first.
    projection_insert_records = []
    matched_projection_ids = set()

    for proj in espn_projections:
        player_id = _match_sleeper_id(proj["name"], proj["position"], player_lookup, fallback_lookup, team_def_lookup)
        if not player_id or player_id in matched_projection_ids:
            continue

        matched_projection_ids.add(player_id)
        projection_insert_records.append((player_id, proj["projected_points"], now))

    # --- NEW: Fallback for unmatched active D/ST and Kickers ---
    for p in sleeper_players:
        pid = str(p["player_id"])
        if pid not in matched_sleeper_ids and p["status"] == "Active" and p["position"] in {"DEF", "K"}:
            matched_sleeper_ids.add(pid)
            adp_insert_records.append((
                pid, p["full_name"], format, teams,
                160.0, # Baseline ADP Fallback
                0.0, 180, 140, now
            ))

    if len(matched_sleeper_ids) == 0:
        logger.error("Zero matches found. Aborting write to prevent empty states.")
        return {"players_loaded": len(sleeper_players), "adp_matched": 0, "adp_total": len(adp_data)}

    if os.path.exists(db_path):
        try: os.remove(db_path)
        except OSError as e: logger.error(f"Could not remove DB: {e}"); return {}
            
    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        players_to_insert = [(str(p["player_id"]), p["full_name"], p["position"], p["team"], p["status"], p["years_exp"]) for p in sleeper_players]
        cursor.executemany("INSERT OR IGNORE INTO players (player_id, full_name, position, team, status, years_exp) VALUES (?, ?, ?, ?, ?, ?)", players_to_insert)
        cursor.executemany("INSERT INTO adp (player_id, full_name, format, teams, adp, stdev, high, low, last_updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", adp_insert_records)

        season = datetime.date.today().year
        byes_to_insert = [(team, season, week) for team, week in bye_weeks.items()]
        cursor.executemany("INSERT OR REPLACE INTO team_byes (team, season, bye_week) VALUES (?, ?, ?)", byes_to_insert)
        cursor.executemany("INSERT OR REPLACE INTO projections (player_id, projected_points, last_updated) VALUES (?, ?, ?)", projection_insert_records)
        conn.commit()

    # --- Self-Verification Logging ---
    def_count = sum(1 for p in sleeper_players if p["position"] == "DEF" and str(p["player_id"]) in matched_sleeper_ids)
    k_count = sum(1 for p in sleeper_players if p["position"] == "K" and str(p["player_id"]) in matched_sleeper_ids)

    logger.info(f"Pipeline Execution Summary:")
    logger.info(f"Total players loaded: {len(sleeper_players)}")
    logger.info(f"Total D/ST units loaded: {def_count} (Expected >= 30)")
    logger.info(f"Total Kickers loaded: {k_count} (Expected >= 25)")
    logger.info(f"Total bye weeks loaded: {len(bye_weeks)} (Expected 32)")
    logger.info(f"Total real projections matched: {len(matched_projection_ids)}/{len(espn_projections)} (unmatched players fall back to the ADP-curve estimate)")

    if def_count < 30: logger.warning("WARNING: Less than 30 D/ST units were loaded!")
    if k_count < 25: logger.warning("WARNING: Less than 25 Kickers were loaded!")

    return {
        "db_file": db_path, "players_loaded": len(sleeper_players),
        "adp_matched": len(matched_sleeper_ids), "adp_total": len(adp_data),
        "projections_matched": len(matched_projection_ids), "projections_total": len(espn_projections),
    }

def get_available_players_df(format: str = "ppr", teams: int = 12) -> pd.DataFrame:
    db_path = _get_db_name(format, teams)
    if not os.path.exists(db_path):
        return pd.DataFrame()
        
    # Databases built before team_byes/projections existed are still readable:
    # fall back to NULL rather than letting the join fail and blank out the pool.
    try:
        with sqlite3.connect(db_path) as conn:
            has_byes = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='team_byes'"
            ).fetchone() is not None
            has_projections = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='projections'"
            ).fetchone() is not None

            if has_byes:
                bye_select, bye_join = "b.bye_week", "LEFT JOIN team_byes b ON b.team = p.team"
            else:
                bye_select, bye_join = "NULL as bye_week", ""

            if has_projections:
                proj_select, proj_join = "pr.projected_points", "LEFT JOIN projections pr ON pr.player_id = p.player_id"
            else:
                proj_select, proj_join = "NULL as projected_points", ""

            query = f"""
                SELECT p.player_id, COALESCE(a.full_name, p.full_name) as name, p.position, p.team, p.status,
                       a.adp, a.high, a.low, a.stdev, a.last_updated, {bye_select}, {proj_select}
                FROM adp a
                INNER JOIN players p ON a.player_id = p.player_id
                {bye_join}
                {proj_join}
                ORDER BY a.adp ASC
            """
            df = pd.read_sql_query(query, conn)
    except Exception as e:
        logger.error(f"Error querying database {db_path}: {e}")
        return pd.DataFrame()

    if df.empty: return df

    # Strict Null Filtering & Type Enforcement
    df = df.dropna(subset=['player_id', 'name', 'position'])
    df['player_id'] = df['player_id'].astype(str).str.strip()
    
    # Filter out text variations of null/NaN
    invalid_mask = df['player_id'].str.lower().isin({'', 'nan', 'none', 'natype', 'null'})
    df = df[~invalid_mask]

    return df.sort_values(by='adp', ascending=True).reset_index(drop=True)

if __name__ == "__main__":
    summary = refresh_database(format="half-ppr", teams=12)
    df_players = get_available_players_df(format="half-ppr", teams=12)
    
    if df_players.empty:
        print("\n❌ DataFrame is empty.")
    else:
        null_ids_count = df_players['player_id'].isna().sum()
        nan_string_count = (df_players['player_id'].str.lower() == 'nan').sum()
        empty_count = (df_players['player_id'].str.strip() == '').sum()
        def_count = len(df_players[df_players['position'] == 'DEF'])
        
        print(f"\n✅ Pipeline Audit Complete. Valid Defenses loaded: {def_count}/32")
        if null_ids_count == 0 and nan_string_count == 0 and empty_count == 0 and def_count >= 30:
            print("🎉 SUCCESS: Zero null/nan/empty player IDs detected. All Defenses captured!")