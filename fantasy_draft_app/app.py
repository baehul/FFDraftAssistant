import streamlit as st
import pandas as pd
import datetime
import os

# ---------------------------------------------------------
# 1. PAGE CONFIG MUST BE THE VERY FIRST COMMAND
# ---------------------------------------------------------
st.set_page_config(page_title="Draft War Room", page_icon="🏈", layout="wide")

# ---------------------------------------------------------
# 2. Modular Imports
# ---------------------------------------------------------
try:
    import data_pipeline as dp
    import valuation as val
    import draft_manager as dm
    import roster_projection as rp
except ImportError:
    st.error("⚠️ Ensure data_pipeline.py, valuation.py, draft_manager.py, and roster_projection.py are in the same directory.")
    st.stop()


# ---------------------------------------------------------
# Callbacks (State Management)
# ---------------------------------------------------------
def init_draft():
    """Initializes or resets the draft session and engine."""
    try:
        # Fetch base data
        df = dp.get_available_players_df(
            st.session_state.format_choice, 
            st.session_state.team_count
        )
        
        # Setup Settings & Valuation Engine
        roster_slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1, "BENCH": 6}
        settings = val.LeagueSettings(
            format=st.session_state.format_choice,
            teams=st.session_state.team_count,
            pass_td_pts=st.session_state.pass_td_pts,
            roster_slots=roster_slots
        )
        
        st.session_state.league_settings = settings
        st.session_state.valuation_engine = val.ValuationEngine(settings)
        
        # Setup Draft Session
        st.session_state.draft_session = dm.DraftSession(
            players_df=df,
            user_team_id=st.session_state.user_pos,
            teams=st.session_state.team_count,
            rounds=st.session_state.total_rounds,
            format=st.session_state.format_choice
        )
        st.session_state.draft_active = True
        
    except Exception as e:
        st.error(f"Failed to start draft. Do you need to Refresh / Download data first? Error: {e}")

def handle_draft_player(player_id):
    st.session_state.draft_session.make_pick(player_id)

def handle_undo():
    st.session_state.draft_session.undo_last_pick()

def handle_redo():
    st.session_state.draft_session.redo_pick()

def handle_simulate_pick(temp):
    st.session_state.draft_session.simulate_bot_pick(temperature=temp)

def handle_fast_forward(temp):
    st.session_state.draft_session.simulate_until_user_turn(temperature=temp)

def connect_sleeper_draft():
    """
    Links this session to a live Sleeper draft: pulls its team count/rounds
    so local pick numbering lines up with Sleeper's, and (if a username was
    given) auto-detects the user's own draft slot.
    """
    raw = st.session_state.get("sleeper_draft_input", "")
    try:
        draft_id = dp.resolve_sleeper_draft_id(raw)
        meta = dp.fetch_sleeper_draft(draft_id)
    except Exception as e:
        st.session_state.sleeper_connect_error = f"Couldn't connect: {e}"
        return

    warnings = []
    st.session_state.sleeper_draft_id = draft_id
    st.session_state.sleeper_draft_type = meta.get("type", "snake")

    settings = meta.get("settings") or {}
    if settings.get("teams"):
        st.session_state.team_count = int(settings["teams"])
    if settings.get("rounds"):
        st.session_state.total_rounds = int(settings["rounds"])

    username = st.session_state.get("sleeper_username_input", "").strip()
    if username:
        user_id = dp.resolve_sleeper_username(username)
        slot = (meta.get("draft_order") or {}).get(user_id) if user_id else None
        if slot:
            st.session_state.user_pos = int(slot)
        else:
            warnings.append(
                "Couldn't find your draft slot yet (draft order may not be randomized). "
                "Set 'Your Draft Position' manually below."
            )

    # A draft session started *before* connecting was built with whatever
    # team/round settings were on screen at the time - it won't retroactively
    # pick up Sleeper's real numbers, and a mismatch here would misattribute
    # every pick's team for the rest of the draft. Flag it instead of
    # silently wiping an in-progress session the user may not want reset.
    if st.session_state.get("draft_active"):
        live = st.session_state.draft_session
        if live.teams != st.session_state.team_count or live.rounds != st.session_state.total_rounds:
            warnings.append(
                f"Your active draft is set up for {live.teams} teams / {live.rounds} rounds, but "
                f"Sleeper reports {st.session_state.team_count}/{st.session_state.total_rounds}. "
                "Click **Start / Reset Draft** again before syncing, or picks will be attributed to the wrong team."
            )

    st.session_state.sleeper_connect_error = " ".join(warnings) if warnings else None
    st.session_state.draft_mode = "Real Draft"

def disconnect_sleeper_draft():
    st.session_state.sleeper_draft_id = None
    st.session_state.sleeper_connect_error = None

def sync_sleeper_picks() -> int:
    """Pulls the latest picks from the connected Sleeper draft and applies any new ones locally."""
    draft_id = st.session_state.get("sleeper_draft_id")
    if not draft_id or not st.session_state.get("draft_active"):
        return 0
    picks = dp.fetch_sleeper_draft_picks(draft_id)
    try:
        applied = st.session_state.draft_session.sync_from_sleeper_picks(picks)
    except ValueError as e:
        st.session_state.sleeper_connect_error = str(e)
        return 0
    st.session_state.sleeper_last_sync = datetime.datetime.now().strftime("%H:%M:%S")
    return len(applied)


# ---------------------------------------------------------
# Draft Summary
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_league_projections(_draft, _engine, roster_slots_items, weeks, signature):
    """
    Projects every team. The draft session and engine are passed with a leading
    underscore so Streamlit skips hashing them; `signature` (the drafted player
    ids) and `weeks` are what actually key the cache.
    """
    return rp.project_all_teams(_draft, _engine, dict(roster_slots_items), list(weeks))


def render_draft_summary(draft, engine, roster_slots, user_team_id):
    """Renders the post-draft summary: starters vs. bye/injury-adjusted team value."""
    st.subheader("📊 Draft Summary")

    window = st.radio(
        "Projection window",
        ["Full regular season (Wk 1-17)", "Fantasy regular season (Wk 1-14)"],
        horizontal=True,
        key="summary_window"
    )
    weeks = tuple(range(1, 15)) if "1-14" in window else tuple(range(1, 18))

    signature = tuple(p.player_id for p in draft.drafted_picks)
    projections = compute_league_projections(
        draft, engine, tuple(sorted(roster_slots.items())), weeks, signature
    )

    mine = next((p for p in projections if p.team_id == user_team_id), None)
    if mine is None or mine.lineup.empty:
        st.info("Your roster is empty - nothing to summarize yet.")
        return

    if mine.per_player['bye_week'].isna().all():
        st.warning(
            "⚠️ Bye week data unavailable - click **🔄 Refresh / Download Data** in the sidebar. "
            "Projections below currently reflect injury risk only."
        )

    # -- Headline numbers
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "🟢 Starters (Ceiling)", f"{mine.ceiling:,.0f}",
        help="Your best lineup if nobody ever missed a game."
    )
    m2.metric(
        "🏈 Team Projection", f"{mine.team_projection:,.0f}",
        delta=f"{-mine.absence_cost:,.0f} vs ceiling",
        help="Expected points once byes and injury risk are priced in and your bench fills the holes."
    )
    m3.metric(
        "🛟 Bench Insurance", f"+{mine.bench_insurance:,.0f}",
        help="What your backups are worth: Team Projection minus what you'd score if holes went unfilled."
    )
    m4.metric(
        "🩹 Absence Cost", f"-{mine.absence_cost:,.0f}",
        help="Points lost to byes and injuries that even your bench cannot recover."
    )

    st.caption(
        f"Starters project for **{mine.ceiling:,.0f}** points across {len(weeks)} weeks. "
        f"Accounting for byes, injury risk, and backups stepping in, your team projects for "
        f"**{mine.team_projection:,.0f}** - your bench recovers **{mine.bench_insurance:,.0f}** "
        f"of the **{mine.ceiling - mine.no_bench_floor:,.0f}** points those absences would otherwise cost."
    )

    st.divider()

    # -- Starting lineup & weekly outlook
    left, right = st.columns([1, 1])

    with left:
        st.markdown("##### 🥇 Optimal Starting Lineup")
        lineup_view = mine.lineup.rename(columns={
            'slot': 'Slot', 'name': 'Player', 'position': 'Pos',
            'team': 'NFL', 'bye_week': 'Bye', 'projected_points': 'Proj Pts'
        })
        lineup_view['Proj Pts'] = lineup_view['Proj Pts'].round(1)
        st.dataframe(lineup_view, use_container_width=True, hide_index=True)

    with right:
        st.markdown("##### 📅 Week-by-Week Outlook")
        weekly_view = mine.weekly.rename(columns={
            'week': 'Wk', 'expected_points': 'Exp Pts',
            'on_bye': 'On Bye', 'expected_holes': 'Exp Holes'
        })
        weekly_view['Exp Pts'] = weekly_view['Exp Pts'].round(1)
        weekly_view['Exp Holes'] = weekly_view['Exp Holes'].round(2)
        st.dataframe(
            weekly_view,
            use_container_width=True, hide_index=True, height=380,
            column_config={
                "Exp Pts": st.column_config.ProgressColumn(
                    "Exp Pts",
                    help="Expected starting lineup points that week",
                    format="%.1f",
                    min_value=0,
                    max_value=float(weekly_view['Exp Pts'].max()) if not weekly_view.empty else 1.0,
                )
            }
        )

    # -- Per-player contribution
    st.markdown("##### 👥 Player Contribution")
    player_view = mine.per_player[[
        'name', 'position', 'team', 'bye_week', 'role',
        'projected_points', 'exp_games_started', 'exp_points'
    ]].rename(columns={
        'name': 'Player', 'position': 'Pos', 'team': 'NFL', 'bye_week': 'Bye',
        'role': 'Role', 'projected_points': 'Season Proj',
        'exp_games_started': 'Exp Starts', 'exp_points': 'Exp Pts Contributed'
    })
    for col in ('Season Proj', 'Exp Starts', 'Exp Pts Contributed'):
        player_view[col] = player_view[col].round(1)
    st.dataframe(player_view, use_container_width=True, hide_index=True)

    st.divider()

    # -- League leaderboard
    st.markdown("##### 🏆 League Leaderboard")
    board = pd.DataFrame([{
        'Rank': rank,
        'Team': f"Team {p.team_id}" + (" (You)" if p.team_id == user_team_id else ""),
        'Team Projection': round(p.team_projection, 1),
        'Starters (Ceiling)': round(p.ceiling, 1),
        'Bench Insurance': round(p.bench_insurance, 1),
        'Absence Cost': round(p.absence_cost, 1),
    } for rank, p in enumerate(projections, start=1)])

    st.dataframe(
        board.style.apply(
            lambda row: ['background-color: rgba(255, 215, 0, 0.25)' if "(You)" in row['Team'] else '' for _ in row],
            axis=1
        ),
        use_container_width=True, hide_index=True
    )

    my_rank = next((i for i, p in enumerate(projections, start=1) if p.team_id == user_team_id), None)
    if my_rank:
        st.success(f"Your team ranks **#{my_rank} of {len(projections)}** by bye/injury-adjusted projection.")

    with st.expander("ℹ️ Model assumptions"):
        st.markdown(f"""
Season projections are converted to a per-game rate (season total ÷ {rp.GAMES_PER_SEASON}) and the
season is walked one week at a time. Each week, the optimal startable lineup is computed
**exactly** over every availability outcome, so bench players slide up whenever a starter is out.

- **Bye week** — a certain absence for that player that week.
- **Injury risk** — a baseline per-game probability of missing a game, by position:

{" · ".join(f"`{pos}` {rate:.0%}" for pos, rate in rp.POSITION_MISS_RATE.items())}

Absences are treated as independent across players and weeks. Current real-world injury
designations are **not** factored in — a player on IR today is projected as healthy.
""")


@st.fragment(run_every="20s")
def render_sleeper_sync_bar():
    """
    Auto-polls the connected Sleeper draft every 20s and applies any new
    picks - only this fragment reruns on the timer, not the whole page, so
    it's cheap to leave running. Phone browsers pause background-tab timers
    once the screen locks or another app takes focus, so the auto-refresh
    can silently stop there; the manual button (any click inside a fragment
    reruns just that fragment) is the reliable fallback to catch back up.

    If sync ever breaks (bad network at the venue, Sleeper API hiccup, an
    out-of-order pick this session can't reconcile), the error and a
    one-tap way back to fully manual entry live right here in the main body -
    not just in the sidebar, which is collapsed by default on a phone and
    easy to miss mid-draft.
    """
    applied = sync_sleeper_picks()
    error = st.session_state.get("sleeper_connect_error")

    c1, c2, c3 = st.columns([3, 1, 1.6])
    with c1:
        last = st.session_state.get("sleeper_last_sync")
        status = f"🔗 Synced to Sleeper draft `{st.session_state.sleeper_draft_id}`"
        st.caption(status + (f" · last checked {last}" if last else ""))
    with c2:
        st.button("🔄 Sync Now", key="manual_sync_btn", width="stretch")
    with c3:
        switch_to_manual = st.button("🔓 Switch to Manual", key="manual_fallback_btn", width="stretch")

    if error:
        st.error(
            f"⚠️ Sync issue: {error}\n\nSleeper itself is unaffected - keep drafting there. "
            "Tap **Switch to Manual** to enter picks here yourself instead."
        )

    if switch_to_manual:
        disconnect_sleeper_draft()
        st.rerun()  # escapes the fragment - the whole page needs to reflect manual mode now

    if applied:
        st.rerun()


# ---------------------------------------------------------
# Sidebar: Configuration & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("🏆 League Setup")
    st.selectbox("Scoring Format", ["ppr", "half-ppr", "standard"], key="format_choice")
    st.slider("Number of Teams", min_value=8, max_value=16, value=12, key="team_count")
    st.slider("Total Rounds", min_value=10, max_value=20, value=15, key="total_rounds")
    
    user_max = st.session_state.get("team_count", 12)
    st.slider("Your Draft Position (Team ID)", min_value=1, max_value=user_max, value=1, key="user_pos")
    st.number_input("Passing TD Points", min_value=0.0, max_value=6.0, value=4.0, step=0.5, key="pass_td_pts")

    st.divider()

    st.header("⚙️ Data Management")
    if st.button("🔄 Refresh / Download Data", width="stretch"):
        with st.spinner(f"Fetching {st.session_state.format_choice} data for {st.session_state.team_count} teams..."):
            try:
                dp.refresh_database(
                    format=st.session_state.format_choice, 
                    teams=st.session_state.team_count
                ) 
                st.success("✅ Data refreshed successfully!")
            except Exception as e:
                st.error(f"Error refreshing data: {e}")
                
    st.divider()

    st.header("🔗 Live Sync (Sleeper)")
    if st.session_state.get("sleeper_draft_id"):
        st.success(f"Connected: draft `{st.session_state.sleeper_draft_id}`")
        if st.session_state.get("sleeper_draft_type") not in (None, "snake"):
            st.warning("⚠️ This isn't a snake draft - pick order/team tracking may not line up.")
        st.button("Disconnect", on_click=disconnect_sleeper_draft, width="stretch")
    else:
        st.text_input(
            "Sleeper Draft or League URL/ID", key="sleeper_draft_input",
            placeholder="https://sleeper.com/leagues/.../predraft",
            help="Either your league link (e.g. the predraft page) or a direct draft link works."
        )
        st.text_input(
            "Your Sleeper Username (optional)", key="sleeper_username_input",
            help="Auto-detects your draft slot below. Leave blank to set it manually."
        )
        st.button("🔗 Connect", on_click=connect_sleeper_draft, width="stretch")

    if st.session_state.get("sleeper_connect_error"):
        st.error(st.session_state.sleeper_connect_error)

    st.divider()
    st.header("🎮 Draft Mode")
    st.radio("Mode", ["Practice (Mock Draft)", "Real Draft"], key="draft_mode")

    st.button("🚀 Start / Reset Draft", on_click=init_draft, type="primary", width="stretch")


# ---------------------------------------------------------
# Main Application Body
# ---------------------------------------------------------
if not st.session_state.get("draft_active", False):
    st.title("🏈 Draft War Room")
    st.info("👈 Please configure your league settings and click **Start / Reset Draft** in the sidebar to begin.")
    st.stop()

# -- Retrieve Current Draft State
draft = st.session_state.draft_session
engine = st.session_state.valuation_engine
state = draft.get_current_state()
synced = bool(st.session_state.get("sleeper_draft_id"))

# Safely extract pick information (protecting against different dictionary keys)
current_round = state.get('round', 1)
current_pick_in_round = state.get('current_pick', state.get('pick', 1))
overall_pick = state.get("overall_pick", state.get("pick_number", 1))
is_complete = state.get("is_complete", False)

# ---------------------------------------------------------
# Top Banner: Draft Status
# ---------------------------------------------------------
st.title("🏈 Draft War Room")

col1, col2, col3, col4, col5 = st.columns([1.5, 1.5, 2.5, 1.5, 2])
with col1:
    st.metric("Current Pick", f"{current_round}.{current_pick_in_round}")
with col2:
    st.metric("Overall", overall_pick)
with col3:
    if is_complete:
        st.success("🏁 **DRAFT COMPLETE** 🏁")
    elif state.get("is_user_turn", False):
        st.error("🚨 **YOUR TURN - YOU ARE ON THE CLOCK** 🚨")
    else:
        st.info(f"⏳ Team {state.get('team_on_clock', 'N/A')} is on the clock")
with col4:
    st.metric("Picks Until You", state.get("picks_until_user", "N/A"))
with col5:
    st.write("Actions")
    c_undo, c_redo = st.columns(2)
    # Disabled while synced: a live pick can't really be "undone" - it's
    # still on Sleeper's own record, so the next poll would just reapply it.
    c_undo.button("↩️ Undo", on_click=handle_undo, width="stretch", disabled=synced)
    c_redo.button("↪️ Redo", on_click=handle_redo, width="stretch", disabled=synced)

if synced:
    render_sleeper_sync_bar()

st.divider()

# ---------------------------------------------------------
# Player Evaluations
# These feed both the Recommendation Center and the Available Player Pool tab,
# so they are computed regardless of which main view is showing.
# ---------------------------------------------------------
avail_df = draft.get_available_players()
current_roster_counts = draft.get_team_roster_counts(st.session_state.user_pos)
next_pick = draft.get_effective_lookahead_pick(st.session_state.user_pos, overall_pick)

user_roster_df = rp.build_roster_df(draft, engine, st.session_state.user_pos)

eval_df = engine.compute_player_evaluations(
    avail_df,
    current_roster_counts,
    current_pick=overall_pick,
    next_pick=next_pick,
    all_team_roster_counts=draft.get_all_teams_roster_counts(),
    user_team_id=st.session_state.user_pos,
    team_at_pick_fn=draft.get_team_at_pick,
    current_roster_df=user_roster_df
)

# Data Sanitization & Rank Differentials for Highlight Insights
eval_df['adp'] = pd.to_numeric(eval_df.get('adp', 999), errors='coerce').fillna(9999)
eval_df['vona_score'] = pd.to_numeric(eval_df.get('vona_score', 0), errors='coerce').fillna(0)

# Calculate Sleeper/Reach value (Positive difference means high VONA rank, but low ADP rank -> Sleeper)
eval_df['vona_rank'] = eval_df['vona_score'].rank(ascending=False, method='min')
eval_df['adp_rank'] = eval_df['adp'].rank(ascending=True, method='min')
eval_df['rank_diff'] = eval_df['adp_rank'] - eval_df['vona_rank']

# Tier Break Detection: flags positions where a handful of players are clearly
# ahead of the pack right now, vs. positions where VONA is high but evenly
# spread (so similar value likely survives to your next pick regardless).
eval_df = engine.identify_tier_gaps(eval_df)

# ---------------------------------------------------------
# Main View: summary takes over once the draft is done
# ---------------------------------------------------------
if is_complete:
    show_summary = True
else:
    show_summary = st.checkbox(
        "📊 Preview draft summary",
        value=False,
        key="preview_summary",
        help="See your projected team totals before the draft ends."
    )

if show_summary:
    if is_complete and not st.session_state.get("celebrated_completion", False):
        st.balloons()
        st.session_state.celebrated_completion = True

    render_draft_summary(
        draft,
        engine,
        st.session_state.league_settings.roster_slots,
        st.session_state.user_pos
    )

elif synced:
    st.subheader("⚡ Action Center")
    st.info(
        "🔗 Synced to Sleeper - make picks there. New picks appear here automatically "
        "within ~20s, or tap **Sync Now** above."
    )

elif state.get("is_user_turn", False) or st.session_state.draft_mode == "Real Draft":
    st.subheader("⚡ Action Center")
    avail_pool = draft.get_available_players()
    
    # Format options nicely and embed Player ID for easy extraction
    options = avail_pool.apply(
        lambda row: f"{row['name']} ({row['position']} - {row['team']}) | ADP: {row.get('adp', 'N/A')} [ID: {row['player_id']}]", 
        axis=1
    ).tolist()
    
    sel_col, btn_col = st.columns([4, 1])
    with sel_col:
        selected_player_str = st.selectbox("Search & Select Player to Draft", options, label_visibility="collapsed")
    with btn_col:
        if st.button("✅ Draft Player", type="primary", width="stretch") and selected_player_str:
            # Extract the ID embedded in the brackets [ID: X]
            pid_str = selected_player_str.split("[ID: ")[-1].replace("]", "")
            handle_draft_player(pid_str)
            st.rerun()

elif st.session_state.draft_mode == "Practice (Mock Draft)":
    st.subheader("⚡ Action Center")
    bot_col1, bot_col2, bot_col3 = st.columns([2, 2, 3])
    temp = bot_col3.slider("Bot Randomness (Temperature)", 0.0, 10.0, 3.0, step=0.5, key="bot_temp")

    bot_col1.button("🤖 Sim 1 Bot Pick", on_click=handle_simulate_pick, args=(temp,), width="stretch")
    bot_col2.button("⏩ Fast Forward to My Turn", on_click=handle_fast_forward, args=(temp,), type="primary", width="stretch")

positions = ["QB", "RB", "WR", "TE", "K", "DEF"]

# ---------------------------------------------------------
# Recommendation Center
# ---------------------------------------------------------
if not show_summary:
    st.divider()
    st.subheader("🎯 Recommendation Center")

    # 0. Tier Break Alerts - surface the most urgent positional cliffs, i.e.
    # where a small group is clearly ahead of the pack right now and likely
    # to be gone (via opponent picks) before your next turn. Shows up to the
    # top 2 distinct positions rather than only the single most urgent one -
    # two tiers can be within a point of each other (e.g. a 2-player RB1
    # tier vs. a 4-player WR1 tier both breaking before your next pick), and
    # showing only the top one would arbitrarily hide an equally real cliff.
    tier_alerts = eval_df[eval_df['is_last_in_tier'] & (eval_df['tier_urgency'] > 0)]
    if not tier_alerts.empty:
        top_alerts = (
            tier_alerts.sort_values('tier_urgency', ascending=False)
            .drop_duplicates(subset='position')
            .head(2)
        )
        for _, top_alert in top_alerts.iterrows():
            alert_pos = top_alert['position']
            tier_names = eval_df[
                (eval_df['position'] == alert_pos) & (eval_df['tier_rank'] > 0)
            ].sort_values('tier_rank')['name'].tolist()

            st.warning(
                f"🧨 **{alert_pos} Tier Break:** {', '.join(tier_names)} form the last strong tier - "
                f"**{top_alert['tier_exhaust_prob'] * 100:.0f}% chance all are gone** by your next pick, "
                f"costing **~{top_alert['tier_gap_value']:.0f} VONA pts** if you wait."
            )

    # 1. View Toggle (Sort by VONA vs ADP vs Tier Urgency)
    sort_mode = st.radio(
        "Sort Recommendations By:",
        ["Value (VONA)", "Consensus ADP", "Tier Urgency"],
        horizontal=True, key="rec_sort"
    )
    if sort_mode == "Tier Urgency":
        sort_cols, sort_ascs = ['tier_urgency', 'vona_score', 'adp'], [False, False, True]
    elif sort_mode == "Consensus ADP":
        sort_cols, sort_ascs = ['adp'], [True]
    else:
        sort_cols, sort_ascs = ['vona_score', 'adp'], [False, True]
    sort_col, sort_asc = sort_cols[0], sort_ascs[0]

    # Extract Top 10 Overall
    top_overall_df = eval_df.sort_values(by=sort_cols, ascending=sort_ascs).head(10)

    # 2. Render Top 10 Overall Targets (Scrollable container for clean UX)
    st.markdown("##### 🏆 Top 10 Overall Recommendations")
    if not top_overall_df.empty:
        with st.container(height=350, border=True):
            for idx, row in top_overall_df.reset_index(drop=True).iterrows():
                c1, c2, c3, c4 = st.columns([3, 2, 2, 1])

                # Identify high-value targets (sleepers) vs reaches vs tier-cliff urgency
                badge = ""
                if row['rank_diff'] >= 10:
                    badge = "🔥 Sleeper"
                elif row['rank_diff'] <= -10:
                    badge = "⚠️ Reach"
                if row.get('is_last_in_tier', False):
                    badge = (badge + " " if badge else "") + "🧨 Last in Tier"
                if row.get('bye_week_conflicts', 0) >= 2:
                    badge = (badge + " " if badge else "") + f"📅 Bye Clash (Wk {int(row['bye_week'])})"

                c1.markdown(f"**{row['name']}** ({row['position']} - {row['team']}) {badge}")
                c2.write(f"VONA: **{row['vona_score']:.2f}** | Proj Pts: {row.get('projected_points', 0):.1f}")
                c3.write(f"ADP: {row['adp'] if row['adp'] != 9999 else 'N/A'} | Survival: {row.get('survival_prob_next_pick', 0) * 100:.0f}%")

                c4.button(
                    "Draft",
                    key=f"draft_overall_{idx}_{row['player_id']}",
                    on_click=handle_draft_player,
                    args=(row['player_id'],),
                    width="stretch",
                    disabled=synced
                )
                st.divider()
    else:
        st.info("No recommendations available. Draft may be over.")

    # 3. Positional Tabs (Top 10 per tab)
    st.markdown("##### 🔍 Top 10 Values by Position")
    tabs = st.tabs(positions)

    for tab_idx, pos in enumerate(positions):
        with tabs[tab_idx]:
            pos_df = eval_df[eval_df['position'] == pos].sort_values(by=sort_cols, ascending=sort_ascs).head(10)

            if not pos_df.empty:
                for idx, row in pos_df.reset_index(drop=True).iterrows():
                    p_c1, p_c2, p_c3, p_c4 = st.columns([3, 2, 2, 1])

                    badge = "🔥" if row['rank_diff'] >= 10 else ("⚠️" if row['rank_diff'] <= -10 else "")
                    if row.get('is_last_in_tier', False):
                        badge = (badge + " " if badge else "") + "🧨"
                    if row.get('bye_week_conflicts', 0) >= 2:
                        badge = (badge + " " if badge else "") + "📅"

                    p_c1.write(f"**{row['name']}** ({row['team']}) {badge}")
                    p_c2.write(f"VONA: **{row['vona_score']:.2f}** | Proj Pts: {row.get('projected_points', 0):.1f}")
                    p_c3.write(f"ADP: {row['adp'] if row['adp'] != 9999 else 'N/A'} | Survival: {row.get('survival_prob_next_pick', 0) * 100:.0f}%")

                    p_c4.button(
                        "Draft",
                        key=f"draft_{pos}_{idx}_{row['player_id']}",
                        on_click=handle_draft_player,
                        args=(row['player_id'],),
                        width="stretch",
                        disabled=synced
                    )
            else:
                st.write(f"No {pos} available.")

    # 4. Positional Runway - multi-pick lookahead across the user's own
    # upcoming picks, not just the very next one. Complements the reactive
    # tier-break alert above with a forward-looking "what can I safely wait
    # on" view spanning several rounds.
    st.markdown("##### 📈 Positional Runway (Next Picks)")
    st.caption(
        "Share of today's best available value at each position expected to still be "
        "there by each of your next picks. A bar that empties fast means draft that "
        "position now; one that stays full is safe to defer."
    )
    upcoming_picks = draft.get_upcoming_user_picks(st.session_state.user_pos, overall_pick, n=4)
    if upcoming_picks:
        runway_df = engine.compute_positional_runway(
            avail_df, overall_pick, upcoming_picks,
            all_team_roster_counts=draft.get_all_teams_roster_counts(),
            user_team_id=st.session_state.user_pos,
            team_at_pick_fn=draft.get_team_at_pick,
        )
        if not runway_df.empty:
            runway_df = runway_df.copy()
            runway_df['retained_pct'] = 1.0 - runway_df['decay_pct']
            runway_df['pick_label'] = runway_df['pick_number'].apply(
                lambda p: f"Pick {int(p)} (Rd {(int(p) - 1) // draft.teams + 1})"
            )

            pivot = runway_df.pivot(index='position', columns='pick_label', values='retained_pct')
            ordered_cols = [f"Pick {int(p)} (Rd {(int(p) - 1) // draft.teams + 1})" for p in upcoming_picks]
            pivot = pivot.reindex(columns=[c for c in ordered_cols if c in pivot.columns])
            pivot = pivot.reindex([p for p in positions if p in pivot.index])

            st.dataframe(
                pivot,
                use_container_width=True,
                column_config={
                    col: st.column_config.ProgressColumn(col, min_value=0.0, max_value=1.0, format="%.0f%%")
                    for col in pivot.columns
                }
            )
        else:
            st.caption("No positional runway data available.")
    else:
        st.caption("No more picks remaining to project.")

st.divider()

# ---------------------------------------------------------
# Bottom Section: Roster & Draft Board Tabs
# ---------------------------------------------------------
tab_roster, tab_history, tab_board, tab_pool = st.tabs([
    "📋 My Roster", 
    "📜 Draft History / Log", 
    "🏟️ League Draft Board / All Teams", 
    "🏊 Available Player Pool"
])

# 1. My Roster
with tab_roster:
    roster_records = draft.get_team_roster(st.session_state.user_pos)
    if roster_records:
        roster_df = pd.DataFrame([vars(p) for p in roster_records])
        
        # Dynamically determine columns to show to prevent KeyErrors
        cols_to_show = []
        for col in ['position', 'player_name', 'name', 'team', 'round', 'overall_pick', 'pick_number', 'pick']:
            if col in roster_df.columns and col not in cols_to_show:
                if col == 'name' and 'player_name' in cols_to_show: continue
                if col == 'player_name' and 'name' in cols_to_show: continue
                cols_to_show.append(col)
                
        st.dataframe(roster_df[cols_to_show], use_container_width=True, hide_index=True)
    else:
        st.info("Your roster is currently empty.")

# 2. Draft History Log
with tab_history:
    all_picks = []
    for t_id in range(1, st.session_state.team_count + 1):
        team_picks = draft.get_team_roster(t_id)
        for p in team_picks:
            pick_dict = vars(p).copy()
            pick_dict['drafting_team'] = t_id
            all_picks.append(pick_dict)
            
    if all_picks:
        log_df = pd.DataFrame(all_picks)
        
        if 'overall_pick' in log_df.columns: log_df = log_df.sort_values(by="overall_pick")
        elif 'pick_number' in log_df.columns: log_df = log_df.sort_values(by="pick_number")
        elif 'pick' in log_df.columns and 'round' in log_df.columns: log_df = log_df.sort_values(by=["round", "pick"])
        else: log_df = log_df.sort_values(by="round")
            
        cols_to_show = ['drafting_team']
        for col in ['overall_pick', 'pick_number', 'pick', 'round', 'player_name', 'name', 'position', 'team']:
            if col in log_df.columns and col not in cols_to_show:
                if col == 'name' and 'player_name' in cols_to_show: continue
                if col == 'player_name' and 'name' in cols_to_show: continue
                cols_to_show.append(col)
                
        st.dataframe(log_df[cols_to_show], use_container_width=True, hide_index=True)
    else:
        st.info("No picks have been made yet.")

# 3. League Draft Board & All Teams
with tab_board:
    st.subheader("Team Roster Inspector")
    inspect_team = st.selectbox(
        "Select Team to Inspect",
        range(1, st.session_state.team_count + 1),
        format_func=lambda x: f"Team {x}" + (" (You)" if x == st.session_state.user_pos else "")
    )
    
    inspect_roster = draft.get_team_roster(inspect_team)
    if inspect_roster:
        inspect_df = pd.DataFrame([vars(p) for p in inspect_roster]).fillna("N/A")
        
        c_show = [c for c in ['round', 'pick', 'position', 'player_name', 'name', 'team', 'bye_week'] if c in inspect_df.columns]
        if 'player_name' in c_show and 'name' in c_show: c_show.remove('name')
            
        st.dataframe(inspect_df[c_show], use_container_width=True, hide_index=True)
        
        if 'position' in inspect_df.columns:
            pos_counts = inspect_df['position'].value_counts().to_dict()
            st.write(f"**Positions Drafted:** {', '.join([f'{k}: {v}' for k, v in pos_counts.items()])}")
    else:
        st.info(f"Team {inspect_team} roster is currently empty.")
        
    st.divider()
    
    st.subheader("Full League Draft Grid")
    try:
        # Check API support
        board_df = draft.get_draft_board_df()
        st.dataframe(board_df, use_container_width=True)
    except AttributeError:
        # Safely pivot the board if get_draft_board_df() is missing
        board_data = []
        for t_id in range(1, st.session_state.team_count + 1):
            team_picks = draft.get_team_roster(t_id)
            for p in team_picks:
                p_dict = vars(p).copy()
                rnd = p_dict.get('round', 0)
                name = p_dict.get('player_name', p_dict.get('name', 'Unknown'))
                pos = p_dict.get('position', '')
                board_data.append({
                    "Round": rnd, 
                    "Team": f"Team {t_id}", 
                    "Pick": f"{name} ({pos})"
                })
        if board_data:
            df_board = pd.DataFrame(board_data).pivot(index="Round", columns="Team", values="Pick").fillna("")
            st.dataframe(df_board, use_container_width=True)
        else:
            st.info("Draft board is empty.")

# 4. Available Player Pool (with Dual Sort Support)
with tab_pool:
    st.markdown("##### Filter and Sort Available Players")
    c_sort, c_filter = st.columns(2)
    with c_sort:
        pool_sort_mode = st.radio(
            "Sort Pool By:", ["Value (VONA)", "Consensus ADP", "Tier Urgency"],
            horizontal=True, key="pool_sort"
        )
    with c_filter:
        filter_pos = st.selectbox("Filter Position", ["ALL"] + positions)

    # Format values before rendering
    display_pool = eval_df[[
        'name', 'position', 'team', 'adp', 'vona_score', 'projected_points',
        'survival_prob_next_pick', 'rank_diff', 'is_last_in_tier', 'tier_urgency',
        'bye_week', 'bye_week_conflicts'
    ]].copy()
    display_pool['survival_prob_next_pick'] = (display_pool['survival_prob_next_pick'] * 100).round(1).astype(str) + "%"
    display_pool['is_last_in_tier'] = display_pool['is_last_in_tier'].map({True: "🧨", False: ""})
    display_pool = display_pool.rename(columns={
        'is_last_in_tier': 'tier_break', 'tier_urgency': 'tier_urgency_pts',
        'bye_week_conflicts': 'bye_clash_count'
    })

    # Hide our artificial 9999 value assigned to undrafted ADPs
    display_pool.loc[display_pool['adp'] == 9999, 'adp'] = None

    if filter_pos != "ALL":
        display_pool = display_pool[display_pool['position'] == filter_pos]

    if pool_sort_mode == "Tier Urgency":
        sort_cols_pool, sort_ascs_pool = ['tier_urgency_pts', 'vona_score', 'adp'], [False, False, True]
    elif pool_sort_mode == "Consensus ADP":
        sort_cols_pool, sort_ascs_pool = ['adp'], [True]
    else:
        sort_cols_pool, sort_ascs_pool = ['vona_score', 'adp'], [False, True]

    st.dataframe(
        display_pool.sort_values(by=sort_cols_pool, ascending=sort_ascs_pool).drop(columns=['rank_diff']),
        use_container_width=True,
        hide_index=True
    )