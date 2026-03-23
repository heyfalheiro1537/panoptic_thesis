"""Page 1 — Org-wide team overview.

Shows:
- Org-level KPIs (avg RoDI, avg acceptance rate, avg revert rate)
- Team ranking: DORA rank vs RoDI rank with rank delta
- Multi-team RoDI trend over sprints
"""

import streamlit as st
import pandas as pd

from data.load import load_sprint_data, TEAM_COLORS
from components.sidebar import render_sidebar, filter_df
from components.metrics import kpi_row
from components.charts import trend_chart

st.set_page_config(page_title="Overview — Panoptic", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_sprint_data()


df_all = get_data()
selected_teams, sprint_range = render_sidebar(df_all)
df = filter_df(df_all, selected_teams, sprint_range)

st.title("Org-wide Overview")
st.caption("Team health summary across all teams and selected sprint window.")

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

avg_rodi      = df.groupby("sprint_index")["rodi"].mean().iloc[-1]
avg_acc       = df["agent_acceptance_rate"].mean()
avg_revert    = df["revert_rate"].mean()
avg_block_rev = df["blocking_review_ratio"].mean()

kpi_row([
    {"label": "Avg RoDI (latest sprint)",    "value": avg_rodi,      "fmt": "{:.2f}"},
    {"label": "Avg Agent Acceptance Rate",   "value": avg_acc,       "fmt": "{:.1%}"},
    {"label": "Avg Revert Rate",             "value": avg_revert,    "fmt": "{:.1%}", "inverse": True},
    {"label": "Avg Blocking Review Ratio",   "value": avg_block_rev, "fmt": "{:.1%}", "inverse": True},
])

st.divider()

# ---------------------------------------------------------------------------
# Team ranking table
# ---------------------------------------------------------------------------

st.subheader("Team Ranking: DORA Rank vs RoDI Rank")
st.caption(
    "DORA rank = ranked by deploy_frequency (higher = better). "
    "RoDI rank = ranked by quality-adjusted output per cost. "
    "Green delta = RoDI rank is better than DORA rank (undervalued). "
    "Red delta = DORA rank is better than RoDI rank (inflated perception)."
)

# Use the latest sprint for each team
latest = df.sort_values("sprint_index").groupby("team_name").last().reset_index()

latest["dora_rank"] = latest["deploy_frequency"].rank(ascending=False).astype(int)
latest["rodi_rank"] = latest["rodi"].rank(ascending=False).astype(int)
latest["rank_delta"] = latest["dora_rank"] - latest["rodi_rank"]

ranking = latest[[
    "team_name", "dora_rank", "rodi_rank", "rank_delta",
    "rodi", "agent_acceptance_rate", "revert_rate",
    "blocking_review_ratio", "usage_heterogeneity_cv",
]].sort_values("rodi_rank")

ranking.columns = [
    "Team", "DORA Rank", "RoDI Rank", "Rank Δ",
    "RoDI", "Acceptance Rate", "Revert Rate",
    "Blocking Review Ratio", "Heterogeneity CV",
]


def colour_delta(val):
    if val > 0:
        return "color: #2ecc71; font-weight: bold"
    elif val < 0:
        return "color: #e74c3c; font-weight: bold"
    return ""


styled = (
    ranking.style
    .format({
        "RoDI":                  "{:.2f}",
        "Acceptance Rate":       "{:.1%}",
        "Revert Rate":           "{:.1%}",
        "Blocking Review Ratio": "{:.1%}",
        "Heterogeneity CV":      "{:.2f}",
        "Rank Δ":                "{:+d}",
    })
    .applymap(colour_delta, subset=["Rank Δ"])
)

st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Multi-team RoDI trend
# ---------------------------------------------------------------------------

st.subheader("RoDI over Sprints")
fig = trend_chart(df, "rodi", title="Return on Dev Investment (RoDI)", y_label="RoDI", height=340)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Acceptance rate trend
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Agent Acceptance Rate")
    fig = trend_chart(
        df, "agent_acceptance_rate",
        title="Agent Acceptance Rate over Sprints",
        y_label="Acceptance Rate",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Revert Rate")
    fig = trend_chart(
        df, "revert_rate",
        title="Revert Rate over Sprints",
        y_label="Revert Rate",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)
