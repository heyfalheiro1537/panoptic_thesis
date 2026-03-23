"""Page 2 — Single team deep dive.

Shows:
- Team selector
- Sprint-over-sprint trend charts (RoDI, acceptance rate, revert rate, blocking reviews)
- Stacked variable cost chart
- SHAP waterfall for latest sprint
- Recommendations
"""

import streamlit as st
import pandas as pd

from data.load import load_sprint_data, TEAM_COLORS
from data.model import SHAP_VALUES, RECOMMENDATIONS
from components.sidebar import render_sidebar, filter_df
from components.metrics import kpi_row
from components.charts import trend_chart, stacked_cost_chart, shap_waterfall

st.set_page_config(page_title="Team Detail — Panoptic", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_sprint_data()


df_all = get_data()
_, sprint_range = render_sidebar(df_all)

# Team selector in main area (not filtered by sidebar multiselect here)
all_teams = sorted(df_all["team_name"].unique().tolist())
team_name = st.selectbox("Select team", options=all_teams, index=0)

df_team = filter_df(df_all, [team_name], sprint_range)

st.title(f"Team Detail — {team_name}")

if df_team.empty:
    st.warning("No data for the selected sprint range.")
    st.stop()

# ---------------------------------------------------------------------------
# Latest sprint KPIs
# ---------------------------------------------------------------------------

latest = df_team.sort_values("sprint_index").iloc[-1]
prev   = df_team.sort_values("sprint_index").iloc[-2] if len(df_team) > 1 else None

kpi_row([
    {
        "label": "RoDI",
        "value": latest["rodi"],
        "delta": latest["rodi"] - prev["rodi"] if prev is not None else None,
        "fmt": "{:.2f}",
    },
    {
        "label": "Agent Acceptance Rate",
        "value": latest["agent_acceptance_rate"],
        "delta": latest["agent_acceptance_rate"] - prev["agent_acceptance_rate"] if prev is not None else None,
        "fmt": "{:.1%}",
    },
    {
        "label": "Revert Rate",
        "value": latest["revert_rate"],
        "delta": latest["revert_rate"] - prev["revert_rate"] if prev is not None else None,
        "fmt": "{:.1%}",
        "inverse": True,
    },
    {
        "label": "Blocking Review Ratio",
        "value": latest["blocking_review_ratio"],
        "delta": latest["blocking_review_ratio"] - prev["blocking_review_ratio"] if prev is not None else None,
        "fmt": "{:.1%}",
        "inverse": True,
    },
    {
        "label": "MCP Adoption",
        "value": latest["mcp_adoption_rate"],
        "delta": latest["mcp_adoption_rate"] - prev["mcp_adoption_rate"] if prev is not None else None,
        "fmt": "{:.1%}",
    },
])

st.divider()

# ---------------------------------------------------------------------------
# Trend charts (2 × 2 grid)
# ---------------------------------------------------------------------------

st.subheader("Sprint-over-sprint trends")

col1, col2 = st.columns(2)

with col1:
    fig = trend_chart(
        df_team, "rodi",
        title="RoDI",
        y_label="Return on Dev Investment",
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = trend_chart(
        df_team, "revert_rate",
        title="Revert Rate",
        y_label="Revert Rate",
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = trend_chart(
        df_team, "agent_acceptance_rate",
        title="Agent Acceptance Rate",
        y_label="Acceptance Rate",
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = trend_chart(
        df_team, "blocking_review_ratio",
        title="Blocking Review Ratio",
        y_label="Blocking Review Ratio",
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Additional Cursor signals
# ---------------------------------------------------------------------------

st.subheader("Cursor usage signals")

col3, col4 = st.columns(2)

with col3:
    fig = trend_chart(
        df_team, "usage_heterogeneity_cv",
        title="Usage Heterogeneity (CV)",
        y_label="Coefficient of Variation",
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True)

with col4:
    fig = trend_chart(
        df_team, "ai_deliberation_ratio",
        title="AI Deliberation Ratio",
        y_label="ask + plan / total messages",
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------

st.subheader("Variable cost breakdown")
st.caption("c_ai_effective = AI spend adjusted for acceptance quality. c_rework = blocking reviews + reverts. c_human is excluded (fixed, dominates scale).")
fig = stacked_cost_chart(df_team, team_name)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# SHAP waterfall
# ---------------------------------------------------------------------------

st.subheader("Why this prediction? (SHAP decomposition)")
st.caption("Showing predicted revert rate for the latest sprint. Red bars increase risk, green bars decrease risk.")

if team_name in SHAP_VALUES:
    shap = SHAP_VALUES[team_name]
    fig  = shap_waterfall(
        contributions=shap["contributions"],
        base_value=shap["base_value"],
        prediction=shap["prediction"],
        team_name=team_name,
    )
    st.plotly_chart(fig, use_container_width=True)

    pred = shap["prediction"]
    if pred < 0.07:
        st.success(f"Predicted revert rate: **{pred:.1%}** — Low risk. Team is operating well.")
    elif pred < 0.12:
        st.warning(f"Predicted revert rate: **{pred:.1%}** — Moderate risk. Monitor key signals.")
    else:
        st.error(f"Predicted revert rate: **{pred:.1%}** — High risk. Intervention recommended.")
else:
    st.info("SHAP values not available for this team.")

st.divider()

# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

st.subheader("Recommendations")

recs = RECOMMENDATIONS.get(team_name, [])
if not recs:
    st.info("No recommendations at this time.")
else:
    for rec in recs:
        if rec["severity"] == "error":
            st.error(f"**{rec['icon']} {rec['title']}**\n\n{rec['detail']}")
        elif rec["severity"] == "warning":
            st.warning(f"**{rec['icon']} {rec['title']}**\n\n{rec['detail']}")
        else:
            st.info(f"**{rec['icon']} {rec['title']}**\n\n{rec['detail']}")
