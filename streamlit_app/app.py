"""Panoptic — AI-Calibrated DORA Dashboard.

Home page: org-wide headline metrics + navigation guide.

Run with:
    streamlit run streamlit_app/app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from data.load import load_sprint_data, TEAM_COLORS
from data.model import ABLATION_RESULTS
from components.sidebar import render_sidebar, filter_df
from components.metrics import kpi_row
from components.charts import trend_chart

st.set_page_config(
    page_title="Panoptic — AI-Calibrated DORA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_sprint_data()


df_all = get_data()
selected_teams, sprint_range = render_sidebar(df_all)
df = filter_df(df_all, selected_teams, sprint_range)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Panoptic")
st.markdown(
    "**AI-Calibrated DORA** · Predicting team health from AI tool usage signals · "
    "Phase 1 — Hard-coded mock data"
)
st.caption(
    "We used machine learning to discover which operational signals from AI coding tool usage "
    "predict team health outcomes — the first empirical, team-level diagnostic framework "
    "for AI tool ROI in software engineering."
)
st.divider()

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Org-wide headline KPIs
# ---------------------------------------------------------------------------

st.subheader("Organisation snapshot")

latest_per_team = df.sort_values("sprint_index").groupby("team_name").last()

avg_rodi   = latest_per_team["rodi"].mean()
avg_acc    = df["agent_acceptance_rate"].mean()
avg_revert = df["revert_rate"].mean()
avg_mcp    = df["mcp_adoption_rate"].mean()
n_teams    = df["team_name"].nunique()
n_sprints  = df["sprint_index"].nunique()

kpi_row([
    {"label": "Teams tracked",            "value": n_teams,    "fmt": "{:.0f}"},
    {"label": "Sprints in window",        "value": n_sprints,  "fmt": "{:.0f}"},
    {"label": "Avg RoDI (latest)",        "value": avg_rodi,   "fmt": "{:.2f}"},
    {"label": "Avg Acceptance Rate",      "value": avg_acc,    "fmt": "{:.1%}"},
    {"label": "Avg Revert Rate",          "value": avg_revert, "fmt": "{:.1%}", "inverse": True},
    {"label": "Avg MCP Adoption",         "value": avg_mcp,    "fmt": "{:.1%}"},
])

st.divider()

# ---------------------------------------------------------------------------
# Org-wide RoDI trend
# ---------------------------------------------------------------------------

col_chart, col_info = st.columns([3, 1])

with col_chart:
    st.subheader("RoDI over time — all teams")
    fig = trend_chart(
        df, "rodi",
        title="Return on Dev Investment per Sprint",
        y_label="RoDI",
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.subheader("What is RoDI?")
    st.markdown(
        """
        **Return on Dev Investment**

        ```
        RoDI = V_output / C_total
        ```

        Where:
        - `V_output` = quality-adjusted PR volume (weighted_prs × quality_mult × stability_mult)
        - `C_total` = c_human + c_ai_effective + c_rework

        A higher RoDI means the team is producing more quality-adjusted output
        per unit of total engineering spend.
        """
    )
    abl = ABLATION_RESULTS
    st.metric(
        "Cursor signal improvement over DORA",
        f"+{abl['r2_improvement_pct']:.0f}% R²",
        help="Adding Cursor operational signals improves revert-rate prediction from R²=0.41 to R²=0.73",
    )

st.divider()

# ---------------------------------------------------------------------------
# Navigation guide
# ---------------------------------------------------------------------------

st.subheader("Navigate the dashboard")

nav_cols = st.columns(4)

nav_items = [
    {
        "page":   "01_Overview",
        "icon":   "🏆",
        "title":  "Team Overview",
        "detail": "Org-wide team ranking. Compare DORA rank vs RoDI rank. Spot teams where AI usage is inflating or suppressing perceived performance.",
    },
    {
        "page":   "02_Team_Detail",
        "icon":   "🔍",
        "title":  "Team Detail",
        "detail": "Single-team deep dive. Sprint-over-sprint trends, variable cost breakdown, SHAP waterfall explaining each prediction, and actionable recommendations.",
    },
    {
        "page":   "03_Findings",
        "icon":   "📈",
        "title":  "Model Findings",
        "detail": "Which signals matter most? Where are the risk thresholds? How much do Cursor signals add over DORA alone? The core thesis results.",
    },
    {
        "page":   "04_ROI_Curve",
        "icon":   "💡",
        "title":  "ROI Curve",
        "detail": "AI spend vs quality-adjusted output for every team-sprint. Identify teams above and below the efficiency frontier.",
    },
]

for col, item in zip(nav_cols, nav_items):
    with col:
        st.markdown(
            f"""
            <div style="
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 16px;
                height: 160px;
            ">
                <div style="font-size: 1.8em;">{item['icon']}</div>
                <strong>{item['title']}</strong><br/>
                <small style="color: #666;">{item['detail']}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Team archetypes quick reference
# ---------------------------------------------------------------------------

st.subheader("Team archetypes in this dataset")

archetypes = pd.DataFrame([
    {"Team": "Alpha",   "Profile": "High AI / Good Quality",       "Acceptance": "72%", "Revert": "4%",  "Heterogeneity": "Low",    "Key trait": "High deliberation, broad MCP adoption"},
    {"Team": "Beta",    "Profile": "High AI / Bad Quality",        "Acceptance": "41%", "Revert": "18%", "Heterogeneity": "Medium", "Key trait": "Pure generation, no Ask/Plan mode usage"},
    {"Team": "Gamma",   "Profile": "Low AI / Good Quality",        "Acceptance": "62%", "Revert": "5%",  "Heterogeneity": "Low",    "Key trait": "Careful adoption, high ask_to_total_ratio"},
    {"Team": "Delta",   "Profile": "Uneven Adoption / Declining",  "Acceptance": "↓65→57%", "Revert": "↑6→14%", "Heterogeneity": "High ↑", "Key trait": "One power user carrying AI; degrading trend"},
    {"Team": "Epsilon", "Profile": "Low Adoption",                 "Acceptance": "58%", "Revert": "8%",  "Heterogeneity": "Medium", "Key trait": "35% DAU, low MCP, room to grow"},
])

st.dataframe(archetypes, use_container_width=True, hide_index=True)
