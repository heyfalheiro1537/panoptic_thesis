"""Page 4 — AI spend vs quality-adjusted output (ROI curve).

Shows:
- Scatter of all team-sprints: effective AI spend vs RoDI
- Degree-2 polynomial fit (diminishing returns curve)
- Latest-sprint team annotations
- Interpretation guide
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from data.load import load_sprint_data, TEAM_COLORS
from components.sidebar import render_sidebar, filter_df
from components.charts import roi_scatter

st.set_page_config(page_title="ROI Curve — Panoptic", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_sprint_data()


df_all = get_data()
selected_teams, sprint_range = render_sidebar(df_all)
df = filter_df(df_all, selected_teams, sprint_range)

st.title("AI Spend vs Output: ROI Curve")
st.caption(
    "Each dot is one team-sprint observation. "
    "Teams above the fitted curve extract above-average output per AI dollar. "
    "Teams below are over-spending relative to quality-adjusted output."
)

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Axis selector
# ---------------------------------------------------------------------------

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    x_option = st.selectbox(
        "X-axis (AI spend proxy)",
        options=["c_ai_effective", "agent_waste_ratio", "usage_heterogeneity_cv"],
        index=0,
        format_func=lambda c: {
            "c_ai_effective":      "Effective AI spend (BRL)",
            "agent_waste_ratio":   "Agent waste ratio",
            "usage_heterogeneity_cv": "Usage heterogeneity (CV)",
        }[c],
    )
with col_sel2:
    y_option = st.selectbox(
        "Y-axis (output quality)",
        options=["rodi", "v_output", "revert_rate"],
        index=0,
        format_func=lambda c: {
            "rodi":       "RoDI (quality-adjusted output / cost)",
            "v_output":   "V_output (quality-adjusted PR volume)",
            "revert_rate": "Revert Rate (lower = better)",
        }[c],
    )

st.divider()

# ---------------------------------------------------------------------------
# Main scatter + fitted curve
# ---------------------------------------------------------------------------

fig = roi_scatter(df, x_col=x_option, y_col=y_option)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Latest sprint team annotations table
# ---------------------------------------------------------------------------

st.subheader("Team positions (latest sprint)")

latest = df.sort_values("sprint_index").groupby("team_name").last().reset_index()

# Compute position relative to population mean
mean_x = latest[x_option].mean()
mean_y = latest[y_option].mean()
latest["above_curve"] = (
    (latest[x_option] < mean_x) & (latest[y_option] > mean_y)
    | (latest[x_option] > mean_x) & (latest[y_option] > mean_y)
)

display = latest[["team_name", x_option, y_option, "agent_acceptance_rate", "usage_heterogeneity_cv"]].copy()
display.columns = ["Team", x_option, y_option, "Acceptance Rate", "Heterogeneity CV"]


def colour_position(val):
    return ""


styled = (
    display.style
    .format({
        x_option:          "{:.2f}",
        y_option:          "{:.3f}",
        "Acceptance Rate": "{:.1%}",
        "Heterogeneity CV": "{:.2f}",
    })
    .background_gradient(subset=[y_option], cmap="RdYlGn")
)
st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

st.subheader("How to read this chart")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        """
        **Teams above the curve** are extracting more quality-adjusted output
        per unit of effective AI spend than average. This often correlates with:
        - High acceptance rate (good prompt quality)
        - Low usage heterogeneity (broad team adoption)
        - High deliberation ratio (thinking before generating)
        """
    )
with col_b:
    st.markdown(
        """
        **Teams below the curve** are spending more (in waste, rework, or
        high effective spend from low acceptance) than their output justifies.
        Common causes:
        - Low acceptance rate (many rejected suggestions)
        - Concentrated usage (one person driving all AI output)
        - Low MCP adoption (not integrating external context)
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Acceptance rate segmentation
# ---------------------------------------------------------------------------

st.subheader("Output by acceptance rate band")
st.caption("Teams are split into three bands based on agent_acceptance_rate. Shows whether high acceptance → higher output.")

df["acc_band"] = pd.cut(
    df["agent_acceptance_rate"],
    bins=[0, 0.52, 0.65, 1.0],
    labels=["Low (<52%)", "Medium (52–65%)", "High (>65%)"],
)

band_summary = (
    df.groupby("acc_band", observed=True)[["rodi", "revert_rate", "v_output"]]
    .mean()
    .round(3)
    .reset_index()
)
band_summary.columns = ["Acceptance Band", "Avg RoDI", "Avg Revert Rate", "Avg V_output"]

fig_band = px.bar(
    band_summary,
    x="Acceptance Band",
    y="Avg RoDI",
    color="Acceptance Band",
    color_discrete_sequence=["#e74c3c", "#f39c12", "#2ecc71"],
    title="Avg RoDI by Acceptance Rate Band",
    height=320,
)
fig_band.update_layout(
    showlegend=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_band, use_container_width=True)
