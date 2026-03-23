"""Page 3 — Model findings (thesis results page).

Shows:
- Global feature importance bar chart
- Threshold discovery scatter plots
- Ablation study results (GitHub-only vs GitHub+Cursor)
- Key finding cards
"""

import streamlit as st
import pandas as pd

from data.load import load_sprint_data
from data.model import (
    FEATURE_IMPORTANCE,
    GROUP_COLORS,
    ABLATION_RESULTS,
    KEY_FINDINGS,
)
from components.sidebar import render_sidebar, filter_df
from components.charts import feature_importance_bar, threshold_scatter

st.set_page_config(page_title="Findings — Panoptic", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_sprint_data()


df_all = get_data()
selected_teams, sprint_range = render_sidebar(df_all)
df = filter_df(df_all, selected_teams, sprint_range)

st.title("Model Findings")
st.caption(
    "This page presents what the trained model LEARNS from the data — "
    "which signals matter, where the thresholds are, and how much Cursor signals "
    "add over DORA metrics alone."
)
st.info(
    "**Phase 1 note:** Values below are hard-coded placeholders grounded in the thesis hypotheses. "
    "They will be replaced by actual trained model outputs in Phase 2."
)

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

st.subheader("Which signals predict next-sprint revert rate?")
st.caption("Mean absolute SHAP value across all team-sprint observations. Higher = more predictive.")

fig = feature_importance_bar(FEATURE_IMPORTANCE, GROUP_COLORS)
st.plotly_chart(fig, use_container_width=True)

# Legend for groups
st.caption(
    "Colour groups: "
    + "  ".join(
        f"<span style='color:{v}'>■ {k}</span>"
        for k, v in GROUP_COLORS.items()
    ),
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Threshold discovery scatters
# ---------------------------------------------------------------------------

st.subheader("Where are the risk thresholds?")

col1, col2 = st.columns(2)

with col1:
    st.caption("Agent acceptance rate vs revert rate — threshold at 0.52")
    fig = threshold_scatter(
        df,
        x_col="agent_acceptance_rate",
        y_col="revert_rate",
        x_label="Agent Acceptance Rate",
        y_label="Revert Rate",
        title="Acceptance Rate vs Revert Rate",
        threshold_x=0.52,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.caption("Usage heterogeneity vs blocking review ratio — threshold at 0.50")
    fig = threshold_scatter(
        df,
        x_col="usage_heterogeneity_cv",
        y_col="blocking_review_ratio",
        x_label="Usage Heterogeneity (CV)",
        y_label="Blocking Review Ratio",
        title="Heterogeneity vs Blocking Reviews",
        threshold_x=0.50,
    )
    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.caption("AI deliberation ratio vs revert rate — threshold at 0.30")
    fig = threshold_scatter(
        df,
        x_col="ai_deliberation_ratio",
        y_col="revert_rate",
        x_label="AI Deliberation Ratio",
        y_label="Revert Rate",
        title="Deliberation Ratio vs Revert Rate",
        threshold_x=0.30,
    )
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.caption("MCP adoption rate vs blocking review ratio")
    fig = threshold_scatter(
        df,
        x_col="mcp_adoption_rate",
        y_col="blocking_review_ratio",
        x_label="MCP Adoption Rate",
        y_label="Blocking Review Ratio",
        title="MCP Adoption vs Review Friction",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

st.subheader("How much do Cursor signals add over DORA alone?")
st.caption(
    "Ablation: train model on GitHub-only signals, then on GitHub + all Cursor signals. "
    "Measures how much predictive power the operational Cursor signals contribute."
)

abl = ABLATION_RESULTS
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("### GitHub signals only")
    st.metric("R²",   f"{abl['github_only']['r2']:.2f}")
    st.metric("RMSE", f"{abl['github_only']['rmse']:.3f}")
    st.metric("AUC (incident classification)", f"{abl['github_only']['auc']:.2f}")

with col_b:
    st.markdown("### GitHub + Cursor signals")
    st.metric("R²",   f"{abl['github_cursor']['r2']:.2f}",  delta=f"+{abl['github_cursor']['r2'] - abl['github_only']['r2']:.2f}")
    st.metric("RMSE", f"{abl['github_cursor']['rmse']:.3f}", delta=f"{abl['github_cursor']['rmse'] - abl['github_only']['rmse']:.3f}")
    st.metric("AUC",  f"{abl['github_cursor']['auc']:.2f}",  delta=f"+{abl['github_cursor']['auc'] - abl['github_only']['auc']:.2f}")

with col_c:
    st.markdown("### Relative improvement")
    st.metric("R² improvement",   f"+{abl['r2_improvement_pct']:.0f}%")
    st.metric("RMSE reduction",   f"{abl['improvement_pct']:.0f}%")
    st.markdown(
        "> Cursor operational signals are **not redundant with DORA**. "
        "They capture process, not just output — and they explain variance "
        "that GitHub metrics alone cannot."
    )

st.divider()

# ---------------------------------------------------------------------------
# Key findings cards
# ---------------------------------------------------------------------------

st.subheader("Key Findings")

rows = [KEY_FINDINGS[:2], KEY_FINDINGS[2:]]
for row in rows:
    cols = st.columns(len(row))
    for col, finding in zip(cols, row):
        group_color = GROUP_COLORS.get(finding.get("group", ""), "#95a5a6")
        with col:
            st.markdown(
                f"""
                <div style="
                    border-left: 4px solid {group_color};
                    padding: 12px 16px;
                    border-radius: 4px;
                    background: rgba(0,0,0,0.03);
                    margin-bottom: 12px;
                ">
                    <strong>{finding['title']}</strong><br/>
                    <small>{finding['detail']}</small>
                """,
                unsafe_allow_html=True,
            )
            if finding.get("threshold") is not None:
                st.caption(f"Signal: `{finding['signal']}` · Threshold: {finding['threshold']}")
            st.markdown("</div>", unsafe_allow_html=True)
