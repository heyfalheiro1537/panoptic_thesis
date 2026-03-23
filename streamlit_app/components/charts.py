"""Reusable Plotly chart builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data.load import TEAM_COLORS


# ---------------------------------------------------------------------------
# SHAP waterfall
# ---------------------------------------------------------------------------

def shap_waterfall(
    contributions: list[tuple[str, float, float]],
    base_value: float,
    prediction: float,
    team_name: str,
) -> go.Figure:
    """Horizontal bar chart showing SHAP contributions to predicted revert risk.

    Parameters
    ----------
    contributions : [(feature_name, shap_value, feature_value)]
        shap_value > 0 increases predicted revert rate (bad → red)
        shap_value < 0 decreases predicted revert rate (good → green)
    """
    features = [c[0] for c in contributions]
    values   = [c[1] for c in contributions]
    feat_vals = [c[2] for c in contributions]

    colors  = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]
    labels  = [
        f"{'+' if v > 0 else ''}{v:.3f}  ({f}={fv:.2f})"
        for v, f, fv in zip(values, features, feat_vals)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=features,
            x=values,
            orientation="h",
            marker_color=colors,
            text=[f"{'+' if v > 0 else ''}{v:.3f}" for v in values],
            textposition="outside",
            hovertext=labels,
            hoverinfo="text",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#aaaaaa", line_dash="dash")
    fig.update_layout(
        title={
            "text": (
                f"<b>Predicted revert risk — {team_name}</b><br>"
                f"<sup>Base rate: {base_value:.3f} → Prediction: {prediction:.3f}</sup>"
            ),
            "x": 0,
        },
        xaxis_title="SHAP contribution to next-sprint revert rate",
        height=320,
        showlegend=False,
        margin=dict(l=170, r=60, t=80, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Global feature importance
# ---------------------------------------------------------------------------

def feature_importance_bar(importance_df: pd.DataFrame, group_colors: dict) -> go.Figure:
    """Horizontal bar chart of global SHAP feature importances, coloured by group."""
    df = importance_df.sort_values("importance")
    colors = [group_colors.get(g, "#95a5a6") for g in df["group"]]

    fig = go.Figure(
        go.Bar(
            y=df["feature"],
            x=df["importance"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2f}" for v in df["importance"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="<b>Global feature importance</b> (mean |SHAP|, target = revert_rate_next)",
        xaxis_title="Mean |SHAP value|",
        height=420,
        showlegend=False,
        margin=dict(l=180, r=60, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Multi-team trend line
# ---------------------------------------------------------------------------

def trend_chart(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    y_label: str,
    height: int = 300,
) -> go.Figure:
    """Multi-line chart of ``y_col`` over sprint_index, one line per team."""
    fig = px.line(
        df,
        x="sprint_index",
        y=y_col,
        color="team_name",
        color_discrete_map=TEAM_COLORS,
        markers=True,
        title=title,
        labels={"sprint_index": "Sprint", y_col: y_label, "team_name": "Team"},
        height=height,
    )
    fig.update_layout(
        legend_title_text="Team",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Stacked cost chart (per team, over sprints)
# ---------------------------------------------------------------------------

def stacked_cost_chart(df: pd.DataFrame, team_name: str) -> go.Figure:
    """Stacked bar: c_ai_effective + c_rework per sprint for one team.

    c_human is excluded because it dominates and obscures variance.
    """
    team_df = df[df["team_name"] == team_name].sort_values("sprint_index")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=team_df["sprint_index"],
        y=team_df["c_ai_effective"],
        name="c_ai_effective",
        marker_color="#3498db",
    ))
    fig.add_trace(go.Bar(
        x=team_df["sprint_index"],
        y=team_df["c_rework"],
        name="c_rework",
        marker_color="#e74c3c",
    ))
    fig.update_layout(
        barmode="stack",
        title=f"<b>Variable costs per sprint — {team_name}</b><br>"
              "<sup>c_human (fixed) excluded to show variance</sup>",
        xaxis_title="Sprint",
        yaxis_title="Cost (BRL)",
        legend_title_text="Component",
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Threshold scatter
# ---------------------------------------------------------------------------

def threshold_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    threshold_x: float | None = None,
    threshold_y: float | None = None,
    height: int = 340,
) -> go.Figure:
    """Scatter plot with optional threshold lines and team colouring."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="team_name",
        color_discrete_map=TEAM_COLORS,
        opacity=0.75,
        title=title,
        labels={x_col: x_label, y_col: y_label, "team_name": "Team"},
        height=height,
    )
    if threshold_x is not None:
        fig.add_vline(
            x=threshold_x,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=f"Risk threshold ({threshold_x})",
            annotation_position="top right",
        )
    if threshold_y is not None:
        fig.add_hline(
            y=threshold_y,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=f"Risk threshold ({threshold_y})",
            annotation_position="top right",
        )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# ROI scatter with fitted curve
# ---------------------------------------------------------------------------

def roi_scatter(df: pd.DataFrame, x_col: str = "c_ai_effective", y_col: str = "rodi") -> go.Figure:
    """Scatter of all team-sprints with a degree-2 polynomial fit overlaid."""
    x = df[x_col].values
    y = df[y_col].values

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="team_name",
        color_discrete_map=TEAM_COLORS,
        hover_data=["team_name", "sprint_index", "agent_acceptance_rate", "revert_rate"],
        opacity=0.70,
        title="<b>AI spend vs quality-adjusted output</b><br>"
              "<sup>Each dot = one team-sprint. Curve = degree-2 polynomial fit.</sup>",
        labels={x_col: "Effective AI spend (BRL)", y_col: "RoDI", "team_name": "Team"},
        height=480,
    )

    # Polynomial fit
    coeffs = np.polyfit(x, y, deg=2)
    x_fit  = np.linspace(x.min(), x.max(), 200)
    y_fit  = np.polyval(coeffs, x_fit)
    fig.add_trace(go.Scatter(
        x=x_fit,
        y=y_fit,
        mode="lines",
        name="Fitted curve",
        line=dict(color="#2c3e50", width=2, dash="dot"),
    ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Team",
    )
    return fig
