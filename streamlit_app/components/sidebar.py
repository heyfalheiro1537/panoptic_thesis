"""Shared sidebar renderer used by every page."""

from __future__ import annotations

import streamlit as st
import pandas as pd


def render_sidebar(df: pd.DataFrame) -> tuple[list[str], tuple[int, int]]:
    """Render the left sidebar and return current filter selections.

    Returns
    -------
    selected_teams : list[str]
        Team names currently selected (may be empty).
    sprint_range : tuple[int, int]
        (min_sprint_index, max_sprint_index) inclusive.
    """
    with st.sidebar:
        st.markdown("## Panoptic")
        st.caption("AI-Calibrated DORA")
        st.divider()

        all_teams = sorted(df["team_name"].unique().tolist())
        selected_teams = st.multiselect(
            "Teams",
            options=all_teams,
            default=all_teams,
            help="Filter which teams appear in charts and tables.",
        )

        n_sprints = int(df["sprint_index"].max()) + 1
        sprint_range = st.slider(
            "Sprint range",
            min_value=0,
            max_value=n_sprints - 1,
            value=(0, n_sprints - 1),
            help="Sprints are 2-week periods starting 2024-07-01.",
        )

        st.divider()
        st.caption("📊 Data source: hard-coded mock (Phase 1)")
        st.caption("Model: hard-coded SHAP stubs (Phase 2 pending)")

    return selected_teams, sprint_range


def filter_df(
    df: pd.DataFrame,
    selected_teams: list[str],
    sprint_range: tuple[int, int],
) -> pd.DataFrame:
    """Apply sidebar selections to the full DataFrame."""
    lo, hi = sprint_range
    mask = (
        df["team_name"].isin(selected_teams)
        & df["sprint_index"].between(lo, hi)
    )
    return df[mask].copy()
