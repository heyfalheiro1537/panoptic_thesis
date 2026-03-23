"""Reusable metric card helpers."""

from __future__ import annotations

import streamlit as st


def kpi_row(items: list[dict]) -> None:
    """Render a horizontal row of st.metric cards.

    Parameters
    ----------
    items : list of dicts with keys:
        label   : str  — metric label
        value   : any  — current value (formatted with ``fmt``)
        delta   : any  — delta vs previous period (optional, None to hide)
        fmt     : str  — Python format string, default "{:.2f}"
        inverse : bool — if True, positive delta is shown red (e.g. revert rate)
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        fmt     = item.get("fmt", "{:.2f}")
        inverse = item.get("inverse", False)
        delta   = item.get("delta")

        val_str = fmt.format(item["value"]) if item["value"] is not None else "—"

        if delta is not None:
            delta_str   = f"{delta:+.2f}"
            delta_color = "inverse" if inverse else "normal"
        else:
            delta_str   = None
            delta_color = "normal"

        col.metric(
            label=item["label"],
            value=val_str,
            delta=delta_str,
            delta_color=delta_color,
        )
