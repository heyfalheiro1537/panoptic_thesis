"""Hard-coded sprint metrics dataset for the Phase 1 dashboard demo.

5 team archetypes × 12 sprints = 60 rows. Generated deterministically with
a fixed seed so charts are stable across runs. Replace this file with real
client calls once Phase 4 real-API integration is complete.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEAM_COLORS = {
    "Alpha": "#2ecc71",
    "Beta": "#e74c3c",
    "Gamma": "#3498db",
    "Delta": "#f39c12",
    "Epsilon": "#9b59b6",
}


def load_sprint_data() -> pd.DataFrame:
    """Return a DataFrame with 5 teams × 12 sprints of synthetic sprint metrics.

    Column groups
    -------------
    Identity:   team_name, sprint_index, sprint_start
    Group A:    agent_acceptance_rate, agent_waste_ratio, tab_acceptance_rate,
                ask_to_total_ratio, mcp_adoption_rate, usage_heterogeneity_cv,
                dau_mean, adoption_rate, cloud_agent_dau, skills_adoption_rate,
                command_diversity, plan_mode_usage, ai_deliberation_ratio,
                bugbot_resolved_rate
    Group B:    deploy_frequency, median_lead_time_hours, revert_rate,
                blocking_review_ratio, n_prs_merged, weighted_prs
    Group C:    n_incidents, mttr_hours
    Group D:    team_size
    Computed:   c_human, c_ai_effective, c_rework, c_total,
                quality_multiplier, stability_multiplier, v_output, rodi
    """
    rng = np.random.default_rng(42)

    teams = [
        _make_team(rng, "Alpha",   team_size=10, profile="high_ai_good"),
        _make_team(rng, "Beta",    team_size=8,  profile="high_ai_bad"),
        _make_team(rng, "Gamma",   team_size=12, profile="low_ai_good"),
        _make_team(rng, "Delta",   team_size=9,  profile="uneven"),
        _make_team(rng, "Epsilon", team_size=7,  profile="low_adoption"),
    ]
    df = pd.concat(teams, ignore_index=True)
    df = _compute_derived(df)
    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _noise(rng, center, sigma, n, lo=0.0, hi=1.0):
    return np.clip(rng.normal(center, sigma, n), lo, hi)


def _make_team(rng, name: str, team_size: int, profile: str) -> pd.DataFrame:
    N = 12
    t = np.arange(N)
    sprint_starts = pd.date_range("2024-07-01", periods=N, freq="2W")

    if profile == "high_ai_good":
        agent_acc  = _noise(rng, 0.72, 0.03, N)
        tab_acc    = _noise(rng, 0.68, 0.02, N)
        ask_ratio  = _noise(rng, 0.35, 0.05, N)
        mcp_rate   = _noise(rng, 0.80, 0.05, N)
        het_cv     = _noise(rng, 0.18, 0.03, N, lo=0.05)
        dau        = _noise(rng, team_size * 0.85, 0.8, N, lo=1, hi=team_size)
        revert     = _noise(rng, 0.04, 0.01, N)
        block_rev  = _noise(rng, 0.18, 0.03, N)
        deploy     = _noise(rng, 18, 2, N, lo=5, hi=40).astype(int)
        lead       = _noise(rng, 22, 4, N, lo=4, hi=72)
        incidents  = rng.poisson(0.3, N)
        mttr       = _noise(rng, 2.5, 0.5, N, lo=0.5, hi=24)
        cloud_dau  = _noise(rng, team_size * 0.40, 0.5, N, lo=0, hi=team_size)
        skill_rate = _noise(rng, 0.60, 0.05, N)
        cmd_div    = _noise(rng, 4.5, 0.4, N, lo=1, hi=8)
        plan_use   = _noise(rng, 12, 2, N, lo=0, hi=50)

    elif profile == "high_ai_bad":
        agent_acc  = _noise(rng, 0.41, 0.04, N)
        tab_acc    = _noise(rng, 0.52, 0.03, N)
        ask_ratio  = _noise(rng, 0.08, 0.03, N)
        mcp_rate   = _noise(rng, 0.20, 0.05, N)
        het_cv     = _noise(rng, 0.45, 0.05, N, lo=0.10)
        dau        = _noise(rng, team_size * 0.90, 0.8, N, lo=1, hi=team_size)
        revert     = _noise(rng, 0.18, 0.03, N)
        block_rev  = _noise(rng, 0.42, 0.05, N)
        deploy     = _noise(rng, 22, 3, N, lo=5, hi=40).astype(int)
        lead       = _noise(rng, 42, 8, N, lo=4, hi=96)
        incidents  = rng.poisson(2.0, N)
        mttr       = _noise(rng, 8.0, 2.0, N, lo=1.0, hi=24)
        cloud_dau  = _noise(rng, team_size * 0.15, 0.5, N, lo=0, hi=team_size)
        skill_rate = _noise(rng, 0.15, 0.05, N)
        cmd_div    = _noise(rng, 2.0, 0.5, N, lo=1, hi=8)
        plan_use   = _noise(rng, 3, 1, N, lo=0, hi=20)

    elif profile == "low_ai_good":
        agent_acc  = _noise(rng, 0.62, 0.04, N)
        tab_acc    = _noise(rng, 0.55, 0.03, N)
        ask_ratio  = _noise(rng, 0.45, 0.05, N)
        mcp_rate   = _noise(rng, 0.30, 0.05, N)
        het_cv     = _noise(rng, 0.25, 0.03, N, lo=0.05)
        dau        = _noise(rng, team_size * 0.50, 1.0, N, lo=1, hi=team_size)
        revert     = _noise(rng, 0.05, 0.01, N)
        block_rev  = _noise(rng, 0.18, 0.03, N)
        deploy     = _noise(rng, 12, 2, N, lo=3, hi=30).astype(int)
        lead       = _noise(rng, 18, 3, N, lo=4, hi=48)
        incidents  = rng.poisson(0.4, N)
        mttr       = _noise(rng, 3.0, 0.5, N, lo=0.5, hi=12)
        cloud_dau  = _noise(rng, team_size * 0.20, 0.3, N, lo=0, hi=team_size)
        skill_rate = _noise(rng, 0.35, 0.05, N)
        cmd_div    = _noise(rng, 3.5, 0.5, N, lo=1, hi=8)
        plan_use   = _noise(rng, 8, 2, N, lo=0, hi=30)

    elif profile == "uneven":
        agent_acc  = np.clip(0.65 - t * 0.008 + rng.normal(0, 0.02, N), 0.35, 0.80)
        tab_acc    = _noise(rng, 0.60, 0.03, N)
        ask_ratio  = _noise(rng, 0.20, 0.05, N)
        mcp_rate   = _noise(rng, 0.40, 0.05, N)
        het_cv     = np.clip(0.55 + t * 0.015 + rng.normal(0, 0.02, N), 0.40, 0.95)
        dau        = _noise(rng, team_size * 0.60, 1.0, N, lo=1, hi=team_size)
        revert     = np.clip(0.06 + t * 0.007 + rng.normal(0, 0.01, N), 0.03, 0.30)
        block_rev  = np.clip(0.25 + t * 0.008 + rng.normal(0, 0.02, N), 0.10, 0.60)
        deploy     = _noise(rng, 16, 3, N, lo=4, hi=35).astype(int)
        lead       = _noise(rng, 30, 6, N, lo=4, hi=72)
        incidents  = rng.poisson(0.8, N)
        mttr       = _noise(rng, 5.0, 1.0, N, lo=1.0, hi=18)
        cloud_dau  = _noise(rng, team_size * 0.25, 0.5, N, lo=0, hi=team_size)
        skill_rate = _noise(rng, 0.20, 0.05, N)
        cmd_div    = _noise(rng, 2.5, 0.5, N, lo=1, hi=6)
        plan_use   = _noise(rng, 5, 2, N, lo=0, hi=20)

    else:  # low_adoption
        agent_acc  = _noise(rng, 0.58, 0.04, N)
        tab_acc    = _noise(rng, 0.50, 0.04, N)
        ask_ratio  = _noise(rng, 0.28, 0.06, N)
        mcp_rate   = _noise(rng, 0.15, 0.04, N)
        het_cv     = _noise(rng, 0.30, 0.04, N, lo=0.05)
        dau        = _noise(rng, team_size * 0.35, 0.8, N, lo=1, hi=team_size)
        revert     = _noise(rng, 0.08, 0.02, N)
        block_rev  = _noise(rng, 0.28, 0.04, N)
        deploy     = _noise(rng, 10, 2, N, lo=3, hi=25).astype(int)
        lead       = _noise(rng, 26, 5, N, lo=4, hi=60)
        incidents  = rng.poisson(0.6, N)
        mttr       = _noise(rng, 4.0, 1.0, N, lo=0.5, hi=12)
        cloud_dau  = _noise(rng, team_size * 0.10, 0.2, N, lo=0, hi=team_size)
        skill_rate = _noise(rng, 0.10, 0.03, N)
        cmd_div    = _noise(rng, 1.5, 0.4, N, lo=1, hi=5)
        plan_use   = _noise(rng, 2, 1, N, lo=0, hi=10)

    return pd.DataFrame({
        "team_name":              name,
        "sprint_index":           t,
        "sprint_start":           sprint_starts,
        "team_size":              team_size,
        # Group A
        "agent_acceptance_rate":  agent_acc,
        "agent_waste_ratio":      1 - agent_acc,
        "tab_acceptance_rate":    tab_acc,
        "ask_to_total_ratio":     ask_ratio,
        "mcp_adoption_rate":      mcp_rate,
        "usage_heterogeneity_cv": het_cv,
        "dau_mean":               dau,
        "adoption_rate":          dau / team_size,
        "cloud_agent_dau":        cloud_dau,
        "skills_adoption_rate":   skill_rate,
        "command_diversity":      cmd_div,
        "plan_mode_usage":        plan_use,
        "ai_deliberation_ratio":  np.clip(ask_ratio + plan_use / 100, 0, 1),
        "bugbot_resolved_rate":   _noise(rng, 0.60, 0.10, N),
        # Group B
        "deploy_frequency":       deploy,
        "median_lead_time_hours": lead,
        "revert_rate":            revert,
        "blocking_review_ratio":  block_rev,
        "n_prs_merged":           deploy,
        "weighted_prs":           deploy * _noise(rng, 1.8, 0.2, N, lo=1.0, hi=4.0),
        # Group C
        "n_incidents":            incidents.astype(float),
        "mttr_hours":             mttr,
    })


def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    avg_daily_rate   = 300_000 / 220        # R$1,364/day per engineer
    working_days     = 10                   # 2-week sprint
    cursor_per_seat  = 40 / 2              # ~USD 20 per seat per sprint

    df["c_human"] = df["team_size"] * avg_daily_rate * working_days
    df["c_ai_effective"] = (
        cursor_per_seat * df["team_size"]
        / ((df["agent_acceptance_rate"] + df["tab_acceptance_rate"]) / 2)
    )
    df["c_rework"] = (
        df["blocking_review_ratio"] * df["n_prs_merged"] * 200
        + df["revert_rate"]          * df["n_prs_merged"] * 800
    )
    df["c_total"] = df["c_human"] + df["c_ai_effective"] + df["c_rework"]

    baseline_revert    = 0.08
    baseline_incidents = 1.0
    df["quality_multiplier"]   = baseline_revert    / (baseline_revert    + df["revert_rate"])
    df["stability_multiplier"] = baseline_incidents / (baseline_incidents + df["n_incidents"])
    df["v_output"] = df["weighted_prs"] * df["quality_multiplier"] * df["stability_multiplier"]
    df["rodi"]     = df["v_output"] / (df["c_total"] / 10_000)

    return df
