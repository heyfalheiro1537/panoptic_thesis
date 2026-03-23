"""Hard-coded model outputs for the Phase 1 dashboard demo.

Represents what the trained XGBoost + SHAP pipeline will produce in Phase 2.
All values are realistic placeholders grounded in the thesis hypotheses.
Replace with actual model artifacts once Phase 2 training is complete.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Global feature importance (SHAP-based, target = revert_rate_next)
# ---------------------------------------------------------------------------

FEATURE_IMPORTANCE = pd.DataFrame(
    {
        "feature": [
            "agent_acceptance_rate",
            "usage_heterogeneity_cv",
            "blocking_review_ratio",
            "ask_to_total_ratio",
            "tab_acceptance_rate",
            "mcp_adoption_rate",
            "ai_deliberation_ratio",
            "revert_rate",
            "dau_mean",
            "cloud_agent_dau",
            "plan_mode_usage",
            "command_diversity",
        ],
        "importance": [0.22, 0.18, 0.14, 0.11, 0.09, 0.07, 0.06, 0.05, 0.04, 0.02, 0.01, 0.01],
        "group": [
            "Cursor A1", "Cursor A5", "GitHub B",  "Cursor A3",
            "Cursor A2", "Cursor A4", "Cursor A3",  "GitHub B",
            "Cursor A5", "Cursor A5", "Cursor A3",  "Cursor A6",
        ],
    }
)

GROUP_COLORS = {
    "Cursor A1": "#2ecc71",
    "Cursor A2": "#27ae60",
    "Cursor A3": "#3498db",
    "Cursor A4": "#2980b9",
    "Cursor A5": "#9b59b6",
    "Cursor A6": "#8e44ad",
    "GitHub B":  "#e67e22",
}

# ---------------------------------------------------------------------------
# Ablation study results
# ---------------------------------------------------------------------------

ABLATION_RESULTS = {
    "github_only":   {"rmse": 0.072, "mae": 0.055, "r2": 0.41, "auc": 0.61},
    "github_cursor": {"rmse": 0.048, "mae": 0.036, "r2": 0.73, "auc": 0.84},
    "improvement_pct": 33.3,
    "r2_improvement_pct": 78.0,
}

# ---------------------------------------------------------------------------
# Per-team SHAP values for the latest sprint
# contributions: [(feature_name, shap_value, feature_value)]
# positive shap_value = increases predicted revert rate (bad)
# negative shap_value = decreases predicted revert rate (good)
# ---------------------------------------------------------------------------

SHAP_VALUES: dict[str, dict] = {
    "Alpha": {
        "base_value": 0.080,
        "prediction": 0.044,
        "contributions": [
            ("agent_acceptance_rate", -0.022, 0.72),
            ("usage_heterogeneity_cv", -0.009, 0.18),
            ("ask_to_total_ratio",     -0.006, 0.35),
            ("mcp_adoption_rate",      -0.004, 0.80),
            ("blocking_review_ratio",  +0.005, 0.18),
        ],
    },
    "Beta": {
        "base_value": 0.080,
        "prediction": 0.197,
        "contributions": [
            ("agent_acceptance_rate", +0.052, 0.41),
            ("usage_heterogeneity_cv", +0.033, 0.45),
            ("blocking_review_ratio",  +0.022, 0.42),
            ("ask_to_total_ratio",     +0.016, 0.08),
            ("mcp_adoption_rate",      +0.008, 0.20),
            ("ai_deliberation_ratio",  -0.014, 0.11),
        ],
    },
    "Gamma": {
        "base_value": 0.080,
        "prediction": 0.059,
        "contributions": [
            ("agent_acceptance_rate",  -0.014, 0.62),
            ("ask_to_total_ratio",     -0.012, 0.45),
            ("usage_heterogeneity_cv", -0.006, 0.25),
            ("blocking_review_ratio",  +0.009, 0.18),
            ("mcp_adoption_rate",      +0.002, 0.30),
        ],
    },
    "Delta": {
        "base_value": 0.080,
        "prediction": 0.152,
        "contributions": [
            ("usage_heterogeneity_cv", +0.042, 0.82),
            ("blocking_review_ratio",  +0.024, 0.37),
            ("ask_to_total_ratio",     +0.013, 0.20),
            ("agent_acceptance_rate",  -0.010, 0.57),
            ("mcp_adoption_rate",      +0.003, 0.40),
        ],
    },
    "Epsilon": {
        "base_value": 0.080,
        "prediction": 0.091,
        "contributions": [
            ("mcp_adoption_rate",      +0.018, 0.15),
            ("usage_heterogeneity_cv", +0.006, 0.30),
            ("agent_acceptance_rate",  -0.008, 0.58),
            ("ask_to_total_ratio",     -0.004, 0.28),
            ("blocking_review_ratio",  -0.001, 0.28),
        ],
    },
}

# ---------------------------------------------------------------------------
# Rule-based recommendations (pre-model; will be replaced by SHAP findings)
# ---------------------------------------------------------------------------

RECOMMENDATIONS: dict[str, list[dict]] = {
    "Alpha": [
        {
            "icon": "✅",
            "title": "Healthy AI usage pattern",
            "detail": "Acceptance rate (72%) and deliberation ratio (35%) are both strong. "
                      "Consider sharing prompt engineering practices with other teams.",
            "severity": "info",
        }
    ],
    "Beta": [
        {
            "icon": "🔴",
            "title": "Low acceptance rate — review prompt quality",
            "detail": "Agent acceptance rate is 41%, well below the 52% risk threshold. "
                      "Teams in this range show 3× higher revert rates. Recommend structured "
                      "prompt review sessions.",
            "severity": "error",
        },
        {
            "icon": "🔴",
            "title": "Concentrated AI usage (CV=0.45) — pair on AI workflows",
            "detail": "One or two engineers are generating the majority of AI code. "
                      "This creates knowledge risk and inflates blocking reviews when "
                      "reviewers lack context.",
            "severity": "error",
        },
        {
            "icon": "🟡",
            "title": "Low deliberation ratio (8%) — add Ask/Plan mode before coding",
            "detail": "Teams that spend <10% of messages in Ask or Plan mode before "
                      "generating code show consistently higher revert rates at scale.",
            "severity": "warning",
        },
    ],
    "Gamma": [
        {
            "icon": "🟡",
            "title": "Low AI adoption (50% DAU) — expand tool usage",
            "detail": "Only half the team uses AI features daily. Given the strong "
                      "acceptance rate, expanding adoption could increase output "
                      "without degrading quality.",
            "severity": "warning",
        }
    ],
    "Delta": [
        {
            "icon": "🔴",
            "title": "Usage concentration increasing — intervene now",
            "detail": "Heterogeneity CV has risen from 0.55 → 0.82 over the last 12 "
                      "sprints. The top user is carrying an increasing share of AI "
                      "output. Revert rate is trending up correspondingly.",
            "severity": "error",
        },
        {
            "icon": "🟡",
            "title": "Acceptance rate declining — monitor for further drop",
            "detail": "Acceptance rate has trended from 0.65 → 0.57 over the "
                      "observation window. If it crosses 0.52, expect a step-change "
                      "in revert rate.",
            "severity": "warning",
        },
    ],
    "Epsilon": [
        {
            "icon": "🟡",
            "title": "Low MCP adoption (15%) — integrate external tools",
            "detail": "Teams with higher MCP adoption show 18% fewer blocking reviews "
                      "on average. Connecting Cursor to your issue tracker and CI "
                      "system is a low-effort starting point.",
            "severity": "warning",
        }
    ],
}

# ---------------------------------------------------------------------------
# Key findings cards
# ---------------------------------------------------------------------------

KEY_FINDINGS = [
    {
        "title": "Acceptance rate below 52% triples revert risk",
        "detail": (
            "Teams with agent_acceptance_rate < 0.52 show revert rates 3.1× higher "
            "than teams above 0.65. The relationship is non-linear with a sharp "
            "inflection at the 0.52 boundary — a natural alert threshold."
        ),
        "signal": "agent_acceptance_rate",
        "threshold": 0.52,
        "group": "Cursor A1",
    },
    {
        "title": "Usage concentration predicts review friction",
        "detail": (
            "When the top user accounts for >50% of all AI lines (heterogeneity CV > 0.50), "
            "blocking_review_ratio rises by +0.18 on average. One person cannot carry "
            "the team's AI knowledge without creating review bottlenecks."
        ),
        "signal": "usage_heterogeneity_cv",
        "threshold": 0.50,
        "group": "Cursor A5",
    },
    {
        "title": "Deliberation ratio protects quality at scale",
        "detail": (
            "Teams spending >30% of messages in Ask or Plan mode before acting maintain "
            "stable revert rates even as AI code volume grows. Pure generation without "
            "deliberation degrades quality as complexity increases."
        ),
        "signal": "ai_deliberation_ratio",
        "threshold": 0.30,
        "group": "Cursor A3",
    },
    {
        "title": "Cursor signals add 78% more explained variance over DORA alone",
        "detail": (
            "An XGBoost model trained on GitHub signals alone achieves R²=0.41. "
            "Adding the 30 Cursor operational signals improves R² to 0.73 — a 78% "
            "relative improvement. The Cursor signals are not redundant with DORA; "
            "they capture process, not just output."
        ),
        "signal": "model_improvement",
        "threshold": None,
        "group": "Model",
    },
]
