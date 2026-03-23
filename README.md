# AI-Calibrated DORA: Predicting Team Health from AI Tool Usage Signals



## TL;DR

"Every company with AI coding tools asks: is this working? We built a system that answers that question using real production data — not surveys, not vendor benchmarks — and it reveals which specific usage patterns predict whether teams benefit from AI or not."

### The contribution

1. An economic model that treats engineers as production units, making AI tool ROI measurable at the team level
2. Empirical evidence of which AI operational signals predict team health outcomes — the first such analysis in the literature
3. A practical, deployable diagnostic tool that any engineering org can use to make evidence-based AI investment decisions

### The validation

- Ablation study: model with Cursor signals vs without → quantified improvement
- Leave-one-team-out CV: model generalizes across teams, not memorizing
- Comparison with existing frameworks: our signals predict outcomes that DORA alone misses

---

## Project overview

Build a ML system that learns which aspects of AI coding tool usage (Cursor) predict whether engineering teams are healthy or degrading, using real production data from a engineering org.

**Core thesis:** AI coding tools make internal team operations observable for the first time. By modeling software engineers as economic production units, we can use operational signals from AI tool usage to predict team health outcomes — and learn which signals matter most, enabling evidence-based investment decisions.

**One-sentence pitch:** "We used machine learning to discover which operational signals from AI coding tool usage predict team health outcomes, producing the first empirical, team-level diagnostic framework for AI tool ROI in software engineering."

---

## Context and motivation

### The problem

Currently, leadership cannot connect AI tool usage to engineering outcomes. A team could have 35% acceptance rate, rising reverts, concentrated usage in one developer, and nobody sees the connection until incidents spike.

### The gap in existing research


| Study                 | What it found                                     | What it doesn't answer                    |
| --------------------- | ------------------------------------------------- | ----------------------------------------- |
| METR RCT (2025)       | AI makes experienced devs 19% slower              | Which usage patterns cause the slowdown?  |
| DORA Report (2024)    | 25% more AI adoption → 7.2% less stability        | What should a specific team do about it?  |
| Faros AI (2025)       | 21% more tasks, 98% more PRs, 91% longer reviews  | Which teams benefit and which don't? Why? |
| Stack Overflow (2025) | 84% adoption, only 16.3% report significant gains | What predicts whether gains materialize?  |


All existing research stops at "here's the problem." This project goes further: "here's which specific signals predict it, and here's what to do about it."

### Key insight

The cost of AI tokens is economically irrelevant (~0.2% of team cost). The real question is: does AI usage make  engineers MORE or LESS effective? The answer is in the operational signals — acceptance rate, waste, adoption patterns, review friction — not in the dollar cost of tokens.

---

## Data architecture

### Input signals

**Group A — Operational signals (Cursor Enterprise API)**

These describe HOW the team uses AI internally.

**API STATUS:** Cursor Enterprise Analytics API is fully documented with 13 team-level endpoints and 10 by-user endpoints. All signals below map to real API responses. Rate limits: 100 req/min (team), 50 req/min (by-user). Date ranges capped at 30 days; data refreshes every 2 minutes.

**Group A1 — Agent Edits** (`/analytics/team/agent-edits`)


| Signal                  | Description                                 | Why it matters                                                             |
| ----------------------- | ------------------------------------------- | -------------------------------------------------------------------------- |
| `agent_suggested_diffs` | Total diffs suggested by Agent per sprint   | Volume of AI code generation                                               |
| `agent_accepted_diffs`  | Total diffs accepted by team                | Useful output volume                                                       |
| `agent_acceptance_rate` | `accepted_diffs / suggested_diffs`          | Core quality signal — low = bad prompts or poor domain fit                 |
| `agent_waste_ratio`     | `1 - agent_acceptance_rate`                 | Direct measure of unproductive AI generation                               |
| `agent_lines_accepted`  | Total lines (green + red) accepted          | Scale of AI-written code actually kept                                     |
| `agent_net_additions`   | `green_lines_accepted - red_lines_accepted` | Net code growth from Agent (positive = generating, negative = refactoring) |


**Group A2 — Tab Completions** (`/analytics/team/tabs`)


| Signal                | Description                                   | Why it matters                           |
| --------------------- | --------------------------------------------- | ---------------------------------------- |
| `tab_suggestions`     | Total Tab autocomplete suggestions per sprint | Inline completion engagement             |
| `tab_accepts`         | Tab completions accepted                      | Actual inline adoption                   |
| `tab_acceptance_rate` | `accepts / suggestions`                       | Inline fit quality — distinct from Agent |
| `tab_lines_accepted`  | Lines of code accepted via Tab                | Scale of Tab-written code                |


**Group A3 — Interaction Modes** (`/analytics/team/ask-mode`, `/analytics/team/plans`, `/analytics/team/models`)


| Signal                | Description                           | Why it matters                                             |
| --------------------- | ------------------------------------- | ---------------------------------------------------------- |
| `ask_mode_messages`   | Messages sent in Ask mode (read-only) | Measures deliberation / learning before acting             |
| `total_messages`      | Total messages across all models      | Overall chat engagement                                    |
| `ask_to_total_ratio`  | `ask_mode_messages / total_messages`  | High = team thinks before acting; low = pure generation    |
| `plan_mode_usage`     | Plan mode invocations per sprint      | Structured planning behavior                               |
| `model_diversity`     | Count of distinct models used         | Teams using multiple models may be optimizing cost/quality |
| `primary_model_share` | % of messages on the most-used model  | Concentration risk in model selection                      |


**Group A4 — MCP Adoption** (`/analytics/team/mcp`)


| Signal               | Description                             | Why it matters                         |
| -------------------- | --------------------------------------- | -------------------------------------- |
| `mcp_total_calls`    | Total MCP tool invocations per sprint   | Extent of external tool integration    |
| `mcp_unique_tools`   | Distinct MCP tools used                 | Breadth of tooling integration         |
| `mcp_unique_servers` | Distinct MCP servers connected          | Infrastructure sophistication          |
| `mcp_adoption_rate`  | Users invoking MCP / total active users | How broadly the team uses integrations |


**Group A5 — Usage Distribution** (`/analytics/team/dau`, `/analytics/by-user/`*)


| Signal                   | Description                            | Why it matters                                       |
| ------------------------ | -------------------------------------- | ---------------------------------------------------- |
| `dau_mean`               | Average daily active users over sprint | Sustained engagement (not just total)                |
| `dau_cv`                 | Coefficient of variation of daily DAU  | Usage consistency — spiky = unstable adoption        |
| `adoption_rate`          | Active AI users / team size            | Tool penetration                                     |
| `usage_heterogeneity_cv` | CV of per-member lines accepted        | Concentrated risk — one person carrying all AI usage |
| `cloud_agent_dau`        | Daily users of Cloud/Background Agents | Advanced feature adoption                            |


**Group A6 — Commands and Skills** (`/analytics/team/commands`, `/analytics/team/skills`)


| Signal                 | Description                                         | Why it matters                         |
| ---------------------- | --------------------------------------------------- | -------------------------------------- |
| `commands_total`       | Total command invocations (explain, refactor, etc.) | Active use of structured capabilities  |
| `command_diversity`    | Distinct commands used                              | Breadth of feature adoption            |
| `skills_total`         | Total skills invocations per sprint                 | Use of org-defined workflows           |
| `skills_adoption_rate` | Users invoking skills / total active users          | How broadly shared best-practices land |


**Group A7 — AI Code Attribution and Bugbot** (`/analytics/team/bugbot`, dashboard CSV export)


| Signal                 | Description                               | Why it matters                                       |
| ---------------------- | ----------------------------------------- | ---------------------------------------------------- |
| `ai_code_share`        | % of committed lines attributed to Cursor | Ground truth: how much production code is AI-written |
| `bugbot_issues_total`  | Total issues found by BugBot in PRs       | AI-detected quality problems before merge            |
| `bugbot_resolved_rate` | `issues_resolved / issues_total`          | Whether teams act on AI-found bugs                   |
| `bugbot_high_severity` | High-severity issues found per sprint     | Critical quality signal                              |


**Group B — Output signals (GitHub API)**

These describe WHAT the team produces:


| Signal                                | Description                                            | Source                                            |
| ------------------------------------- | ------------------------------------------------------ | ------------------------------------------------- |
| `deploy_frequency`                    | Merged PRs per sprint (proxy for deploys)              | `GET /repos/{org}/{repo}/pulls?state=closed`      |
| `median_lead_time_hours`              | Median time from PR creation to merge                  | PR `created_at` → `merged_at` delta               |
| `change_failure_rate`                 | Reverted PRs / total merged PRs                        | Revert detection in commit messages or linked PRs |
| `revert_rate`                         | Same as CFR — key quality signal                       | Most reliable quality metric, hard to game        |
| `blocking_review_ratio`               | Reviews with CHANGES_REQUESTED / total reviews         | `GET /repos/{org}/{repo}/pulls/{n}/reviews`       |
| `blocking_review_count`               | Absolute count of blocking reviews                     | Filter by `state: CHANGES_REQUESTED`              |
| `total_review_rounds`                 | Total review events per sprint                         | All review types counted                          |
| `n_prs_merged`                        | Raw PR count                                           | Direct from API                                   |
| `weighted_prs`                        | Complexity-weighted PR count (small=1, med=2, large=4) | Based on `changed_files` count                    |
| `total_additions` / `total_deletions` | Lines changed                                          | PR stats                                          |


**Group C — Incident signals (PagerDuty/Opsgenie)**


| Signal        | Description                                                   |
| ------------- | ------------------------------------------------------------- |
| `n_incidents` | Count of incidents per sprint (can be dummy flag 0/1 for MVP) |
| `mttr_hours`  | Mean time to resolve                                          |


**Group D — Team context**


| Signal      | Description                                                         |
| ----------- | ------------------------------------------------------------------- |
| `team_size` | Number of engineers (denominator for all per-capita normalizations) |


### Output (what the model predicts)

**Target variables (next sprint):**


| Target                       | Type                 | Description                       |
| ---------------------------- | -------------------- | --------------------------------- |
| `revert_rate_next`           | Regression (0-1)     | Next sprint's revert rate         |
| `blocking_review_ratio_next` | Regression (0-1)     | Next sprint's review friction     |
| `incident_flag_next`         | Classification (0/1) | Whether next sprint has incidents |


**But the prediction is not the deliverable.** The deliverable is what the model LEARNS:

- Which operational signals most strongly predict quality degradation
- What thresholds matter (e.g., acceptance rate below X → reverts spike)
- How signals interact (e.g., high `agent_lines_accepted` + low `agent_acceptance_rate` = more code generated but low quality fit)
- Whether different team domains have different signal-outcome relationships

### Computed metrics (derived, no API needed)


| Metric                  | Formula                                                             | Purpose                                             |
| ----------------------- | ------------------------------------------------------------------- | --------------------------------------------------- |
| `c_human`               | `team_size × avg_daily_rate × working_days_per_sprint`              | Fixed cost baseline                                 |
| `c_ai_effective`        | `ai_cost_raw / ((agent_acceptance_rate + tab_acceptance_rate) / 2)` | Effective AI spend adjusted for acceptance quality  |
| `c_rework`              | `blocking_reviews × review_cost + reverts × revert_cost`            | Quality debt cost                                   |
| `v_output`              | `weighted_prs × quality_mult × stability_mult`                      | Quality-adjusted output                             |
| `quality_multiplier`    | `baseline_revert / (baseline_revert + actual_revert)`               | Bounded 0-1 decay                                   |
| `stability_multiplier`  | `baseline_incidents / (baseline_incidents + actual_incidents)`      | Bounded 0-1 decay                                   |
| `rodi`                  | `v_output / c_total`                                                | Return on dev investment                            |
| `ai_deliberation_ratio` | `(ask_mode_messages + plan_mode_usage) / total_messages`            | Measures "thinking before doing" vs pure generation |
| `mcp_integration_depth` | `mcp_unique_servers × mcp_adoption_rate`                            | Breadth × penetration of external tool integrations |


---

## Model architecture

### Primary model: XGBoost

Best fit for the data reality: tabular features, small N (~100-200 observations), need for feature importance. SHAP decomposition gives per-team, per-prediction feature contributions.

### Comparison models


| Model                 | Purpose                                                            |
| --------------------- | ------------------------------------------------------------------ |
| Linear regression     | Baseline — can simple correlations explain outcomes?               |
| Bayesian hierarchical | Uncertainty estimates on predictions, partial pooling across teams |
| Random forest         | Ensemble comparison point                                          |


### Feature selection pipeline

1. Correlation matrix — drop features with r > 0.9 (e.g., `agent_lines_accepted` and `agent_accepted_diffs`, `tab_lines_accepted` and `tab_accepts`)
2. VIF check — remove multicollinear features
3. XGBoost feature importance — first pass ranking
4. SHAP analysis — per-feature, per-prediction decomposition

### Validation strategy

- Time-based split: train on sprints 1-8, test on sprints 9-12
- Leave-one-team-out cross-validation: train on N-1 teams, test on held-out team
- Compare: metrics-only baseline vs operational-signals-included model

---

## Project phases

### Phase 1 — Mock data and metric pipeline

**Status:** In progress
**Dependencies:** None
**Goal:** Build the complete data pipeline with client classes that support both mock and real data. Each client takes `mock: bool = True` in its constructor — when `mock=True` it generates synthetic data matching real API schemas, when `mock=False` it calls the real API. No separate mock layer to swap.

#### Files to create

```
src/
├── config.py              # Sprint cadence, cost parameters, team profiles
├── clients/
│   ├── __init__.py        # Exports CursorClient, GitHubClient, IncidentClient
│   ├── cursor.py          # CursorClient(mock=True) — 12 methods matching Cursor Enterprise API endpoints
│   ├── github.py          # GitHubClient(mock=True) — PR and review methods
│   └── incident.py        # IncidentClient(mock=True) — PagerDuty/Opsgenie incidents
├── metrics/
│   ├── __init__.py
│   ├── extraction.py      # extract_sprint_metrics() — clients → flat row
│   ├── costs.py           # compute_costs() — C_human, C_ai, C_rework
│   ├── output_value.py    # compute_output_rodi() — V_output, RoDI
│   └── diagnostics.py     # generate_recommendations() — pattern matching
├── data/
│   └── generate.py        # generate_dataset() — teams × sprints → CSV
└── notebooks/
    └── 01_explore.ipynb   # EDA on generated dataset
```

#### Step by step

1. Create `src/config.py`
  - `SprintConfig` dataclass: `duration_days`, `num_sprints`, `start_date`
  - `TeamProfile` dataclass: all team parameters (size, AI adoption rates, quality baselines)
  - Define 5-8 team archetypes covering: high AI/good quality, high AI/bad quality, low AI/good quality, low AI/bad quality, uneven adoption, etc.
  - Cost constants: `AVG_YEARLY_SALARY_BRL = 300_000`, `CURSOR_MONTHLY_SEAT_USD = 40`, `AVG_REVIEW_COST_USD`, `AVG_REVERT_COST_USD`
2. Create `src/clients/github.py` — `GitHubClient(mock: bool = True, token: str | None = None)`
  - `get_pull_requests(org, repo, start_date, end_date, state="closed") -> list[dict]` — `GET /repos/{org}/{repo}/pulls`
  - Return schema matches GitHub API: `id`, `number`, `title`, `state`, `created_at`, `merged_at`, `merged`, `user`, `changed_files`, `additions`, `deletions`
  - When `mock=True`: add `_meta` field for simulation-only data: `complexity`, `is_reverted`, `team`, `sprint_start`
  - `get_reviews(org, repo, pr_number) -> list[dict]` — `GET /repos/{org}/{repo}/pulls/{n}/reviews`
  - When `mock=False`: uses PyGithub or raw requests, handles pagination and rate limiting
  - Each method docstring documents the real API endpoint path and response fields
3. Create `src/clients/cursor.py` — `CursorClient(mock: bool = True, api_key: str | None = None, base_url: str = "https://api.cursor.com")`
  - One public method per real Cursor Enterprise API endpoint, each returning data matching the documented JSON schema. When `mock=True`, generates synthetic data; when `mock=False`, calls the real endpoint with Basic auth:
    - `get_agent_edits(start_date, end_date, users?) -> list[dict]` — `/analytics/team/agent-edits`; fields: `event_date`, `total_suggested_diffs`, `total_accepted_diffs`, `total_rejected_diffs`, `total_green_lines_accepted`, `total_red_lines_accepted`, `total_lines_suggested`, `total_lines_accepted`
    - `get_tabs(start_date, end_date, users?) -> list[dict]` — `/analytics/team/tabs`; same line-count structure
    - `get_dau(start_date, end_date, users?) -> list[dict]` — `/analytics/team/dau`; fields: `date`, `dau`, `cli_dau`, `cloud_agent_dau`, `bugbot_dau`
    - `get_models(start_date, end_date, users?) -> list[dict]` — `/analytics/team/models`; fields: `date`, `model_breakdown` (model → messages + users dict)
    - `get_mcp(start_date, end_date, users?) -> list[dict]` — `/analytics/team/mcp`; fields: `event_date`, `tool_name`, `mcp_server_name`, `usage`
    - `get_ask_mode(start_date, end_date, users?) -> list[dict]` — `/analytics/team/ask-mode`; fields: `event_date`, `model`, `usage`
    - `get_plans(start_date, end_date, users?) -> list[dict]` — `/analytics/team/plans`; fields: `event_date`, `model`, `usage`
    - `get_commands(start_date, end_date, users?) -> list[dict]` — `/analytics/team/commands`; fields: `event_date`, `command_name`, `usage`
    - `get_skills(start_date, end_date, users?) -> list[dict]` — `/analytics/team/skills`; fields: `event_date`, `skill_name`, `usage`
    - `get_bugbot(start_date, end_date, repo?, pr_state?) -> list[dict]` — `/analytics/team/bugbot`; fields: `pr_number`, `timestamp`, `reviews`, `issues` (total + by_severity), `issues_resolved`
    - `get_agent_edits_by_user(start_date, end_date) -> dict[str, list]` — `/analytics/by-user/agent-edits`; keyed by user email
    - `get_tabs_by_user(start_date, end_date) -> dict[str, list]` — `/analytics/by-user/tabs`; used to compute `usage_heterogeneity_cv`
  - Mock mode models usage heterogeneity via by-user methods: power-law distribution for high-heterogeneity team profiles, near-uniform for low
4. Create `src/clients/incident.py` — `IncidentClient(mock: bool = True, api_key: str | None = None, provider: str = "pagerduty")`
  - `get_incidents(service, start_date, end_date) -> list[dict]`
  - Schema mirrors PagerDuty: `id`, `title`, `status`, `severity`, `created_at`, `resolved_at`, `service`, `time_to_resolve_hours`
  - Mock mode: Poisson-distributed incident count per sprint
5. Create `src/metrics/extraction.py`
  - `extract_sprint_metrics(team, sprint_start, sprint_end, sprint_index, *, cursor_client, github_client, incident_client) -> dict`
  - Receives client instances (callers pass `mock=True` or `mock=False`) — extraction logic is identical regardless
  - Calls all three clients, computes all raw features, returns flat dict (one row)
6. Create `src/metrics/costs.py`
  - `compute_costs(df, config) -> DataFrame`
  - Add columns: `c_human`, `c_ai_effective` (raw / acceptance), `c_rework`, `c_total`
  - Use realistic BRL-based cost constants
7. Create `src/metrics/output_value.py`
  - `compute_output_and_rodi(df) -> DataFrame`
  - Quality multiplier: `baseline / (baseline + actual)` — bounded, no negative values
  - Stability multiplier: same pattern on incidents
  - V_output = weighted_prs × quality × stability
  - RoDI = V_output / C_total
  - AI marginal efficiency = V_output / C_ai_effective
8. Create `src/metrics/diagnostics.py`
  - `generate_recommendations(df) -> DataFrame`
  - Rule-based recommendations as starting point (later replaced by model findings)
  - Flag: low acceptance, high heterogeneity, high revert rate, low adoption, high cost per task
9. Create `src/data/generate.py`
  - `generate_dataset(teams, config) -> DataFrame`
  - Iterate teams × sprints, call extraction, compute costs, compute RoDI
  - Export to `data/sprint_metrics.csv`
10. Create `notebooks/01_explore.ipynb`
  - Load CSV, basic EDA
    - Correlation matrix across all features
    - Distribution plots per team
    - Raw DORA rank vs RoDI rank comparison
    - Identify which features have meaningful variance

#### Done when

- All files created with clean interfaces
- `python -m src.data.generate` produces a CSV with 60+ rows (5 teams × 12 sprints)
- Notebook runs end-to-end showing team comparisons
- Every client method docstring documents the real API endpoint and response fields
- Switching from mock to real data requires only `mock=False` + credentials — no code changes downstream

---

### Phase 2 — Feature engineering and model training

**Dependencies:** Phase 1 complete, CSV dataset available
**Goal:** Train the prediction model, extract feature importance, learn which signals matter.

#### Files to create

```
src/
├── features/
│   ├── __init__.py
│   ├── selection.py       # Correlation filter, VIF, importance ranking
│   └── engineering.py     # Derived features, interaction terms, lag features
├── models/
│   ├── __init__.py
│   ├── baseline.py        # Linear regression baseline
│   ├── xgboost_model.py   # Primary model + SHAP analysis
│   └── evaluation.py      # Cross-validation, metrics, comparison
└── notebooks/
    ├── 02_features.ipynb   # Feature selection analysis
    └── 03_model.ipynb      # Model training and SHAP interpretation
```

#### Step by step

1. Create `src/features/engineering.py`
  - Add lag features: previous sprint's revert rate, acceptance rate, etc.
  - Add rolling averages: 3-sprint rolling mean for key signals
  - Add interaction terms: `agent_suggested_diffs × agent_waste_ratio` = total wasted agent effort; `tab_suggestions × (1 - tab_acceptance_rate)` = total wasted inline completions
  - Add rate-of-change features: is acceptance rate trending up or down?
  - Create target columns: `revert_rate_next`, `blocking_review_ratio_next`, `incident_flag_next` (shifted by one sprint)
2. Create `src/features/selection.py`
  - `correlation_filter(df, threshold=0.9) -> list[str]` — drop highly correlated features
  - `vif_filter(df, threshold=10) -> list[str]` — drop multicollinear features
  - `importance_ranking(df, target) -> DataFrame` — quick XGBoost fit, return feature importances
  - `select_features(df, target) -> tuple[list[str], DataFrame]` — full pipeline, return selected features and analysis
3. Create `src/models/baseline.py`
  - `train_baseline(X_train, y_train) -> dict` — linear regression
  - Return: model, coefficients, R², predictions
  - This is the "can simple correlations explain it?" check
4. Create `src/models/xgboost_model.py`
  - `train_xgboost(X_train, y_train, X_test, y_test) -> dict`
  - Return: model, predictions, feature importance, SHAP values
  - `explain_team(model, team_data) -> dict` — per-team SHAP decomposition
  - `explain_prediction(model, single_row) -> dict` — per-prediction SHAP waterfall
  - Key hyperparameters: keep simple, use early stopping, max_depth=4-6
5. Create `src/models/evaluation.py`
  - `time_split_cv(df, train_sprints, test_sprints) -> dict` — temporal validation
  - `leave_team_out_cv(df, models) -> dict` — generalization check
  - `compare_models(results) -> DataFrame` — side by side: baseline vs XGBoost
  - `ablation_study(df, feature_groups) -> DataFrame` — train with/without Cursor signals to show they add predictive value
  - Metrics: RMSE, MAE, R² for regression targets; AUC, F1 for incident classification
6. Create `notebooks/02_features.ipynb`
  - Run feature selection pipeline
  - Visualize: correlation heatmap, VIF scores, importance bar chart
  - Document which features survive and why
  - Show the feature groups: Cursor-only, GitHub-only, combined
7. Create `notebooks/03_model.ipynb`
  - Train all models
  - **Key experiment:** compare GitHub-only model vs GitHub+Cursor model
  - If Cursor signals improve prediction → thesis validated
  - SHAP summary plot: which features matter globally
  - SHAP per-team: which features matter for each team's predictions
  - Identify thresholds: at what acceptance rate do reverts spike?
  - Identify interactions: does team size moderate the effect of `agent_lines_accepted` volume? Does `ai_deliberation_ratio` interact with `agent_acceptance_rate`?

#### Done when

- Ablation study shows whether Cursor signals add predictive power over GitHub-only
- Top 5 most important features identified with SHAP
- At least one non-obvious finding (interaction, threshold, or pattern not predicted by intuition)
- Model generalizes in leave-one-team-out CV (not just memorizing team identity)

---

### Phase 3 — Streamlit dashboard

**Dependencies:** Phase 2 complete, trained model available
**Goal:** Present findings to engineering leadership in an actionable interface.

#### Files to create

```
streamlit_app/
├── app.py                 # Main entry, sidebar navigation, data source toggle
├── pages/
│   ├── 01_overview.py     # Org-wide health summary, team ranking
│   ├── 02_team_detail.py  # Single team deep dive, SHAP waterfall
│   ├── 03_findings.py     # Global findings: which signals matter, thresholds
│   └── 04_roi_curve.py    # AI spend vs output curve, team positions
├── components/
│   ├── charts.py          # Plotly/Altair chart helpers
│   └── metrics.py         # Metric card components
└── data/
    ├── load.py            # Data loading, caching
    └── model.py           # Model loading, prediction, SHAP computation
```

#### Step by step

1. Create `streamlit_app/app.py`
  - Sidebar: `mock` toggle (sets `mock=True/False` on all clients), sprint range selector, team filter
  - Page navigation using Streamlit multipage
  - Load dataset and model on startup with `@st.cache_data`
2. Create `streamlit_app/pages/01_overview.py`
  - Org-wide metrics: average RoDI, average acceptance rate, total AI spend
  - Team ranking table: raw DORA rank vs RoDI rank, with rank delta highlighted
  - Color-coded: green for teams where RoDI rank > DORA rank (undervalued), red for opposite (inflated)
  - Trend sparklines per team over sprints
3. Create `streamlit_app/pages/02_team_detail.py`
  - Team selector dropdown
  - Sprint-over-sprint charts: RoDI, acceptance rate, revert rate, review churn
  - Cost breakdown: C_human (fixed bar) + C_ai_effective + C_rework (variable, stacked)
  - SHAP waterfall for latest sprint: "here's why your team's prediction is what it is"
  - Model-generated recommendations (from Phase 2 findings, not hard-coded rules)
4. Create `streamlit_app/pages/03_findings.py`
  - This is the thesis results page
  - Global SHAP summary: ranked feature importance bar chart
  - Threshold discovery: scatter plots showing inflection points (e.g., acceptance rate vs revert rate)
  - Ablation results: "adding Cursor signals improved prediction by X%"
  - Key finding cards: the 3-5 most important learnings from the model
5. Create `streamlit_app/pages/04_roi_curve.py`
  - X-axis: AI effective spend (`c_ai_effective`) or total lines suggested per sprint
  - Y-axis: quality-adjusted output (V_output) or RoDI
  - Each dot = one team-sprint observation, colored by team
  - Fitted curve showing diminishing returns
  - Team position annotations: "your team is HERE on the curve"
  - If domains differ, show separate curves per domain
6. Create `streamlit_app/components/charts.py` and `metrics.py`
  - Reusable chart functions: SHAP waterfall, feature importance bar, trend line, scatter with fitted curve
  - Metric card component: value, delta from previous sprint, color coding

#### Done when

- Dashboard runs with `streamlit run streamlit_app/app.py`
- All four pages render with mock data
- SHAP visualizations are interpretable by non-technical reader
- Switching from mock to real requires only `mock=False` + credentials in sidebar toggle

---

### Phase 4 — Real API integration

**Dependencies:** Data access approved by SumUp leadership
**Goal:** Implement the `mock=False` code paths in each client, validate findings on production data.

No new files to create — Phase 1 clients already define the full interface. This phase fills in the `_fetch`_* methods.

#### Step by step

1. **Validate Cursor API endpoint mapping against production**
  - For each of the 12 methods in `CursorClient`, confirm the real endpoint returns the expected schema
  - Note the 30-day date range limit: add date-range chunking inside `CursorClient._fetch`_* methods when date spans > 30 days
  - Note that `ai_code_share` (Group A7) is not available via API — retrieve via dashboard CSV export or derive from `agent_lines_accepted + tab_lines_accepted` vs total committed lines
  - Document confirmed field mappings in `docs/cursor_api_mapping.md`
2. Implement `mock=False` path in `src/clients/github.py`
  - Fill `_fetch_pull_requests` and `_fetch_reviews` using PyGithub or raw requests
  - Handle pagination, rate limiting, error recovery
  - Map GitHub response to the same dict schema the mock path returns
3. Implement `mock=False` path in `src/clients/cursor.py`
  - Fill all 12 `_fetch_`* methods using `requests` with Basic auth
  - Handle 30-day date-range chunking and 100 req/min rate limits
4. Implement `mock=False` path in `src/clients/incident.py`
  - Fill `_fetch_incidents` using PagerDuty REST API or Opsgenie equivalent
5. **Draft data access request for SumUp leadership**
  - Exactly which GitHub API scopes needed (read-only: repos, pulls, reviews)
  - Exactly which Cursor enterprise endpoints needed
  - Data governance: team-level aggregation only, no individual tracking
  - LGPD compliance statement
  - Expected data volume: N teams × M sprints × K features
  - What leadership gets: the dashboard and findings
6. **Run on real data and compare**
  - Instantiate clients with `mock=False` + credentials; no other code changes needed
  - Retrain model on real data
  - Compare: do the same signals matter? Same thresholds? Same patterns?
  - If synthetic and real agree → strong validation
  - If they diverge → document why, adjust model, discuss in thesis

#### Done when

- All `_fetch_`* methods implemented and tested
- `mock=False` produces real data through the same pipeline
- Model retrained on real SumUp data
- Findings compared: synthetic vs real
- Dashboard running on production data

---

## Technical stack


| Component      | Tool                  | Why                                    |
| -------------- | --------------------- | -------------------------------------- |
| Language       | Python 3.11+          | Ecosystem for ML + data                |
| Data           | pandas, numpy         | Standard tabular data                  |
| ML             | scikit-learn, xgboost | XGBoost primary, sklearn for baselines |
| Explainability | shap                  | Per-prediction feature decomposition   |
| Stats          | scipy, statsmodels    | VIF, correlation, statistical tests    |
| Visualization  | plotly, altair        | Interactive charts in Streamlit        |
| Dashboard      | streamlit             | Fast prototyping, thesis demo          |
| API clients    | requests, PyGithub    | GitHub API access                      |
| Notebooks      | jupyter               | Exploration and thesis figures         |
| Testing        | pytest                | Unit tests for metric computation      |


## Environment setup

```bash
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn xgboost shap scipy statsmodels plotly altair streamlit requests PyGithub jupyter pytest
```

---

## File tree (complete)

```
thesis-ai-dora/
├── PROJECT_PLAN.md            # This file
├── README.md                  # Quick start
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project config
├── src/
│   ├── __init__.py
│   ├── config.py              # All configuration and team profiles
│   ├── clients/               # mock=True for Phase 1, mock=False for Phase 4
│   │   ├── __init__.py
│   │   ├── cursor.py          # CursorClient — 12 methods, Cursor Enterprise API
│   │   ├── github.py          # GitHubClient — PRs and reviews
│   │   └── incident.py        # IncidentClient — PagerDuty/Opsgenie
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── extraction.py
│   │   ├── costs.py
│   │   ├── output_value.py
│   │   └── diagnostics.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── selection.py
│   │   └── engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── xgboost_model.py
│   │   └── evaluation.py
│   └── data/
│       └── generate.py
├── streamlit_app/
│   ├── app.py
│   ├── pages/
│   │   ├── 01_overview.py
│   │   ├── 02_team_detail.py
│   │   ├── 03_findings.py
│   │   └── 04_roi_curve.py
│   ├── components/
│   │   ├── charts.py
│   │   └── metrics.py
│   └── data/
│       ├── load.py
│       └── model.py
├── notebooks/
│   ├── 01_explore.ipynb
│   ├── 02_features.ipynb
│   └── 03_model.ipynb
├── data/
│   ├── sprint_metrics.csv     # Generated dataset
│   └── models/                # Saved trained models
├── docs/
│   ├── cursor_api_mapping.md
│   ├── data_access_request.md
│   └── thesis_proposal.md
└── tests/
    ├── test_metrics.py
    ├── test_clients.py
    └── test_models.py
```

