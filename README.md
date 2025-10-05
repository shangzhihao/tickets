# Intelligent Ticket Analytics Platform

## Overview
This repository will host an end-to-end system for learning from historical support tickets and assisting agents in real time. The planned solution combines classical analytics, machine learning, and retrieval-augmented generation (RAG) to triage tickets, recommend resolutions, surface anomalies, and measure team performance.

## What We Will Build
- **Hydra-driven CLI** exposed through `python main.py analytics ...` that orchestrates ingestion, analytics, modeling, and reporting pipelines.
- **Analytics bundle** that computes resolution speed, SLA compliance, CSAT drivers, and agent performance metrics with optional artifact persistence under `outputs/`.
- **Intelligent processing engine** featuring:
  - Multi-model ticket categorization (gradient boosting + deep learning) with experiment tracking.
  - Hybrid RAG + graph-powered retrieval tying tickets, products, issues, knowledge base entries, and resolutions together.
  - Anomaly detection to flag emerging issues, sentiment shifts, and retrieval failures.
- **Lifecycle automation** covering model retraining, evaluation, and deployment-ready packaging (containers, configs, monitoring hooks).

## Repository Layout
```
src/
  tickets/
    analytics/         # Upcoming analytics pipeline modules (args, loader, metrics, drivers, reporting, pipeline)
    api/               # Future API endpoints for ticket intake and solution delivery
    data/              # Data adapters and feature loading utilities
    models/            # Training scripts, metadata, and model registries
    schemas/           # Pydantic/Dataclass schemas for configs and artifacts
    utils/             # Shared helpers (date parsing, logging wrappers, etc.)
assets/                # Sample datasets and configuration docs (to be documented when added)
outputs/               # Persisted analytics reports, models, logs
conf/                  # Hydra configuration tree (to be introduced)
```

## Getting Started
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -U pip`
3. `pip install -e .[dev]`  ↳ once `pyproject.toml` exposes `[project.optional-dependencies]` entries.
4. Run formatting and linting with `python -m black .` and `ruff check .` before committing.

## CLI Usage (planned)
```bash
python main.py analytics --start-date 2024-01-01 --end-date 2024-01-31 \
  --agent-id AGENT-101 --output-dir outputs/analytics --refresh-models
```
- `--dry-run` prints the report without writing artifacts.
- `--refresh-models` retrains satisfaction driver models and updates stored weights.
- Arguments will be parsed via `tickets.analytics.args.parse_analytics_args` to keep Hydra configs decoupled from CLI concerns.

## Data Requirements
- Primary input: bronze/offline parquet or JSON files with columns such as `ticket_id`, `created_at`, `resolved_at`, `status`, `priority`, `channel`, `queue`, `agent_id`, `satisfaction_score`, `sla_breach`, and `reopened_count`.
- Optional joins: agent roster, customer segmentation, SLA targets by queue. Place small fixtures in `assets/` and document them here when committed.
- Analytics pipeline outputs JSON/CSV/Parquet bundles in `outputs/analytics/` and serialized models (e.g., `outputs/models/satisfaction_model.pkl`).

## Testing Strategy
- Use `pytest` for unit/integration coverage; target critical-path behaviors (date parsing, SLA math, model fallbacks).
- Run `pytest --maxfail=1 --disable-warnings -q` locally and in CI.
- Provide fixture datasets under `tests/fixtures/` for deterministic analytics and modeling tests.

## Roadmap Highlights
1. Extend Hydra config schemas to cover analytics toggles, output targets, and data sources.
2. Implement CLI argument parsing and command dispatch in `main.py`.
3. Build data loaders, transformers, and metric calculators in `tickets/analytics/`.
4. Train satisfaction driver models with feature importance reporting and artifact persistence.
5. Wire reporting layer for CLI output plus JSON/CSV/Parquet exports.
6. Add anomaly detection and graph-enhanced RAG retrieval for ticket recommendations.
7. Document architecture, model benchmarks, and operational playbooks alongside the code.

## Contributing & Next Steps
- Keep modules focused and documented; favor composition over monoliths.
- Maintain clean commit history with imperative subjects (e.g., `Add analytics pipeline skeleton`).
- Use issues/PRs to track open questions (dataset gaps, modeling trade-offs, deployment needs).
- Upcoming tasks include adding `pyproject.toml`, codifying dev dependencies, and standing up the first analytics module skeletons.
