# Intelligent Ticket Support Platform

This repository hosts the foundation of an intelligent product support system that learns from historical ticket data to triage, resolve, and monitor incoming requests. The implementation prioritizes reproducible data flows, modular machine-learning pipelines, and API-ready components that can be composed into production deployments.

> **Current status:** Data ingestion, offline analytics, configuration, and logging pieces are functional. Modeling, retrieval, anomaly detection, and FastAPI orchestration are under active development with configuration scaffolding already in place.

## Platform Highlights
- **Prefect-powered ingestion** creates bronze/offline/online Parquet datasets in S3/MinIO and publishes online-ready slices into Redis JSON.
- **Pydantic-based configuration** keeps environments reproducible; defaults live in code and `.env` entries or environment variables override them at runtime.
- **Structured logging with Loguru** outputs module-scoped logs (`data`, `ml`, `api`) to the `logs/` directory and stdout.
- **Experiment infrastructure** combines MLflow tracking targets with S3 artifact storage and Redis feature serving.
- **Model configuration stubs** for CatBoost, XGBoost, and deep learning pipelines support forthcoming Optuna/Bayesian tuning and ensemble workflows.

## Repository Layout
```
docker/                    # Local stack (MinIO, MLflow, Redis) via Docker Compose
logs/                      # Structured logs written by Loguru handlers
outputs/                   # Placeholder for generated artifacts and analysis exports
src/tickets/
  data/                    # Prefect flows/tasks for ingestion and offline analytics
  schemas/                 # Pydantic v2 models (tickets, metrics, task enums)
  utils/                   # Shared configuration helpers, logging, and clients
tests/                     # Pytest suite (add tests alongside new features)
uv.lock                    # Locked dependency set managed by uv
```

## Quickstart
1. Install [uv](https://github.com/astral-sh/uv) (recommended via `pipx install uv`).
2. Sync the project environment:
   ```bash
   uv sync
   ```
3. Start local dependencies:
   ```bash
   docker compose -f docker/docker-compose.yaml up
   ```
   - MinIO S3 API: http://127.0.0.1:9000 (console at :9001)
   - MLflow UI: http://127.0.0.1:5001
   - Redis Stack: redis://127.0.0.1:6379
4. Run the data maintenance tasks (ingest → check → analyze) using the defaults or environment overrides from your `.env`:
   ```bash
   uv run -m tickets.main
   ```
5. Execute the ingestion flow when new raw data arrives (adjust env vars inline or in `.env` first):
   ```bash
   DATA__RAW_FILE=raw/tickets.json DATA__BUCKET=tickets uv run python -c "from tickets.data.runner import runner; \
from tickets.schemas.tasks import Task; runner(Task.INGEST)"
   ```

## Configuration
- Typed defaults live in `src/tickets/schemas/config.py` inside the `AppConfig` settings model.
- Override any value via environment variables (e.g., `DATA__BUCKET=tickets`, `REDIS_HOST=redis.internal`) or update the project `.env`.
- Secrets such as MinIO credentials are sourced from `.env` or real environment variables; export them before running flows in other shells.
- All modules import a shared `CONFIG` instance that is frozen to prevent accidental mutation during runtime.

## Data Pipelines
### Prefect ingestion (`tickets.data.ingest`)
1. `bronze`: reads raw ticket JSON from S3/MinIO, writes a compressed Parquet snapshot.
2. `offline`: applies domain cleaning (extend `clean`) and persists a curated offline Parquet dataset.
3. `online`: generates a recency-ordered slice, converts datetimes to ISO strings, and writes records to Redis JSON with pipeline batching.

The flow is orchestrated by Prefect tasks, enabling retries, scheduling, and Prefect Cloud integration without code changes.

### Offline analytics (`tickets.data.analyze`)
- `OfflineMetricsAnalyzer` loads the offline Parquet snapshot, validates required columns, and computes:
  - Response time percentiles (overall and by sentiment).
  - Satisfaction score percentiles (overall and by sentiment).
- Metrics are logged and saved back to S3/MinIO as JSON for downstream dashboards or alerting.

## Modeling & Retrieval Roadmap
- **Classical ML**: CatBoost/XGBoost training pipelines with Optuna/Bayesian optimization and feature importance tracking (hyper-parameters managed via `AppConfig.xgboost`).
- **Deep learning**: Torch-based text encoders with DataLoader abstractions, early stopping, and scheduler support (defaults surfaced by `AppConfig.dnn`).
- **Hybrid retrieval**: Planned graph-enhanced RAG stack combining semantic search (Sentence Transformers), metadata filters, and success-aware re-ranking.
- **Continual learning**: Agent feedback loops will push outcomes into an MLflow model registry and surface drift diagnostics.

## Observability & Logging
- Loguru writes structured logs at the module level (console + rotating files in `logs/`).
- MLflow within Docker Compose stores artifacts in MinIO (`s3://mlflow`); point training scripts to the container endpoint.
- Planned extensions include Prometheus exporters, anomaly detection on ticket velocity/sentiment, and retrieval-failure alerts.

## Quality & Tooling
- Format and lint:
  ```bash
  uv run ruff check .
  uv run ruff format .
  ```
- Static typing:
  ```bash
  uv run mypy src
  ```
- Tests:
  ```bash
  uv run pytest
  ```
- Target Python: 3.12 (enforced via pyproject).

## Deployment Notes
- **Docker Compose** (provided) bootstraps the local dependency stack. Add the application container when FastAPI services are ready.
- **Docker images**: prefer multi-stage builds with `uv` or `pip wheel` stages; integrate health checks mirroring the Redis/MinIO probes.
- **GitLab CI**: automate lint → type check → tests → image build/push → deployment promotion; reuse the commands above within CI jobs.
- **Kubernetes**: manifests will live under `deploy/k8s` (planned) with Helm/Kustomize overlays for cluster rollouts.

## Contribution Checklist
- Keep modules functional, typed, and documented (`PEP 8`, `PEP 257`).
- Favor pure functions and RORO patterns; avoid hidden mutations.
- Persist intermediate datasets as Parquet and log S3/Redis interactions with contextual metadata.
- Add pytest coverage for each new flow, utility, or schema.
- Update this README and architecture docs as new components become production-ready.
