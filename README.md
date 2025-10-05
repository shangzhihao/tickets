# Intelligent Ticket Analytics Platform

This repository implements the ingestion and serving backbone for an intelligent ticket analytics platform. The current focus -- and the only production-ready component today -- is a Prefect-powered data ingestion flow that hydrates S3/MinIO storage with bronze, offline, and online datasets, then publishes the freshest tickets to Redis for low-latency access. Supporting infrastructure can be provisioned locally via Docker Compose or in-cluster with the provided Kubernetes manifests. Additional analytics, modeling, and API layers are under active development.

## Key Capabilities Today
- Hydra-based CLI entry point (`python -m tickets.main`) that locks configuration using `OmegaConf` and dispatches work through a lightweight task runner.
- Prefect flow `tickets.data.ingest.ingest` that: reads raw JSON tickets, materializes a bronze parquet snapshot, produces a cleaned offline dataset, and curates a recency-ordered online slice.
- S3-compatible storage integration via `boto3`, targeting MinIO by default (`conf/data/minio.yaml`) with environment overrides for production credentials.
- Redis JSON persistence for the online dataset, pushed through a pipeline for efficient bulk writes.
- Pydantic `Ticket` schema describing the normalized ticket contract, plus Loguru-backed structured logging to `logs/app.log`.

## Project Layout
```
conf/                  # Hydra configuration tree (data sources, model defaults, logging, serving)
docker/                # Local dependency stack (MinIO, MLflow, Redis)
k8s/                   # Kustomize manifests for MLflow + MinIO deployments
src/tickets/
  data/                # Prefect flow and task implementations
  schemas/             # Pydantic models (Ticket domain schema)
  utils/               # Shared utilities such as logging setup
tests/                 # Pytest suite placeholder for upcoming coverage
```

## Getting Started
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -U pip`
3. `pip install -e .[dev]`
4. Run `python -m black .` and `ruff check .` before committing, then use `pytest` (or `pytest -k pattern`) once tests are in place.

## Local Dependencies with Docker Compose
- Start services: `docker compose -f docker/docker-compose.yaml up`
- Exposed endpoints:
  - MLflow UI: http://127.0.0.1:5001
  - MinIO S3 API: http://127.0.0.1:9000 (console on 9001)
  - Redis Stack: redis://127.0.0.1:6379
- Data persists under `docker/data/`; stop with `docker compose -f docker/docker-compose.yaml down`.
- Override MinIO credentials by exporting `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` before launch.

## Running the Ingestion Flow
```bash
python -m tickets.main \
  data=minio \
  redis_host=127.0.0.1 \
  redis_port=6379
```
- Hydra reads defaults from `conf/config.yaml`. Override any setting inline (`data.bucket=tickets-dev`) or by swapping configs (`data=aws`).
- The flow expects the raw ticket payload at `data.raw_file`; upload sample data to MinIO (see `docker` stack) before running.
- Successful runs emit structured logs to `logs/app.log` and cache S3 writes via `tickets.data.ingest.get_s3_client`.

## Configuration Notes
- `conf/data/minio.yaml` uses `${oc.env:MINIO_ACCESS_KEY}` and `${oc.env:MINIO_SECRET_KEY}` lookups with sensible defaults for local work.
- `conf/config.yaml` also sets Redis host/port and data split ratios to keep experiments deterministic.
- Additional model configurations (`conf/catboost`, `conf/dnn`, `conf/xgboost`) are present for upcoming training pipelines and can be overridden using standard Hydra syntax.

## Kubernetes Deployment (Optional)
- Apply the Kustomize overlay after configuring secrets: `kubectl apply -k k8s`
- Default hostPorts expose:
  - MLflow tracking: http://127.0.0.1:5001
  - MinIO S3 API: http://127.0.0.1:30900
  - MinIO console: http://127.0.0.1:30901
- Update credentials in `k8s/minio.yaml` prior to any shared or production environment rollouts.

## Development Checklist
- Add regression tests under `tests/` as new features land; follow `test_<topic>.py` naming.
- Keep modules small and typed—public functions already expose type hints and leverage `from __future__ import annotations`.
- Use `mypy`, `ruff`, and `black` to enforce style/quality; configurations live in `pyproject.toml`.
- Review `AGENTS.md` and `nogit-task.md` for contributor guidance and assignment context.

## Next Steps
- Expand `tickets.data.clean` with domain-specific cleansing, enrichments, and SLA derivations.
- Stand up analytics and modeling pipelines in new `tickets/analytics` modules, keeping CLI orchestration under `tickets.main`.
- Introduce fixture datasets in `assets/` and corresponding tests to lock in expected flow outputs.
- Document progress milestones in this README as additional components graduate from WIP to usable modules.