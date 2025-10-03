# Intelligent Product Support System

This repository houses an intelligent ticket support platform designed to help product support teams handle large ticket volumes with automation and decision support. With historical tickets and arriving daily, the system learns from past resolutions to automate routine issues, surface emerging problems, and accelerate complex case handling.

## Deploying MLflow and MinIO on Kubernetes

Apply manifests with:

```
kubectl apply -k k8s
```

Default access (after pods are Running):
- MLflow tracking server: http://127.0.0.1:5001 (hostPort)
- MLflow NodePort (for remote clients): http://<node-ip>:31500
- MinIO S3 API: http://127.0.0.1:30900
- MinIO console: http://127.0.0.1:30901 (login using `mlflowadmin`/`mlflowadmin123`)

Update the credentials in `k8s/minio.yaml` before production deployments.

### Running Locally with Docker Compose

Launch the stack from the repository root:

```
docker compose -f docker/docker-compose.yaml up
```

The compose file persists MinIO and MLflow data under `docker/data/`.
Override credentials by exporting `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` before running `docker compose` if needed.
Access points:
- MLflow tracking UI: http://127.0.0.1:5001
- MinIO S3 API: http://127.0.0.1:9000
- MinIO console: http://127.0.0.1:9001 (login with the same credentials as Kubernetes)

Stop the stack with `docker compose -f docker/docker-compose.yaml down`.
