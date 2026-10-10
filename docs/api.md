# API Reference

## Authentication
- `POST /api/auth/register`: Register a new user (`email`, `password`, `full_name`).
- `POST /api/auth/login`: Authenticate and receive a JWT access token.
- `GET /api/auth/me`: Get current authenticated user details.

## Projects
- `GET /api/projects`: List user projects.
- `POST /api/projects`: Create a new project.
- `GET /api/projects/{id}`: Get project details.
- `DELETE /api/projects/{id}`: Delete a project.

## Models
- `POST /api/models/upload`: Upload a serialized model file (`.pkl` or `.joblib`).
- `GET /api/models`: List uploaded models (optionally filtered by `project_id`).
- `GET /api/models/{id}`: Get model metadata and inspection attributes.
- `DELETE /api/models/{id}`: Delete a model.

## Datasets
- `POST /api/datasets/upload`: Upload a tabular dataset (`.csv`) with `target_column` and optional `sensitive_columns`.
- `GET /api/datasets`: List uploaded datasets (optionally filtered by `project_id`).
- `GET /api/datasets/{id}`: Get dataset metadata, schema, and sample preview.
- `DELETE /api/datasets/{id}`: Delete a dataset.

## Validation Runs & Findings
- `POST /api/validation/runs`: Trigger a new multi-dimensional validation run (`model_id`, `dataset_id`, optional config).
- `GET /api/validation/runs`: List validation runs.
- `GET /api/validation/runs/{id}`: Get run status, duration, overall health score, and category scores.
- `GET /api/validation/runs/{id}/findings`: Retrieve detailed findings filtered by severity or category.

## Audit Reports (JSON & PDF Export)
- `GET /api/reports/{id}` or `GET /api/reports/{id}/json`: Download structured 15-section JSON audit report attachment.
- `GET /api/reports/{id}/pdf`: Download audit-grade PDF validation report (`application/pdf`).

## Health Check
- `GET /api/health`: Service liveness and readiness probe.

