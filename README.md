# Industrial Visual Anomaly Detection System — MLOps Pipeline on Azure ML

End-to-end MLOps pipeline for training, serving, and deploying a [PatchCore](https://github.com/openvinotoolkit/anomalib) anomaly detection model (via [anomalib](https://github.com/open-edge-platform/anomalib)) on the MVTecAD dataset, with automated CI/CD to an Azure Machine Learning managed online endpoint.

## Architecture

```
.github/workflows/
├── build-image.yml        # Builds & pushes the inference Docker image to ACR
├── deploy-endpoint.yml     # Creates/updates the Azure ML environment + online deployment
├── deploy.yml               # Orchestrator: test -> build -> deploy
└── tests.yml                 # Runs the pytest suite

docker/
├── .dockerignore
└── Dockerfile              # Multi-stage build for the inference server image

terraform/
├── main.tf                 # Resource group, ACR, Key Vault, storage, Azure ML workspace
├── ml_resources.tf          # ML environment version, online endpoint, ACR role assignments
├── outputs.tf               # scoring_uri, acr_login_server, workspace_name
├── providers.tf              # azurerm + azapi provider setup
└── variables.tf              # Configurable names/locations (defaults included)

tests/
├── conftest.py               # Shared pytest fixtures / anomalib import shims
├── test_endpoint_score.py    # Unit tests for endpoint_score.py
└── test_server.py            # Integration tests for the FastAPI server

deployment.yml                # Azure ML online deployment manifest
endpoint.yml                  # Azure ML online endpoint manifest
endpoint_score.py             # Scoring script (init/run) used by the deployed endpoint
environment.yml                # Base Azure ML environment definition
find_and_register_best.py      # Picks the best MLflow run and registers it as a model
pytest.ini
requirements.txt              # Runtime dependencies (training/scoring)
requirements-test.txt          # Test-only dependencies
score.py                       # CLI batch scoring script
server.py                      # FastAPI app for local testing of the scoring logic
test-request.json              # Sample payload for smoke-testing the deployed endpoint
train.py                       # Training script (PatchCore + MVTecAD via anomalib/MLflow)
```

## How it fits together

1. **Train** (`train.py`) — trains a PatchCore model on an MVTecAD category, evaluates it, exports it (ONNX/Torch/OpenVINO), and logs parameters, metrics, and the checkpoint to MLflow. It can track locally (`./mlruns`) or remotely against an Azure ML workspace if Azure credentials are supplied.
2. **Select & register** (`find_and_register_best.py`) — queries MLflow for the best run of an experiment by a chosen metric (default `image_AUROC`), downloads its checkpoint, registers it as a versioned Azure ML model, bumps `deployment.yml` to point at the new version, and pushes the change to trigger CI/CD.
3. **Serve** (`endpoint_score.py`, `server.py`) — `endpoint_score.py` implements the `init()`/`run()` contract expected by Azure ML managed endpoints (decode the image, run inference, log prediction + input-quality metrics, return JSON). `server.py` wraps it in a FastAPI app (`/health`, `/score`) so it can be run and tested locally or in Docker before deploying.
4. **Package** (`docker/Dockerfile`) — a 3-stage build (base → builder → runtime) that installs PyTorch (CPU) and anomalib, pre-downloads PatchCore weights, then produces a slim runtime image running the FastAPI server on port 8080.
5. **Infrastructure** (`terraform/`) — provisions the Azure resource group, container registry (ACR), Key Vault, storage account, and Azure ML workspace, plus the role assignments needed for the workspace and the online endpoint's managed identity to pull images from ACR.
6. **CI/CD** (`.github/workflows/`) — on a push to `main` touching relevant paths:
   - `tests.yml` runs the pytest suite;
   - `build-image.yml` builds and pushes the Docker image to ACR, tagged with the short git SHA and `latest`;
   - `deploy-endpoint.yml` creates/updates the Azure ML environment with the new image, creates or updates the online deployment, forces 100% traffic to it, and runs a smoke test against the live endpoint.

   `deploy.yml` chains these three workflows together (`test → build → deploy`) and can also be triggered manually via `workflow_dispatch`.

## Prerequisites

- Python 3.11
- Docker
- [Terraform](https://developer.hashicorp.com/terraform) >= 1.5.0
- Azure CLI with the `ml` extension (`az extension add -n ml`)
- An Azure subscription 

## Getting started

### 1. Provision infrastructure

```bash
cd terraform
terraform init
terraform apply
```

This creates the resource group, ACR, Azure ML workspace, and supporting resources defined in `variables.tf`.

### 2. Train a model

```bash
pip install -r requirements.txt
python train.py --category bottle --data-root data/mvtec --epochs 1
```

Pass `--subscription-id`, `--resource-group`, and `--workspace-name` (or set the corresponding `AZURE_*` env vars) to have MLflow track the run against the Azure ML workspace instead of locally.

### 3. Select and register the best run

```bash
python find_and_register_best.py \
  --subscription-id <sub-id> \
  --resource-group <rg-name> \
  --workspace-name <ws-name> \
  --metric-name image_AUROC \
  --threshold 0.90
```

This registers the best-performing model in the Azure ML model registry, updates `deployment.yml`, and (unless `--skip-push` is set) commits and pushes the change — which triggers `deploy.yml` in CI.

### 4. Run the scoring service locally

```bash
uvicorn server:app --reload --port 8080
```

or via Docker:

```bash
docker build -f docker/Dockerfile -t patchcore-inference:latest .
docker run --rm -p 8080:8080 -e AZUREML_MODEL_DIR=./outputs patchcore-inference:latest
```

Test it:

```bash
curl -X POST http://localhost:8080/score \
  -H "Content-Type: application/json" \
  -d @test-request.json
```

### 5. Batch scoring (no server)

```bash
python score.py --input path/to/images_or_folder --ckpt path/to/model.ckpt --save-maps
```

## Testing

```bash
pip install -r requirements-test.txt
pytest
```

Tests cover:
- payload validation and error handling in `endpoint_score.run()` (invalid JSON, missing/invalid base64, corrupt/truncated images, empty predictions, inference exceptions);
- the FastAPI `/health` and `/score` routes in `server.py`, including error propagation as HTTP 400 and Pydantic validation as HTTP 422.

## CI/CD secrets

The GitHub Actions workflows expect the following repository secrets:

| Secret | Used for |
|---|---|
| `AZURE_CREDENTIALS` | Azure login (`azure/login@v2`) |
| `ACR_NAME` | Docker build/push target and environment image reference |
| `RESOURCE_GROUP` | Target resource group for the Azure ML workspace |
| `WORKSPACE_NAME` | Target Azure ML workspace |

## Endpoint contract

**Request** (`POST /score`):
```json
{
  "image_b64": "<base64-encoded PNG/JPEG>",
  "image_size": [256, 256]
}
```

**Response**:
```json
{
  "pred_score": 0.312,
  "pred_label": 0,
  "is_anomalous": false
}
```

**Health check**: `GET /health` → `{"status": "ok"}`