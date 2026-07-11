# PatchCore Anomaly Detection — MLOps Pipeline on Azure ML

## Overview

This project deploys a **PatchCore** anomaly detection model (trained on the MVTecAD "bottle" category) as a managed inference API on **Azure Machine Learning**. It uses a hybrid infrastructure-as-code approach: **Terraform** manages the underlying Azure resources, while **model registration, deployment, and traffic routing are handled manually or via dedicated GitHub Actions workflows** — deliberately kept separate from Terraform's scope.

The core design principle throughout this project: **Terraform manages infrastructure (the control plane); it never manages data artifacts (the data plane)**. Docker images and trained model weights are binary data that must already exist before Terraform can reference them — Terraform builds pointers to them, it does not create or upload them.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Azure Resource Group                     │
│                        (rg-patchcore)                        │
│                                                                │
│  ┌──────────────┐   ┌──────────────────┐   ┌───────────────┐ │
│  │  Container   │   │   Azure ML        │   │   Storage /   │ │
│  │  Registry    │──▶│   Workspace        │◀──│   Key Vault / │ │
│  │  (ACR)       │   │   (ws-patchcore)   │   │   App Insights│ │
│  └──────────────┘   └────────┬──────────┘   └───────────────┘ │
│                               │                                │
│                      ┌────────▼──────────┐                     │
│                      │  Online Endpoint   │                     │
│                      │ (patchcore-endpoint│                     │
│                      └────────┬──────────┘                     │
│                               │                                │
│                      ┌────────▼──────────┐                     │
│                      │  Online Deployment │                     │
│                      │ (patchcore-deploy) │                     │
│                      │  Standard_DS3_v2   │                     │
│                      └───────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### What Terraform manages

| Resource | Purpose |
|---|---|
| Resource Group | Container for all project resources |
| Container Registry (ACR) | Stores the Docker inference image |
| ML Workspace (+ Storage, Key Vault, App Insights) | Azure ML control plane |
| Role Assignments (AcrPull) | Grants workspace and endpoint identities pull access to ACR |
| Environment | Points to a specific Docker image tag |
| Online Endpoint | The stable HTTPS scoring URL |

### What Terraform explicitly does NOT manage

| Item | Managed by | Why |
|---|---|---|
| Docker image build & push | GitHub Actions (`deploy.yml`) or manual `docker build/push` | Binary artifact — a build/push action, not an infrastructure declaration |
| Trained model (`.ckpt`) upload | Manual `az ml model create`, or `register-and-deploy.yml` | Binary data transfer; Terraform's ARM-based providers cannot upload files |
| Online Deployment | Manual `az ml online-deployment create/update`, or GitHub Actions | Kept out of Terraform by design, per project decision, so deployment timing stays under explicit manual control |
| Traffic routing (100%) | Manual `az ml online-endpoint update --traffic`, or GitHub Actions | Same reasoning as above |

---

## Repository Structure

```
detecti/
├── .github/workflows/
│   ├── deploy.yml                 # Push-triggered: build+push image, create/update deployment, set traffic
│   ├── register-and-deploy.yml    # Manual: register a new model version + deploy + traffic
│   └── infra.yml                  # Manual only: full infra recreate/destroy via Terraform
├── terraform/
│   ├── providers.tf               # azurerm + azapi provider config
│   ├── variables.tf                # All configurable inputs
│   ├── main.tf                     # Resource Group, ACR, ML Workspace + dependencies
│   ├── ml_resources.tf             # Environment + Online Endpoint (via azapi)
│   └── outputs.tf                  # scoring_uri, acr_login_server, workspace_name
├── docker/Dockerfile               # Multi-stage build; bakes in the timm/HF backbone weights
├── endpoint_score.py                # Azure ML scoring script (init/run)
├── server.py                        # Local FastAPI wrapper for testing outside Azure ML
├── train.py                         # PatchCore training script (anomalib)
├── deployment.yml                   # Managed online deployment spec (model + environment refs)
├── environment.yml                  # (legacy/manual) environment spec
├── test-request.json                # Sample payload for smoke testing
└── requirements.txt
```

---

## Prerequisites

- Azure CLI (`az`) with the `ml` extension: `az extension add -n ml`
- Terraform >= 1.5.0
- Docker Desktop
- An active Azure subscription (this project was built and tested on Azure for Students)

---

## Local Setup — First-Time Deployment

### 1. Authenticate

```powershell
az login
```

### 2. Provision infrastructure with Terraform

```powershell
cd terraform
terraform init
terraform plan -var="image_tag=latest"
terraform apply -var="image_tag=latest"
```

This creates the Resource Group, ACR, ML Workspace, Environment, Online Endpoint, and the AcrPull role assignments. It does **not** create a Deployment or route any traffic.

> **Known issue and fix:** if `terraform apply` fails with a 403 `AuthenticationFailed` error on the Storage Account, add `storage_use_azuread = true` to the `azurerm` provider block in `providers.tf`. This is required on subscriptions (such as Azure for Students) where storage account key-based authentication is restricted by policy.

> **Known issue and fix:** if the Online Deployment step later fails with *"workspace enables a private link feature and it blocks your online managed endpoint"*, ensure `public_network_access_enabled = true` is set on the `azurerm_machine_learning_workspace` resource (recent `azurerm` provider versions default this more restrictively than the CLI does).

### 3. Build and push the Docker image

```powershell
az acr login --name acrpatchcore
docker build -f docker/Dockerfile -t acrpatchcore.azurecr.io/patchcore-inference:latest .
docker push acrpatchcore.azurecr.io/patchcore-inference:latest
```

### 4. Register the trained model (manual — not managed by Terraform)

```powershell
az ml model create --name patchcore-bottle --version 1 `
  --path outputs/Patchcore/v3/weights/lightning/model.ckpt `
  --type custom_model `
  --workspace-name ws-patchcore --resource-group rg-patchcore
```

### 5. Create the deployment and route traffic (manual — not managed by Terraform)

```powershell
az ml online-deployment create --file deployment.yml `
  --workspace-name ws-patchcore --resource-group rg-patchcore --all-traffic

az ml online-endpoint update --name patchcore-endpoint `
  --workspace-name ws-patchcore --resource-group rg-patchcore `
  --traffic "patchcore-deploy=100"
```

### 6. Test

```powershell
az ml online-endpoint invoke --name patchcore-endpoint `
  --request-file test-request.json `
  --workspace-name ws-patchcore --resource-group rg-patchcore
```

Expected response shape:
```json
{"pred_score": 0.907076, "pred_label": 1, "is_anomalous": true}
```

---

## Docker Image Notes

The Dockerfile downloads the PatchCore backbone (WideResNet50, pretrained on ImageNet via `timm`) **at build time**, not at container runtime, by directly instantiating `Patchcore()` inside the build:

```dockerfile
ENV TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/hf \
    HF_HUB_OFFLINE=0
RUN mkdir -p /app/.cache/torch /app/.cache/hf \
 && python -c "from anomalib.models import Patchcore; Patchcore()"
```

This approach was chosen over guessing a HuggingFace `repo_id` because `timm` may resolve pretrained weights via either `torch.hub` or the HuggingFace Hub depending on version — instantiating the model directly captures whichever mechanism is actually used, without assumptions. Both cache directories are created unconditionally (`mkdir -p`) so the multi-stage `COPY --from=builder` step never fails even if only one cache path is populated.

At runtime, `HF_HUB_OFFLINE=1` ensures the container never attempts a network call to HuggingFace Hub — the image is fully self-contained.

The trained model (`model.ckpt`) is **not** baked into the image. It is mounted by Azure ML from the registered model asset at `AZUREML_MODEL_DIR`, kept intentionally decoupled from the code/image lifecycle.

---

## GitHub Actions Workflows

### `deploy.yml` — triggered on push to `main`

Builds and pushes the Docker image, updates the Azure ML Environment to the new image tag, then creates the deployment if it doesn't exist or updates it if it does, and finally re-asserts 100% traffic.

Required secrets: `AZURE_CREDENTIALS`, `ACR_NAME`, `RESOURCE_GROUP`, `WORKSPACE_NAME`.

### `register-and-deploy.yml` — manual trigger only

Registers a newly trained model version, then creates/updates the deployment and sets traffic to 100%. Used after a retraining run, independent of any code push.

### `infra.yml` — manual trigger only

Runs `terraform apply` or `terraform destroy` against the full infrastructure. This is intentionally **not** wired to any push trigger — it exists solely for full environment recreation after a deliberate teardown, or full teardown itself.

---

## Cost Management (Azure for Students)

The **Online Deployment is the only continuously-billed resource** (a running `Standard_DS3_v2` VM behind the endpoint). Everything else (ACR, Storage, Key Vault, App Insights, the Workspace shell itself) costs negligibly at rest.

**To pause spending without losing infrastructure:**

```powershell
az ml online-endpoint update --name patchcore-endpoint `
  --workspace-name ws-patchcore --resource-group rg-patchcore `
  --traffic "patchcore-deploy=0"

az ml online-deployment delete --name patchcore-deploy `
  --endpoint-name patchcore-endpoint `
  --workspace-name ws-patchcore --resource-group rg-patchcore --yes
```

Resume later via `register-and-deploy.yml` or the manual deployment command — no need to touch Terraform.

**To tear down everything** (full reset, e.g. before a long break):

```powershell
cd terraform
terraform destroy -var="image_tag=latest"
```

> **Note:** destroying the Resource Group cascades to delete *everything* inside it, including resources created manually (the Deployment and the registered Model), not just what Terraform created directly.

> **Note on soft-delete:** Azure ML Workspaces are soft-deleted, not immediately purged. If you `terraform apply` again after a destroy and get `BadRequest: Soft-deleted workspace exists`, purge it first:
> ```powershell
> az ml workspace list-deleted --resource-group rg-patchcore
> az ml workspace delete --name ws-patchcore --resource-group rg-patchcore --permanently-delete --yes
> ```

---

## Troubleshooting Reference

| Symptom | Cause | Fix |
|---|---|---|
| `BadArgument: Endpoint identity does not have pull permission on the registry` | Endpoint's system-assigned identity lacks `AcrPull` | Terraform now grants this automatically (`azurerm_role_assignment.endpoint_acr_pull`) |
| `Can't delete deployment with non-zero traffic weight` | Traffic must be zeroed before deleting a deployment | Run `az ml online-endpoint update --traffic "<name>=0"` first |
| `unexpected status 403 ... AuthenticationFailed` on Storage Account (Terraform) | Shared key auth blocked by subscription policy | Set `storage_use_azuread = true` in the `azurerm` provider block |
| `workspace enables a private link feature and it blocks your online managed endpoint` | Workspace network isolation too restrictive for a managed endpoint | Set `public_network_access_enabled = true` on the workspace |
| `Soft-deleted workspace exists` | A workspace with the same name was deleted but not purged | `az ml workspace delete --permanently-delete` |
| `properties.path is not expected here` / `properties.environmentType is not expected here, it's read only` (azapi) | Incorrect ARM schema property names in Terraform `body` blocks | Use `modelUri` (not `path`) for models; omit `environmentType` (read-only, inferred automatically) |
| Docker build fails on `COPY hf_cache/hf` | Local HF cache directory never committed to the repo | Download model weights at build time instead of copying a local cache (see Docker Image Notes above) |

---

## Design Decisions Log

- **Model registry vs. baked-in model**: the trained `.ckpt` is registered as an Azure ML model asset rather than baked into the Docker image, to allow independent versioning and rollback without a Docker rebuild. The generic ImageNet backbone, by contrast, *is* baked into the image, since it never changes independently of the code.
- **Deployment/traffic kept out of Terraform**: by explicit project decision, Online Deployment and traffic routing remain manual (CLI or GitHub Actions) rather than Terraform-managed, to keep deployment timing under direct human control rather than tied to infrastructure `apply` cycles.
- **Local Terraform state (no remote backend, for now)**: given single-operator usage, the state file remains local rather than in a remote Azure Storage backend. This should be revisited if collaborators are added or if Terraform is later wired into automated CI/CD.