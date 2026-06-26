"""
deploy_endpoint.py — Déploie le modèle enregistré sur un Azure ML Managed Online Endpoint.

Usage:
    python deploy_endpoint.py --model-name NAME --model-version VER
                              [--endpoint-name STR] [--instance-type STR]
                              [--instance-count INT]
"""

import argparse
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy PatchCore to Azure ML Online Endpoint")
    parser.add_argument("--model-name",     required=True)
    parser.add_argument("--model-version",  required=True)
    parser.add_argument("--endpoint-name",  default="patchcore-endpoint")
    parser.add_argument("--deployment-name",default="blue")
    parser.add_argument("--instance-type",  default="Standard_DS3_v2",
                        help="Azure VM SKU — GPU: Standard_NC6s_v3")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--traffic",        type=int, default=100,
                        help="% traffic routed to this deployment (0–100)")
    parser.add_argument("--subscription",   default=os.getenv("AML_SUBSCRIPTION_ID"))
    parser.add_argument("--resource-group", default=os.getenv("AML_RESOURCE_GROUP"))
    parser.add_argument("--workspace",      default=os.getenv("AML_WORKSPACE"))
    return parser.parse_args()


def wait_for_endpoint(ml, endpoint_name: str, timeout: int = 600) -> None:
    """Poll until the endpoint provisioning state is Succeeded."""
    from azure.ai.ml.entities import ManagedOnlineEndpoint
    deadline = time.time() + timeout
    while time.time() < deadline:
        ep = ml.online_endpoints.get(endpoint_name)
        state = ep.provisioning_state
        log.info("Endpoint state: %s", state)
        if state == "Succeeded":
            return
        if state in ("Failed", "Canceled"):
            log.error("Endpoint provisioning %s", state)
            sys.exit(1)
        time.sleep(20)
    log.error("Timeout waiting for endpoint %s", endpoint_name)
    sys.exit(1)


def wait_for_deployment(ml, endpoint_name: str, deployment_name: str, timeout: int = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        dep = ml.online_deployments.get(deployment_name, endpoint_name=endpoint_name)
        state = dep.provisioning_state
        log.info("Deployment state: %s", state)
        if state == "Succeeded":
            return
        if state in ("Failed", "Canceled"):
            log.error("Deployment provisioning %s", state)
            sys.exit(1)
        time.sleep(30)
    log.error("Timeout waiting for deployment")
    sys.exit(1)


def main() -> None:
    args = parse_args()

    for var, val in [
        ("subscription",   args.subscription),
        ("resource-group", args.resource_group),
        ("workspace",      args.workspace),
    ]:
        if not val:
            log.error("Missing --%s", var)
            sys.exit(1)

    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import (
        ManagedOnlineEndpoint,
        ManagedOnlineDeployment,
        Environment,
        CodeConfiguration,
    )
    from azure.core.exceptions import ResourceNotFoundError
    from azure.identity import DefaultAzureCredential

    ml = MLClient(
        DefaultAzureCredential(),
        subscription_id=args.subscription,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace,
    )

    # ── 1. Créer ou récupérer l'endpoint ─────────────────────────────────────
    try:
        endpoint = ml.online_endpoints.get(args.endpoint_name)
        log.info("Endpoint '%s' already exists — reusing.", args.endpoint_name)
    except ResourceNotFoundError:
        log.info("Creating endpoint '%s' …", args.endpoint_name)
        endpoint = ManagedOnlineEndpoint(
            name=args.endpoint_name,
            description="PatchCore anomaly detection endpoint",
            auth_mode="key",
            tags={"framework": "anomalib", "model": args.model_name},
        )
        ml.online_endpoints.begin_create_or_update(endpoint).result()
        wait_for_endpoint(ml, args.endpoint_name)
        log.info("Endpoint created.")

    # ── 2. Environnement Docker (référence l'image custom) ───────────────────
    env = Environment(
        name="anomalib-inference-env",
        image="patchcore-inference:latest",   # image buildée par Dockerfile
        # Fallback : utiliser un curated env si l'image n'est pas pushée
        # conda_file="conda_env.yml",
        description="Anomalib PatchCore inference environment",
    )

    # ── 3. Déploiement ────────────────────────────────────────────────────────
    log.info(
        "Deploying model %s v%s → %s/%s …",
        args.model_name, args.model_version,
        args.endpoint_name, args.deployment_name,
    )
    deployment = ManagedOnlineDeployment(
        name=args.deployment_name,
        endpoint_name=args.endpoint_name,
        model=f"azureml:{args.model_name}:{args.model_version}",
        environment=env,
        code_configuration=CodeConfiguration(
            code=".",
            scoring_script="endpoint_score.py",   # voir ci-dessous
        ),
        instance_type=args.instance_type,
        instance_count=args.instance_count,
    )
    ml.online_deployments.begin_create_or_update(deployment).result()
    wait_for_deployment(ml, args.endpoint_name, args.deployment_name)
    log.info("Deployment ready.")

    # ── 4. Router le trafic ───────────────────────────────────────────────────
    endpoint = ml.online_endpoints.get(args.endpoint_name)
    endpoint.traffic = {args.deployment_name: args.traffic}
    ml.online_endpoints.begin_create_or_update(endpoint).result()
    log.info("Traffic: %d%% → %s", args.traffic, args.deployment_name)

    # ── 5. Afficher l'URI de scoring ─────────────────────────────────────────
    endpoint = ml.online_endpoints.get(args.endpoint_name)
    scoring_uri = endpoint.scoring_uri
    log.info("✅ Endpoint live: %s", scoring_uri)

    gha_output = os.getenv("GITHUB_OUTPUT")
    if gha_output:
        with open(gha_output, "a") as fh:
            fh.write(f"scoring_uri={scoring_uri}\n")


if __name__ == "__main__":
    main()
