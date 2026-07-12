import argparse
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find best MLflow run and register it as a model")
    p.add_argument("--subscription-id", required=True)
    p.add_argument("--resource-group", required=True)
    p.add_argument("--workspace-name", required=True)
    p.add_argument("--experiment-name", default="patchcore-anomaly-detection")
    p.add_argument("--model-name", default="patchcore-bottle")
    p.add_argument("--metric-name", default="image_AUROC")
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--deployment-file", default="deployment.yml")
    p.add_argument("--top-n", type=int, default=5, help="Nombre de runs a afficher pour comparaison")
    p.add_argument("--skip-push", action="store_true")
    return p.parse_args()


def run(cmd: list[str], capture: bool = False) -> str:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if result.returncode != 0:
        if capture:
            print(result.stdout)
            print(result.stderr)
        sys.exit(f"Commande echouee: {' '.join(cmd)}")
    return result.stdout.strip() if capture else ""


def main() -> None:
    args = parse_args()

    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    import mlflow
    from mlflow.tracking import MlflowClient

    # ── Connexion au tracking Azure ML (meme mecanisme que train.py) ───────────
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )
    tracking_uri = ml_client.workspaces.get(args.workspace_name).mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # ── Chercher l'experiment et trier les runs par metrique ────────────────────
    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        sys.exit(f"ERREUR: experiment '{args.experiment_name}' introuvable dans le workspace.")

    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=max(args.top_n, 20),
    )
    if not runs:
        sys.exit(f"ERREUR: aucun run trouve dans l'experiment '{args.experiment_name}'.")

    matching = [r for r in runs if args.metric_name in r.data.metrics]
    if not matching:
        available = sorted({k for r in runs for k in r.data.metrics.keys()})
        print(f"ERREUR: aucun run n'a la metrique '{args.metric_name}'.")
        print(f"Metriques disponibles trouvees sur les {len(runs)} runs les plus recents:")
        for m in available:
            print(f"  - {m}")
        sys.exit(f"\nRelance avec --metric-name <une-des-metriques-ci-dessus>.")

    matching.sort(key=lambda r: r.data.metrics[args.metric_name], reverse=True)
    runs = matching[: args.top_n]

    print(f"\nTop {len(runs)} runs par {args.metric_name} :")
    for r in runs:
        print(f"  {r.info.run_id}  {args.metric_name}={r.data.metrics[args.metric_name]:.4f}  ({r.info.run_name})")

    best = runs[0]
    best_value = best.data.metrics[args.metric_name]
    print(f"\n>>> Meilleur run: {best.info.run_id} ({best.info.run_name})  {args.metric_name}={best_value:.4f}")

    if best_value < args.threshold:
        sys.exit(f"REJETE: meilleur score {best_value:.4f} < seuil {args.threshold}. Rien enregistre.")

    # ── Telecharger le checkpoint DE CE RUN PRECIS (pas un rglob local) ────────
    with TemporaryDirectory() as tmp:
        local_ckpt_dir = mlflow.artifacts.download_artifacts(
            run_id=best.info.run_id, artifact_path="checkpoint", dst_path=tmp,
        )
        ckpt_candidates = list(Path(local_ckpt_dir).rglob("*.ckpt"))
        if not ckpt_candidates:
            sys.exit(f"ERREUR: aucun .ckpt dans les artefacts du run {best.info.run_id}.")
        model_dir = ckpt_candidates[0].parent
        print(f"Checkpoint recupere depuis MLflow: {ckpt_candidates[0]}")

        # ── Enregistrer dans le registre Azure ML ────────────────────────────
        version = run(
            [
                r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd", "ml", "model", "create",
                "--name", args.model_name,
                "--path", str(model_dir),
                "--type", "custom_model",
                "--workspace-name", args.workspace_name,
                "--resource-group", args.resource_group,
                "--query", "version",
                "-o", "tsv",
            ],
            capture=True,
        )

    print(f"ACCEPTE: {args.model_name}:{version} enregistre (run {best.info.run_id}).")

    # ── Bump deployment.yml ──────────────────────────────────────────────────
    deployment_path = Path(args.deployment_file)
    content = deployment_path.read_text()
    new_content = re.sub(
        r"^model: .*$",
        f"model: azureml:{args.model_name}:{version}",
        content,
        flags=re.MULTILINE,
    )
    deployment_path.write_text(new_content)
    print(f"{args.deployment_file} -> azureml:{args.model_name}:{version}")

    if args.skip_push:
        print("--skip-push: pas de commit/push, arret ici.")
        return

    # ── git commit + push -> declenche deploy.yml en CI ─────────────────────
    run(["git", "add", args.deployment_file])
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if diff.returncode == 0:
        print("Aucun changement a commiter (deja sur cette version).")
        return

    run(["git", "commit", "-m",
         f"chore: bump {args.model_name} to v{version} (run {best.info.run_id}, {args.metric_name}={best_value:.4f})"])
    run(["git", "push"])
    print("Push effectue -> deploy.yml devrait se declencher automatiquement en CI.")


if __name__ == "__main__":
    main()