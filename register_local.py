"""
register_local.py — Equivalent du job CI "train-and-register", mais SANS
soumettre de job Azure ML. A executer sur TON PC juste apres un
entrainement local (python train.py ...).

Pourquoi pas de job Azure ML : pas de compute GPU/CPU disponible sur cet
abonnement pour l'instant (cf. quota BatchAI). Ca n'empeche ni le tracking
MLflow (train.py peut ecrire directement dans le workspace, voir
--subscription-id/--resource-group/--workspace-name dans train.py) ni
l'enregistrement du modele : `az ml model create` est un appel API sur le
registre du workspace, il ne consomme AUCUN compute.

Prerequis :
  - az login  (une fois, pour que az/az-identity trouvent tes credentials)
  - le repo git est clone en local et ce script est lance depuis sa racine
    (il fait git add/commit/push sur deployment.yml a la fin)

Usage :
    python register_local.py \
        --model-dir outputs \
        --resource-group rg-patchcore \
        --workspace-name ws-patchcore \
        --model-name patchcore-bottle \
        --metric-name image_AUROC \
        --threshold 0.90

Comportement :
  1. Lit metrics.json et le .ckpt dans --model-dir (sortie de train.py)
  2. Si metric >= threshold :
       - az ml model create  (nouvelle version dans le registre Azure ML)
       - met a jour deployment.yml avec cette nouvelle version
       - git commit + git push  -> declenche deploy.yml en CI
  3. Sinon : exit 1, rien n'est touche
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Register PatchCore model from a local training run")
    p.add_argument("--model-dir", required=True, help="Dossier de sortie de train.py (--output-dir), contient metrics.json + *.ckpt")
    p.add_argument("--resource-group", required=True)
    p.add_argument("--workspace-name", required=True)
    p.add_argument("--model-name", default="patchcore-bottle")
    p.add_argument("--metric-name", default="image_AUROC")
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--deployment-file", default="deployment.yml", help="Chemin vers deployment.yml dans le repo")
    p.add_argument("--skip-push", action="store_true", help="Registre le modele et met a jour deployment.yml, mais ne fait pas git commit/push (pour tester)")
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
    model_dir = Path(args.model_dir)

    # ── 1. Lire les metriques locales ───────────────────────────────────────
    metrics_candidates = list(model_dir.rglob("metrics.json"))
    if not metrics_candidates:
        sys.exit(f"ERREUR: aucun metrics.json trouve sous {model_dir}. As-tu bien lance train.py avec --output-dir {model_dir} ?")

    metrics = json.loads(metrics_candidates[0].read_text())
    print(f"Metriques trouvees: {metrics}")

    if args.metric_name not in metrics:
        sys.exit(f"ERREUR: metrique '{args.metric_name}' absente ({list(metrics.keys())}).")

    value = float(metrics[args.metric_name])
    print(f"{args.metric_name} = {value:.4f}  (seuil requis: >= {args.threshold})")

    if value < args.threshold:
        sys.exit(f"REJETE: {args.metric_name}={value:.4f} < seuil={args.threshold}. Modele NON enregistre.")

    # ── 2. Localiser le checkpoint ──────────────────────────────────────────
    ckpt_candidates = list(model_dir.rglob("*.ckpt"))
    if not ckpt_candidates:
        sys.exit(f"ERREUR: aucun .ckpt trouve sous {model_dir}.")
    ckpt_dir = ckpt_candidates[0].parent
    print(f"Checkpoint: {ckpt_candidates[0]}")

    # ── 3. Enregistrer le modele via az cli ─────────────────────────────────
    # (subprocess + az cli plutot que le SDK: reutilise directement ta
    # session `az login` locale, pas de config credentials supplementaire)
    output = run(
        [
            r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd", "ml", "model", "create",
            "--name", args.model_name,
            "--path", str(ckpt_dir),
            "--type", "custom_model",
            "--workspace-name", args.workspace_name,
            "--resource-group", args.resource_group,
            "--query", "version",
            "-o", "tsv",
        ],
        capture=True,
    )
    model_version = output.strip()
    if not model_version:
        sys.exit("ERREUR: az ml model create n'a pas retourne de version.")
    print(f"ACCEPTE: {args.model_name}:{model_version} enregistre dans Azure ML.")

    # ── 4. Bump deployment.yml ──────────────────────────────────────────────
    deployment_path = Path(args.deployment_file)
    content = deployment_path.read_text()
    new_content = re.sub(
        r"^model: .*$",
        f"model: azureml:{args.model_name}:{model_version}",
        content,
        flags=re.MULTILINE,
    )
    if new_content == content:
        print("Attention: aucune ligne 'model: ...' trouvee/modifiee dans deployment.yml, verifie le fichier.")
    deployment_path.write_text(new_content)
    print(f"{args.deployment_file} mis a jour -> azureml:{args.model_name}:{model_version}")

    if args.skip_push:
        print("--skip-push: pas de commit/push, arret ici.")
        return

    # ── 5. git commit + push -> declenche deploy.yml en CI ──────────────────
    run(["git", "add", args.deployment_file])
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if diff.returncode == 0:
        print("Aucun changement a commiter (deja sur cette version).")
        return

    run(["git", "commit", "-m",
         f"chore: bump {args.model_name} to v{model_version} (entrainement local, {args.metric_name}={value:.4f})"])
    run(["git", "push"])
    print("Push effectue -> deploy.yml devrait se declencher automatiquement en CI.")


if __name__ == "__main__":
    main()