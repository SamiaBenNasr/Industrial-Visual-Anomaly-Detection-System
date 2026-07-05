# -----------------------------------------------------------------
# Terraform gère ici UNIQUEMENT l'infrastructure : environment,
# endpoint, deployment, traffic. Ce sont des ressources ARM pures,
# aucune donnée binaire ne transite ici.
#
# Le MODELE (.ckpt) N'EST PAS géré par Terraform : il doit déjà être
# enregistré au préalable via une commande séparée :
#
#   az ml model create --name patchcore-bottle --version 1 \
#     --path <local path to model.ckpt> --type custom_model \
#     --workspace-name ws-patchcore --resource-group rg-patchcore
#
# Terraform référence juste ce pointeur (var.model_version) --
# il ne le crée jamais, ne l'upload jamais, n'y touche jamais.
# -----------------------------------------------------------------

# ---------- Environment: nouvelle version à chaque nouveau image_tag ----------

resource "azapi_resource" "environment_version" {
  type      = "Microsoft.MachineLearningServices/workspaces/environments/versions@2023-10-01"
  name      = var.image_tag
  parent_id = "${azurerm_machine_learning_workspace.ws.id}/environments/${var.environment_name}"

  body = {
    properties = {
      image = "${azurerm_container_registry.acr.login_server}/${var.image_name}:${var.image_tag}"
      inferenceConfig = {
        livenessRoute = {
          path = "/health"
          port = 8080
        }
        readinessRoute = {
          path = "/health"
          port = 8080
        }
        scoringRoute = {
          path = "/score"
          port = 8080
        }
      }
    }
  }
}

# ---------- Online endpoint ----------
resource "azapi_resource" "online_endpoint" {
  type      = "Microsoft.MachineLearningServices/workspaces/onlineEndpoints@2023-10-01"
  name      = var.endpoint_name
  parent_id = azurerm_machine_learning_workspace.ws.id
  location  = var.location

  identity {
    type = "SystemAssigned"
  }

  body = {
    properties = {
      authMode = "Key"
    }
  }

  response_export_values = ["identity.principalId"]
}
# Permission AcrPull de l'endpoint -- c'est exactement le bug
# "BadArgument: Endpoint identity does not have pull permission"
# qu'on a rencontré en manuel. Terraform la garantit à chaque apply.
resource "azurerm_role_assignment" "endpoint_acr_pull" {
  scope                = azurerm_container_registry.acr.id
  role_definition_name = "AcrPull"
  principal_id         = azapi_resource.online_endpoint.output.identity.principalId
}

# -----------------------------------------------------------------
# Online Deployment et Traffic NE SONT PAS gérés par Terraform.
# Créés/mis à jour manuellement, après chaque apply, avec :
#
#   az ml online-deployment create --file deployment.yml `
#     --workspace-name ws-patchcore --resource-group rg-patchcore --all-traffic
#
# ou, si le deployment existe déjà :
#
#   az ml online-deployment update --file deployment.yml `
#     --workspace-name ws-patchcore --resource-group rg-patchcore
#
#   az ml online-endpoint update --name patchcore-endpoint `
#     --workspace-name ws-patchcore --resource-group rg-patchcore `
#     --traffic "patchcore-deploy=100"
#
# deployment.yml doit référencer :
#   environment: azureml:patchcore-env:<version = image_tag utilisé dans terraform>
#   model: azureml:patchcore-bottle:<model_version>
# -----------------------------------------------------------------
