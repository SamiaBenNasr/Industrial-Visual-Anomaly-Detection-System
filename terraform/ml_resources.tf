# -----------------------------------------------------------------
# These AzureML resources aren't natively supported by the azurerm
# provider, so we manage them via azapi (generic ARM/REST access).
# Terraform's state diffing replaces all the manual
# "if exists -> update else -> create" bash logic.
# -----------------------------------------------------------------

# ---------- Environment: new version each time image_tag changes ----------
resource "azapi_resource" "environment_version" {
  type      = "Microsoft.MachineLearningServices/workspaces/environments/versions@2023-10-01"
  name      = var.image_tag
  parent_id = "${azurerm_machine_learning_workspace.ws.id}/environments/${var.environment_name}"

  body = {
    properties = {
      environmentType = "UserCreated"
      image           = "${azurerm_container_registry.acr.login_server}/${var.image_name}:${var.image_tag}"
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

# ---------- Model: pointer to an already-uploaded model in the datastore ----------
# NOTE: Terraform does not upload the .ckpt binary. Upload it once with:
#   az ml model create --name <model_name> --version <model_version> \
#     --path <local path to model.ckpt> --type custom_model \
#     --workspace-name <workspace> --resource-group <rg>
# Terraform then only registers/references that version -- it will not
# touch it again as long as model_version doesn't change.
resource "azapi_resource" "model_version" {
  type      = "Microsoft.MachineLearningServices/workspaces/models/versions@2023-10-01"
  name      = var.model_version
  parent_id = "${azurerm_machine_learning_workspace.ws.id}/models/${var.model_name}"

  body = {
    properties = {
      modelType = "CustomModel"
      path = {
        uri = "azureml://datastores/workspaceblobstore/paths/models/${var.model_name}/${var.model_version}/model.ckpt"
      }
    }
  }

  lifecycle {
    ignore_changes = [body] # don't re-touch an already-registered model version
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

# Endpoint identity's own AcrPull grant -- this is the exact permission
# gap that caused the "BadArgument: Endpoint identity does not have pull
# permission" error during manual setup. Terraform wires it automatically
# every time, so it can never be forgotten again.
resource "azurerm_role_assignment" "endpoint_acr_pull" {
  scope                = azurerm_container_registry.acr.id
  role_definition_name = "AcrPull"
  principal_id         = azapi_resource.online_endpoint.output.identity.principalId
}

# ---------- Online deployment ----------
resource "azapi_resource" "online_deployment" {
  type      = "Microsoft.MachineLearningServices/workspaces/onlineEndpoints/deployments@2023-10-01"
  name      = var.deployment_name
  parent_id = azapi_resource.online_endpoint.id
  location  = var.location

  body = {
    properties = {
      endpointComputeType = "Managed"
      model                = azapi_resource.model_version.id
      environmentId         = azapi_resource.environment_version.id
      instanceType          = var.instance_type
      appInsightsEnabled    = true
      environmentVariables = {
        AZUREML_MODEL_DIR = "/var/azureml-app/azureml-models/${var.model_name}/${var.model_version}"
      }
    }
    sku = {
      name     = "Default"
      capacity = var.instance_count
    }
  }

  depends_on = [azurerm_role_assignment.endpoint_acr_pull]
}

# ---------- Traffic: 100% to this deployment ----------
# Replaces the manual `az ml online-endpoint update --traffic ...` step,
# and the "can't delete deployment with non-zero traffic" gotcha is
# handled automatically by Terraform on destroy (it reverses order).
resource "azapi_update_resource" "endpoint_traffic" {
  type        = "Microsoft.MachineLearningServices/workspaces/onlineEndpoints@2023-10-01"
  resource_id = azapi_resource.online_endpoint.id

  body = {
    properties = {
      traffic = {
        (var.deployment_name) = 100
      }
    }
  }

  depends_on = [azapi_resource.online_deployment]
}
