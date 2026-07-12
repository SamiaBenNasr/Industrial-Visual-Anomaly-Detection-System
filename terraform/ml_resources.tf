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

resource "azurerm_role_assignment" "endpoint_acr_pull" {
  scope                = azurerm_container_registry.acr.id
  role_definition_name = "AcrPull"
  principal_id         = azapi_resource.online_endpoint.output.identity.principalId
}


