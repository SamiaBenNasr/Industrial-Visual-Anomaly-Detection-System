output "scoring_uri" {
  description = "URL to POST inference requests to"
  value       = "https://${var.endpoint_name}.${var.location}.inference.ml.azure.com/score"
}

output "acr_login_server" {
  description = "ACR login server, used for docker tag/push"
  value       = azurerm_container_registry.acr.login_server
}

output "workspace_name" {
  value = azurerm_machine_learning_workspace.ws.name
}


