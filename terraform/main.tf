data "azurerm_client_config" "current" {}

# -----------------------------------------------------------------
# Resource group
# -----------------------------------------------------------------
resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location
}

# -----------------------------------------------------------------
# Container registry
# -----------------------------------------------------------------
resource "azurerm_container_registry" "acr" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Basic"
  admin_enabled       = false
}

# -----------------------------------------------------------------
# Workspace dependencies (mirrors what `az ml workspace create` auto-provisions)
# -----------------------------------------------------------------
resource "azurerm_application_insights" "appi" {
  name                = "${var.workspace_name}-appi"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  application_type    = "web"

  lifecycle {
    ignore_changes = [workspace_id]
  }
}

resource "azurerm_key_vault" "kv" {
  name                = "${replace(var.workspace_name, "-", "")}kv"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
}

resource "azurerm_storage_account" "storage" {
  name                     = "${replace(var.workspace_name, "-", "")}storage"
  location                 = azurerm_resource_group.rg.location
  resource_group_name      = azurerm_resource_group.rg.name
  account_tier             = "Standard"
  account_replication_type = "LRS"
  shared_access_key_enabled = true
}

# -----------------------------------------------------------------
# Azure ML workspace
# -----------------------------------------------------------------
resource "azurerm_machine_learning_workspace" "ws" {
  name                    = var.workspace_name
  location                = azurerm_resource_group.rg.location
  resource_group_name     = azurerm_resource_group.rg.name
  application_insights_id = azurerm_application_insights.appi.id
  key_vault_id            = azurerm_key_vault.kv.id
  storage_account_id      = azurerm_storage_account.storage.id

  identity {
    type = "SystemAssigned"
  }
}

# Workspace identity needs AcrPull too (covers workspace-triggered pulls,
# e.g. notebook/compute-instance image pulls, separate from the endpoint's
# own identity below).
resource "azurerm_role_assignment" "workspace_acr_pull" {
  scope                = azurerm_container_registry.acr.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_machine_learning_workspace.ws.identity[0].principal_id
}
