terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
    azapi = {
      source  = "azure/azapi"
      version = "~> 1.13"
    }
  }

  # Remote state -- required so your local runs and GitHub Actions runs
  # see the same state and don't try to recreate resources in duplicate.
  # Create this storage account once (see backend-bootstrap.tf) then
  # uncomment this block and run `terraform init -migrate-state`.
  #
  # backend "azurerm" {
  #   resource_group_name  = "rg-tfstate"
  #   storage_account_name = "sttfstatepatchcore"
  #   container_name       = "tfstate"
  #   key                  = "patchcore.tfstate"
  # }
}


provider "azurerm" {
  storage_use_azuread = true

  features {}
}

provider "azapi" {}
