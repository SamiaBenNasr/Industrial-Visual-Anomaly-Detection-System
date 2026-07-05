variable "resource_group_name" {
  description = "Resource group holding all PatchCore MLOps resources"
  type        = string
  default     = "rg-patchcore"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "polandcentral"
}

variable "acr_name" {
  description = "Container registry name (must be globally unique, alphanumeric only)"
  type        = string
  default     = "acrpatchcore"
}

variable "workspace_name" {
  description = "Azure ML workspace name"
  type        = string
  default     = "ws-patchcore"
}

variable "environment_name" {
  description = "Azure ML environment name"
  type        = string
  default     = "patchcore-env"
}




variable "endpoint_name" {
  description = "Azure ML managed online endpoint name"
  type        = string
  default     = "patchcore-endpoint"
}



variable "image_name" {
  description = "Docker image repository name in ACR"
  type        = string
  default     = "patchcore-inference"
}

variable "image_tag" {
  description = "Docker image tag to deploy (set by CI/CD, e.g. git SHA). Each new value creates a new environment version."
  type        = string
}


