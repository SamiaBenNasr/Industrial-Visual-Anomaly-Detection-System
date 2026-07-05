# -----------------------------------------------------------------
# RUN THIS ONCE, SEPARATELY, BEFORE your main Terraform project uses
# a remote backend. This creates the storage account that will hold
# the Terraform state file itself, so it can't be managed by the
# same state it stores (chicken-and-egg problem).
#
# Usage:
#   az group create --name rg-tfstate --location polandcentral
#
#   az storage account create --name sttfstatepatchcore \
#     --resource-group rg-tfstate --location polandcentral \
#     --sku Standard_LRS --encryption-services blob
#
#   az storage container create --name tfstate \
#     --account-name sttfstatepatchcore --auth-mode login
#
# Then uncomment the backend block in providers.tf and run:
#   terraform init -migrate-state
# -----------------------------------------------------------------
