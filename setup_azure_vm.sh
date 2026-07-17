#!/usr/bin/env bash
# Provision + bootstrap an Azure T4 spot VM for this project.
# Run the "az vm create" block from your local machine (needs Azure CLI + login).
# Run everything after "ssh in" ON the VM itself.

set -euo pipefail

# ---------- 1. Provision (run locally, requires: az login) ----------
RESOURCE_GROUP="slm-rg"
LOCATION="eastus"          # change to a region with T4 spot capacity near you
VM_NAME="slm-t4-vm"
ADMIN_USER="azureuser"

az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

az vm create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --image "microsoft-dsvm:ubuntu-hpc:2204:latest" \
  --size "Standard_NC4as_T4_v3" \
  --priority Spot \
  --eviction-policy Deallocate \
  --max-price -1 \
  --admin-username "$ADMIN_USER" \
  --generate-ssh-keys \
  --os-disk-size-gb 128

# Enable auto-shutdown at 8pm local time (optional, saves cost)
az vm auto-shutdown --resource-group "$RESOURCE_GROUP" --name "$VM_NAME" --time 2000

echo "VM created. SSH in with:"
az vm show -d --resource-group "$RESOURCE_GROUP" --name "$VM_NAME" --query publicIps -o tsv

# ---------- 2. Bootstrap (run ON the VM after ssh-ing in) ----------
cat <<'EOF'
Run these on the VM (the Data Science VM image already has NVIDIA driver + CUDA + conda):

  nvidia-smi                      # confirm GPU is visible
  git clone <your-repo-url> slm-from-scratch
  cd slm-from-scratch
  conda env create -f environment.yml
  conda activate slm
  jupyter lab --no-browser --port=8888

Then from your LOCAL machine, tunnel in:
  ssh -L 8888:localhost:8888 azureuser@<VM_PUBLIC_IP>
EOF
