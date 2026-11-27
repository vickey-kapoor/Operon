variable "project_id" {
  description = "Google Cloud project that holds the OSWorld runner."
  type        = string
  default     = "propane-cubist-479515-c5"
}

variable "region" {
  description = "Region of the network."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Zone of the VM (the workflow's OSWORLD_GCP_ZONE must match)."
  type        = string
  default     = "us-central1-a"
}

variable "vm_name" {
  description = "Name of the runner VM (the workflow's OSWORLD_VM_NAME must match)."
  type        = string
  default     = "operon-osworld"
}

variable "machine_type" {
  description = "VM size. N2 runs nested KVM, which OSWorld's task VM needs."
  type        = string
  default     = "n2-standard-4"
}

variable "disk_gb" {
  description = "Boot disk size in GB (OSWorld's image is about 11 GB, its qcow2 grows to 32 GB per task VM)."
  type        = number
  default     = 150
}

variable "github_repository_id" {
  description = "Numeric id of the GitHub repository that may sign in (survives a rename, unlike owner/name)."
  type        = string
  default     = "1177308224"
}

variable "github_workflow_ref" {
  description = "The only workflow file and ref whose jobs may act as the CI service account."
  type        = string
  default     = "vickey-kapoor/Operon/.github/workflows/osworld-bench.yml@refs/heads/main"
}

variable "import_existing_wif" {
  description = "true when the workload identity pool `github` and its provider `operon-repo` already exist (KAN-21): they are imported instead of created."
  type        = bool
  default     = false
}
