# One-time bootstrap for infra/gcp (KAN-224): the versioned bucket that holds infra/gcp's Terraform state.
# Its own state stays local (terraform.tfstate in this folder, gitignored): it is one bucket, and
# `terraform import` brings it back if that file is lost (see infra/gcp/README.md).
terraform {
  required_version = ">= 1.10"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "= 8.12.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" {
  description = "Google Cloud project that holds the OSWorld runner."
  type        = string
  default     = "propane-cubist-479515-c5"
}

variable "region" {
  description = "Region of the state bucket."
  type        = string
  default     = "us-central1"
}

resource "google_project_service" "storage" {
  service            = "storage.googleapis.com"
  disable_on_destroy = false
}

resource "google_storage_bucket" "tfstate" {
  name     = "operon-tfstate-${var.project_id}"
  location = upper(var.region)

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false

  versioning {
    enabled = true
  }

  # Keep the 10 newest old versions of each state file; older ones are deleted.
  lifecycle_rule {
    condition {
      num_newer_versions = 10
      with_state         = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.storage]
}

output "state_bucket" {
  description = "The bucket that infra/gcp/versions.tf names in its gcs backend."
  value       = google_storage_bucket.tfstate.name
}
