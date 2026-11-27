# The OSWorld benchmark runner on Google Cloud (KAN-224). See infra/gcp/README.md.
terraform {
  required_version = ">= 1.10"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "= 8.12.0"
    }
  }

  # Made by infra/gcp/bootstrap. A backend block cannot read variables, so the bucket is spelled out.
  backend "gcs" {
    bucket = "operon-tfstate-propane-cubist-479515-c5"
    prefix = "osworld"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}
