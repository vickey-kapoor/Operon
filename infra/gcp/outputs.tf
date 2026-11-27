# Set the first two as repo variables GCP_WORKLOAD_IDENTITY_PROVIDER and GCP_SERVICE_ACCOUNT.
output "workload_identity_provider" {
  description = "Repo variable GCP_WORKLOAD_IDENTITY_PROVIDER."
  value       = google_iam_workload_identity_pool_provider.operon_repo.name
}

output "service_account_email" {
  description = "Repo variable GCP_SERVICE_ACCOUNT."
  value       = google_service_account.osworld_ci.email
}

output "vm_name" {
  description = "Repo variable OSWORLD_VM_NAME (only when not operon-osworld)."
  value       = google_compute_instance.osworld.name
}

output "zone" {
  description = "Repo variable OSWORLD_GCP_ZONE (only when not us-central1-a)."
  value       = google_compute_instance.osworld.zone
}

output "project_id" {
  description = "Repo variable GCP_PROJECT_ID (only when not propane-cubist-479515-c5)."
  value       = var.project_id
}
