# GitHub Actions signs in with OIDC (workload identity federation): no key file exists anywhere.
# The pool and provider may already exist (KAN-21): set import_existing_wif = true to adopt them.
locals {
  wif_imports = var.import_existing_wif ? { this = true } : {}
}

import {
  for_each = local.wif_imports
  to       = google_iam_workload_identity_pool.github
  id       = "projects/${var.project_id}/locations/global/workloadIdentityPools/github"
}

import {
  for_each = local.wif_imports
  to       = google_iam_workload_identity_pool_provider.operon_repo
  id       = "projects/${var.project_id}/locations/global/workloadIdentityPools/github/providers/operon-repo"
}

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"

  depends_on = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "operon_repo" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "operon-repo"
  display_name                       = "Operon repository"

  attribute_mapping = {
    "google.subject"          = "assertion.sub"
    "attribute.repository_id" = "assertion.repository_id"
    "attribute.ref"           = "assertion.ref"
    # A job of a called workflow (benchmark.yml -> osworld-bench.yml) carries job_workflow_ref; fall back to
    # workflow_ref so a run started directly still maps, in case GitHub leaves job_workflow_ref out there.
    # Both are "<owner>/<repo>/.github/workflows/<file>@<ref>".
    "attribute.workflow_ref" = "\"job_workflow_ref\" in assertion ? assertion.job_workflow_ref : assertion.workflow_ref"
  }

  # Only this repository (by its numeric id, which a rename or a new repo with the same name cannot take)
  # and only runs of the main branch get a token at all.
  attribute_condition = "assertion.repository_id == \"${var.github_repository_id}\" && assertion.ref == \"refs/heads/main\""

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# The identity the OSWorld workflow acts as. It can do nothing but start, stop and hand a runner
# script to the one VM (the roles below).
resource "google_service_account" "osworld_ci" {
  account_id   = "operon-osworld-ci"
  display_name = "Operon OSWorld workflow (GitHub Actions)"

  depends_on = [google_project_service.apis]
}

# Only jobs of osworld-bench.yml on main (also when benchmark.yml calls it) may act as it.
resource "google_service_account_iam_member" "osworld_ci_wif" {
  service_account_id = google_service_account.osworld_ci.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.workflow_ref/${var.github_workflow_ref}"
}

resource "google_project_iam_custom_role" "osworld_operator" {
  role_id     = "operonOsworldOperator"
  title       = "Operon OSWorld VM operator"
  description = "Start and stop the OSWorld runner VM, hand it a one-job runner script, read its serial console (KAN-224)."
  permissions = [
    "compute.instances.get",
    "compute.instances.getSerialPortOutput",
    "compute.instances.setMetadata",
    "compute.instances.start",
    "compute.instances.stop",
  ]
}

# gcloud waits for the operation a start, stop or metadata change returns; reading an operation is a
# zone-level permission, so this one is granted on the project.
resource "google_project_iam_custom_role" "zone_ops_reader" {
  role_id     = "operonZoneOpsReader"
  title       = "Operon zone operation reader"
  description = "Read zone operations, so gcloud can wait for its own VM calls (KAN-224)."
  permissions = ["compute.zoneOperations.get"]
}

# On the one instance only, not the project.
resource "google_compute_instance_iam_member" "osworld_ci_operator" {
  zone          = google_compute_instance.osworld.zone
  instance_name = google_compute_instance.osworld.name
  role          = google_project_iam_custom_role.osworld_operator.id
  member        = "serviceAccount:${google_service_account.osworld_ci.email}"
}

resource "google_project_iam_member" "osworld_ci_zone_ops" {
  project = var.project_id
  role    = google_project_iam_custom_role.zone_ops_reader.id
  member  = "serviceAccount:${google_service_account.osworld_ci.email}"
}
