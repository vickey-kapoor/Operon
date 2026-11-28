locals {
  services = toset([
    "cloudresourcemanager.googleapis.com",
    "compute.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
  ])
}

resource "google_project_service" "apis" {
  for_each           = local.services
  service            = each.value
  disable_on_destroy = false
}

# A network of its own. Nothing comes in: the VM only reaches out (GitHub, Docker Hub, PyPI, the model APIs).
resource "google_compute_network" "bench" {
  name                    = "operon-bench"
  auto_create_subnetworks = false

  depends_on = [google_project_service.apis]
}

resource "google_compute_subnetwork" "bench" {
  name          = "operon-bench-${var.region}"
  region        = var.region
  network       = google_compute_network.bench.id
  ip_cidr_range = "10.20.0.0/24"
}

# A custom network already drops incoming traffic; this rule says so explicitly, so a later
# allow rule has to be written above it on purpose. Outgoing traffic is allowed (implied rule).
resource "google_compute_firewall" "deny_all_ingress" {
  name      = "operon-bench-deny-all-ingress"
  network   = google_compute_network.bench.id
  direction = "INGRESS"
  priority  = 65534

  deny {
    protocol = "all"
  }

  source_ranges = ["0.0.0.0/0"]
}

# The runner VM. Terraform creates it stopped and never starts or stops it afterwards: the workflow does.
resource "google_compute_instance" "osworld" {
  name         = var.vm_name
  zone         = var.zone
  machine_type = var.machine_type

  # Created, then stopped (the first boot's provisioning is cut short; startup.sh is restart-safe).
  desired_status = "TERMINATED"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
      size  = var.disk_gb
      type  = "pd-balanced"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.bench.id
    # An ephemeral public address for outgoing traffic only (cheaper than Cloud NAT); the firewall lets nothing in.
    access_config {
      network_tier = "STANDARD"
    }
  }

  # OSWorld runs its task VM under KVM inside this VM.
  advanced_machine_features {
    enable_nested_virtualization = true
  }

  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }

  # Billing backstop: whatever happens to the workflow, a run longer than 17 hours is stopped
  # (the longest bench job, 25 tasks, is sized at 905 minutes).
  scheduling {
    provisioning_model          = "STANDARD"
    instance_termination_action = "STOP"
    max_run_duration {
      seconds = 61200
    }
  }

  # No service_account block: the VM has no Google identity and no token on its metadata server.

  metadata = {
    enable-oslogin         = "TRUE"
    block-project-ssh-keys = "TRUE"
    startup-script         = replace(file("${path.module}/startup.sh"), "\r\n", "\n")
  }

  lifecycle {
    # desired_status: the workflow starts and stops the VM. operon-jit-script: the one-job runner
    # script the workflow puts on the VM for one run (scripts/bench-runners/jit-runner.sh).
    ignore_changes = [desired_status, metadata["operon-jit-script"]]
  }

  depends_on = [google_project_service.apis]
}
