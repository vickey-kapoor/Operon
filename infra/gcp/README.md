# OSWorld runner on Google Cloud (KAN-224)

Terraform for the VM that runs the OSWorld benchmark (`.github/workflows/osworld-bench.yml`, see
`docs/osworld-runner.md`). Project `propane-cubist-479515-c5`, region `us-central1`, zone `us-central1-a`.
Terraform 1.10 or newer (CI uses 1.17.2), provider `hashicorp/google` 8.12.0 (pinned; `.terraform.lock.hcl` is committed
for Linux and Windows).

| Folder | What it makes | State |
|---|---|---|
| `bootstrap/` | the versioned bucket `operon-tfstate-<project>` that holds the state of `infra/gcp` | local, in `bootstrap/terraform.tfstate` (gitignored) |
| `infra/gcp` | APIs, network `operon-bench` (nothing comes in), VM `operon-osworld`, the GitHub OIDC sign-in, the workflow's service account and its two custom roles | in the bucket, prefix `osworld` |

What `infra/gcp` makes:

- **APIs**: compute, iam, iamcredentials, sts, cloudresourcemanager (never turned off on destroy).
- **Network** `operon-bench` with subnet `operon-bench-us-central1` (10.20.0.0/24) and one firewall rule that
  denies all incoming traffic. The VM has an ephemeral public address (Standard tier) for outgoing traffic only.
- **VM** `operon-osworld`: n2-standard-4 (4 vCPU, 16 GB), Debian 12, 150 GB pd-balanced disk, nested KVM on,
  Shielded VM, OS Login, project SSH keys blocked, **no service account** (no Google token on the VM).
  Terraform creates it stopped and never starts or stops it afterwards (`desired_status` is ignored after
  creation). A run longer than 17 hours is stopped by Google Cloud (`max_run_duration`), and the VM powers
  itself off after 20 minutes with no runner (`operon-idle-off`): two backstops in case a run never reaches
  its `stop-vm` job.
- **Startup script** `startup.sh` (every boot, restart-safe): Docker, Python, git, an 8 GB swap file, the user
  `runner` (groups docker and kvm), the GitHub Actions runner 2.338.2 checked against its sha256 (unpacked
  only: each run brings a one-job config), a memory log on the serial console every 30 s, and the watcher that
  runs the one-job runner script the workflow puts on the metadata key `operon-jit-script`.
- **Sign-in**: workload identity pool `github`, provider `operon-repo` (GitHub's OIDC issuer). Only repository
  id 1177308224 on `refs/heads/main` gets a token. Service account `operon-osworld-ci` may be used only by
  jobs of `osworld-bench.yml@refs/heads/main` (also when `benchmark.yml` calls it).
- **Permissions** of `operon-osworld-ci`: custom role `operonOsworldOperator` (`compute.instances.get`,
  `start`, `stop`, `setMetadata`, `getSerialPortOutput`) on the VM only, and `operonZoneOpsReader`
  (`compute.zoneOperations.get`, so gcloud can wait for its own calls) on the project. Nothing else.

## Cost

About $0.20 an hour while a run is going (VM and address), about $15 a month for the 150 GB disk while it
is stopped, and cents for the state bucket. A 25-task run takes about 3-5 hours: about $0.60-1.00 of VM time,
on top of the model cost.

## One-time setup

On a PC with `gcloud auth application-default login` done for an owner of the project:

1. Ask Vickey for the cost OK first (see Cost).
2. Check whether the sign-in pool already exists (an earlier story may have made it):

   ```bash
   gcloud iam workload-identity-pools describe github --location global --project propane-cubist-479515-c5
   gcloud iam workload-identity-pools providers describe operon-repo --workload-identity-pool github --location global --project propane-cubist-479515-c5
   ```

   If both exist, pass `-var import_existing_wif=true` to the plan and apply in step 4: Terraform then takes
   them over instead of creating them (and changes them to the settings here). If only the pool exists,
   import it by hand: `terraform -chdir=infra/gcp import google_iam_workload_identity_pool.github projects/propane-cubist-479515-c5/locations/global/workloadIdentityPools/github`.
   A pool deleted less than 30 days ago blocks the name: undelete it (`gcloud iam workload-identity-pools undelete github --location global`) and import it.
3. The state bucket:

   ```bash
   terraform -chdir=infra/gcp/bootstrap init
   terraform -chdir=infra/gcp/bootstrap plan -out bootstrap.tfplan
   terraform -chdir=infra/gcp/bootstrap apply bootstrap.tfplan
   ```

   If `bootstrap/terraform.tfstate` is ever lost, `terraform -chdir=infra/gcp/bootstrap import google_storage_bucket.tfstate operon-tfstate-propane-cubist-479515-c5` brings it back.
4. The VM and the sign-in:

   ```bash
   terraform -chdir=infra/gcp init
   terraform -chdir=infra/gcp plan -out osworld.tfplan
   terraform -chdir=infra/gcp apply osworld.tfplan
   terraform -chdir=infra/gcp output
   ```

5. Repo variables (Settings > Secrets and variables > Actions > Variables):
   `GCP_WORKLOAD_IDENTITY_PROVIDER` = output `workload_identity_provider`,
   `GCP_SERVICE_ACCOUNT` = output `service_account_email`. `OSWORLD_VM_NAME`, `OSWORLD_GCP_ZONE` and
   `GCP_PROJECT_ID` only when they differ from `operon-osworld`, `us-central1-a` and the project above.
   The repo secret `RUNNER_ADMIN_TOKEN` stays as it is (docs/osworld-runner.md).

The first run after the apply installs everything on the VM (a few minutes; `start-vm` waits up to 15 for the
runner) and then downloads OSWorld's image (about 11 GB), so it takes longer than later runs.

## Looking at the VM

```bash
gcloud compute instances describe operon-osworld --zone us-central1-a --format='value(status)'
gcloud compute instances get-serial-port-output operon-osworld --zone us-central1-a | grep 'operon-'
gcloud compute instances stop operon-osworld --zone us-central1-a      # if a run left it on
gcloud compute ssh operon-osworld --zone us-central1-a --tunnel-through-iap   # needs an allow rule for IAP first: none exists
```

Changing `startup.sh` is an in-place metadata update; it runs on the next start. Changing the machine type
needs the VM stopped.
