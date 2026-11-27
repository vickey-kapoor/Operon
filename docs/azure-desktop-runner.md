# Azure desktop benchmark runner

The live desktop benchmark (`.github/workflows/live-desktop-bench.yml`) can run on
GitHub's hosted `windows-latest` runner or on our own Azure VM. Pick the VM with
the workflow's `runner: azure` input. (Paid desktop runs of Operon's own suites were stopped, KAN-184; this runner is kept for the cleanup decision.)

On `azure`, the workflow starts the VM, mints a one-job runner for the run under a
label unique to it (`operon-desktop-<run id>-<attempt>-<random>`, KAN-195; see `scripts/bench-runners/jit-runner.sh`),
waits for it to come online, runs the benchmark, and deallocates the VM again, so the VM costs nothing between runs
except its disk and IP (a few dollars a month). Only one Azure run happens at a time.

The VM is a Standard_D4s_v5 (4 vCPU, 16 GB) running Windows Server 2022 with the
desktop experience. The setup script creates a standard (non-admin) account,
`benchrunner`, signs it in automatically at boot and starts the runner, unelevated,
inside that interactive session at 1920x1080, with Windows Update, notifications
and the lock screen off. The admin account `operon` is only for you, over RDP. Server has no Calculator
app, so `calculator_multiply` fails there, as it does on the hosted runner.

## One-time setup

You need the Azure CLI (`az login` done) and admin rights on the repository. The
commands are PowerShell; set your own values on the first lines.

### 1. Create the VM

```powershell
$rg = "operon-bench"; $vm = "operon-desktop"; $location = "eastus"
$adminPassword = Read-Host "VM admin password (12+ chars, upper, lower, digit, symbol)"

az group create --name $rg --location $location
az vm create --resource-group $rg --name $vm `
  --image MicrosoftWindowsServer:WindowsServer:2022-datacenter-g2:latest `
  --size Standard_D4s_v5 --storage-sku StandardSSD_LRS --os-disk-size-gb 128 `
  --admin-username operon --admin-password $adminPassword `
  --public-ip-sku Standard --nsg-rule NONE

# RDP from your current IP only. The runner itself needs no inbound ports.
$myIp = (Invoke-RestMethod https://api.ipify.org)
az network nsg rule create --resource-group $rg --nsg-name "${vm}NSG" --name rdp-from-me `
  --priority 1000 --protocol Tcp --destination-port-ranges 3389 --source-address-prefixes $myIp

# Backstop: deallocate every day at 07:00 UTC in case a run never reaches its stop job.
az vm auto-shutdown --resource-group $rg --name $vm --time 0700
```

### 2. Run the setup script on the VM

1. Nothing to register: the script unpacks the runner but does not register it, and needs no
   token. Create the repo secret `RUNNER_ADMIN_TOKEN` (a fine-grained token for this repository only,
   permission Administration: read and write) so each run can mint its one-job runner.
2. RDP in: `mstsc /v:$(az vm show -d -g $rg -n $vm --query publicIps -o tsv)`. Sign in as
   `.\operon`; the leading `.\` picks the VM's local account instead of your Microsoft account.
3. Get the script onto the VM. Paste `scripts/azure-desktop-runner/setup-runner.ps1`
   into Notepad and save it as `C:\setup-runner.ps1`, or download it from the repository.
4. In an elevated PowerShell on the VM:

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   C:\setup-runner.ps1
   ```

The script installs PowerShell 7, Git and Python 3.14, quiets the desktop, creates
the `benchrunner` account with a random password nobody sees, sets up auto-logon for
it, unpacks the runner with a logon task that waits for a run's runner config
(`C:\actions-runner\.jitconfig`, readable only by `benchrunner`) and restarts the VM, which
drops your RDP session. The runner shows under Settings > Actions > Runners only while a
run's job is on it.

RDP in as `.\operon` when you need the VM. That is a separate session, so the
runner's desktop keeps running. Don't sign in as `benchrunner`.

### Edge for the Hybrid tasks

The Hybrid benchmark (KAN-73, no longer a paid run since KAN-184) has the agent use Microsoft Edge
on the VM. Run `scripts/edge-policies.ps1` once, in an elevated PowerShell as
`.\operon` (paste it into Notepad as `C:\edge-policies.ps1` like the setup script):

```powershell
Set-ExecutionPolicy -Scope Process Bypass
C:\edge-policies.ps1
```

It sets machine-wide Edge policies: no first-run wizard, no "restore pages" prompt,
no password or address autofill, no shopping assistant or sidebar. The runner account
can't set them itself, since it is not an administrator, and GitHub's hosted runner gets
them from the workflow before each Hybrid run. Do it before the first Hybrid run, then
retake the clean snapshot (step 4).

### Why a standard account

A desktop run on the Azure runner executes the benchmarked branch's code on this VM, and
the VM is not wiped between runs. Under a standard account that code can't install
services, change system settings or read the admin's files. Each run also starts by
removing the usual places it could leave something to start later (the account's
Run keys, Startup folder, its own scheduled tasks and PowerShell profiles), so a
leftover can't see the next run's API keys. That cleanup is best effort. Rebuilding
the VM from a fresh image is the only full reset, so rebuild it if a branch you
didn't write was ever benchmarked on it.

**Moving an existing VM to the standard account.** Older setups ran the runner as
the admin account. RDP in as `.\operon` (no token is needed now), copy the current
`setup-runner.ps1` to the VM and run step 2 again. It creates `benchrunner`, removes
the old runner and its logon task and restarts. The next run deletes the old lasting
runner's registration from GitHub. Do it
while no benchmark is running.

**An existing `operon-desktop` VM must re-run `setup-runner.ps1` before its next `azure` run (KAN-195).**
The old script registered a lasting runner and started it without waiting for a one-job config; the new
logon task is what picks up each run's config, so a VM set up the old way never comes online for a run.

### 3. Let the workflow start and stop the VM

The workflow signs in to Azure with GitHub's OIDC token, so no Azure secret is
stored in GitHub, and the identity can only start, stop and read the VM and run
one command on it: a custom role, because the built-in Virtual Machine Contributor
role also allows password resets and more. Run Command (code as SYSTEM on the VM) is
now in the role, because that is how a run hands the VM its one-job runner config
(KAN-195). The trade-off: the identity is trusted for `main` only, and it now delivers
a one-job runner instead of the VM holding a lasting one that any approved workflow could use.
The run-command right is granted on the whole resource group, so it also covers every other VM in it.
(The OSWorld runner moved to Google Cloud in KAN-224 and no longer uses this role; see `docs/osworld-runner.md`.)

```powershell
$sub = az account show --query id -o tsv
$tenant = az account show --query tenantId -o tsv
$scope = az group show --name $rg --query id -o tsv

# A role with only what the workflow does.
@"
{
  "Name": "Operon bench VM operator",
  "Description": "Start, deallocate, read and hand a one-job runner config to the benchmark VMs",
  "Actions": [
    "Microsoft.Compute/virtualMachines/read",
    "Microsoft.Compute/virtualMachines/start/action",
    "Microsoft.Compute/virtualMachines/deallocate/action",
    "Microsoft.Compute/virtualMachines/instanceView/read",
    "Microsoft.Compute/virtualMachines/runCommand/action"
  ],
  "AssignableScopes": ["$scope"]
}
"@ | Set-Content operon-role.json
az role definition create --role-definition operon-role.json   # an existing role: az role definition update

# An app registration GitHub can sign in as, trusted only for this repo's main branch
# (/bench comments and manual runs both run main's workflow).
$appId = az ad app create --display-name operon-desktop-bench --query appId -o tsv
az ad sp create --id $appId
az ad app federated-credential create --id $appId --parameters (@"
{"name":"operon-main","issuer":"https://token.actions.githubusercontent.com","subject":"repo:vickey-kapoor/Operon:ref:refs/heads/main","audiences":["api://AzureADTokenExchange"]}
"@)
az role assignment create --assignee $appId --role "Operon bench VM operator" --scope $scope

gh variable set AZURE_CLIENT_ID --repo vickey-kapoor/Operon --body $appId
gh variable set AZURE_TENANT_ID --repo vickey-kapoor/Operon --body $tenant
gh variable set AZURE_SUBSCRIPTION_ID --repo vickey-kapoor/Operon --body $sub

az vm deallocate --resource-group $rg --name $vm
```

Without `gh`, add the three variables under Settings > Secrets and variables >
Actions > Variables. If you named the resource group or VM differently, also add
repo variables `AZURE_VM_RESOURCE_GROUP` and `AZURE_VM_NAME`; the workflow defaults
to `operon-bench` and `operon-desktop`.

**Moving off the older secret.** Earlier setups stored a service principal's
client secret as the `AZURE_CREDENTIALS` repo secret, with Virtual Machine
Contributor. The workflow still uses it while `AZURE_CLIENT_ID` is unset. Once the
OIDC sign-in works, delete the secret (`gh secret delete AZURE_CREDENTIALS`) and the
old service principal (`az ad sp delete --id <its appId>`).

### 4. Save a clean snapshot

Once the runner shows **Idle** and one desktop run on the runner has passed,
snapshot the deallocated VM's OS disk. It is the known-good state to go back to
if an update, a stray dialog or a half-finished run leaves the desktop in a mess.
A Standard snapshot is billed on the disk's used space, a few dollars a month.

```powershell
az vm deallocate --resource-group $rg --name $vm
$disk = az vm show --resource-group $rg --name $vm --query storageProfile.osDisk.managedDisk.id -o tsv
az snapshot create --resource-group $rg --name "$vm-clean" --source $disk --sku Standard_LRS
```

To go back to it, make a disk from the snapshot and swap it in (the VM must be deallocated):

```powershell
az disk create --resource-group $rg --name "$vm-os-restored" --source "$vm-clean" --sku StandardSSD_LRS
az vm update --resource-group $rg --name $vm --os-disk "$vm-os-restored"
```

The runner folder and the logon task are on the disk, so they come back too; there is no registration to
restore, since each run registers its own one-job runner.

## Troubleshooting

- **`az vm create` fails with SkuNotAvailable.** The region has no spare capacity
  for that size on your subscription, which is common on new ones. Add `-l <region>`
  (for example `westus2`) to try another region, or use Standard_D4as_v5 or
  Standard_D4s_v6, which are the same 4 vCPU / 16 GB. The resource group can stay
  where it is.
- **The bench job sits at "Waiting for a runner".** The VM is up but the runner
  isn't online. Usually auto-logon failed (for example after a password change):
  RDP in, rerun the script, and let it restart. The config file
  `C:\actions-runner\.jitconfig` is picked up when the account is signed in.
- **Screenshots are black or the run can't see windows.** The session is locked or
  disconnected, most often after an RDP session was closed. Restart the VM.
- **Runner logs** are in `C:\actions-runner\_diag` and
  `C:\actions-runner\runner-console.log`.
