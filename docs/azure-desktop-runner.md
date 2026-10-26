# Azure desktop benchmark runner

The live desktop benchmark (`.github/workflows/live-desktop-bench.yml`) can run on
GitHub's hosted `windows-latest` runner or on our own Azure VM. Pick the VM with
the workflow's `runner: azure` input, or comment `/bench desktop azure` on a PR.

On `azure`, the workflow starts the VM, waits for its runner to come online, runs
the benchmark, and deallocates the VM again, so the VM costs nothing between runs
except its disk and IP (a few dollars a month). Only one Azure run happens at a time.

The VM is a Standard_D4s_v5 (4 vCPU, 16 GB) running Windows Server 2022 with the
desktop experience. The setup script signs its admin account in automatically at
boot and starts the runner inside that interactive session at 1920x1080, with
Windows Update, notifications and the lock screen off. Server has no Calculator
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

1. Get a runner token: repository **Settings > Actions > Runners > New self-hosted
   runner**, choose Windows, and copy the value after `--token` in the Configure
   step. It expires after an hour.
2. RDP in: `mstsc /v:$(az vm show -d -g $rg -n $vm --query publicIps -o tsv)`. Sign in as
   `.\operon`; the leading `.\` picks the VM's local account instead of your Microsoft account.
3. Get the script onto the VM. Paste `scripts/azure-desktop-runner/setup-runner.ps1`
   into Notepad and save it as `C:\setup-runner.ps1`, or download it from the repository.
4. In an elevated PowerShell on the VM:

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   C:\setup-runner.ps1 -RunnerToken <token> -Password (Read-Host -AsSecureString "operon's password")
   ```

The script installs PowerShell 7, Git and Python 3.14, quiets the desktop, sets up
auto-logon, registers the runner (name `azure-desktop`, label `operon-desktop`)
and restarts the VM, which drops your RDP session. A minute later the runner
should show as **Idle** under Settings > Actions > Runners.

Don't RDP in again unless you need to. Closing an RDP window locks the session,
and a locked session has no desktop for the agent to see. If you do RDP in,
leave by running `C:\actions-runner\leave-rdp.cmd`, which hands the session back
to the console, or restart the VM.

### 3. Let the workflow start and stop the VM

```powershell
$scope = az group show --name $rg --query id -o tsv
az ad sp create-for-rbac --name operon-desktop-bench --role "Virtual Machine Contributor" `
  --scopes $scope --json-auth > azure-credentials.json

gh secret set AZURE_CREDENTIALS --repo vickey-kapoor/Operon < azure-credentials.json
Remove-Item azure-credentials.json

az vm deallocate --resource-group $rg --name $vm
```

The service principal can only manage VMs in this resource group. Without `gh`,
add the secret under Settings > Secrets and variables > Actions. If you named the
resource group or VM differently, also add repo variables `AZURE_VM_RESOURCE_GROUP`
and `AZURE_VM_NAME`; the workflow defaults to `operon-bench` and `operon-desktop`.

## Troubleshooting

- **`az vm create` fails with SkuNotAvailable.** The region has no spare capacity
  for that size on your subscription, which is common on new ones. Add `-l <region>`
  (for example `westus2`) to try another region, or use Standard_D4as_v5 or
  Standard_D4s_v6, which are the same 4 vCPU / 16 GB. The resource group can stay
  where it is.
- **The bench job sits at "Waiting for a runner".** The VM is up but the runner
  isn't online. Usually auto-logon failed (for example after a password change):
  RDP in, rerun the script with a new token, and let it restart.
- **Screenshots are black or the run can't see windows.** The session is locked or
  disconnected, most often after an RDP session was closed. Restart the VM.
- **The runner disappeared from Settings > Actions > Runners.** GitHub removes a
  self-hosted runner that hasn't connected for 14 days, and a deallocated VM
  doesn't connect. If two weeks pass without an Azure run, start the VM
  (`az vm start`), RDP in, rerun the script with a new token, and deallocate it.
- **Runner logs** are in `C:\actions-runner\_diag` and
  `C:\actions-runner\runner-console.log`.
