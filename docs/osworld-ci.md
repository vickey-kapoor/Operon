# Running the OSWorld benchmark from GitHub Actions (KAN-169)

`.github/workflows/osworld-bench.yml` runs `python -m evals.osworld` (Operon on OSWorld's own tasks) on
the Azure Linux VM `operon-linux`. It starts the VM, runs the tasks there, prints one line per task in the
job summary, and always deallocates the VM. Trigger it from the Actions tab: **OSWorld benchmark → Run
workflow**. Manual only: every run spends Gemini credits (the 5-task pilot cost about $0.31).

OSWorld's task machine is a Docker container with its own Ubuntu desktop, so it needs a VM with nested
virtualization (`/dev/kvm`). GitHub-hosted runners can't give it that, hence the Azure VM.

## How it reaches the VM

The workflow uses `az vm run-command create` (the Azure control plane): no SSH, no open port, no private
key in GitHub. `scripts/osworld-vm-run.sh` runs on the VM, fetches the ref to benchmark with the run's
short-lived `GITHUB_TOKEN` (contents: read), reinstalls Operon, runs the tasks and prints a summary
(run-command keeps only the last 4 KB of output, so the full report stays on the VM in `.var/bench/`).
The Gemini key is a *protected parameter*: it is not shown in the run-command's settings, and the script
passes it to the runner only through that process's environment, never to a file.

## One-time setup

1. **Azure sign-in for GitHub** (same as the Desktop benchmark, `docs/azure-desktop-runner.md`): repo
   variables `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` (OIDC), or the older
   `AZURE_CREDENTIALS` secret. The identity needs to start, deallocate and run commands on the VM
   (role **Virtual Machine Contributor** on resource group `operon-bench` is enough).
2. **Repo secret** `GEMINI_API_KEY`.
3. **The VM** (`operon-linux`, Standard_D2s_v4, OS disk 128 GB) needs:
   - nested virtualization: `/dev/kvm` exists (it does on D-v4 and newer), and `azureuser` is in the `kvm` and `docker` groups;
   - Docker (`apt install docker.io`);
   - `~/OSWorld`: `git clone --depth 1 https://github.com/xlang-ai/OSWorld ~/OSWorld`;
   - `~/osw-venv`: Python 3.11 venv with `pip install -r ~/OSWorld/requirements.txt` and `uv` (`pip install uv`).
     OSWorld pins `torch` 2.5, which has no Python 3.14 wheels, so OSWorld and Operon use different Pythons;
   - `~/operon-venv`: `~/osw-venv/bin/uv venv --python 3.14 ~/operon-venv`.
   The first run downloads OSWorld's 11 GB task image (about 5 minutes) into the checkout's `docker_vm_data`.
4. Optional repo variables `AZURE_VM_RESOURCE_GROUP` (default `operon-bench`) and `AZURE_LINUX_VM_NAME`
   (default `operon-linux`).

## Cost

About $0.10 an hour of VM time while a run is going; Azure bills the 128 GB disk (about $10 a month) and
the static IP even while the VM is deallocated. Model spend is capped by `max_cost_usd` (the runner stops
starting tasks past it).

## Not yet validated

The workflow was written without a live GitHub run: dispatch it once with `limit: 1` and `max_cost_usd: 0.5`
to check the Azure sign-in, the run-command parameters and the summary output before relying on it.
