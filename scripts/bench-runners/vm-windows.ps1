# Runs on operon-desktop as SYSTEM (az vm run-command, RunPowerShellScript), started by jit-runner.sh (KAN-195).
# jit-runner.sh puts two lines in front of this file:
#   $JitConfig = '<base64 one-job runner config>'
#   $RunnerLabel = '<operon-desktop-RUNID-ATTEMPT-RANDOM>'
# The runner itself must start in the runner account's own desktop session (a service has no screen to
# drive), so this only hands over the config: the logon task's start-runner.ps1
# (scripts/azure-desktop-runner/setup-runner.ps1) waits for the file, reads it, deletes it and runs
# `run.cmd --jitconfig`.
$ErrorActionPreference = "Stop"

# The standard account setup-runner.ps1 creates. A fixed default, not prepended by jit-runner.sh:
# setup-runner.ps1 has the same default (-RunnerUser), so only a VM set up under another name needs this line changed.
$RunnerUser = "benchrunner"
$RunnerDir = "C:\actions-runner"
$path = Join-Path $RunnerDir ".jitconfig"
$partial = Join-Path $RunnerDir ".jitconfig.tmp"

if (-not (Test-Path $RunnerDir)) { throw "No runner folder at $RunnerDir (run setup-runner.ps1 first)." }

# A registration left by an old lasting runner would make --jitconfig refuse (hidden files: -Force).
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $RunnerDir ".runner"), (Join-Path $RunnerDir ".credentials"), (Join-Path $RunnerDir ".credentials_rsaparams")

# Write to a temporary name and lock it down before the secret goes in (the runner account, SYSTEM and
# the Administrators only, by well-known SID so the account names don't matter), then rename it: the
# logon task only ever sees a complete file.
Remove-Item $path, $partial -Force -ErrorAction SilentlyContinue
New-Item -ItemType File -Path $partial | Out-Null
icacls $partial /inheritance:r /grant "${RunnerUser}:(M)" "*S-1-5-18:(F)" "*S-1-5-32-544:(F)" | Out-Null
if ($LASTEXITCODE -ne 0) { Remove-Item $partial -Force; throw "Could not restrict $partial." }
[IO.File]::WriteAllText($partial, $JitConfig)
Move-Item $partial $path

Write-Output "operon-jit-runner-started $RunnerLabel"
