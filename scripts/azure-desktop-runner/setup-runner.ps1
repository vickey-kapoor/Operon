<#
.SYNOPSIS
Turns a fresh Azure Windows VM into the self-hosted runner for the live desktop
benchmark (.github/workflows/live-desktop-bench.yml, runner input "azure").

.DESCRIPTION
Run once, in an elevated PowerShell, while signed in over RDP as the account the
runner should use (the VM's admin account). It:

  - installs PowerShell 7, Git and Python 3.14 (the workflow's shell and toolchain)
  - turns off Windows Update, notifications, the lock screen, the screen saver,
    sleep, Server Manager at logon and the shutdown reason prompt, so nothing
    pops up over the desktop the agent drives
  - signs that account in automatically at boot (Sysinternals Autologon keeps the
    password as an LSA secret, not in plain text in the registry)
  - registers the GitHub Actions runner with the "operon-desktop" label and starts
    it at logon, hidden, inside the interactive session; a runner installed as a
    Windows service has no desktop to screenshot, so it is not a service
  - sets the screen to 1920x1080 each time the runner starts

Then it restarts the VM. After the restart the account signs in, the runner comes
online, and the VM is ready. See docs/azure-desktop-runner.md for the whole setup.

.PARAMETER RunnerToken
A runner registration token: repository Settings > Actions > Runners > New
self-hosted runner, or `gh api -X POST repos/vickey-kapoor/Operon/actions/runners/registration-token --jq .token`.
It expires after an hour.

.PARAMETER Password
The password of the account you are signed in as, for auto-logon.

.EXAMPLE
.\setup-runner.ps1 -RunnerToken AAAA... -Password (Read-Host -AsSecureString)
#>
param(
    [Parameter(Mandatory)] [string] $RunnerToken,
    [Parameter(Mandatory)] [securestring] $Password,
    [string] $Repo = "vickey-kapoor/Operon",
    [string] $RunnerName = "azure-desktop",
    [string] $Labels = "operon-desktop",
    [string] $RunnerDir = "C:\actions-runner",
    [string] $PythonVersion = "3.14.0",
    [switch] $NoRestart
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest is far slower with the progress bar

$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this from an elevated PowerShell (Run as administrator)."
}
$userName = $env:USERNAME
$downloads = Join-Path $env:TEMP "operon-runner-setup"
New-Item -ItemType Directory -Force -Path $downloads | Out-Null

function Get-LatestAsset([string] $repo, [string] $pattern) {
    $release = Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest"
    $asset = $release.assets | Where-Object { $_.name -match $pattern } | Select-Object -First 1
    if (-not $asset) { throw "No asset matching '$pattern' in the latest $repo release." }
    return $asset
}

function Save-File([string] $url, [string] $name) {
    $path = Join-Path $downloads $name
    Invoke-WebRequest -Uri $url -OutFile $path -UseBasicParsing
    return $path
}

# Every download is checked before it runs: GitHub release assets against the
# SHA-256 digest GitHub publishes for them, other installers by their Authenticode
# signature. A mismatch stops the script.
function Save-Asset($asset) {
    $path = Save-File $asset.browser_download_url $asset.name
    if ($asset.digest -notmatch '^sha256:([0-9a-fA-F]{64})$') { throw "No SHA-256 digest published for $($asset.name)." }
    $expected = $Matches[1]
    $actual = (Get-FileHash -Algorithm SHA256 -Path $path).Hash
    if ($actual -ne $expected.ToUpperInvariant()) { throw "SHA-256 mismatch for $($asset.name): got $actual, expected $expected." }
    return $path
}

function Assert-Signed([string] $path, [string] $publisher) {
    $signature = Get-AuthenticodeSignature -FilePath $path
    if ($signature.Status -ne "Valid" -or $signature.SignerCertificate.Subject -notmatch $publisher) {
        throw "$path is not validly signed by $publisher (status $($signature.Status))."
    }
}

function Set-Reg([string] $path, [string] $name, $value, [string] $type = "DWord") {
    if (-not (Test-Path $path)) { New-Item -Path $path -Force | Out-Null }
    New-ItemProperty -Path $path -Name $name -Value $value -PropertyType $type -Force | Out-Null
}

# --- toolchain ---------------------------------------------------------------
Write-Output "Installing PowerShell 7..."
$asset = Get-LatestAsset "PowerShell/PowerShell" '^PowerShell-[\d.]+-win-x64\.msi$'
$msi = Save-Asset $asset
Start-Process msiexec.exe -Wait -ArgumentList "/i", "`"$msi`"", "/quiet", "/norestart", "ADD_PATH=1"

Write-Output "Installing Git..."
$asset = Get-LatestAsset "git-for-windows/git" '^Git-[\d.]+-64-bit\.exe$'
$exe = Save-Asset $asset
Start-Process $exe -Wait -ArgumentList "/VERYSILENT", "/NORESTART", "/SUPPRESSMSGBOXES"

Write-Output "Installing Python $PythonVersion..."
$exe = Save-File "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe" "python.exe"
Assert-Signed $exe "Python Software Foundation"
Start-Process $exe -Wait -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1", "Include_test=0"

# --- keep the desktop quiet --------------------------------------------------
Write-Output "Turning off updates, notifications, lock screen and sleep..."
$au = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU"
Set-Reg $au "NoAutoUpdate" 1
Set-Reg $au "NoAutoRebootWithLoggedOnUsers" 1
Stop-Service wuauserv -Force -ErrorAction SilentlyContinue
Set-Service wuauserv -StartupType Disabled

Set-Reg "HKLM:\SOFTWARE\Policies\Microsoft\Windows\Explorer" "DisableNotificationCenter" 1
Set-Reg "HKCU:\Software\Policies\Microsoft\Windows\CurrentVersion\PushNotifications" "NoToastApplicationNotification" 1
Set-Reg "HKCU:\Software\Microsoft\Windows\CurrentVersion\PushNotifications" "ToastEnabled" 0
Set-Reg "HKLM:\SOFTWARE\Policies\Microsoft\Windows\Personalization" "NoLockScreen" 1
Set-Reg "HKLM:\SOFTWARE\Policies\Microsoft\Windows\OOBE" "DisablePrivacyExperience" 1
Set-Reg "HKCU:\Control Panel\Desktop" "ScreenSaveActive" "0" "String"
Set-Reg "HKLM:\SOFTWARE\Policies\Microsoft\Windows NT\Reliability" "ShutdownReasonOn" 0
# Server only: no Server Manager window at logon.
Set-Reg "HKLM:\SOFTWARE\Microsoft\ServerManager" "DoNotOpenServerManagerAtLogon" 1
Get-ScheduledTask -TaskName ServerManager -ErrorAction SilentlyContinue | Disable-ScheduledTask | Out-Null
# A logon banner would stop auto-logon at the banner.
$winlogon = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System"
Set-Reg $winlogon "legalnoticecaption" "" "String"
Set-Reg $winlogon "legalnoticetext" "" "String"
powercfg /change monitor-timeout-ac 0
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

# --- auto-logon --------------------------------------------------------------
Write-Output "Setting up auto-logon for $userName..."
$autologon = Save-File "https://live.sysinternals.com/Autologon64.exe" "Autologon64.exe"
Assert-Signed $autologon "Microsoft Corporation"
$plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Password))
Start-Process $autologon -Wait -ArgumentList $userName, $env:COMPUTERNAME, $plain, "/accepteula"
$plain = $null

# --- GitHub Actions runner ---------------------------------------------------
Write-Output "Installing the GitHub Actions runner..."
$asset = Get-LatestAsset "actions/runner" '^actions-runner-win-x64-[\d.]+\.zip$'
$zip = Save-Asset $asset
New-Item -ItemType Directory -Force -Path $RunnerDir | Out-Null
Expand-Archive -Path $zip -DestinationPath $RunnerDir -Force
& "$RunnerDir\config.cmd" --unattended --replace `
    --url "https://github.com/$Repo" --token $RunnerToken `
    --name $RunnerName --labels $Labels --work "_work"
if ($LASTEXITCODE -ne 0) { throw "Runner registration failed (is the token older than an hour?)." }

# Started at logon: sets the screen size, then runs the runner in this session.
@'
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class Display {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct DEVMODE {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmDeviceName;
        public short dmSpecVersion, dmDriverVersion, dmSize, dmDriverExtra;
        public int dmFields, dmPositionX, dmPositionY, dmDisplayOrientation, dmDisplayFixedOutput;
        public short dmColor, dmDuplex, dmYResolution, dmTTOption, dmCollate;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmFormName;
        public short dmLogPixels;
        public int dmBitsPerPel, dmPelsWidth, dmPelsHeight, dmDisplayFlags, dmDisplayFrequency;
        public int dmICMMethod, dmICMIntent, dmMediaType, dmDitherType, dmReserved1, dmReserved2;
        public int dmPanningWidth, dmPanningHeight;
    }
    [DllImport("user32.dll")] static extern bool EnumDisplaySettings(string dev, int mode, ref DEVMODE dm);
    [DllImport("user32.dll")] static extern int ChangeDisplaySettings(ref DEVMODE dm, int flags);
    public static int Set(int w, int h) {
        DEVMODE dm = new DEVMODE();
        dm.dmSize = (short)Marshal.SizeOf(typeof(DEVMODE));
        EnumDisplaySettings(null, -1, ref dm);
        dm.dmPelsWidth = w; dm.dmPelsHeight = h;
        dm.dmFields = 0x80000 | 0x100000;
        return ChangeDisplaySettings(ref dm, 1);  // CDS_UPDATEREGISTRY
    }
}
"@
[Display]::Set(1920, 1080) | Out-Null
Set-Location $PSScriptRoot
& .\run.cmd *> runner-console.log
'@ | Set-Content -Path "$RunnerDir\start-runner.ps1" -Encoding UTF8

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$RunnerDir\start-runner.ps1`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:COMPUTERNAME\$userName"
$taskPrincipal = New-ScheduledTaskPrincipal -UserId "$env:COMPUTERNAME\$userName" -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "OperonRunner" -Action $action -Trigger $trigger `
    -Principal $taskPrincipal -Settings $settings -Force | Out-Null

# Leaving RDP by closing the window locks the session, and a locked session has no
# desktop to capture. This hands the session back to the console instead.
@'
@echo off
for /f "skip=1 tokens=3" %%s in ('query user %USERNAME%') do tscon %%s /dest:console
'@ | Set-Content -Path "$RunnerDir\leave-rdp.cmd" -Encoding ASCII

Write-Output ""
Write-Output "Done. When you leave RDP in future, run $RunnerDir\leave-rdp.cmd instead of closing the window."
if ($NoRestart) {
    Write-Output "Restart the VM to sign in automatically and start the runner."
} else {
    Write-Output "Restarting in 15 seconds..."
    Start-Sleep -Seconds 15
    Restart-Computer -Force
}
