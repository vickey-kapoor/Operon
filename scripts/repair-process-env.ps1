param(
    [switch]$PersistForSession
)

$fixedPathext = ".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC;.CPL"
$fixedComspec = "C:\Windows\System32\cmd.exe"
$fixedSystemRoot = "C:\Windows"

$env:COMSPEC = $fixedComspec
$env:PATHEXT = $fixedPathext
$env:SystemRoot = $fixedSystemRoot
$env:WINDIR = $fixedSystemRoot

if ($PersistForSession) {
    Write-Host "COMSPEC=$env:COMSPEC"
    Write-Host "PATHEXT=$env:PATHEXT"
    Write-Host "SystemRoot=$env:SystemRoot"
    Write-Host "WINDIR=$env:WINDIR"
    Write-Host "Process launch environment repaired for this PowerShell session."
} else {
    Write-Host "COMSPEC, PATHEXT, SystemRoot, and WINDIR repaired for the current script scope."
}
