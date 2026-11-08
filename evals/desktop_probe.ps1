# Desktop benchmark state probe: prints one JSON object describing the open windows.
#
# Used by evals/desktop_state.py to grade the desktop tasks without a judge model.
# This is grading only, run after the agent stops; nothing here is part of the
# agent's perception or policy (which stay vision-only).
#
# For every visible top-level window: title, class, process, process start time,
# and from its UI Automation tree the element names, the selected items' names and
# the text of text controls (Notepad's document, a console's buffer). Also the
# folder each File Explorer window shows.
#
# -Inspect: comma-separated process names whose windows get the UI Automation walk;
#   other windows are listed (title, process, handle) without it, which keeps the probe fast.
# -Close: comma-separated process names to stop first; -CloseExplorer closes File
#   Explorer's folder windows first. Used to reset the desktop before a task.
param([int]$MaxNodes = 2500, [int]$MaxText = 20000, [string]$Inspect = "", [string]$Close = "",
      [switch]$CloseExplorer)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
$A = [System.Windows.Automation.AutomationElement]
$walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker

function Get-Texts($element, [ref]$names, [ref]$selected, [ref]$texts) {
    $stack = New-Object System.Collections.Stack
    $stack.Push($element)
    $seen = 0
    while ($stack.Count -gt 0 -and $seen -lt $MaxNodes) {
        $node = $stack.Pop()
        $seen++
        try {
            $name = $node.Current.Name
            if ($name) { $names.Value.Add($name) | Out-Null }
            $pattern = $null
            if ($node.TryGetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern, [ref]$pattern)) {
                if ($pattern.Current.IsSelected -and $name) { $selected.Value.Add($name) | Out-Null }
            }
            $pattern = $null
            if ($node.TryGetCurrentPattern([System.Windows.Automation.TextPattern]::Pattern, [ref]$pattern)) {
                $text = $pattern.DocumentRange.GetText($MaxText)
                if ($text) { $texts.Value.Add($text) | Out-Null }
            }
            $pattern = $null
            if ($node.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$pattern)) {
                $value = $pattern.Current.Value
                if ($value) { $texts.Value.Add($value) | Out-Null }
            }
        } catch { }
        try {
            $child = $walker.GetFirstChild($node)
            while ($child -ne $null) {
                $stack.Push($child)
                $child = $walker.GetNextSibling($child)
            }
        } catch { }
    }
}

function Split-Names([string]$list) {
    @($list.Split(",") | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })
}
$inspectNames = Split-Names $Inspect
foreach ($name in (Split-Names $Close)) {
    Get-Process -Name $name -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}
if ($CloseExplorer) {
    try {
        foreach ($ew in @((New-Object -ComObject Shell.Application).Windows())) { try { $ew.Quit() } catch { } }
    } catch { }
}
if ($Close -or $CloseExplorer) { Start-Sleep -Milliseconds 1500 }

$windows = @()
$top = $A::RootElement.FindAll([System.Windows.Automation.TreeScope]::Children, [System.Windows.Automation.Condition]::TrueCondition)
foreach ($w in $top) {
    try {
        $c = $w.Current
        if (-not $c.Name) { continue }
        # Minimized windows are listed too (so a reset's baseline includes them), but
        # marked offscreen and not read in depth; the checks only grade visible ones.
        $offscreen = [bool]$c.IsOffscreen
        $proc = Get-Process -Id $c.ProcessId -ErrorAction SilentlyContinue
        $started = $null
        try { $started = $proc.StartTime.ToUniversalTime().ToString("o") } catch { }
        $names = New-Object System.Collections.ArrayList
        $selected = New-Object System.Collections.ArrayList
        $texts = New-Object System.Collections.ArrayList
        $procName = $(if ($proc) { $proc.ProcessName } else { "" })
        if (-not $offscreen -and $inspectNames -contains $procName.ToLower()) {
            Get-Texts $w ([ref]$names) ([ref]$selected) ([ref]$texts)
        }
        $r = $c.BoundingRectangle
        $windows += [ordered]@{
            title = $c.Name; class = $c.ClassName; pid = $c.ProcessId; hwnd = $c.NativeWindowHandle
            process = $procName; started = $started; offscreen = $offscreen
            rect = @([int]$r.Left, [int]$r.Top, [int]$r.Width, [int]$r.Height)
            names = @($names); selected = @($selected); texts = @($texts)
        }
    } catch { }
}

$explorer = @()
try {
    foreach ($ew in (New-Object -ComObject Shell.Application).Windows()) {
        try { $explorer += [ordered]@{ path = $ew.Document.Folder.Self.Path; hwnd = [int64]$ew.HWND } } catch { }
    }
} catch { }

[ordered]@{
    windows = $windows
    explorer = $explorer
    documents = [Environment]::GetFolderPath("MyDocuments")
} | ConvertTo-Json -Depth 6 -Compress
