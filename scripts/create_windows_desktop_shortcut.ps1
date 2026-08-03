# Create XRDmatch shortcut on the Windows Desktop
# Usage: powershell -ExecutionPolicy Bypass -File scripts\create_windows_desktop_shortcut.ps1 -ProjectRoot "C:\path\to\XRDmatch" -EnvName xrdmatch

param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$EnvName = "xrdmatch"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $ProjectRoot.TrimEnd('\', '/')

function Find-Conda {
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        return (Get-Command conda).Source
    }
    $candidates = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

$conda = Find-Conda
if (-not $conda) {
    Write-Error "conda not found. Run install.bat first."
}

$condaBase = (& $conda info --base).Trim()
$pythonw = Join-Path $condaBase "envs\$EnvName\pythonw.exe"
$python = Join-Path $condaBase "envs\$EnvName\python.exe"

if (-not (Test-Path $pythonw)) {
    if (Test-Path $python) {
        $pythonw = $python
    } else {
        Write-Error "Environment '$EnvName' not found. Run install.bat first."
    }
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "XRDmatch.lnk"
$launcherBat = Join-Path $ProjectRoot "launch_xrdmatch.bat"

$launchContent = @"
@echo off
cd /d "$ProjectRoot"
"$pythonw" "$ProjectRoot\main.py"
if errorlevel 1 (
  echo XRDmatch exited with an error.
  pause
)
"@
Set-Content -Path $launcherBat -Value $launchContent -Encoding ASCII

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = $pythonw
$Shortcut.Arguments = "`"$ProjectRoot\main.py`""
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.WindowStyle = 1
$Shortcut.Description = "XRD Phase Matcher"

$ico = Join-Path $ProjectRoot "assets\xrdmatch_icon.ico"
if (Test-Path $ico) {
    $Shortcut.IconLocation = "$ico,0"
} else {
    $Shortcut.IconLocation = "$pythonw,0"
}

$Shortcut.Save()

Write-Host "Created: $shortcutPath"
Write-Host "Created: $launcherBat"
Write-Host "Double-click XRDmatch on your Desktop to launch."
