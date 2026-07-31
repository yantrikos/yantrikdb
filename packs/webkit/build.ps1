# Compile every brief and render every page, FAILING LOUDLY.
#
# Written after a compile crashed for all six briefs while the pipeline
# printed six PASS lines: the renders were stale HTML from the previous
# build, because the loop piped compiler output to Out-Null and never
# checked an exit code. A build step that cannot fail is a build step
# that lies.
#
# ASCII only in this file. A non-ASCII dash in a string broke the
# PowerShell parser outright on this machine's encoding.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path (Split-Path -Parent $root) ".venv\Scripts\python.exe"
$webkit = Join-Path $root "webkit"

Remove-Item (Join-Path $webkit "out\*.html") -ErrorAction SilentlyContinue

$failed = 0
Get-ChildItem (Join-Path $webkit "briefs\*.ops") | ForEach-Object {
    $out = Join-Path $webkit ("out\" + $_.BaseName + ".html")
    & $py (Join-Path $webkit "compiler.py") $_.FullName --out $out --strict
    if ($LASTEXITCODE -ne 0) {
        Write-Host "COMPILE FAILED: $($_.Name)" -ForegroundColor Red
        $failed++
    }
}
if ($failed -gt 0) { throw "$failed brief(s) failed to compile; refusing to render stale output" }

$pages = Get-ChildItem (Join-Path $webkit "out\*.html") | ForEach-Object { $_.FullName }
if (-not $pages) { throw "no pages compiled" }
& C:\Python313\python.exe (Join-Path $webkit "shoot.py") $pages
& $py (Join-Path $webkit "gallery.py")
