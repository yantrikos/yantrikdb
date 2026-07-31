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
$kit = Join-Path $root "letterpress"

Remove-Item (Join-Path $kit "out\*.html") -ErrorAction SilentlyContinue

# BOTH ops directories. The first version rebuilt only briefs\ and left
# generated\ pages on disk from an earlier compiler, so a run reported
# six PASS lines for hand-written pages and six stale FAILs for the
# model's - and the stale ones were the numbers that mattered. Same
# stale-output failure this script exists to prevent, one directory over.
$failed = 0
Get-ChildItem (Join-Path $kit "briefs\*.ops"), (Join-Path $kit "generated\*.ops") | ForEach-Object {
    $out = Join-Path $kit ("out\" + $_.BaseName + ".html")
    & $py (Join-Path $kit "compiler.py") $_.FullName --out $out --strict
    if ($LASTEXITCODE -ne 0) {
        Write-Host "COMPILE FAILED: $($_.Name)" -ForegroundColor Red
        $failed++
    }
}
if ($failed -gt 0) { throw "$failed brief(s) failed to compile; refusing to render stale output" }

& $py (Join-Path $kit "test_palette.py")
if ($LASTEXITCODE -ne 0) { throw "palette invariant violated; refusing to render" }

$pages = Get-ChildItem (Join-Path $kit "out\*.html") | ForEach-Object { $_.FullName }
if (-not $pages) { throw "no pages compiled" }
& C:\Python313\python.exe (Join-Path $kit "shoot.py") $pages
& $py (Join-Path $kit "gallery.py")
