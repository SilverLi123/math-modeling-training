param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d{4}-[A-Z0-9]+-[A-Z0-9]+$')]
    [string]$Id
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$template = Join-Path $repoRoot 'problems\TEMPLATE'
$target = Join-Path $repoRoot (Join-Path 'problems' $Id)

if (-not (Test-Path -LiteralPath $template -PathType Container)) {
    throw "Problem template not found: $template"
}

if (Test-Path -LiteralPath $target) {
    throw "Problem directory already exists: $target"
}

Copy-Item -LiteralPath $template -Destination $target -Recurse

Get-ChildItem -LiteralPath $target -Recurse -Filter '*.md' -File | ForEach-Object {
    $content = Get-Content -LiteralPath $_.FullName -Raw
    $content = $content.Replace('{{PROBLEM_ID}}', $Id)
    Set-Content -LiteralPath $_.FullName -Value $content -Encoding utf8NoBOM
}

Write-Host "Created problem workspace: $target"
Write-Host "Next: add a row to problems/index.md and complete the problem README."
