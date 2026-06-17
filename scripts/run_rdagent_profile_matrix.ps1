param(
    [string]$ProfileName = "",
    [int]$LoopN = 1,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-ProgressLine {
    param(
        [string]$Stage,
        [string]$Message,
        [string]$Color = "Gray"
    )
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] [$Stage] $Message" -ForegroundColor $Color
}

function Get-LatestLogDirName {
    param([string]$ProjectRoot)
    $logDir = Join-Path $ProjectRoot "log"
    if (-not (Test-Path $logDir)) {
        return $null
    }
    $latest = Get-ChildItem -Path $logDir -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latest) {
        return $null
    }
    return $latest.Name
}

function Get-RunHealthSummary {
    param(
        [string]$ProjectRoot,
        [string]$LogDirName
    )
    if ([string]::IsNullOrWhiteSpace($LogDirName)) {
        return "unknown"
    }
    $logPath = Join-Path (Join-Path $ProjectRoot "log") $LogDirName
    if (-not (Test-Path $logPath)) {
        return "unknown"
    }
    $files = Get-ChildItem -Path $logPath -Recurse -File -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    if (-not $files) {
        return "unknown"
    }
    $matches = Select-String -Path $files -Pattern "All tasks are failed" -SimpleMatch -ErrorAction SilentlyContinue
    if ($matches) {
        return "all_tasks_failed"
    }
    return "no_global_failure_marker"
}

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Save-RunArtifacts {
    param(
        [string]$ProjectRoot,
        [string]$ProfileName,
        [string]$LogDirName,
        [string]$Health,
        [double]$ElapsedSeconds
    )
    if ([string]::IsNullOrWhiteSpace($LogDirName)) {
        return $null
    }
    $src = Join-Path (Join-Path $ProjectRoot "log") $LogDirName
    if (-not (Test-Path $src)) {
        return $null
    }

    $artifactRoot = Join-Path $ProjectRoot "run_artifacts"
    Ensure-Dir -Path $artifactRoot
    $dstName = "${LogDirName}__${ProfileName}"
    $dst = Join-Path $artifactRoot $dstName
    if (Test-Path $dst) {
        Remove-Item -Recurse -Force $dst
    }
    Copy-Item -Recurse -Force $src $dst

    $summary = @(
        "profile=$ProfileName",
        "log_dir=$LogDirName",
        "health=$Health",
        ("elapsed_seconds={0}" -f $ElapsedSeconds),
        ("saved_at={0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    )
    $summaryPath = Join-Path $dst "RUN_SUMMARY.txt"
    [System.IO.File]::WriteAllLines($summaryPath, $summary)

    $latestPtr = Join-Path $artifactRoot ("LATEST__{0}.txt" -f $ProfileName)
    [System.IO.File]::WriteAllText($latestPtr, $dst)
    return $dst
}

$root = Split-Path -Parent $PSScriptRoot
$matrixPath = Join-Path $root "config/rdagent_profile_matrix.json"
if (-not (Test-Path $matrixPath)) {
    throw "Profile matrix not found: $matrixPath"
}

$matrix = Get-Content $matrixPath -Raw | ConvertFrom-Json
$profiles = $matrix.profiles
if ($ProfileName) {
    $profiles = $profiles | Where-Object { $_.name -eq $ProfileName }
    if (-not $profiles) {
        throw "Profile '$ProfileName' not found in $matrixPath"
    }
}

$deepseekKey = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "User")
if ([string]::IsNullOrWhiteSpace($deepseekKey)) {
    throw "DEEPSEEK_API_KEY (User scope) is empty. Please set it first."
}

$total = @($profiles).Count
$idx = 0
$totalStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
Write-ProgressLine -Stage "START" -Message "Selected profiles: $total, LoopN=$LoopN, DryRun=$DryRun" -Color "Cyan"

foreach ($p in $profiles) {
    $idx++
    $profileStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $pct = [math]::Round((($idx - 1) / [math]::Max($total, 1)) * 100, 1)
    Write-ProgressLine -Stage "PROFILE" -Message "[$idx/$total][$pct%] $($p.name) - begin" -Color "Cyan"
    Write-ProgressLine -Stage "PROFILE" -Message "Description: $($p.description)"
    Write-ProgressLine -Stage "PROFILE" -Message "Feature sets: $($p.feature_sets -join ', ')"

    $envLines = @(
        "CHAT_MODEL=deepseek/deepseek-chat",
        "DEEPSEEK_API_KEY=$deepseekKey",
        "EMBEDDING_MODEL=openai/text-embedding-3-small",
        "OPENAI_API_KEY=",
        "QLIB_RDAGENT_CONDA_ENV=rdagent",
        "MODEL_CoSTEER_ENV_TYPE=conda",
        "LIVE_OUTPUT=False",
        "QLIB_QUANT_QUANT_HYPOTHESIS_GEN=rdagent_integration.project_quant_proposal.ProjectQlibQuantHypothesisGen",
        "QLIB_QUANT_FACTOR_HYPOTHESIS2EXPERIMENT=rdagent_integration.project_proposal.ProjectQlibFactorHypothesis2Experiment",
        "QLIB_QUANT_MODEL_HYPOTHESIS2EXPERIMENT=rdagent_integration.project_proposal.ProjectQlibModelHypothesis2Experiment",
        "RDAGENT_PROFILE_NAME=$($p.name)",
        "RDAGENT_FEATURE_SET_HINT=$($p.feature_sets -join ',')"
    )

    $p.env.PSObject.Properties | ForEach-Object {
        $envLines += "$($_.Name)=$($_.Value)"
    }

    Write-ProgressLine -Stage "ENV" -Message "Writing .env.wsl/.env for profile '$($p.name)'"
    $envWslPath = Join-Path $root ".env.wsl"
    $envPath = Join-Path $root ".env"
    [System.IO.File]::WriteAllLines($envWslPath, $envLines)
    Copy-Item -Force $envWslPath $envPath

    # rdagent env (micromamba) is symlinked into miniconda3/envs/rdagent so conda can find it.
    # Contains: rdagent 0.8.0 + pyqlib 0.9.7 + qrun
    $cmd = "/home/administrator/miniconda3/bin/conda run -n rdagent python scripts/run_fin_quant.py --loop_n=$LoopN"
    $wslCmd = "cd /mnt/d/quant_project/Qlib_Quant/qlib-VNPY_Quant_Project; $cmd"

    if ($DryRun) {
        Write-ProgressLine -Stage "DRY-RUN" -Message "wsl -d Ubuntu bash -lc `"$wslCmd`"" -Color "Yellow"
        $profileStopwatch.Stop()
        Write-ProgressLine -Stage "PROFILE" -Message "[$idx/$total] $($p.name) dry-run done in $([math]::Round($profileStopwatch.Elapsed.TotalSeconds,1))s" -Color "Yellow"
        continue
    }

    $beforeLatest = Get-LatestLogDirName -ProjectRoot $root
    Write-ProgressLine -Stage "RUN" -Message "Launching fin_quant for '$($p.name)' (latest log before run: $beforeLatest)"
    wsl -d Ubuntu bash -lc $wslCmd
    if ($LASTEXITCODE -ne 0) {
        $profileStopwatch.Stop()
        Write-ProgressLine -Stage "FAILED" -Message "[$idx/$total] $($p.name) failed after $([math]::Round($profileStopwatch.Elapsed.TotalSeconds,1))s, exit=$LASTEXITCODE" -Color "Red"
        throw "Profile '$($p.name)' failed with exit code $LASTEXITCODE"
    }
    $profileStopwatch.Stop()
    $afterLatest = Get-LatestLogDirName -ProjectRoot $root
    $nowPct = [math]::Round(($idx / [math]::Max($total, 1)) * 100, 1)
    Write-ProgressLine -Stage "DONE" -Message "[$idx/$total][$nowPct%] $($p.name) completed in $([math]::Round($profileStopwatch.Elapsed.TotalSeconds,1))s" -Color "Green"
    if ($afterLatest) {
        Write-ProgressLine -Stage "LOG" -Message "Latest run log: log/$afterLatest" -Color "Green"
        $health = Get-RunHealthSummary -ProjectRoot $root -LogDirName $afterLatest
        switch ($health) {
            "all_tasks_failed" {
                Write-ProgressLine -Stage "SUMMARY" -Message "Run health: FAILED (found 'All tasks are failed')." -Color "Yellow"
            }
            "no_global_failure_marker" {
                Write-ProgressLine -Stage "SUMMARY" -Message "Run health: PASS (no global failure marker)." -Color "Green"
            }
            default {
                Write-ProgressLine -Stage "SUMMARY" -Message "Run health: UNKNOWN (could not inspect log markers)." -Color "Gray"
            }
        }

        $saved = Save-RunArtifacts -ProjectRoot $root -ProfileName $p.name -LogDirName $afterLatest -Health $health -ElapsedSeconds $profileStopwatch.Elapsed.TotalSeconds
        if ($saved) {
            Write-ProgressLine -Stage "ARTIFACT" -Message "Saved run artifacts to: $saved" -Color "Green"
        } else {
            Write-ProgressLine -Stage "ARTIFACT" -Message "Skip saving artifacts (log dir missing)." -Color "Gray"
        }
    }
}

$totalStopwatch.Stop()
Write-ProgressLine -Stage "FINISH" -Message "All selected profiles completed in $([math]::Round($totalStopwatch.Elapsed.TotalMinutes,2)) minutes." -Color "Green"
