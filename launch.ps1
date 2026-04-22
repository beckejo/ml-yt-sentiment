# Quick launcher script for ML sentiment analysis project
# Usage: .\launch.ps1 [-SkipTraining] [-SkipApi] [-SkipStreamlit] [-SkipMlflow]

param(
    [switch]$SkipTraining,
    [switch]$SkipApi,
    [switch]$SkipStreamlit,
    [switch]$SkipMlflow
)

# Convert switches to arguments
$args = @()
if ($SkipTraining) { $args += "--skip-training" }
if ($SkipApi) { $args += "--skip-api" }
if ($SkipStreamlit) { $args += "--skip-streamlit" }
if ($SkipMlflow) { $args += "--skip-mlflow" }

# Run launcher
python launcher.py @args
