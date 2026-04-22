# Quick Start Guide

## Unified Launcher

Launch everything with a single command:

### PowerShell (Windows)
```powershell
.\launch.ps1
```

### Python (Cross-platform)
```bash
python launcher.py
```

## Usage Options

**Start everything (default):**
```powershell
.\launch.ps1
```

**Skip training (use existing champion):**
```powershell
.\launch.ps1 -SkipTraining
```

**Start only specific services:**
```powershell
# API + Streamlit only (no training, no MLflow)
.\launch.ps1 -SkipTraining -SkipMlflow

# API only
.\launch.ps1 -SkipTraining -SkipStreamlit -SkipMlflow
```

## What It Does

The launcher will:

1. **Train** (if not skipped)
   - Uses pre-downloaded data: `reddit_local.csv`, `YoutubeCommentsDataSet.csv`
   - Requires `HAND_LABELED_TEST_PATH` env var (hand-labeled validation set)
   - Registers champion model in MLflow with quality gates

2. **Start FastAPI** (if not skipped)
   - Listens on `http://127.0.0.1:8001`
   - Docs available at `/docs`
   - Loads champion model at startup with quality validation

3. **Start Streamlit** (if not skipped)
   - Listens on `http://127.0.0.1:8501`
   - Interactive UI for predictions

4. **Start MLflow UI** (if not skipped)
   - Listens on `http://127.0.0.1:5000`
   - View all training runs and metrics

## Setup (First Time Only)

**Set hand-labeled test data path** (required for training):
```powershell
$env:HAND_LABELED_TEST_PATH = "path/to/hand_labeled_test.csv"
```

**Install dependencies** (if needed):
```powershell
pip install -r requirements.txt
```

## Troubleshooting

**"HAND_LABELED_TEST_PATH not set"**
- You need to provide hand-labeled validation data
- Set: `$env:HAND_LABELED_TEST_PATH = "path/to/file.csv"`
- Expected format: CSV with `comment` and `sentiment` columns

**"Training failed"**
- Check data paths are correct
- Verify `reddit_local.csv` and `YoutubeCommentsDataSet.csv` exist
- Check `HAND_LABELED_TEST_PATH` points to valid file with all 3 sentiment classes

**Port already in use (like "Address already in use port 8001")**
- Kill existing processes: `Get-Process python | Stop-Process -Force`
- Or use custom ports by editing launcher.py

## Example Workflows

**Fresh training + full stack:**
```powershell
$env:HAND_LABELED_TEST_PATH = "data/my_labeled_data.csv"
.\launch.ps1
```

**Iterate on model (skip training):**
```powershell
.\launch.ps1 -SkipTraining
```

**Just test API without UI:**
```powershell
.\launch.ps1 -SkipStreamlit -SkipMlflow
```

**Development: Train + API only (no UI overhead):**
```powershell
.\launch.ps1 -SkipStreamlit -SkipMlflow
```
