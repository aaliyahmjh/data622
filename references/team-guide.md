# Team Guide

Quick reference for environment setup, daily workflows, and git branching.

---

## 1. Environment Setup (do this once)

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.

### Install uv
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone the repo and install
```bash
git clone <repo-url>
cd data622

# Creates a .venv and installs all dependencies from pyproject.toml
uv sync
```

That's it. You don't need to manually activate the virtual environment — `uv run` handles it.

---

## 2. Common uv Commands

| Task | Command |
|---|---|
| Install / sync all dependencies | `uv sync` |
| Add a new package | `uv add <package>` |
| Remove a package | `uv remove <package>` |
| Run a script | `uv run python scripts/my_script.py` |
| Run a notebook | `uv run jupyter notebook` |
| Launch the Shiny app | `uv run shiny run src/data622/app.py` |
| Open a Python shell | `uv run python` |

> After running `uv add` or `uv remove`, commit both `pyproject.toml` and `uv.lock` so teammates stay in sync.

---

## 3. Project Directory Quick Reference

```
data/raw/        → Source data — never edit directly
data/interim/    → In-progress cleaned data
data/processed/  → Final model-ready data
models/          → Saved trained models (.pkl)
notebooks/
  exploratory/   → EDA and experimentation
  presentation/  → Polished notebooks for sharing
references/      → Docs and guides (like this one)
scripts/         → End-to-end pipeline scripts
src/data622/     → Main Python package
```

Import paths using the centralized `paths.py`:
```python
from data622.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR

df = pd.read_csv(RAW_DATA_DIR / "payroll.csv")
```

---

## 4. Git Workflow (Feature Branches)

Each piece of work gets its own branch, merged into `main` via a pull request.

### Starting new work
```bash
# Always branch off an up-to-date main
git checkout main
git pull
git checkout -b feature/your-task-name
```

### While working
```bash
# Stage and commit your changes regularly
git add src/data622/dataset.py
git commit -m "Add load_raw_data function"

# Push to GitHub
git push -u origin feature/your-task-name
```

### Finishing up
1. Go to the repo on GitHub
2. Click **"Compare & pull request"**
3. Add a short description and request a review
4. After it's merged, clean up locally:

```bash
git checkout main
git pull
git branch -d feature/your-task-name
```

### Staying up to date with teammates
```bash
# Pull the latest main into your branch
git checkout main
git pull
git checkout feature/your-branch
git merge main
```

### Suggested branch names
| Work | Branch name |
|---|---|
| Data loading | `feature/dataset` |
| Feature engineering | `feature/features` |
| Model training | `feature/train` |
| Prediction logic | `feature/predict` |
| Shiny app | `feature/app` |
| EDA notebook | `feature/eda` |
