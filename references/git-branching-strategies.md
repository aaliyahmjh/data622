# Git Branching Strategies

Two approaches are outlined below for a small team (4 people) with mixed git experience.

---

## Option 1: Feature Branches (Recommended)

One branch per module or task, merged into `main` via pull request.

### Structure
```
main
├── feature/dataset  # Aali / Mehreen - Setting up the raw data and any basic cleaning / preprocessing
├── feature/features # Aali / Mehreen - Feature selection & engineering, including scaling
├── feature/train    # Mehreen Setting up models and saving trained models down
├── feature/predict  # Miraj Ingesting new data, preprocessing it, and saving results / metrics
└── feature/app      # Kevin - Shiny App for UI
```

### Example Workflow
```bash
# 1. Start from an up-to-date main
git checkout main
git pull

# 2. Create a branch for your task
git checkout -b feature/dataset

# 3. Do your work, commit regularly
git add src/data622/dataset.py
git commit -m "Add load_raw_data function"

# 4. Push and open a pull request on GitHub
git push -u origin feature/dataset
# Then open a PR on GitHub to merge into main

# 5. After PR is merged, clean up
git checkout main
git pull
git branch -d feature/dataset
```

### Pros
- Broken code stays off `main`
- PRs provide a natural review step
- Clean history — each branch has a clear purpose
- Industry-standard workflow

### Cons
- Slightly more overhead creating/switching branches
- Requires PRs to be reviewed and merged promptly

---

## Option 2: Person Branches

One long-lived branch per team member. Each person merges to `main` when their piece is done.

### Structure
```
main
├── dev/kevin
├── dev/aaliyah
├── dev/mehreen
└── dev/miraj
```

### Example Workflow
```bash
# 1. Create your personal branch once (do this once at the start)
git checkout main
git pull
git checkout -b dev/kevin

# 2. Do your work, commit regularly
git add src/data622/train.py
git commit -m "Add baseline linear regression model"

# 3. Push your branch
git push -u origin dev/kevin

# 4. When ready to merge, open a PR on GitHub from dev/kevin -> main

# 5. To stay up to date with teammates' work
git checkout main
git pull
git checkout dev/kevin
git merge main
```

### Pros
- Simpler mental model — one branch to manage
- Less overhead for less experienced git users

### Cons
- Long-lived branches drift from `main` and cause larger merge conflicts
- Harder to isolate individual features if something breaks

---

## Recommendation

**Use Feature Branches (Option 1).** The project files map cleanly to branches, so conflicts will be rare. For teammates less comfortable with git, the day-to-day workflow is straightforward and GitHub Desktop or VS Code's Source Control panel can handle most steps without the command line.
