# Final Project Proposal

## Target Users & Features

**Target Users:** Repo owners and contributors who need to organize and direct their efforts according to issue priorities.

**Core Need:** Users must understand incoming issues and assess the relevant priority and impact they will have.

## Interface

Shiny

## Data Sources

- **Primary:** ITSM Dataset
- **Stretch:** GitHub Issues API (live issues for a more realistic dataset, pending availability)

## Machine Learning Problems

### 1st Priority

- **Classification** — Priority / Issue Type (Bug, Improvement, New Feature)
- **Regression** — Time-to-Resolution

### 2nd Priority

- **Clustering** — Incidence patterns

## Evaluation Strategy

| Dataset | Approach |
|---|---|
| ITSM | Evaluate against established priority and TTR using a train/test split |
| GitHub Issues | Train on old issues, test on new |

## Model Training Strategy

Ideally, new GitHub issues will serve as a mechanism to monitor model drift and fine-tune on more recent data.

## Computational Requirements

No additional compute needed — no large neural networks, LLMs, or GPU/TPU resources required.

## Minimally Viable Product (Midterm Check)

1. EDA
2. Data Sourcing
3. Data Preprocessing
4. Basic Feature Engineering
5. One model (classification or regression) trained but not optimized
