import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

repo_root = Path(__file__).resolve().parents[1]
preferred = repo_root / 'data' / 'processed' / 'nyc_annual_salary_employees_payBasis_perAnuum.csv'

out_dir = repo_root / 'notebooks' / 'exploratory' / 'eda_outputs'
plots_dir = out_dir / 'plots'
os.makedirs(plots_dir, exist_ok=True)

out_file = out_dir / 'summary.txt'
with open(out_file, 'w', encoding='utf-8') as outf:
    if not preferred.exists():
        raise FileNotFoundError(f"Preferred file not found: {preferred}")
    dfpath = str(preferred)
    outf.write(f"Loading: {dfpath}\n")
    print("Loading", dfpath)
    # Read CSV -- may take time/memory
    df = pd.read_csv(dfpath, low_memory=False)
    outf.write(f"Shape: {df.shape}\n\n")

    outf.write("-- Dtypes --\n")
    dtypes = df.dtypes
    for col, dt in dtypes.items():
        outf.write(f"{col}: {dt}\n")

    outf.write("\n-- Top missing values (20) --\n")
    miss = df.isna().sum().sort_values(ascending=False).head(20)
    outf.write(miss.to_string())
    outf.write("\n\n-- Numeric summary (first 20 cols) --\n")
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        descr = num.describe().T.head(20)
        outf.write(descr.to_string())
    else:
        outf.write('No numeric columns found\n')

    outf.write("\n\n-- Top categories for up to 6 object columns --\n")
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in obj_cols[:6]:
        outf.write(f"\n[{col}] top values:\n")
        vc = df[col].value_counts(dropna=False).head(10)
        outf.write(vc.to_string())
        outf.write("\n")

    # Save head sample
    sample_csv = out_dir / 'head_sample.csv'
    df.head(200).to_csv(sample_csv, index=False)
    outf.write(f"\nSaved head sample to: {sample_csv}\n")

    # Save histograms for first up to 8 numeric columns
    if not num.empty:
        num_cols = num.columns.tolist()[:8]
        for c in num_cols:
            try:
                plt.figure(figsize=(6,4))
                sns.histplot(df[c].dropna(), kde=False)
                plt.title(c)
                fname = plots_dir / f"hist_{c}.png"
                plt.tight_layout()
                plt.savefig(fname)
                plt.close()
                outf.write(f"Saved plot: {fname}\n")
            except Exception as e:
                outf.write(f"Failed plotting {c}: {e}\n")

print('EDA complete. Outputs in', out_dir)
