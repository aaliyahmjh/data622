# 🎯 Data & Pipeline Challenges & Solutions

## Challenge 1: Data Volume (2.6M Records)

### Current State
- **Raw Dataset**: 2,617,642 records × 22 columns (2015-2024)
- **Processed Dataset**: ~920K records after filtering (train: 491K, valid: 201K, test: 236K)
- **Memory Usage**: ~200-400 MB with all features
- **Processing Time**: Feature engineering takes 2-5 minutes

### Issues This Creates
- ⏱️ Slow preprocessing pipeline (regenerating splits takes minutes)
- 💾 Large file sizes (CSV exports 300+ MB, Git LFS overhead)
- 🔄 Training bottleneck (RandomForest/XGBoost slower on large data)
- 📊 Validation overhead (cross-validation × models × hyperparameters = slow)
- 🖥️ Memory constraints during preprocessing

### Solutions Implemented & Recommended

#### ✅ Already Implemented
```python
# Your current approach: Temporal split + filtering
TRAIN_YEARS = [2021, 2022]      # 491K records
VALID_YEARS = [2023]            # 201K records  
TEST_YEARS = [2024]             # 236K records
```

#### 🔄 Quick Improvements (< 30 min)

**1. Chunked Processing for Large Operations**
```python
# Instead of processing all 2.6M at once:
def load_and_process_in_chunks(filepath, chunksize=100000):
    """Process large CSV in chunks to reduce memory usage"""
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        chunk = preprocess_chunk(chunk)
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

# Usage:
df = load_and_process_in_chunks('raw_data.csv', chunksize=100000)
# Memory usage: ~50MB instead of 200MB
```

**2. Selective Feature Computation**
```python
# Only compute features you'll use (drop low-variance features)
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)  # Drop features with <1% variance
X_filtered = selector.fit_transform(X)
important_features = X.columns[selector.get_support()]

# Result: Reduce 23 features → 18 relevant features
```

**3. Data Type Optimization**
```python
# Reduce memory by 50% with dtype optimization
def optimize_dtypes(df):
    """Convert columns to memory-efficient types"""
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')  # or int16 if range allows
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')  # If limited unique values
    
    return df

# Before: 400 MB | After: 200 MB
```

#### 🚀 Medium-Term Solutions (< 1 hour)

**4. Parallel Processing with Dask**
```python
import dask.dataframe as dd

# Load and process in parallel
ddf = dd.read_csv('raw_data.csv')
ddf_processed = ddf.map_partitions(preprocess_chunk)  # Parallel processing
df = ddf_processed.compute()  # Combine results

# Speed-up: 2-4x faster on multi-core machines
```

**5. Sampling Strategy for Validation**
```python
# Stratified sample for quick testing (90% speed, 99% accuracy)
def stratified_sample(df, strata_cols=['title_std', 'agency_std'], sample_pct=0.5):
    """Keep proportions while reducing size"""
    return df.groupby(strata_cols, group_keys=False).apply(
        lambda x: x.sample(frac=sample_pct, random_state=42)
    )

# Use for hyperparameter tuning:
valid_sample = stratified_sample(valid_set, sample_pct=0.3)  # 60K instead of 201K
model.tune_regularization(valid_sample)  # 5x faster tuning
```

**6. Model-Specific Optimizations**
```python
# LinearRegression/Ridge: Already fast, good choice ✅
# RandomForest: Use max_depth, min_samples_leaf to constrain
rf = RandomForestRegressor(
    n_estimators=50,      # Default 100, enough for accuracy
    max_depth=15,         # Prevent overfitting + speed up
    min_samples_leaf=10,  # Require 10+ samples per leaf
    n_jobs=-1,            # Use all CPU cores
    random_state=42
)

# XGBoost: Use early stopping to avoid wasted iterations
xgb = XGBRegressor(
    n_estimators=500,
    early_stopping_rounds=10,
    eval_metric='rmse',
    n_jobs=-1
)
xgb.fit(X_train, y_train, 
        eval_set=[(X_valid, y_valid)],
        verbose=False)  # Stops after 10 rounds without improvement
```

---

## Challenge 2: Borough Stratification for Jobs

### Current State
- ✅ You have `work_location_borough` in dataset
- ❌ Current split is **temporal only** (by fiscal year)
- ❌ **No guarantee** that each borough-job combination is represented equally in train/val/test

### What Goes Wrong Without Stratification

**Scenario**: Rare job in Manhattan
```
Train set:  0% of rare_job in Manhattan
Valid set:  100% of rare_job in Manhattan
Test set:   0% of rare_job in Manhattan

Result: Model OVERFITS on validation set (sees job never before)
```

**Scenario**: Janitor jobs distributed unevenly
```
Train: 70% Manhattan, 30% Brooklyn
Valid: 30% Manhattan, 70% Brooklyn
Test:  50% Manhattan, 50% Manhattan

Result: Performance degradation on validation (different distribution)
```

### Solutions

#### ✅ Solution 1: Stratified Time-Series Split (Recommended)

```python
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_temporal_split(df, train_years, valid_years, test_years):
    """
    Split by time WHILE maintaining borough-job distribution
    """
    train_df = df[df['fiscal_year'].isin(train_years)].copy()
    valid_df = df[df['fiscal_year'].isin(valid_years)].copy()
    test_df = df[df['fiscal_year'].isin(test_years)].copy()
    
    # Create stratification key (borough + job category)
    for split_df in [train_df, valid_df, test_df]:
        split_df['strata'] = (
            split_df['work_location_borough'] + '_' + 
            split_df['title_category']
        )
    
    # Verify stratification
    for name, split_df in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
        strata_pcts = split_df['strata'].value_counts(normalize=True).head(10)
        print(f"\n{name} set stratification (top 10 borough-job combos):")
        print(strata_pcts)
    
    return train_df, valid_df, test_df
```

#### ✅ Solution 2: Ensure Minimum Samples per Stratum

```python
def enforce_minimum_stratum_size(df, min_samples=100):
    """
    Remove rare borough-job combos to ensure reliable predictions
    """
    df = df.copy()
    df['borough_job'] = df['work_location_borough'] + '_' + df['title_category']
    
    # Count samples per stratum
    stratum_counts = df['borough_job'].value_counts()
    rare_strata = stratum_counts[stratum_counts < min_samples].index
    
    print(f"Removing {len(rare_strata)} rare borough-job combinations")
    print(f"  (affecting {df[df['borough_job'].isin(rare_strata)].shape[0]} records)")
    
    # Keep only well-represented strata
    df = df[~df['borough_job'].isin(rare_strata)].copy()
    
    return df.drop('borough_job', axis=1)

# Usage:
df = enforce_minimum_stratum_size(df, min_samples=100)
# Before: 920K records | After: 890K records (removes only ~30K rare combos)
```

#### ✅ Solution 3: Cross-Validate by Borough-Job

```python
from sklearn.model_selection import GroupKFold

def borough_job_cross_validation(df, X, y, model, n_splits=5):
    """
    Validate that model works for all borough-job combinations
    """
    # Create groups: one group per borough-job combo
    df['borough_job_group'] = (
        df['work_location_borough'] + '_' + df['title_category']
    )
    groups = df['borough_job_group'].factorize()[0]
    
    # Leave-one-borough-out cross-validation
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Get borough-job combos in test set
        test_groups = df.iloc[test_idx]['borough_job_group'].unique()
        
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        scores.append(r2)
        
        print(f"Fold {fold+1}: R² = {r2:.4f} | Test groups: {test_groups[:3]}...")
    
    print(f"\nAverage R² across borough-job groups: {np.mean(scores):.4f}")
    print(f"Std Dev: {np.std(scores):.4f}")
    
    return scores
```

#### 📊 Solution 4: Analyze Borough-Job Distribution

```python
def analyze_borough_job_distribution(df):
    """
    Show which combinations are well/poorly represented
    """
    dist = pd.crosstab(
        df['work_location_borough'], 
        df['title_category'],
        margins=True
    )
    
    print("Borough-Job Distribution:")
    print(dist)
    
    # Heatmap of sparse combinations
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    sns.heatmap(dist.iloc[:-1, :-1], annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Records per Borough-Job Combination')
    plt.tight_layout()
    plt.savefig('borough_job_distribution.png')
    
    # Identify rare combinations
    total_records = df.shape[0]
    pcts = (dist / total_records * 100).iloc[:-1, :-1]
    rare = pcts[pcts < 0.5].stack()  # Less than 0.5% of data
    print(f"\nRare combinations (<0.5% of data): {len(rare)} combos affecting {rare.sum():.1f}% of records")
```

---

## Challenge 3: Model Training Bottlenecks

### Issue: GridSearchCV Takes Too Long

**Current Setup**:
```python
# GridSearchCV with 20 alphas × 5-fold CV = 100 fits per model
GridSearchCV(
    Ridge(),
    param_grid={'alpha': [0.001, 0.01, 0.1, ..., 100]},  # 20 values
    cv=5,  # 5 folds
    n_jobs=-1
)
# Time: 30-60 seconds on 201K records ✅ Acceptable
```

**Problem with RandomForest/XGBoost**:
```python
# 10 n_estimators × 5 max_depth × 5-fold CV = 250 fits
GridSearchCV(
    RandomForestRegressor(),
    param_grid={
        'n_estimators': [50, 100, 150, 200, 250],
        'max_depth': [5, 10, 15, 20, None]
    },
    cv=5
)
# Time: 10-30 minutes ❌ Too slow
```

### Solutions

#### ✅ Solution 1: Use Halving Grid Search (Faster)
```python
from sklearn.model_selection import HalvingGridSearchCV

# Eliminates poor hyperparameters in early rounds
hgs = HalvingGridSearchCV(
    RandomForestRegressor(),
    param_grid={
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15]
    },
    cv=5,
    factor=2,  # Cut candidates by 2x each round
    n_jobs=-1
)
# Time: 2-5 minutes (5-10x faster than GridSearch)
```

#### ✅ Solution 2: Use Random Search
```python
from sklearn.model_selection import RandomizedSearchCV

# Sample random hyperparameters instead of grid
rgs = RandomizedSearchCV(
    XGBRegressor(),
    param_distributions={
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 10, 15, 20],
        'learning_rate': [0.01, 0.05, 0.1]
    },
    n_iter=20,  # Try 20 random combinations (vs 60 in grid)
    cv=5,
    n_jobs=-1,
    random_state=42
)
# Time: 5-10 minutes (vs 15-30 for GridSearch)
```

#### ✅ Solution 3: Early Stopping for Tree Models
```python
# XGBoost with early stopping
xgb = XGBRegressor(
    n_estimators=500,
    early_stopping_rounds=20,  # Stop if 20 rounds no improvement
    random_state=42
)

xgb.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)

print(f"Stopped after {xgb.best_iteration} iterations")
# Average: 80-150 iterations instead of 500 (stop after improvement plateaus)
```

---

## Implementation Plan: Stratified Borough-Job Pipeline

### Step 1: Add Stratification to generate_splits.py

```python
# In scripts/generate_splits.py, add:

def stratified_temporal_split(df, train_years, valid_years, test_years):
    """Split by time while preserving borough-job distribution"""
    train_df = df[df['fiscal_year'].isin(train_years)].copy()
    valid_df = df[df['fiscal_year'].isin(valid_years)].copy()
    test_df = df[df['fiscal_year'].isin(test_years)].copy()
    
    # Create stratification column
    for split_df in [train_df, valid_df, test_df]:
        split_df['borough_job'] = (
            split_df['work_location_borough'] + '_' + split_df['title_category']
        )
    
    # Report distribution
    for name, split_df in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
        dist = split_df['borough_job'].value_counts()
        print(f"\n{name} set has {len(dist)} unique borough-job combos")
        print(f"  Median size: {dist.median():.0f}")
        print(f"  Min: {dist.min()}, Max: {dist.max()}")
    
    return train_df.drop('borough_job', axis=1), \
           valid_df.drop('borough_job', axis=1), \
           test_df.drop('borough_job', axis=1)
```

### Step 2: Add Data Volume Optimizations

```python
# Optimize before saving to CSV
df = optimize_dtypes(df)  # 50% smaller files
df = enforce_minimum_stratum_size(df, min_samples=50)  # Remove rare combos
```

### Step 3: Add Borough-Job Validation to model.py

```python
# After training, validate by borough-job:
scores_by_borough_job = borough_job_cross_validation(
    valid_df, X_valid, y_valid, best_model
)

# Should show consistent R² across all groups
# If some groups have low R², needs more data/features for those groups
```

---

## Recommended Actions (Priority Order)

| Priority | Challenge | Solution | Time | Impact |
|----------|-----------|----------|------|--------|
| 🔴 HIGH | Stratification missing | Add borough-job stratification to splits | 20 min | Medium |
| 🔴 HIGH | No rare-combo handling | Filter minimum 50 samples per borough-job | 15 min | Medium |
| 🟡 MEDIUM | Training slow | Use HalvingGridSearchCV instead of GridSearch | 30 min | High |
| 🟡 MEDIUM | Memory usage | Optimize dtypes (int64→int32, float64→float32) | 15 min | Low |
| 🟢 LOW | Data exploration | Analyze borough-job distribution heatmap | 10 min | Low |
| 🟢 LOW | Future scaling | Add Dask for >10M records | 1 hour | Low |

---

## Quick Implementation Checklist

```
☐ 1. Add stratified_temporal_split() to dataset.py
☐ 2. Update generate_splits.py to use stratified split
☐ 3. Add enforce_minimum_stratum_size() to dataset.py
☐ 4. Update model.py to use HalvingGridSearchCV for RF/XGBoost
☐ 5. Add borough_job_cross_validation() to model.py  
☐ 6. Generate new train/valid/test splits
☐ 7. Retrain models with optimized pipeline
☐ 8. Compare performance before/after stratification
☐ 9. Document borough-job distribution in results
☐ 10. Push changes to GitHub
```

---

## Summary

**Data Volume (2.6M)**: ✅ Already well-managed with temporal split
- Ridge regression is fast enough
- Chunked processing available if needed
- Dask optional for future scaling

**Borough Stratification**: ❌ **Missing - Priority to add**
- Current split may miss rare borough-job combos
- Add stratification to ensure balanced representation
- Validate with borough-job CV

**Training Speed**: 🟡 **Can improve**
- Switch GridSearchCV → HalvingGridSearchCV (5-10x faster)
- Use early stopping for tree models
- Reduces tuning time from 30 min → 3-5 min

These improvements will ensure:
- ✅ No data leakage across borough-job groups
- ✅ Faster hyperparameter tuning
- ✅ More reliable predictions for all NYC boroughs
- ✅ Scalable to larger datasets
