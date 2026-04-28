# 📈 Linear Regression Performance Improvement Guide

## Current Performance Baseline
- **Test R²**: 0.7453 (explains 74.53% of salary variance)
- **RMSE**: 0.2050 on log scale (~18% average error)
- **MAE**: 0.1654 on log scale
- **CV R²**: 0.6657 ± 0.0689 (minimal overfitting)

---

## 🎯 Top 5 Improvement Strategies

### 1️⃣ REGULARIZATION (Ridge/Lasso Regression) ⭐ QUICK WIN
**Effort**: ⭐ LOW (30 minutes)  
**Expected Impact**: +0-2% R² improvement  
**Why**: Penalizes large coefficients to reduce overfitting

**Implementation**:
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Try different alpha values
alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    r2 = ridge.score(X_test, y_test)
    print(f"Alpha {alpha}: R² = {r2:.4f}")

# Use GridSearchCV to find optimal alpha
from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': np.logspace(-3, 3, 20)}
ridge_cv = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
ridge_cv.fit(X_train, y_train)
best_alpha = ridge_cv.best_params_['alpha']
```

**When to Use**: 
- If coefficients are unstable
- To handle multicollinearity
- For more stable predictions

---

### 2️⃣ FEATURE ENGINEERING - Year-Relative Features (Option 3) ⭐ RECOMMENDED
**Effort**: ⭐⭐ MEDIUM (1-2 hours)  
**Expected Impact**: +2-4% R² improvement  
**Why**: Better handles year-over-year salary growth pattern

**Problem We're Solving**:
- Salary increases by ~3-5% each year
- Raw `fiscal_year` feature may not capture this optimally
- Relative features (vs. year average) are more normalized

**New Features to Add**:

```python
def add_year_relative_features(df):
    """Create salary features relative to year average."""
    
    # Calculate salary statistics by year
    year_stats = df.groupby('fiscal_year')['base_salary'].agg([
        'mean', 'median', 'std'
    ]).reset_index()
    
    # Merge back to data
    df = df.merge(year_stats, on='fiscal_year', how='left', 
                  suffixes=('', '_by_year'))
    
    # Create relative features
    df['salary_vs_year_mean'] = df['base_salary'] / df['mean']
    df['salary_vs_year_median'] = df['base_salary'] / df['median']
    df['salary_std_by_year'] = df['std']
    
    # Percentile within year
    df['salary_percentile_by_year'] = df.groupby('fiscal_year')[
        'base_salary'
    ].transform(lambda x: (x.rank() / len(x)))
    
    # Deviation from year average (in standard deviations)
    df['salary_z_score_by_year'] = (
        (df['base_salary'] - df['mean']) / df['std']
    )
    
    return df[['salary_vs_year_mean', 'salary_vs_year_median',
               'salary_std_by_year', 'salary_percentile_by_year',
               'salary_z_score_by_year']]
```

**Implementation Steps**:
1. Add function to `src/data622/features.py`
2. Call in feature pipeline before preprocessing
3. Include 4-5 new numeric features
4. Retrain model and compare R²

---

### 3️⃣ FEATURE ENGINEERING - Polynomial & Interaction Features
**Effort**: ⭐⭐ MEDIUM (1-2 hours)  
**Expected Impact**: +1-3% R² improvement  
**Why**: Captures non-linear relationships

**Examples**:
```python
def add_interaction_features(df):
    """Create polynomial and interaction features."""
    
    # Polynomial features
    df['tenure_squared'] = df['tenure_years'] ** 2
    df['tenure_cubed'] = df['tenure_years'] ** 3
    
    # Interaction features
    df['tenure_x_agency_size'] = df['tenure_years'] * df['agency_size']
    df['tenure_x_title_freq'] = df['tenure_years'] * df['title_frequency']
    df['agency_size_x_title_freq'] = df['agency_size'] * df['title_frequency']
    
    # Non-linear transformations
    df['log_agency_size'] = np.log1p(df['agency_size'])
    df['sqrt_tenure'] = np.sqrt(df['tenure_years'])
    
    return df
```

**Be Careful**:
- Avoid too many features (curse of dimensionality)
- Test each feature's impact individually
- Check for multicollinearity

---

### 4️⃣ FEATURE SELECTION - Remove Low-Impact Features
**Effort**: ⭐ LOW (30 minutes)  
**Expected Impact**: +0-1% R² improvement  
**Why**: Cleaner model, fewer parameters, better generalization

**Analysis Code**:
```python
# Get feature coefficients
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print(feature_importance)

# Keep only features with |coef| > threshold
threshold = 0.05
important_features = feature_importance[
    feature_importance['abs_coefficient'] > threshold
]['feature'].tolist()

print(f"Keeping {len(important_features)} of {len(feature_cols)} features")
```

**Steps**:
1. Train baseline model
2. Extract and analyze coefficients
3. Remove bottom 10-20% of features
4. Retrain and compare

---

### 5️⃣ DOMAIN-SPECIFIC FEATURES
**Effort**: ⭐⭐⭐ HIGH (2-4 hours)  
**Expected Impact**: +2-5% R² improvement  
**Why**: Leverages NYC payroll domain knowledge

**Potential Features**:
```python
# Job Category Hierarchies
def add_job_seniority(df):
    """Infer seniority from title patterns."""
    seniority_keywords = {
        'chief': 5,
        'director': 4,
        'manager': 3,
        'senior': 2,
        'junior': 1,
        'assistant': 1
    }
    
    df['job_seniority'] = df['job_title'].str.lower().apply(
        lambda x: max([score for kw, score in seniority_keywords.items() 
                      if kw in x], default=1)
    )
    return df

# Borough Cost of Living Adjustment
def add_borough_cola(df):
    """Add cost-of-living adjustment by borough."""
    cola_factors = {
        'Manhattan': 1.15,
        'Brooklyn': 1.05,
        'Queens': 0.95,
        'Bronx': 0.90,
        'Staten Island': 0.88
    }
    df['borough_cola'] = df['borough'].map(cola_factors)
    return df

# Department/Agency Prestige Score
def add_agency_prestige(df):
    """Rank agencies by average salary (proxy for prestige)."""
    prestige_scores = df.groupby('agency')['base_salary'].mean()
    prestige_rank = prestige_scores.rank() / len(prestige_scores)
    df['agency_prestige'] = df['agency'].map(prestige_rank)
    return df
```

---

## 📊 Comparison Matrix

| Strategy | Effort | Expected Impact | Implementation |
|----------|--------|-----------------|-----------------|
| Ridge Regression | ⭐ | +0-2% | Modify model.py |
| Year-Relative Features | ⭐⭐ | +2-4% | Add to features.py |
| Polynomial Features | ⭐⭐ | +1-3% | Add to features.py |
| Feature Selection | ⭐ | +0-1% | Analysis only |
| Domain Features | ⭐⭐⭐ | +2-5% | New functions |

---

## 🚀 Recommended Implementation Path

### Phase 1: Quick Win (30 minutes)
1. **Add Ridge Regression**
   - Test alpha values: [0.001, 0.01, 0.1, 1.0, 10.0]
   - Expected: +0-2% improvement
   - File: Modify `src/data622/model.py`

### Phase 2: Feature Engineering (1-2 hours)
2. **Add Year-Relative Features**
   - salary_vs_year_mean
   - salary_percentile_by_year
   - Expected: +2-4% improvement
   - File: Update `src/data622/features.py`

3. **Add Interaction Features**
   - tenure × agency_size
   - Non-linear transformations
   - Expected: +1-3% improvement

### Phase 3: Optimization (1 hour)
4. **Feature Selection**
   - Remove low-impact features
   - Expected: +0-1% improvement (cleaner model)

---

## ✅ Expected Final Performance

**Optimistic Scenario** (all strategies combined):
- Current R²: 0.7453
- Expected improvement: +5-7%
- **Target R²: 0.78-0.80**

**Conservative Scenario** (Phase 1 + 2):
- Current R²: 0.7453
- Expected improvement: +2-4%
- **Target R²: 0.765-0.785**

---

## 🔧 Next Steps

1. **Decide on approach**: Which strategy fits your timeline?
2. **Implement incrementally**: Add one feature/strategy, retrain, compare
3. **Track results**: Keep record of R² for each change
4. **A/B test**: Compare old vs. new model on validation set
5. **Final evaluation**: Test on held-out test set

---

## 📝 Testing Template

```python
# After implementing improvements:
from data622.model import SalaryPredictionModel

# Train improved model
model = SalaryPredictionModel(model_type="linear")
results = model.full_pipeline()

# Compare with baseline
print(f"Baseline R²: 0.7453")
print(f"Improved R²: {results['r2']:.4f}")
print(f"Improvement: {(results['r2'] - 0.7453) * 100:.2f}%")
```

---

## 💡 Pro Tips

1. **Always use cross-validation** to avoid overfitting
2. **Scale features properly** before polynomial/interaction features
3. **Test incrementally** - add one feature, check impact
4. **Monitor train-test gap** - watch for overfitting
5. **Document everything** - keep track of what helped/hurt
6. **Use domain expertise** - NYC payroll has specific patterns

---

**Current Status**: Ready to implement Phase 1  
**Last Updated**: April 8, 2026
