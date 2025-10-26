# Data Cleaning and EDA Module - Documentation

## Overview

This comprehensive data cleaning and exploratory data analysis (EDA) module provides a complete solution for data preparation and analysis in the AMI Mortality Predictor project.

## Module Structure

### Core Components

1. **`src/data_cleaning.py`** - Data cleaning pipeline
2. **`src/eda.py`** - Exploratory data analysis tools
3. **`pages/01_Data_Preparation_and_EDA.py`** - Streamlit web interface
4. **`DATA/variable_metadata.json`** - Variable metadata storage
5. **`tests/test_data_cleaning.py`** - Unit tests

## Features

### 1. Data Cleaning Pipeline (`data_cleaning.py`)

#### Classes

##### `CleaningConfig`
Configuration dataclass for cleaning operations:
- **Imputation methods**: mean, median, mode, KNN, forward/backward fill, constant
- **Outlier detection**: IQR method, Z-score method
- **Outlier treatment**: capping, removal, none
- **Categorical encoding**: label, one-hot, ordinal, none
- **Discretization**: quantile, uniform, custom bins
- **General options**: drop duplicates, drop fully missing, drop constant columns

##### `DataCleaner`
Main cleaning pipeline class:
```python
from src.data_cleaning import DataCleaner, CleaningConfig

config = CleaningConfig(
    numeric_imputation='median',
    categorical_imputation='mode',
    outlier_method='iqr',
    outlier_treatment='cap',
    categorical_encoding='label'
)

cleaner = DataCleaner(config)
df_clean = cleaner.fit_transform(df, target_column='mortality_inhospital')

# Save metadata and configuration
cleaner.save_metadata('DATA/variable_metadata.json')
cleaner.save_config('DATA/preprocessing_config.json')

# Get cleaning report
report = cleaner.get_cleaning_report()
```

##### `VariableMetadata`
Stores comprehensive metadata for each variable:
- Original and cleaned types
- Value ranges
- Missing value statistics
- Encoding mappings
- Discretization bins
- Quality flags

#### Quick Cleaning Function

```python
from src.data_cleaning import quick_clean

df_clean, cleaner = quick_clean(
    df,
    target_column='mortality_inhospital',
    numeric_imputation='median',
    outlier_method='iqr'
)
```

### 2. Exploratory Data Analysis (`eda.py`)

#### Classes

##### `EDAAnalyzer`
Comprehensive EDA tool:

```python
from src.eda import EDAAnalyzer

analyzer = EDAAnalyzer(df)

# Univariate analysis
univariate_stats = analyzer.analyze_univariate()
fig = analyzer.plot_distribution('age', plot_type='histogram')

# Bivariate analysis
bivariate_stats = analyzer.analyze_bivariate('age', 'mortality_inhospital')
fig = analyzer.plot_correlation_matrix(method='pearson')
fig = analyzer.plot_scatter('age', 'blood_pressure', add_trendline=True)

# Multivariate analysis (PCA)
pca_results = analyzer.perform_pca(n_components=5, scale=True)
fig = analyzer.plot_pca_scree()
fig = analyzer.plot_pca_biplot(pc_x=1, pc_y=2)
importance_df = analyzer.get_feature_importance_pca(n_components=3)

# Save results
analyzer.save_results('DATA/eda_cache.pkl')
```

#### Statistics Classes

##### `UnivariateStats`
For numerical variables:
- count, mean, median, std
- min, max, quartiles
- skewness, kurtosis

For categorical variables:
- number of categories
- mode and frequency
- category counts

##### `BivariateStats`
Relationship statistics:
- **Numerical-Numerical**: Pearson & Spearman correlations
- **Numerical-Categorical**: ANOVA F-statistic
- **Categorical-Categorical**: Chi-square test, Cramér's V

##### `PCAResults`
PCA analysis results:
- Explained variance by component
- Cumulative variance
- Component loadings
- Transformed data

### 3. Streamlit Web Interface

#### Pages

Navigate to: `http://localhost:8501/01_Data_Preparation_and_EDA`

#### Tabs

1. **📂 Data Loading**
   - Upload CSV/Excel files
   - Load from configured path
   - Load existing cleaned datasets
   - Preview data with statistics

2. **🧹 Data Cleaning**
   - Configure cleaning parameters:
     - Imputation methods for numeric and categorical
     - Outlier detection and treatment
     - Categorical encoding
     - Discretization strategies
   - View before/after comparisons
   - Download cleaned data
   - Save metadata and configuration

3. **📈 Univariate Analysis**
   - Select variables to analyze
   - View comprehensive statistics
   - Interactive visualizations:
     - Histograms with KDE
     - Box plots
     - Violin plots
     - Bar charts (categorical)
     - Pie charts (categorical)

4. **📊 Bivariate Analysis**
   - Correlation matrices (heatmaps)
   - Pairwise scatter plots
   - Statistical tests:
     - Pearson/Spearman correlation
     - ANOVA
     - Chi-square tests
   - Scatter plots with trendlines
   - Scatter matrix for multiple variables

5. **🔬 Multivariate Analysis**
   - Principal Component Analysis (PCA)
   - Scree plots
   - Biplots
   - Feature importance rankings
   - Export transformed data

6. **📋 Quality Report**
   - Dataset summary statistics
   - Missing value analysis
   - Duplicate detection
   - Cardinality analysis
   - Outlier detection summary
   - Overall quality score
   - Export JSON report

### 4. Integration with ML Pipeline

#### Updated Components

##### `preprocess.py`
New functions:
```python
from src.preprocess import get_latest_cleaned_dataset, load_data_with_fallback

# Get path to latest cleaned dataset
cleaned_path = get_latest_cleaned_dataset(task_name='mortality')

# Load with automatic fallback
df, is_cleaned = load_data_with_fallback(
    'DATA/recuima-020425.csv',
    use_cleaned=True,
    task_name='mortality'
)
```

##### `train.py`
New arguments:
```bash
python -m src.train --data DATA/recuima-020425.csv --task mortality --use-cleaned
python -m src.train --data DATA/recuima-020425.csv --task mortality --force-raw
```

##### `predict.py`
New arguments:
```bash
python -m src.predict --model models/best_classifier_mortality.joblib \
    --input DATA/new_data.csv --output predictions.csv --use-cleaned
```

## Usage Workflows

### Workflow 1: Complete Data Preparation

```python
import pandas as pd
from src.data_cleaning import DataCleaner, CleaningConfig
from src.eda import EDAAnalyzer

# 1. Load raw data
df = pd.read_csv('DATA/recuima-020425.csv')

# 2. Configure and clean
config = CleaningConfig(
    numeric_imputation='median',
    categorical_imputation='mode',
    outlier_method='iqr',
    outlier_treatment='cap',
    categorical_encoding='label',
    drop_duplicates=True
)

cleaner = DataCleaner(config)
df_clean = cleaner.fit_transform(df, target_column='mortality_inhospital')

# 3. Save cleaned data
df_clean.to_csv('DATA/cleaned/cleaned_dataset_20250101_120000.csv', index=False)
cleaner.save_metadata('DATA/variable_metadata.json')

# 4. Perform EDA
analyzer = EDAAnalyzer(df_clean)
analyzer.analyze_univariate()
pca_results = analyzer.perform_pca(variance_threshold=0.95)
analyzer.save_results('DATA/eda_cache.pkl')

# 5. Generate report
report = cleaner.get_cleaning_report()
print(f"Cleaned {report['variables_cleaned']} variables")
```

### Workflow 2: Using Streamlit Interface

```bash
# Start the Streamlit app
cd Tools
streamlit run streamlit_app.py

# Navigate to "01_Data_Preparation_and_EDA" in the sidebar
# 1. Upload or load your dataset
# 2. Configure cleaning parameters
# 3. Apply cleaning and download results
# 4. Explore data with interactive visualizations
# 5. Generate and download quality reports
```

### Workflow 3: Training with Cleaned Data

```bash
# After cleaning data in Streamlit or Python:

# Train models (automatically uses cleaned data if available)
cd Tools
python -m src.train --data ../DATA/recuima-020425.csv --task mortality --use-cleaned

# Or force raw data usage
python -m src.train --data ../DATA/recuima-020425.csv --task mortality --force-raw
```

## Configuration Files

### `DATA/variable_metadata.json`
Stores metadata for all variables after cleaning:
```json
{
  "edad": {
    "name": "edad",
    "description": "Edad del paciente en años",
    "original_type": "float64",
    "cleaned_type": "float64",
    "is_numerical": true,
    "original_min": 18.0,
    "original_max": 95.0,
    "cleaned_min": 18.0,
    "cleaned_max": 95.0,
    "missing_count_original": 5,
    "missing_percent_original": 0.5,
    "imputation_method": "median",
    "outliers_detected": 10,
    "outliers_treated": 10,
    "quality_flags": ["outliers_capped_10", "no_missing_values"]
  }
}
```

### `DATA/preprocessing_config.json`
Stores cleaning configuration:
```json
{
  "numeric_imputation": "median",
  "categorical_imputation": "mode",
  "outlier_method": "iqr",
  "outlier_treatment": "cap",
  "categorical_encoding": "label",
  "drop_duplicates": true,
  "drop_fully_missing": true,
  "drop_constant": true
}
```

## Directory Structure

```
mortality-ami-predictor/
├── DATA/
│   ├── recuima-020425.csv (raw data)
│   ├── variable_metadata.json (variable info)
│   ├── preprocessing_config.json (cleaning config)
│   ├── eda_cache.pkl (cached EDA results)
│   └── cleaned/
│       └── cleaned_dataset_*.csv (cleaned datasets)
├── Tools/
│   ├── src/
│   │   ├── data_cleaning.py
│   │   ├── eda.py
│   │   ├── config.py (updated)
│   │   ├── preprocess.py (updated)
│   │   ├── train.py (updated)
│   │   └── predict.py (updated)
│   ├── pages/
│   │   └── 01_Data_Preparation_and_EDA.py
│   ├── tests/
│   │   └── test_data_cleaning.py
│   └── requirements.txt (updated)
```

## Testing

Run unit tests:
```bash
cd Tools
pytest tests/test_data_cleaning.py -v
```

Run all tests:
```bash
pytest tests/ -v --cov=src
```

## Best Practices

1. **Always preserve target column**: Pass `target_column` to cleaning functions
2. **Save metadata**: Keep track of transformations with metadata files
3. **Use cleaned data for training**: Enable `--use-cleaned` flag
4. **Version control cleaned datasets**: Use timestamps in filenames
5. **Document custom bins**: Store discretization mappings in metadata
6. **Review quality reports**: Check quality scores before model training
7. **Cache EDA results**: Save analyzer results for large datasets

## Troubleshooting

### Issue: Missing scipy dependency
```bash
pip install scipy
```

### Issue: Plotly figures not displaying
```bash
pip install kaleido
```

### Issue: Cleaned data not loading
- Check `DATA/cleaned/` directory exists
- Verify file naming convention: `cleaned_dataset_*.csv`
- Check file permissions

### Issue: Memory errors with large datasets
- Use smaller chunks for analysis
- Reduce number of PCA components
- Limit scatter matrix to fewer variables

## API Reference

See docstrings in:
- `src/data_cleaning.py` - Cleaning functions and classes
- `src/eda.py` - Analysis functions and classes

## Contributing

When adding new features:
1. Update relevant classes in `data_cleaning.py` or `eda.py`
2. Add corresponding UI elements in Streamlit page
3. Write unit tests in `tests/test_data_cleaning.py`
4. Update this documentation

## License

Same as parent project.

## Contact

For questions or issues, refer to the main project README.
