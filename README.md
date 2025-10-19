**Master Thesis - Evaluating genre dependence in machine learning models for predicting Spotify popularity from audio features**

## Project Overview

This project compares the performance of global machine learning models versus genre-specific models for predicting music success on Spotify. The analysis includes both regression (predicting popularity scores) and classification (predicting hit/non-hit status).

## Dataset

- **Source:** Kaggle - Spotify 1 Million Tracks Dataset
- **Size:** 1,159,763 tracks initially
- **Time Period:** 2000-2023
- **Features:** 13 audio features from Spotify API
- **Genres:** 8 selected genres with sufficient data

## Project Structure

```
project/
├── 01_Complete_EDA.ipynb           # Main EDA notebook (run first)
├── 02_train_global_models.py       # Train global models
├── 03_train_genre_specific_models.py  # Train genre-specific models
├── 04_cross_genre_validation_and_comparison.py  # Cross-validation & comparison
├── 05_shap_analysis_and_summary.py # SHAP analysis & final report
├── data/                           # Generated CSV files
├── models/                         # Saved model files (.pkl)
├── results/                        # Result CSV files
└── plots/                          # Generated visualizations
```

## Execution Order

### Step 1: Exploratory Data Analysis
**File:** `01_Complete_EDA.ipynb`

Run this notebook first. It performs:
- Data loading and cleaning
- Missing value handling
- Outlier removal
- Genre selection
- Target variable creation
- Feature correlation analysis
- Temporal train/test split

**Output Files (10 CSV files):**
- `spotify_cleaned.csv` - Cleaned data
- `spotify_filtered.csv` - Filtered to selected genres
- `spotify_final.csv` - Final dataset with target variables
- `spotify_train.csv` - Training set (years < 2019)
- `spotify_test.csv` - Test set (years >= 2019)
- `selected_genres.csv` - List of selected genres
- `feature_names.csv` - List of feature names
- `genre_statistics.csv` - Statistics by genre
- `feature_correlation_matrix.csv` - Feature correlations
- `dataset_metadata.csv` - Dataset metadata

**Output Plots (5 PNG files):**
- `01_initial_eda.png`
- `02_genre_distribution.png`
- `03_target_distribution.png`
- `04_feature_correlation.png`
- `05_genre_statistics.png`

### Step 2: Train Global Models
**File:** `02_train_global_models.py`

Trains models across all genres:
- Random Forest Regressor
- XGBoost Regressor
- Random Forest Classifier (with SMOTE)
- XGBoost Classifier (with SMOTE)

**Output Files:**
- `scaler_global.pkl` - Feature scaler
- `rf_regression_global.pkl` - RF regression model
- `xgb_regression_global.pkl` - XGB regression model
- `rf_classification_global.pkl` - RF classification model
- `xgb_classification_global.pkl` - XGB classification model
- `global_regression_results.csv` - Regression metrics
- `global_classification_results.csv` - Classification metrics
- `feature_importance_rf_regression_global.csv`
- `feature_importance_rf_classification_global.csv`

### Step 3: Train Genre-Specific Models
**File:** `03_train_genre_specific_models.py`

Trains separate Random Forest models for each genre:
- One regression model per genre
- One classification model per genre (with SMOTE)

**Output Files (per genre):**
- `rf_regression_{genre}.pkl` - Genre-specific regression model
- `rf_classification_{genre}.pkl` - Genre-specific classification model
- `feature_importance_regression_{genre}.csv`
- `feature_importance_classification_{genre}.csv`

**Additional Files:**
- `genre_regression_scalers.pkl` - All genre scalers
- `genre_classification_scalers.pkl` - All genre scalers
- `genre_specific_regression_results.csv` - All regression results
- `genre_specific_classification_results.csv` - All classification results

### Step 4: Cross-Genre Validation & Comparison
**File:** `04_cross_genre_validation_and_comparison.py`

Performs comprehensive analysis:
- Tests each genre model on all other genres
- Compares global vs genre-specific performance
- Statistical significance testing
- Feature importance comparison across genres

**Output Files:**
- `cross_genre_transfer_regression.csv` - Transfer matrix (MAE)
- `cross_genre_transfer_classification.csv` - Transfer matrix (F1)
- `comparison_global_vs_genre_regression.csv`
- `comparison_global_vs_genre_classification.csv`
- `improvements_regression.csv` - Performance improvements
- `improvements_classification.csv` - Performance improvements
- `feature_importance_matrix.csv` - Combined importance matrix
- `feature_ranking_correlation.csv` - Ranking correlations

**Output Plots:**
- `cross_genre_transfer_regression.png` - Heatmap of transfer performance
- `cross_genre_transfer_classification.png` - Heatmap of transfer performance
- `comparison_regression.png` - Global vs genre comparison
- `comparison_classification.png` - Global vs genre comparison
- `feature_importance_heatmap.png` - Feature importance across genres
- `feature_ranking_correlation.png` - Feature ranking correlation

### Step 5: SHAP Analysis & Final Report
**File:** `05_shap_analysis_and_summary.py`

Advanced interpretability and comprehensive summary:
- SHAP analysis for each genre
- Overall SHAP importance comparison
- Final comprehensive summary report

**Output Files:**
- `shap_importance_{genre}.csv` - SHAP values per genre
- `shap_importance_matrix.csv` - Combined SHAP matrix
- `shap_importance_overall.csv` - Overall importance ranking
- `FINAL_SUMMARY_REPORT.txt` - Comprehensive text report
- `executive_summary.csv` - Key metrics summary

**Output Plots:**
- `shap_summary_{genre}.png` - SHAP plot per genre (8 plots)
- `shap_importance_comparison.png` - Heatmap comparison

## Key Features

### Audio Features Used
1. Danceability
2. Energy
3. Key
4. Loudness
5. Mode
6. Speechiness
7. Acousticness
8. Instrumentalness
9. Liveness
10. Valence
11. Tempo
12. Duration (ms)
13. Time Signature

### Target Variables
- **Regression:** Popularity score (0-100)
- **Classification:** Hit/Non-hit (threshold = 70)

### Models Trained
- Random Forest (Regression & Classification)
- XGBoost (Regression & Classification)
- Class balancing using SMOTE for classification

## Key Metrics

### Regression
- MAE (Mean Absolute Error) - lower is better
- RMSE (Root Mean Squared Error) - lower is better
- R² (R-squared) - higher is better

### Classification
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC (Precision-Recall AUC)

## Requirements

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn (for SMOTE)
shap
scipy
kagglehub
```

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap scipy kagglehub
```

## Usage

1. **Run EDA Notebook:**
   ```python
   # Execute 01_Complete_EDA.ipynb in Jupyter/Colab
   ```

2. **Train Global Models:**
   ```bash
   python 02_train_global_models.py
   ```

3. **Train Genre-Specific Models:**
   ```bash
   python 03_train_genre_specific_models.py
   ```

4. **Cross-Validation & Comparison:**
   ```bash
   python 04_cross_genre_validation_and_comparison.py
   ```

5. **SHAP Analysis & Summary:**
   ```bash
   python 05_shap_analysis_and_summary.py
   ```

## Expected Results

The project demonstrates:
- Genre-specific models typically outperform global models
- Limited transferability of models across genres
- Genre context significantly impacts prediction accuracy
- Feature importance varies by genre
- Statistical significance of genre-specific approach

## Output Summary

### Total Files Generated
- **Data files:** 10 CSV files
- **Model files:** 18+ .pkl files (2 global + 16 genre-specific)
- **Results files:** 15+ CSV files
- **Visualizations:** 20+ PNG files
- **Reports:** 2 text files (summary + executive)

### Key Deliverables
1. Trained models (global and genre-specific)
2. Performance comparison results
3. Cross-genre transfer analysis
4. Feature importance analysis
5. SHAP interpretability analysis
6. Comprehensive final report

## Notes

- All intermediate results are saved as CSV files
- Models are saved as pickle files for reuse
- Plots are saved at 300 DPI for publication quality
- Temporal split ensures realistic evaluation (train on past, test on recent)
- SMOTE is applied to handle class imbalance in classification

## Author

Paolina CASTANY  
Master's Thesis - Big Data & Business Analytics

## License

Academic use only
