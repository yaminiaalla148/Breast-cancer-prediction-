# Breast Cancer Prediction using Machine Learning

Short-term internship project (App Genesis Soft Solutions Pvt Ltd).

## Problem
Predict whether a tumour is malignant or benign from 30 measurements
of cell nuclei (Wisconsin Breast Cancer dataset, 569 records).

## Files
- `dataset.csv` - dataset (30 features + `target`: 0 = malignant, 1 = benign)
- `breast_cancer_prediction.py` - complete project code
- `requirements.txt` - libraries needed
- `outputs/` - graphs and results created when you run the code
- `WEEKS_7_8_ACTIVITY_LOG.md` - ready-to-paste activity log for weeks 7 and 8

## How to run
    pip install -r requirements.txt
    python app.py

Open http://127.0.0.1:5000 for the local web dashboard. On first startup, the app trains and tunes the models; this may take a short while.

To run the command-line analysis and regenerate the plots and model comparison:
    python breast_cancer_prediction.py

## Methodology
1. Load and explore data (EDA, missing values, class balance, correlation)
2. Train/test split (80/20, stratified) and StandardScaler
3. Train Logistic Regression, KNN, Decision Tree, Random Forest, SVM
4. Evaluate: accuracy, precision, recall, F1-score, 5-fold cross validation
5. Tune SVM with Grid Search and a leakage-safe scikit-learn pipeline
6. Confusion matrix and PCA visualisation

## Results (test set)
| Model | Accuracy |
|---|---|
| Logistic Regression | 98.25% |
| SVM (tuned, linear) | 98.25% |
| KNN | 95.61% |
| Random Forest | 95.61% |
| Decision Tree | 91.23% |

## Note
This is an educational project and not a medical diagnosis tool.
