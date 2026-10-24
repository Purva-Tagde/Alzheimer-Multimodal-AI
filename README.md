# Multimodal Alzheimer's Disease Risk Assessment Using Neuroimaging, Cognitive Data, and Explainable Machine Learning

## Overview

This project presents a machine learning framework for Alzheimer's disease risk assessment using neuroimaging-derived features, cognitive assessments, and explainable artificial intelligence (XAI). It investigates multimodal learning using the OASIS-3 dataset and evaluates an independent MRI-feature classification experiment using ADNI data.

The framework combines predictive modelling, feature selection, model evaluation, and explainability techniques to study the potential of integrating different clinical data modalities for Alzheimer's disease classification.

**Important:** This is a research prototype for educational and experimental purposes. It is not a clinically validated diagnostic system and must not be used to make medical decisions.

## Objectives

- Develop a multimodal classification framework using MRI-derived and cognitive features.
- Evaluate Logistic Regression, Random Forest, XGBoost, and a weighted ensemble.
- Apply feature selection to identify informative predictors.
- Evaluate performance using accuracy, precision, recall, specificity, F1-score, ROC-AUC, AUPRC, and Brier score.
- Use explainability techniques such as SHAP to investigate model feature importance.
- Conduct a separate MRI-feature experiment on ADNI data to assess model performance on an independent dataset.

## Datasets

### OASIS-3

OASIS-3 provides neuroimaging and clinical information used in the multimodal experiment.

- Records used in the final experiment: 787
- Unique participants: 612
- Candidate features after data preparation: 224
- Selected features: 50
- Modalities: MRI-derived features and cognitive assessments

### ADNI

ADNI was used for a separate MRI-feature classification experiment.

- Participants in the final baseline experiment: 328
- Cognitively normal (CN): 195
- Alzheimer's disease (AD): 133
- MRI-derived features: 37
- Evaluation split: participant-level training and testing

**Data access:** The original datasets are not included in this repository. Access and reuse must follow the respective dataset providers' registration, licensing, and data-use requirements.

## Methodology

1. **Data preparation:** Prepare MRI-derived and cognitive features for analysis.
2. **Feature engineering:** Align relevant features and prepare classification targets.
3. **Feature selection:** Reduce the OASIS-3 candidate feature set from 224 to 50 selected features.
4. **Model training:** Evaluate Logistic Regression, Random Forest, and XGBoost models.
5. **Ensemble learning:** Combine model predictions using a weighted multimodal ensemble.
6. **Independent experiment:** Train and evaluate MRI-feature models using ADNI data.
7. **Evaluation:** Compare predictive performance using classification and probabilistic metrics.
8. **Explainability:** Examine feature importance using SHAP-based analysis.
9. **Visualization:** Generate model comparison charts, ROC curves, precision-recall curves, confusion matrices, and feature-importance plots.

## Results

### OASIS-3 multimodal experiment

| Metric | Weighted ensemble |
|---|---:|
| Accuracy | 96.34% |
| Precision | 77.78% |
| Recall | 63.64% |
| Specificity | 98.69% |
| F1-score | 70.00% |
| ROC-AUC | 0.8990 |
| AUPRC | 0.5555 |
| Brier score | 0.0400 |

Confusion matrix:

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 151 | 2 |
| Actual positive | 4 | 7 |

### ADNI MRI-feature experiment

The Random Forest model achieved the following results in the separate ADNI experiment.

| Metric | Result |
|---|---:|
| Accuracy | 87.88% |
| Precision | 85.19% |
| Recall | 85.19% |
| Specificity | 89.74% |
| F1-score | 85.19% |
| ROC-AUC | 0.9269 |
| AUPRC | 0.9178 |
| Brier score | 0.1040 |

These are results from two distinct experiments with different data, feature sets, and evaluation settings. They should not be interpreted as a direct comparison between datasets.

## Explainable AI

SHAP-based feature importance is used to examine how selected predictors contribute to model outputs. The resulting feature rankings support model interpretation and help identify potentially informative MRI-derived and cognitive variables.

Feature importance does not establish causation or clinical significance.

## Technology Stack

- **Language:** Python
- **Machine learning:** scikit-learn, XGBoost
- **Data processing:** Pandas, NumPy
- **Explainability:** SHAP
- **Visualization:** Matplotlib and other plotting tools used in the experiments
- **Dashboard:** Streamlit
- **MRI demonstration:** PyTorch, Torchvision, Grad-CAM, and pydicom

## Repository Structure

```text
Alzheimer-Multimodal-AI/
├── dashboard/
│   ├── app_updated.py
│   ├── requirements.txt
│   └── README.md
├── models/
│   ├── adni_logistic_regression.pkl
│   ├── adni_random_forest.pkl
│   ├── adni_xgboost.pkl
│   ├── final_multimodal_ensemble.pkl
│   └── ...
├── reports/
│   ├── final_experiment_summary.csv
│   ├── final_model_results.csv
│   ├── adni_model_results.csv
│   ├── final_confusion_matrix.png
│   ├── final_roc_curve.png
│   ├── final_precision_recall_curve.png
│   └── ...
├── src/
│   ├── plot_adni_comparison.py
│   └── train_mri_features.py
├── data/                 # Local only; not included in Git
├── notebooks/            # Local research notebook excluded from Git
└── .gitignore
```

The exact files in the repository may evolve as the project is maintained.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Purva-Tagde/Alzheimer-Multimodal-AI.git
cd Alzheimer-Multimodal-AI
```

If the repository is private, sign in to GitHub and ensure you have access before cloning.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dashboard dependencies

```powershell
pip install -r dashboard/requirements.txt
```

### 4. Run the dashboard

```powershell
streamlit run dashboard/app_updated.py
```

**Note:** The dashboard also requires its expected model files and compatible dependencies. The large raw-MRI demonstration checkpoint, `models/adni_resnet18_5slice_baseline.pth`, is intentionally not included in this repository. The raw-MRI demo will require that checkpoint to be obtained separately through an authorized source and placed at the expected path.

Model files serialized with Python libraries can depend on compatible library versions. Use trusted model files only.

## Reports and Visualizations

The `reports/` directory contains aggregate model metrics, comparison tables, and evaluation visualizations, including:

- Confusion matrices
- ROC curves
- Precision-recall curves
- Model comparison charts
- Selected feature lists
- SHAP feature-importance summaries
- Reproducibility summaries

These outputs describe experimental results and should be interpreted in the context of their evaluation datasets and protocols.

## Privacy and Responsible Use

- Raw healthcare datasets and processed participant-level data are excluded from this repository.
- Do not upload participant identifiers, MRI images, DICOM metadata, clinical records, credentials, or restricted datasets.
- Follow the dataset providers' data-use agreements.
- This project is intended for research and educational purposes only.
- Predictions are experimental and are not a substitute for professional medical evaluation.

## Limitations

- High classification accuracy does not by itself establish clinical usefulness.
- Class imbalance can affect precision, recall, and other metrics.
- Dataset composition and evaluation procedures can affect generalizability.
- Performance across OASIS-3 and ADNI is not directly comparable because the experiments differ.
- Independent external validation and clinical evaluation would be necessary before considering clinical use.

## Future Work

- Conduct further external validation and robustness testing.
- Evaluate calibration and threshold selection.
- Improve transparency and reproducibility of model training.
- Investigate explainability across different patient subgroups.
- Improve dashboard usability and model-dependency management.
- Document software versions and experimental configurations more comprehensively.

## Author

**Purva Tagde**  
B.Tech, Computer Science and Engineering  
SIT Nagpur

## Disclaimer

This repository is an academic research project. It is not intended to diagnose, treat, cure, or prevent Alzheimer's disease or any other medical condition.
