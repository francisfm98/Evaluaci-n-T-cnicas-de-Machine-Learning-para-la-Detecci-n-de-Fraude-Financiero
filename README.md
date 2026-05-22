# Evaluaci-n-T-cnicas-de-Machine-Learning-para-la-Detecci-n-de-Fraude-Financiero
 Análisis Comparativo de Técnicas de Machine Learning para la Detección de Fraude Financiero en la Apertura de Cuencas Bancarias: Rendimiento, Explicabilidad y Sesgo

# TFM: Bank Account Fraud Detection - Comparative Analysis of Machine Learning and Deep Learning Models

**Author:** Francisco Fajardo

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Installation & Requirements](#installation--requirements)
5. [Usage](#usage)
6. [Key Sections](#key-sections)
7. [Models Implemented](#models-implemented)
8. [Explainability Analysis](#explainability-analysis)
9. [Results & Findings](#results--findings)
10. [Contact & References](#contact--references)

---

## Project Overview

This Master's thesis project presents a comprehensive analysis of bank account fraud detection using both classical machine learning and advanced deep learning approaches. The project includes:

- **Exploratory Data Analysis (EDA)** of the bank fraud dataset
- **Data Preprocessing & Balancing** techniques
- **Classical ML Models** comparison and hyperparameter tuning
- **Deep Learning Models** including MLP and TabTransformer
- **Explainability Analysis** using SHAP and LIME
- **Algorithmic Bias Detection** and mitigation strategies

The main objective is to develop, compare, and interpret predictive models for identifying fraudulent bank account openings while addressing model explainability and fairness.

---

## Dataset

**Source:** Kaggle - Bank Account Fraud Dataset (NeurIPS 2022)

- **URL:** `sgpjesus/bank-account-fraud-dataset-neurips-2022`
- **Target Variable:** `fraud_bool` (Binary classification: Legitimate vs. Fraudulent)
- **Primary File:** `Base.csv`

### Key Characteristics:
- Multiple numerical and categorical features
- Class imbalance present (fraud detection is typically imbalanced)
- No missing values
- Includes outlier detection analysis

---

## Project Structure

```
TFM_CFajardo.ipynb
├── Data Loading & EDA
├── Data Preprocessing & Balancing
├── Classical ML Models
├── Advanced DL Models
├── Explainability Analysis
├── Algorithmic Bias Analysis
└── Comparative Results
```

---

## Installation & Requirements

### Dependencies:

```bash
pip install kagglehub
pip install pandas numpy matplotlib seaborn
pip install scikit-learn imblearn
pip install xgboost lightgbm catboost
pip install torch pytorch-lightning
pip install shap lime
pip install imbalanced-learn
```

### Python Version:
- **Python 3.7+** (recommended: Python 3.9+)

### Environment:
- **Jupyter Notebook** or **JupyterLab**
- GPU support (optional but recommended for deep learning models)

---

## Usage

1. **Clone or download** the notebook file
2. **Install dependencies** using the requirements above
3. **Launch Jupyter:**
   ```bash
   jupyter notebook TFM_CFajardo.ipynb
   ```
4. **Run cells sequentially** from top to bottom
5. The notebook will automatically download the dataset from Kaggle

---

## Key Sections

### 1. **Descargar Dataset** (Download Dataset)
Downloads the bank fraud dataset directly from Kaggle using the `kagglehub` API.

### 2. **EDA** (Exploratory Data Analysis)
- **Estructura general del dataset:** Dataset overview, shape, data types, and target variable distribution
- Fraud vs. Legitimate case distribution
- Missing value detection (None detected)
- Outlier detection using Interquartile Range (IQR) method
- Correlation analysis and multicollinearity detection

### 3. **Preprocesado y Balanceo de Datos** (Data Preprocessing & Balancing)

#### Tratamiento de valores nulos (Null Value Treatment)
- Verified no missing values in dataset

#### Tratamiento de valores atípicos (Outlier Treatment)
- IQR-based outlier detection
- Handling strategy for identified outliers

#### División del dataset (Data Splitting)
- Training, validation, and test set creation
- Stratified sampling to maintain class distribution

#### Codificación de variables (Variable Encoding)
- Categorical variable encoding (One-Hot, Label encoding, etc.)
- Feature scaling and normalization

#### Balanceado de la base de datos (Class Balancing)
- Techniques to handle class imbalance
- SMOTE or other oversampling/undersampling methods

### 4. **Modelos Clásicos de Machine Learning** (Classical ML Models)

Includes implementation and evaluation of:
- Logistic Regression
- Random Forest
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)

#### Features:
- Model training with cross-validation
- Confusion matrix generation
- Hyperparameter tuning
- Comparative performance tables

### 5. **Modelos Avanzados** (Advanced Deep Learning Models)

#### Preparación de datos para PyTorch
- Data conversion to PyTorch tensors
- DataLoader creation for batch processing

#### Modelo 1: MLP (Multilayer Perceptron)
- Fully connected neural network
- Architecture: Multiple hidden layers with activation functions
- Training loop with validation

#### Modelo 2: TabTransformer
- State-of-the-art architecture for tabular data
- Transformer-based feature learning
- Attention mechanisms for feature interaction

### 6. **Análisis SHAP** (SHAP Explainability Analysis)

#### SHAP Global Analysis
- Feature importance at global level
- Identifies most impactful features across all predictions

#### SHAP Local Analysis
- Individual prediction explanation
- SHAP values for specific instances

#### SHAP Dependence
- Feature value variation impact
- How feature values affect predictions

#### Models Covered:
- Classical ML models
- Deep Learning models (MLP, TabTransformer)

### 7. **LIME Local Analysis** (Local Interpretable Model-agnostic Explanations)
- Local approximation of model decisions
- Per-instance feature contributions

### 8. **Análisis del Sesgo Algorítmico** (Algorithmic Bias Analysis)
- Fairness assessment across protected attributes
- Bias detection and quantification
- Mitigation strategies

---

## Models Implemented

### Classical Models:
| Model | Type | Purpose |
|-------|------|---------|
| Logistic Regression | Linear | Baseline model |
| Random Forest | Ensemble | Tree-based non-linear |
| XGBoost | Gradient Boosting | Advanced ensemble |
| LightGBM | Gradient Boosting | Fast gradient boosting |
| CatBoost | Gradient Boosting | Categorical feature handling |
| SVM | Kernel-based | Non-linear classification |
| KNN | Instance-based | Distance-based classification |

### Deep Learning Models:
| Model | Architecture | Advantages |
|-------|-------------|-----------|
| MLP | Multilayer Perceptron | Simple, interpretable |
| TabTransformer | Transformer-based | State-of-the-art for tabular data |

---

## Explainability Analysis

The notebook employs multiple explainability methods to make model predictions interpretable:

### SHAP (SHapley Additive exPlanations)
- **Global explanations:** Feature importance rankings
- **Local explanations:** Per-prediction feature contributions
- **Dependence plots:** Feature value impact patterns

### LIME (Local Interpretable Model-agnostic Explanations)
- Model-agnostic local approximations
- Easy-to-understand feature contributions
- Works with any model type

### Comparative Tables
- Comprehensive comparison of SHAP global, SHAP local, and LIME results
- Helps identify consistent vs. model-specific feature importance patterns

---

## Results & Findings

The notebook generates:

1. **Performance Metrics:**
   - Accuracy, Precision, Recall, F1-Score
   - ROC-AUC curves
   - Confusion matrices

2. **Model Comparison:**
   - Classical ML models vs. Deep Learning models
   - Performance tradeoffs
   - Training time and efficiency

3. **Explainability Insights:**
   - Most important fraud indicators
   - Feature interaction patterns
   - Model decision mechanisms

4. **Fairness Assessment:**
   - Bias across demographic groups
   - Disparate impact analysis
   - Fairness-performance tradeoffs

---

## Technologies Used

- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Classical ML:** Scikit-learn, XGBoost, LightGBM, CatBoost
- **Deep Learning:** PyTorch, PyTorch Lightning
- **Explainability:** SHAP, LIME
- **Data Balancing:** Imbalanced-learn (SMOTE)
- **Data Source:** Kaggle API (kagglehub)

---

## Notes

- The notebook downloads data automatically; requires internet connection
- Deep learning sections benefit from GPU acceleration
- Total execution time: 1-3 hours depending on hardware
- All cells should be run sequentially to maintain proper variable states

---

## Dataset Reference

**Dataset:** Bank Account Fraud Dataset - NeurIPS 2022  
**Source:** Kaggle  
**Citation:** Jesus, S. G. et al. (2022). Bank Account Fraud Dataset

---

## Contact

**Author:** Francisco Fajardo  
**Project Type:** Master's Thesis (TFM)

---

## License

Please refer to the original dataset's license terms on Kaggle.

---

**Last Updated:** 2024-2025  
**Version:** 1.0
