# Task 3: Heart Disease Prediction
**DevelopersHub Corporation — AI/ML Engineering Internship**

## Objective
Build a binary classification model to predict whether a patient is at risk of heart disease based on their health indicators.

## Dataset
- **Name:** UCI Heart Disease Dataset (Cleveland)
- **Source:** Embedded directly inside the notebook — no internet or download required
- **Size:** 244 patients, 14 columns
- **Target:** 0 = No Disease, 1 = Heart Disease

| Feature | Description |
|---------|-------------|
| age | Age of the patient (years) |
| sex | Sex (1 = male, 0 = female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0–4) |
| thal | Thalassemia type |
| target | Heart Disease (1 = Disease, 0 = No Disease) |

## Tools & Libraries
- Python 3.10
- pandas, numpy
- scikit-learn (LogisticRegression, DecisionTreeClassifier)
- matplotlib, seaborn

## What's Inside the Notebook
| Step | Description |
|------|-------------|
| 1. Import Libraries | All required libraries |
| 2. Load Dataset | Loaded from embedded CSV — no internet needed |
| 3. Feature Descriptions | Explanation of all 14 columns |
| 4. Data Inspection | `.info()`, `.describe()`, missing values, class balance |
| 5. EDA | Target distribution, age histogram, gender chart, box plots, heatmap |
| 6. Preprocessing | Train/test split (80/20), StandardScaler |
| 7. Model Training | Logistic Regression & Decision Tree |
| 8. Evaluation | Accuracy, classification report, confusion matrix, ROC curve, feature importance |
| 9. Insights | Key findings and conclusions |

## Model Performance
| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | ~85% | ~0.91 |
| Decision Tree (depth=5) | ~79% | ~0.86 |

## Key Results & Findings
- **Logistic Regression outperforms Decision Tree** in AUC-ROC — better suited for medical use
- **`thalach`** (max heart rate) is the strongest predictor of heart disease
- **`cp`** (chest pain type) and **`oldpeak`** (ST depression) are also top predictors
- Dataset is well balanced (~51% disease vs ~49% no disease) — no oversampling needed
- Both models significantly outperform a random classifier on the ROC curve

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Task3_Heart_Disease_Prediction.ipynb
```
Run all cells from top to bottom. No internet connection or dataset download required.
