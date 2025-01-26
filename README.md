# Heart Disease Prediction

## Overview

This project predicts the likelihood of heart disease in patients using the K-Nearest Neighbors (KNN) machine learning algorithm. The dataset, sourced from Kaggle, contains patient information such as age, cholesterol levels and exercise-induced angina, which are used to train the model.

## Key Features

- **Exploratory Data Analysis (EDA):** Visualizations and statistical analysis to understand the dataset.
- **Data Cleaning:** Removal of invalid or missing values and encoding of categorical features.
- **Feature Selection:** Identification of the most relevant predictors of heart disease.
- **Model Building:** Implementation of a KNN classifier with hyperparameter tuning.
- **Evaluation:** Model performance is assessed using accuracy on a test dataset.

## Dataset

The dataset includes the following features:

- **Age:** Age of the patient.
- **Sex:** Gender (Male/Female).
- **ChestPainType:** Types of chest pain (e.g., Asymptomatic, Atypical Angina).
- **RestingBP:** Resting blood pressure.
- **Cholesterol:** Serum cholesterol levels.
- **FastingBS:** Fasting blood sugar (>120 mg/dl or not).
- **RestingECG:** Resting electrocardiogram results.
- **MaxHR:** Maximum heart rate achieved.
- **ExerciseAngina:** Exercise-induced angina (Yes/No).
- **Oldpeak:** ST depression induced by exercise relative to rest.
- **ST\_Slope:** The slope of the peak exercise ST segment.
- **HeartDisease:** Target variable (1: Heart disease, 0: No heart disease).

## Visualisations and Conclusions

- **Correlation Heatmap:** Identifies relationships between numerical features.
- **Categorical Distributions:** Highlights data imbalances and feature relevance.
- **Grouped Bar Plots:** Show how categorical variables relate to heart disease.

- **Figure 1:** Correlation Heatmap (Original Dataset)

- Strong Predictors: Oldpeak, MaxHR, and HeartDisease show significant correlations.
- Directionality: Oldpeak is positively correlated with HeartDisease, while MaxHR is negatively correlated.
- Weak Correlations: Features like Cholesterol and RestingBP show minimal correlation with HeartDisease.


- **Figure 2:** Correlation Heatmap (Cleaned Dataset)

- Dummy Variables Added: New variables like ST_Slope_Flat and ExerciseAngina_Y emerge as strongly correlated with HeartDisease.
- Expanded Analysis: Dummy variables reveal additional insights, such as the strong role of ST_Slope_Flat and ExerciseAngina_Y.
- Key Finding: Cleaning and encoding data improves the identification of critical features.


- **Figure 3:** Filtered Correlation Heatmap (Threshold > 0.3)

- Focus on Strong Correlations: Only features with significant correlations (e.g., ST_Slope_Flat, ExerciseAngina_Y) are highlighted.
- Simplified Insights: Weakly correlated variables are excluded, making it easier to focus on impactful predictors.
- Predictor Importance: Confirms the key role of Oldpeak and ST_Slope_Flat in predicting heart disease.


- **Figure 4:** Categorical Feature Distribution

- Gender Imbalance: The dataset is skewed towards male patients, which could bias the model.
- Prominent Chest Pain Type: Most patients with heart disease have asymptomatic (ASY) chest pain.
- Exercise Angina: A significant portion of patients with heart disease experienced exercise-induced angina.


- **Figure 5:** Categorical Features vs. Heart Disease

- Exercise Angina: Patients with exercise-induced angina are more likely to have heart disease.
- ST Slope: A flat ST slope is strongly associated with heart disease.
- Fasting Blood Sugar: Elevated fasting blood sugar is moderately linked to heart disease.


## Key Takeaways

- Strong predictors: `Oldpeak`, `ST_Slope_Flat`, `ExerciseAngina_Y`, `MaxHR`.
- Weak predictors: `Cholesterol`, `RestingBP`.
- The model performs well but is influenced by dataset imbalances.
- Model Accuracy on test set: **86.96%**

## Prerequisites

- Python 3.8+
- Libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`

## Usage

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python heart_disease_prediction.py
   ```

## Future Improvements

- Address data imbalance (e.g., oversampling or undersampling).
- Experiment with other classifiers like Random Forest or Logistic Regression.

## Acknowledgments

Dataset: [Kaggle](https://www.kaggle.com/)
