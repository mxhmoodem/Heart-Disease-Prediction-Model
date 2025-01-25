import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv("heart_disease_prediction.csv")

# Data Cleaning: Remove rows with invalid values for RestingBP
df_clean = df.copy()
df_clean = df_clean[df_clean["RestingBP"] != 0]

# Handle zero values in Cholesterol by replacing them with the median
heartdisease_mask = df_clean["HeartDisease"] == 0
cholesterol_without_heartdisease = df_clean.loc[heartdisease_mask, "Cholesterol"]
cholesterol_with_heartdisease = df_clean.loc[~heartdisease_mask, "Cholesterol"]

df_clean.loc[heartdisease_mask, "Cholesterol"] = cholesterol_without_heartdisease.replace(
    to_replace=0, value=cholesterol_without_heartdisease.median())
df_clean.loc[~heartdisease_mask, "Cholesterol"] = cholesterol_with_heartdisease.replace(
    to_replace=0, value=cholesterol_with_heartdisease.median())

# Feature Selection: Separate target and predictors
df_clean = pd.get_dummies(df_clean, drop_first=True)
X = df_clean.drop(["HeartDisease"], axis=1)
y = df_clean["HeartDisease"]

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=417)

# Selected features for training
features = ["Oldpeak", "Sex_M", "ExerciseAngina_Y", "ST_Slope_Flat", "ST_Slope_Up"]

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train[features])

# Hyperparameter Tuning: Define KNN classifier and grid search parameters
knn = KNeighborsClassifier()
grid_params = {"n_neighbors": range(1, 20), "metric": ["minkowski", "manhattan"]}
knn_grid = GridSearchCV(knn, grid_params, scoring='accuracy')
knn_grid.fit(X_train_scaled, y_train)
