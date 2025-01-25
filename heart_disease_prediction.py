import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
