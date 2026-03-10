# ==============================================================================
# MACHINE LEARNING PROJECT - MODELING PHASE
# Author: Your ML Professor
# Task: Used car price prediction (Regression)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time 

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# Added MAPE for business evaluation (as per syllabus)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Plot settings
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==============================================================================
# 1. DATA LOADING AND PREPARATION
# ==============================================================================
print("Loading cleaned dataset...")
try:
    df = pd.read_csv('cleaned_car_data.csv')
    print(f"Dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns.\n")
except FileNotFoundError:
    print("ERROR: Ensure that 'cleaned_car_data.csv' is in the same folder as the script.")
    exit()

X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================================================================
# 2. PREPROCESSING PIPELINE
# ==============================================================================
numeric_features = ['year', 'tax', 'enginesize', 'km_per_litre', 'mileage_km']
categorical_features = ['model', 'transmission', 'fueltype', 'brand']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ==============================================================================
# 3. MODEL DEFINITION AND TRAINING
# ==============================================================================
models = {
    "Linear Regression (Baseline)": LinearRegression(),
    "Ridge Regression (L2)": Ridge(alpha=1.0),
    "Lasso Regression (L1)": Lasso(alpha=0.1, max_iter=5000), 
    "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1),
    "Gradient Boosting (GBM)": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

def evaluate_model(y_true, y_pred, model_name, exec_time):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) # Added Business Metric
    
    print(f"--- {model_name} ---")
    print(f"Training Time: {exec_time:.2f} seconds")
    print(f"MAE:  £{mae:.2f}")
    print(f"RMSE: £{rmse:.2f}")
    print(f"MAPE: {mape * 100:.2f}% (Average percentage error)")
    print(f"R2:   {r2:.4f}\n")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

results = {}

print("Starting model training. Please wait, tree-based models might take some time...\n")

for name, model in models.items():
    print(f"-> Currently training: {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    end_time = time.time()
    
    y_pred = pipeline.predict(X_test)
    results[name] = evaluate_model(y_test, y_pred, name, end_time - start_time)

# ==============================================================================
# 4. CROSS-VALIDATION ON BEST MODEL (Random Forest)
# ==============================================================================
print("-> Performing 3-Fold Cross-Validation on Random Forest to verify stability...")
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', RandomForestRegressor(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1))])

cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
print(f"Cross-Validation R2 Scores: {cv_scores}")
print(f"Average CV R2 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")

# ==============================================================================
# 5. INTERPRETABILITY AND FEATURE IMPORTANCE (Random Forest)
# ==============================================================================
print("Extracting feature importance...")

best_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', RandomForestRegressor(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1))])
best_pipeline.fit(X_train, y_train)

final_rf = best_pipeline.named_steps['regressor']
final_prep = best_pipeline.named_steps['preprocessor']

num_names = numeric_features
cat_names = final_prep.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([num_names, cat_names])

feature_importances = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': final_rf.feature_importances_
})

top_features = feature_importances.sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=top_features, hue='Feature', palette='magma', legend=False)
plt.title('Top 15 Most Important Features for Price Prediction')
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'.")

# ==============================================================================
# 6. PREDICTIONS vs ACTUAL VALUES PLOT
# ==============================================================================
y_pred_best = best_pipeline.predict(X_test)

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_best, alpha=0.3, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Values (Random Forest)')
plt.xlabel('Actual Price (£)')
plt.ylabel('Predicted Price (£)')

# Aggiungiamo un testo esplicativo sul limite del modello nel grafico
plt.text(y.min(), y.max() * 0.9, 
         f"Overall MAPE: {mean_absolute_percentage_error(y_test, y_pred_best)*100:.2f}%\n"
         "Note: Higher percentage error expected on low-end cars.", 
         fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
print("Actual vs predicted plot saved as 'actual_vs_predicted.png'.")
print("\nPipeline successfully completed! Excellent work.")