import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Load engineered dataset
data = pd.read_csv("engineered_loyalty_data.csv")

# Target column
y = data["Strict_Churn"]   # or use Behavioral_Churn

# Drop target + problematic columns
X = data.drop(columns=[
    "Strict_Churn", "Behavioral_Churn",
    "Cancellation Year", "Cancellation Month", "Tier_Code"
])

# Encode categorical variables
cat_cols = ["Gender", "Education", "Marital Status", "Season"]
for col in cat_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Identify numeric columns
num_cols = ["Salary", "Distance", "Flight_Frequency", "Redemption_Ratio",
            "Points Accumulated", "Points Redeemed", "CLV", "Total Flights"]

# Impute missing values
num_imputer = SimpleImputer(strategy="median")
X[num_cols] = num_imputer.fit_transform(X[num_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# Drop any remaining non-numeric columns
X = X.select_dtypes(include=["int64", "float64"])

# Scale numeric features
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# Random Forest Hyperparameter Tuning
# -------------------------------
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

print("Best Random Forest Params:", grid_rf.best_params_)

# Evaluate tuned Random Forest
y_pred_rf = best_rf.predict(X_test)
print("\nRandom Forest Tuned Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Feature importance (Random Forest)
importances = best_rf.feature_importances_
feat_names = X_train.columns
sorted_idx = importances.argsort()[::-1]
print("\nTop 10 Random Forest Features:")
for i in range(10):
    print(feat_names[sorted_idx[i]], importances[sorted_idx[i]])

# -------------------------------
# XGBoost Hyperparameter Tuning
# -------------------------------
xgb = XGBClassifier(eval_metric="logloss", random_state=42)
param_dist_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
random_xgb = RandomizedSearchCV(xgb, param_distributions=param_dist_xgb,
                                n_iter=10, cv=3, scoring='f1', n_jobs=-1)
random_xgb.fit(X_train, y_train)
best_xgb = random_xgb.best_estimator_

print("\nBest XGBoost Params:", random_xgb.best_params_)

# Evaluate tuned XGBoost
y_pred_xgb = best_xgb.predict(X_test)
print("\nXGBoost Tuned Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("F1 Score:", f1_score(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# Feature importance (XGBoost)
xgb_importances = best_xgb.feature_importances_
sorted_idx = xgb_importances.argsort()[::-1]
print("\nTop 10 XGBoost Features:")
for i in range(10):
    print(feat_names[sorted_idx[i]], xgb_importances[sorted_idx[i]])
