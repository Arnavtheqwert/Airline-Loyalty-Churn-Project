import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# Load engineered dataset
data = pd.read_csv("engineered_loyalty_data.csv")
y = data["Strict_Churn"]

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

num_cols = ["Salary", "Distance", "Flight_Frequency", "Redemption_Ratio",
            "Points Accumulated", "Points Redeemed", "CLV", "Total Flights"]

num_imputer = SimpleImputer(strategy="median")
X[num_cols] = num_imputer.fit_transform(X[num_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

X = X.select_dtypes(include=["int64", "float64"])
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# Fast Random Forest (fewer trees)
# -------------------------------
best_rf = RandomForestClassifier(
    n_estimators=100,   # reduced from 300
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1           # parallel processing
)
best_rf.fit(X_train, y_train)

# -------------------------------
# Fast XGBoost (fewer rounds)
# -------------------------------
best_xgb = XGBClassifier(
    n_estimators=100,   # reduced from 300
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1           # parallel processing
)
best_xgb.fit(X_train, y_train)

# -------------------------------
# Quick 3-fold CV for speed
# -------------------------------
cv_f1_rf = cross_val_score(best_rf, X, y, cv=3, scoring='f1', n_jobs=-1)
cv_f1_xgb = cross_val_score(best_xgb, X, y, cv=3, scoring='f1', n_jobs=-1)

print("Random Forest CV F1:", cv_f1_rf.mean(), "+/-", cv_f1_rf.std())
print("XGBoost CV F1:", cv_f1_xgb.mean(), "+/-", cv_f1_xgb.std())

# -------------------------------
# Overfitting check (train vs test F1)
# -------------------------------
train_pred_rf = best_rf.predict(X_train)
test_pred_rf = best_rf.predict(X_test)
print("\nRandom Forest Train F1:", f1_score(y_train, train_pred_rf))
print("Random Forest Test F1:", f1_score(y_test, test_pred_rf))

train_pred_xgb = best_xgb.predict(X_train)
test_pred_xgb = best_xgb.predict(X_test)
print("\nXGBoost Train F1:", f1_score(y_train, train_pred_xgb))
print("XGBoost Test F1:", f1_score(y_test, test_pred_xgb))
