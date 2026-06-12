import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load engineered dataset (from Day 2)
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

# Final check for NaNs
print("Remaining NaNs per column:\n", X.isna().sum())

# Align target with X
y = y.loc[X.index]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
