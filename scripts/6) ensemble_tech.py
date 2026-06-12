import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -------------------------------
# Load engineered dataset
# -------------------------------
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

# Impute missing values
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
# Fast Random Forest (lighter model)
# -------------------------------
best_rf = RandomForestClassifier(
    n_estimators=100,   # reduced from 300
    max_depth=10,       # limit depth
    random_state=42,
    n_jobs=-1
)
best_rf.fit(X_train, y_train)

# -------------------------------
# Fast XGBoost (lighter model)
# -------------------------------
best_xgb = XGBClassifier(
    n_estimators=100,   # reduced from 300
    learning_rate=0.1,
    max_depth=3,        # shallower trees
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
best_xgb.fit(X_train, y_train)

# -------------------------------
# Voting Classifier (soft voting)
# -------------------------------
voting_clf = VotingClassifier(
    estimators=[('rf', best_rf), ('xgb', best_xgb)],
    voting='soft',
    n_jobs=-1
)
voting_clf.fit(X_train, y_train)
y_pred_vote = voting_clf.predict(X_test)

print("\nVoting Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_vote))
print("Precision:", precision_score(y_test, y_pred_vote))
print("Recall:", recall_score(y_test, y_pred_vote))
print("F1 Score:", f1_score(y_test, y_pred_vote))
print("ROC AUC:", roc_auc_score(y_test, y_pred_vote))

# -------------------------------
# Stacking Classifier (simplified, no passthrough)
# -------------------------------
stack_clf = StackingClassifier(
    estimators=[('rf', best_rf), ('xgb', best_xgb)],
    final_estimator=LogisticRegression(max_iter=500),
    passthrough=False,   # faster
    n_jobs=-1
)
stack_clf.fit(X_train, y_train)
y_pred_stack = stack_clf.predict(X_test)

print("\nStacking Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_stack))
print("Precision:", precision_score(y_test, y_pred_stack))
print("Recall:", recall_score(y_test, y_pred_stack))
print("F1 Score:", f1_score(y_test, y_pred_stack))
print("ROC AUC:", roc_auc_score(y_test, y_pred_stack))
