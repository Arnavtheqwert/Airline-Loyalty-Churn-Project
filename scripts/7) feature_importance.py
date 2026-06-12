import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

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
# Permutation Importance (quick run)
# -------------------------------
perm_importance = permutation_importance(
    best_rf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
)

print("\nTop 10 Permutation Importance Features:")
for i in perm_importance.importances_mean.argsort()[::-1][:10]:
    print(f"{X.columns[i]}: {perm_importance.importances_mean[i]:.4f}")
