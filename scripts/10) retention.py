import pandas as pd

# Load dataset
data = pd.read_csv("engineered_loyalty_data.csv")

# Define CLV segments
data['CLV_Segment'] = pd.qcut(data['CLV'], q=3, labels=['Low', 'Medium', 'High'])

# Strategy mapping
def assign_strategy(row):
    if row['CLV_Segment'] == 'High' and row['Strict_Churn'] == 1:
        return "Concierge outreach + tier upgrade"
    elif row['CLV_Segment'] == 'Medium' and row['Strict_Churn'] == 1:
        return "Bundle offers + loyalty progression"
    elif row['CLV_Segment'] == 'Low' and row['Strict_Churn'] == 1:
        return "Discounted fares + gamification"
    else:
        return "Maintain engagement"

data['Retention_Strategy'] = data.apply(assign_strategy, axis=1)

# Preview strategy assignments
print(data[['Loyalty Number', 'CLV_Segment', 'Strict_Churn', 'Retention_Strategy']].head())
