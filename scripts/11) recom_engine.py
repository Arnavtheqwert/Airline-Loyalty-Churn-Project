import pandas as pd

# Load dataset
data = pd.read_csv("engineered_loyalty_data.csv")

# Define CLV segments
data['CLV_Segment'] = pd.qcut(data['CLV'], q=3, labels=['Low', 'Medium', 'High'])

# Recommendation engine rules
def recommend_action(row):
    if row['CLV_Segment'] == 'High' and row['Strict_Churn'] == 1:
        if row['Province'] == 'British Columbia' or row['City'] == 'Calgary':
            return "High-value churner in BC/Calgary → Concierge outreach + localized perks"
        else:
            return "High-value churner → Concierge outreach + tier upgrade"
    elif row['CLV_Segment'] == 'Medium' and row['Strict_Churn'] == 1:
        return "Medium-value churner → Bundle offers + loyalty progression"
    elif row['CLV_Segment'] == 'Low' and row['Strict_Churn'] == 1:
        return "Low-value churner → Discounts + gamification"
    else:
        return "Non-churner → Maintain engagement"

data['Recommendation'] = data.apply(recommend_action, axis=1)

# Preview recommendations
print(data[['Loyalty Number', 'CLV_Segment', 'Strict_Churn', 'Province', 'City', 'Recommendation']].head())
