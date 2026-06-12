import pandas as pd

# Load engineered dataset
data = pd.read_csv("engineered_loyalty_data.csv")

# Define CLV buckets
data['CLV_Segment'] = pd.qcut(data['CLV'], q=3, labels=['Low', 'Medium', 'High'])

# Group by CLV segment and churn
segment_profile = data.groupby(['CLV_Segment', 'Strict_Churn']).agg({
    'CLV': 'mean',
    'Loyalty Number': 'count'
}).rename(columns={'Loyalty Number': 'Customer_Count'}).reset_index()

# Calculate churn rate within each CLV segment
total_counts = data.groupby('CLV_Segment')['Loyalty Number'].count()
segment_profile['Churn_Rate'] = segment_profile.apply(
    lambda row: row['Customer_Count'] / total_counts[row['CLV_Segment']], axis=1
)

print(segment_profile)

# Optional: add demographic profiling
demo_profile = data.groupby(['CLV_Segment', 'Gender', 'Education']).agg({
    'CLV': 'mean',
    'Strict_Churn': 'mean'
}).reset_index()

print("\nDemographic Profile:")
print(demo_profile.head())
