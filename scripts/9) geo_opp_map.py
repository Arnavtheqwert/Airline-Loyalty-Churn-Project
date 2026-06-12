import pandas as pd

# Load engineered dataset
data = pd.read_csv("engineered_loyalty_data.csv")

# Country-level profiling
country_profile = data.groupby(['Country', 'Strict_Churn']).agg({
    'CLV': 'mean',
    'Loyalty Number': 'count'
}).rename(columns={'Loyalty Number': 'Customer_Count'}).reset_index()

# Calculate churn rate per country
total_counts = data.groupby('Country')['Loyalty Number'].count()
country_profile['Churn_Rate'] = country_profile.apply(
    lambda row: row['Customer_Count'] / total_counts[row['Country']], axis=1
)

print("\nCountry Profile:")
print(country_profile.head())

# Province-level profiling
province_profile = data.groupby(['Province', 'Strict_Churn']).agg({
    'CLV': 'mean',
    'Loyalty Number': 'count'
}).rename(columns={'Loyalty Number': 'Customer_Count'}).reset_index()

total_counts_prov = data.groupby('Province')['Loyalty Number'].count()
province_profile['Churn_Rate'] = province_profile.apply(
    lambda row: row['Customer_Count'] / total_counts_prov[row['Province']], axis=1
)

print("\nProvince Profile:")
print(province_profile.head())

# City-level profiling
city_profile = data.groupby(['City', 'Strict_Churn']).agg({
    'CLV': 'mean',
    'Loyalty Number': 'count'
}).rename(columns={'Loyalty Number': 'Customer_Count'}).reset_index()

total_counts_city = data.groupby('City')['Loyalty Number'].count()
city_profile['Churn_Rate'] = city_profile.apply(
    lambda row: row['Customer_Count'] / total_counts_city[row['City']], axis=1
)

print("\nCity Profile:")
print(city_profile.head())
