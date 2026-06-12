import pandas as pd

# Load cleaned dataset from Day 1
data = pd.read_csv("cleaned_loyalty_data.csv")

# Flight frequency: flights per year since enrollment
data["Flight_Frequency"] = data["Total Flights"] / (
    (2026 - data["Enrollment Year"]).clip(lower=1)
)

# Redemption ratio: redeemed ÷ accumulated points
data["Redemption_Ratio"] = (
    data["Points Redeemed"] / data["Points Accumulated"].replace(0, 1)
)

# Season: enrollment month binned into Winter/Spring/Summer/Fall
data["Season"] = pd.cut(
    data["Enrollment Month"],
    bins=[0, 3, 6, 9, 12],
    labels=["Winter", "Spring", "Summer", "Fall"]
)

# Tier encoding from Loyalty Card
tier_map = {"Blue": 1, "Silver": 2, "Gold": 3, "Platinum": 4}
data["Tier_Code"] = data["Loyalty Card"].map(tier_map)

# CLV segmentation: binary based on median CLV
median_clv = data["CLV"].median()
data["CLV_Segment"] = (data["CLV"] >= median_clv).astype(int)

# Strict churn: cancellation date exists
data["Strict_Churn"] = data["Cancellation Year"].notna().astype(int)

# Behavioral churn: inactivity (no flights, no accumulation, no redemption)
data["Behavioral_Churn"] = (
    (data["Total Flights"] == 0) &
    (data["Points Accumulated"] == 0) &
    (data["Points Redeemed"] == 0)
).astype(int)

# Save engineered dataset
data.to_csv("engineered_loyalty_data.csv", index=False)
print("✅ Feature engineering complete. Saved as engineered_loyalty_data.csv")
