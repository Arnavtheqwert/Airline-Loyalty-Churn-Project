import pandas as pd

flights = pd.read_csv("Customer Flight Activity.csv")
loyalty = pd.read_csv("Customer Loyalty History.csv")

merged = pd.merge(flights, loyalty, on="Loyalty Number", how="inner")

mask = merged["Points Redeemed"] > merged["Points Accumulated"]
merged.loc[mask, "Points Redeemed"] = merged.loc[mask, "Points Accumulated"]

mask = (merged["Cancellation Year"].notna()) & (merged["Total Flights"] > 0)
merged.loc[mask, ["Cancellation Year","Cancellation Month"]] = None

merged["Redeem_Without_Flights_Flag"] = (
    (merged["Total Flights"] == 0) & (merged["Points Redeemed"] > 0)
).astype(int)

merged["Invalid_Distance_Flag"] = (
    (merged["Total Flights"] > 0) & (merged["Distance"] <= 0)
).astype(int)

mask = (merged["Cancellation Year"].notna()) & (merged["Cancellation Year"] < merged["Enrollment Year"])
merged.loc[mask, ["Cancellation Year","Cancellation Month"]] = None

merged["Salary"] = merged.groupby("Education")["Salary"].transform(
    lambda x: x.fillna(x.median())
)
if merged["Education"].isnull().any():
    mode_value = merged["Education"].mode()[0]
    merged["Education"].fillna(mode_value, inplace=True)

merged["Missing_Demo_Flag"] = merged[["Salary","Education","Gender","Marital Status"]].isnull().any(axis=1).astype(int)

merged.to_csv("cleaned_loyalty_data.csv", index=False)
print("Cleaning complete. Saved as cleaned_loyalty_data.csv")
