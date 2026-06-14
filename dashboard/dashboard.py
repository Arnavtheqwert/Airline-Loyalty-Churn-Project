import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Airline Loyalty Dashboard", layout="wide")

# Dashboard header
st.title("Airline Loyalty Dashboard")
st.markdown("### Executive Insights into Customer Retention")

# Load dataset
data = pd.read_csv("engineered_loyalty_data.csv")
data['CLV_Segment'] = pd.qcut(data['CLV'], q=3, labels=['Low', 'Medium', 'High'])

# Sidebar filters
st.sidebar.header("Filters")
segment_filter = st.sidebar.selectbox("Select CLV Segment", ["All", "Low", "Medium", "High"])
province_filter = st.sidebar.selectbox("Select Province", ["All"] + list(data['Province'].unique()))

filtered_data = data.copy()
if segment_filter != "All":
    filtered_data = filtered_data[filtered_data['CLV_Segment'] == segment_filter]
if province_filter != "All":
    filtered_data = filtered_data[filtered_data['Province'] == province_filter]

# --- KPI Summary Panel ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", f"{filtered_data['Loyalty Number'].nunique():,}")
col2.metric("Churn Rate", f"{(filtered_data['Strict_Churn'].mean()*100):.1f}%")
col3.metric("Average CLV", f"${filtered_data['CLV'].mean():,.0f}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Churn Analysis", "📈 Customer Segments", "🌍 Geographic Insights", "🎯 Retention Strategies"])

# --- Tab 1: Churn Analysis ---
with tab1:
    st.subheader("Customer Retention Snapshot")
    churn_counts = filtered_data.groupby('Strict_Churn')['Loyalty Number'].nunique()
    churn_percent = churn_counts / churn_counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(
        x=churn_counts.index,
        y=churn_counts.values,
        palette=["#1f77b4", "#ff7f0e"],
        ax=ax,
        width=0.6
    )

    ax.set_title("Churn vs Non-Churn Customers", fontsize=14, fontweight='bold')
    ax.set_xlabel("Churn Risk (0 = Active, 1 = Churned)", fontsize=12)
    ax.set_ylabel("Number of Unique Customers", fontsize=12)

    # Center labels inside bars
    for i, (count, pct) in enumerate(zip(churn_counts.values, churn_percent.values)):
        ax.text(i, count/2, f"{count} ({pct:.1f}%)",
                ha='center', va='center', color='white',
                fontsize=10, fontweight='bold')

    st.pyplot(fig)

# --- Tab 2: Customer Segments ---
with tab2:
    st.subheader("Customer Segment Profiles")
    segment_profile = filtered_data.groupby(['CLV_Segment','Strict_Churn']).agg({
        'CLV':'mean','Loyalty Number':'nunique'
    }).reset_index()

    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.barplot(
        data=segment_profile,
        x='CLV_Segment', y='Loyalty Number',
        hue='Strict_Churn',
        palette=["#1f77b4", "#ff7f0e"],
        ax=ax2, width=0.6
    )
    ax2.set_title("Unique Customers by Segment & Churn", fontsize=14, fontweight='bold')
    ax2.set_xlabel("CLV Segment", fontsize=12)
    ax2.set_ylabel("Number of Unique Customers", fontsize=12)

    # Labels on top of bars in black
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:
            ax2.text(
                p.get_x() + p.get_width()/2,
                height + 20,
                f"{int(height)}",
                ha='center', va='bottom',
                color='black', fontsize=9, fontweight='bold'
            )

    st.pyplot(fig2)
    st.dataframe(segment_profile)
    st.download_button("Export Segment Profiles", segment_profile.to_csv(index=False), "segment_profile.csv")

# --- Tab 3: Geographic Insights ---
with tab3:
    st.subheader("Regional Retention Opportunities")
    geo_profile = filtered_data.groupby(['Province','Strict_Churn']).agg({
        'CLV':'mean','Loyalty Number':'nunique'
    }).reset_index()

    fig3, ax3 = plt.subplots(figsize=(8,5))
    sns.barplot(
        data=geo_profile,
        x='Province', y='Loyalty Number',
        hue='Strict_Churn',
        palette=["#1f77b4", "#ff7f0e"],
        ax=ax3, width=0.6
    )
    ax3.set_title("Unique Customers by Province & Churn", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Province", fontsize=12)
    ax3.set_ylabel("Number of Unique Customers", fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', linestyle='--', alpha=0.6)
    ax3.legend(title="Churn Status", loc='upper right')

    # Labels on top of bars in black
    for p in ax3.patches:
        height = p.get_height()
        if height > 0:
            ax3.text(
                p.get_x() + p.get_width()/2,
                height + 20,
                f"{int(height)}",
                ha='center', va='bottom',
                color='black', fontsize=9, fontweight='bold'
            )

    st.pyplot(fig3)
    st.dataframe(geo_profile)
    st.download_button("Export Geographic Profiles", geo_profile.to_csv(index=False), "geo_profile.csv")

# --- Tab 4: Retention Strategies ---
with tab4:
    st.subheader("Retention Strategies")
    def recommend_action(row):
        if row['CLV_Segment'] == 'High' and row['Strict_Churn'] == 1:
            return "Concierge outreach + tier upgrade"
        elif row['CLV_Segment'] == 'Medium' and row['Strict_Churn'] == 1:
            return "Bundle offers + loyalty progression"
        elif row['CLV_Segment'] == 'Low' and row['Strict_Churn'] == 1:
            return "Discounts + gamification"
        else:
            return "Maintain engagement"

    filtered_data['Recommendation'] = filtered_data.apply(recommend_action, axis=1)
    recommendations = filtered_data[['Loyalty Number','CLV_Segment','Strict_Churn','Province','City','Recommendation']].drop_duplicates()

    st.dataframe(recommendations.head(20))
    st.download_button("Export Recommendations", recommendations.to_csv(index=False), "recommendations.csv")
