import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Airline Loyalty Dashboard", layout="wide")

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

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Churn Risk", "📈 Segment Profiles", "🌍 Geographic Map", "🎯 Recommendations"])

with tab1:
    st.subheader("Churn Risk Distribution")
    churn_counts = filtered_data['Strict_Churn'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=churn_counts.index, y=churn_counts.values, palette=["blue","red"], ax=ax)
    ax.set_title("Churn vs Non-Churn")
    st.pyplot(fig)

with tab2:
    st.subheader("Segment Profiles")
    segment_profile = filtered_data.groupby(['CLV_Segment','Strict_Churn']).agg({
        'CLV':'mean','Loyalty Number':'count'
    }).reset_index()
    st.dataframe(segment_profile)
    st.download_button("Download Segment Profile CSV", segment_profile.to_csv(index=False), "segment_profile.csv")

with tab3:
    st.subheader("Geographic Opportunity Map")
    geo_profile = filtered_data.groupby(['Province','Strict_Churn']).agg({
        'CLV':'mean','Loyalty Number':'count'
    }).reset_index()
    st.dataframe(geo_profile)
    st.download_button("Download Geographic Profile CSV", geo_profile.to_csv(index=False), "geo_profile.csv")

with tab4:
    st.subheader("Retention Recommendations")
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
    st.dataframe(filtered_data[['Loyalty Number','CLV_Segment','Strict_Churn','Province','City','Recommendation']].head(20))
    st.download_button("Download Recommendations CSV", filtered_data[['Loyalty Number','CLV_Segment','Strict_Churn','Province','City','Recommendation']].to_csv(index=False), "recommendations.csv")
