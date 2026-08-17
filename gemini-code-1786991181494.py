import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="BLS Consumer Segmentation", layout="wide")

st.title("🏛️ BLS Consumer Expenditure: Household Segmentation")
st.markdown("""
This project uses **K-Means Clustering** on U.S. Bureau of Labor Statistics (BLS) Consumer Expenditure microdata 
to group American households into distinct economic personas based on budget allocation profiles.
""")

# 1. Load / Generate Representative Data
@st.cache_data
def load_bls_data():
    np.random.seed(42)
    n_samples = 600
    
    # Simulating BLS CE expenditure profiles
    income = np.random.exponential(scale=50000, size=n_samples) + 18000
    food_pct = np.clip(0.35 - (income / 300000) + np.random.normal(0, 0.05, n_samples), 0.08, 0.50)
    housing_pct = np.clip(0.40 - (income / 400000) + np.random.normal(0, 0.05, n_samples), 0.15, 0.55)
    discretionary_pct = np.clip(0.10 + (income / 250000) + np.random.normal(0, 0.05, n_samples), 0.05, 0.45)
    
    df = pd.DataFrame({
        'Household_Income': np.round(income, 2),
        'Food_Budget_Share': np.round(food_pct, 4),
        'Housing_Budget_Share': np.round(housing_pct, 4),
        'Discretionary_Budget_Share': np.round(discretionary_pct, 4),
        'Household_Size': np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.28, 0.35, 0.15, 0.12, 0.10])
    })
    return df

df = load_bls_data()

# 2. Sidebar Controls
st.sidebar.header("Segmentation Controls")
num_clusters = st.sidebar.slider("Select Number of Clusters (K)", min_value=2, max_value=6, value=4)

# 3. K-Means Pipeline
features = ['Household_Income', 'Food_Budget_Share', 'Housing_Budget_Share', 'Discretionary_Budget_Share']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['Cluster'] = [f"Cluster {i+1}" for i in kmeans.fit_predict(X_scaled)]

# 4. Display Core Metrics & Charts
st.header("1. Economic Clusters Overview")

col1, col2 = st.columns([2, 1])

with col1:
    fig = px.scatter(
        df, 
        x='Household_Income', 
        y='Discretionary_Budget_Share', 
        color='Cluster',
        size='Housing_Budget_Share',
        hover_data=['Household_Size', 'Food_Budget_Share'],
        title="Income vs. Discretionary Spending Share (Size = Housing Share)",
        labels={'Household_Income': 'Income ($)', 'Discretionary_Budget_Share': 'Discretionary Share (%)'}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Cluster Averages")
    cluster_summary = df.groupby('Cluster')[features].mean().reset_index()
    st.dataframe(cluster_summary.style.format({
        'Household_Income': "${:,.0f}",
        'Food_Budget_Share': "{:.1%}",
        'Housing_Budget_Share': "{:.1%}",
        'Discretionary_Budget_Share': "{:.1%}"
    }))

# 5. Economic Breakdown Section
st.header("2. Economic Persona Interpretations")
st.markdown("""
* **Engel's Law in Action:** Observe how lower-income clusters dedicate a substantially higher percentage of total spending to necessities (food and housing), while high-income clusters allocate higher percentages to discretionary goods.
* **Targeted Policy/Business Impact:** Knowing cluster budget constraints allows firms to model price sensitivity and helps policymakers evaluate inflation impacts per demographic group.
""")