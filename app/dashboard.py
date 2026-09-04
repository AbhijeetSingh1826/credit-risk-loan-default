import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("💳 Credit Risk & Loan Default Analytics")
st.markdown("Portfolio-level risk analysis with expected loss modeling")

# --- Load results ---
@st.cache_data
def load_data():
    return pd.read_csv("data/risk_analysis_results.csv")

df = load_data()

# --- Top-level portfolio metrics ---
col1, col2, col3, col4 = st.columns(4)

total_exposure = df['loan_amount_actual'].sum()
total_expected_loss = df['expected_loss'].sum()
loss_pct = (total_expected_loss / total_exposure) * 100
actual_default_rate = df['actual_default'].mean() * 100

col1.metric("Total Exposure", f"${total_exposure/1e9:.2f}B")
col2.metric("Total Expected Loss", f"${total_expected_loss/1e6:.1f}M")
col3.metric("Expected Loss %", f"{loss_pct:.2f}%")
col4.metric("Actual Default Rate", f"{actual_default_rate:.1f}%")

st.divider()

# --- Risk tier breakdown ---
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Portfolio Distribution by Risk Tier")
    tier_counts = df['risk_tier_calibrated'].value_counts().reindex(
        ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    )
    fig = px.bar(x=tier_counts.index, y=tier_counts.values,
                 color=tier_counts.index,
                 color_discrete_map={'Low Risk': '#51cf66', 'Medium Risk': '#ffd43b',
                                      'High Risk': '#ffa94d', 'Very High Risk': '#ff4b4b'},
                 labels={'x': 'Risk Tier', 'y': 'Number of Loans'})
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Expected Loss by Risk Tier")
    loss_by_tier = df.groupby('risk_tier_calibrated')['expected_loss'].sum().reindex(
        ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    )
    fig2 = px.bar(x=loss_by_tier.index, y=loss_by_tier.values,
                  color=loss_by_tier.index,
                  color_discrete_map={'Low Risk': '#51cf66', 'Medium Risk': '#ffd43b',
                                       'High Risk': '#ffa94d', 'Very High Risk': '#ff4b4b'},
                  labels={'x': 'Risk Tier', 'y': 'Total Expected Loss ($)'})
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- Calibration check visualization ---
st.subheader("Model Calibration: Predicted vs Actual Default Rate")
calib = df.groupby('risk_tier_calibrated').agg(
    predicted=('default_probability_calibrated', 'mean'),
    actual=('actual_default', 'mean')
).reindex(['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']).reset_index()

fig3 = px.bar(calib, x='risk_tier_calibrated', y=['predicted', 'actual'],
              barmode='group',
              labels={'value': 'Default Rate', 'risk_tier_calibrated': 'Risk Tier'})
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# --- Borrower-level explorer ---
st.subheader("Borrower Risk Explorer")
idx = st.slider("Select a loan from the portfolio", 0, len(df)-1, 0)
selected = df.iloc[idx]

col_x, col_y, col_z = st.columns(3)
col_x.metric("Loan Amount", f"${selected['loan_amount_actual']:,.0f}")
col_y.metric("Default Probability", f"{selected['default_probability_calibrated']*100:.1f}%")
col_z.metric("Risk Tier", selected['risk_tier_calibrated'])

st.write(f"**Expected Loss:** ${selected['expected_loss']:,.2f}")
st.write(f"**Actual Outcome:** {'Defaulted' if selected['actual_default'] == 1 else 'Fully Paid'}")