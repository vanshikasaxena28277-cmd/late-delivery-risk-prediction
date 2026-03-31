# ============================================================
# APL Logistics - Late Delivery Risk Dashboard
# Streamlit Web Application
# ============================================================
# HOW TO RUN:
#   1. First run: python model_training.py  (to create model files)
#   2. Then run:  streamlit run app.py
# ============================================================
 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
 
from sklearn.preprocessing import LabelEncoder
 
 
# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="APL Logistics - Delivery Risk Dashboard",
    page_icon="🚚",
    layout="wide"
)
 
st.title("🚚 APL Logistics — Late Delivery Risk Dashboard")
st.markdown("**Predict and manage shipment delay risk before orders are shipped.**")
st.divider()
 
 
# ============================================================
# LOAD MODEL AND DATA
# ============================================================
@st.cache_resource
def load_model():
    """Load the trained model, scaler, and feature list."""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('feature_names.pkl')
        return model, scaler, features
    except FileNotFoundError:
        return None, None, None
 
 
@st.cache_data
def load_data():
    """Load the raw supply chain dataset."""
    try:
        df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='latin-1')
        return df
    except FileNotFoundError:
        return None
 
 
model, scaler, feature_names = load_model()
df_raw = load_data()
 
# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.header("🔧 Filters")
 
# Shipping mode filter
shipping_modes = ['All', 'Standard Class', 'First Class', 'Second Class', 'Same Day']
selected_mode = st.sidebar.selectbox("Shipping Mode", shipping_modes)
 
# Market filter
markets = ['All', 'LATAM', 'Europe', 'Pacific Asia', 'USCA', 'Africa']
selected_market = st.sidebar.selectbox("Market / Region", markets)
 
# Customer segment filter
segments = ['All', 'Consumer', 'Corporate', 'Home Office']
selected_segment = st.sidebar.selectbox("Customer Segment", segments)
 
# Risk threshold slider
risk_threshold = st.sidebar.slider(
    "High-Risk Threshold", min_value=0.3, max_value=0.9, value=0.6, step=0.05,
    help="Orders above this probability are flagged as High Risk"
)
 
st.sidebar.divider()
st.sidebar.info("Train the model first by running:\n`python model_training.py`")
 
 
# ============================================================
# HELPER: ASSIGN RISK CATEGORY
# ============================================================
def get_risk_category(prob):
    if prob >= risk_threshold:
        return "🔴 High Risk"
    elif prob >= 0.4:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"
 
 
# ============================================================
# TAB LAYOUT
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Risk Overview",
    "🔍 Predict Single Order",
    "🗺️ Region & Mode Analysis",
    "⚡ Action Panel"
])
 
 
# ============================================================
# TAB 1: RISK OVERVIEW
# ============================================================
with tab1:
    st.header("Overall Delivery Risk Overview")
 
    if df_raw is None:
        st.error("Dataset not found. Please add DataCoSupplyChainDataset.csv.")
    elif model is None:
        st.warning("Model not trained yet. Run `python model_training.py` first.")
    else:
        df = df_raw.copy()
 
        # Apply sidebar filters to raw data
        if selected_mode != 'All' and 'Shipping Mode' in df.columns:
            df = df[df['Shipping Mode'] == selected_mode]
        if selected_market != 'All' and 'Market' in df.columns:
            df = df[df['Market'] == selected_market]
        if selected_segment != 'All' and 'Customer Segment' in df.columns:
            df = df[df['Customer Segment'] == selected_segment]
 
        total = len(df)
        late_count = df['Late_delivery_risk'].sum() if 'Late_delivery_risk' in df.columns else 0
        late_pct = late_count / total * 100 if total > 0 else 0
 
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Orders", f"{total:,}")
        with col2:
            st.metric("Late Delivery Risk", f"{late_count:,}", f"{late_pct:.1f}%")
        with col3:
            on_time = total - late_count
            st.metric("On-Time Orders", f"{on_time:,}")
        with col4:
            st.metric("Risk Rate", f"{late_pct:.1f}%")
 
        st.divider()
 
        col_left, col_right = st.columns(2)
 
        # Pie chart
        with col_left:
            st.subheader("Risk Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            counts = [on_time, late_count]
            labels = ['On Time', 'At Risk']
            colors = ['#2ecc71', '#e74c3c']
            ax.pie(counts, labels=labels, colors=colors,
                   autopct='%1.1f%%', startangle=90)
            ax.set_title("Late Delivery Risk Split")
            st.pyplot(fig)
            plt.close()
 
        # Risk by shipping mode
        with col_right:
            st.subheader("Risk by Shipping Mode")
            if 'Shipping Mode' in df.columns and 'Late_delivery_risk' in df.columns:
                mode_risk = df.groupby('Shipping Mode')['Late_delivery_risk'].mean() * 100
                fig, ax = plt.subplots(figsize=(5, 4))
                mode_risk.sort_values().plot(kind='barh', color='steelblue', ax=ax)
                ax.set_xlabel('Late Delivery Rate (%)')
                ax.set_title('Risk % by Shipping Mode')
                st.pyplot(fig)
                plt.close()
 
        # Feature importance chart (from saved file)
        if os.path.exists('plots/feature_importance.png'):
            st.subheader("Top Risk Drivers (Feature Importance)")
            st.image('plots/feature_importance.png')
 
        # Model comparison
        if os.path.exists('plots/model_comparison.png'):
            st.subheader("Model Performance Comparison")
            st.image('plots/model_comparison.png')
 
 
# ============================================================
# TAB 2: PREDICT SINGLE ORDER
# ============================================================
with tab2:
    st.header("Predict Risk for a Single Order")
 
    if model is None or scaler is None:
        st.warning("Model files not found. Please run `python model_training.py` first.")
    else:
        st.markdown("Fill in the order details below to get a late delivery risk prediction.")
 
        col1, col2, col3 = st.columns(3)
 
        with col1:
            shipping_mode_input = st.selectbox(
                "Shipping Mode",
                ['Standard Class', 'First Class', 'Second Class', 'Same Day']
            )
            days_real = st.number_input("Actual Shipping Days", min_value=0, max_value=60, value=4)
            days_scheduled = st.number_input("Scheduled Shipping Days", min_value=0, max_value=60, value=3)
 
        with col2:
            market_input = st.selectbox(
                "Market / Region",
                ['LATAM', 'Europe', 'Pacific Asia', 'USCA', 'Africa']
            )
            order_quantity = st.number_input("Order Quantity", min_value=1, max_value=100, value=2)
            discount_rate = st.slider("Discount Rate", 0.0, 1.0, 0.1, 0.01)
 
        with col3:
            customer_segment_input = st.selectbox(
                "Customer Segment",
                ['Consumer', 'Corporate', 'Home Office']
            )
            product_price = st.number_input("Product Price ($)", min_value=0.0, value=50.0)
            profit_ratio = st.slider("Profit Ratio", -1.0, 1.0, 0.2, 0.01)
 
        if st.button("🔮 Predict Risk", type="primary"):
            # Build input based on feature names
            # We'll create a row with defaults and override known fields
            input_dict = {feat: 0.0 for feat in feature_names}
 
            # Map what we know
            field_map = {
                'Days for shipping (real)': days_real,
                'Days for shipment (scheduled)': days_scheduled,
                'Order Item Quantity': order_quantity,
                'Order Item Discount Rate': discount_rate,
                'Order Item Product Price': product_price,
                'Order Item Profit Ratio': profit_ratio,
                'shipping_delay_gap': days_real - days_scheduled,
                'shipping_pressure_ratio': days_scheduled / (days_real + 1),
                'order_complexity': order_quantity * discount_rate,
                'low_profit_flag': 1 if profit_ratio < 0 else 0,
                'high_discount_flag': 1 if discount_rate > 0.2 else 0,
            }
 
            for k, v in field_map.items():
                if k in input_dict:
                    input_dict[k] = v
 
            input_df = pd.DataFrame([input_dict])
            input_scaled = scaler.transform(input_df)
            prob = model.predict_proba(input_scaled)[0][1]
            category = get_risk_category(prob)
 
            st.divider()
            st.subheader("Prediction Result")
 
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("Late Delivery Probability", f"{prob:.1%}")
            with col_res2:
                st.metric("Risk Category", category)
            with col_res3:
                confidence = abs(prob - 0.5) * 2
                st.metric("Confidence Score", f"{confidence:.1%}")
 
            # Risk bar
            fig, ax = plt.subplots(figsize=(8, 1.5))
            ax.barh(['Risk Level'], [prob], color='#e74c3c' if prob >= risk_threshold else '#f39c12' if prob >= 0.4 else '#2ecc71')
            ax.barh(['Risk Level'], [1 - prob], left=[prob], color='#ecf0f1')
            ax.axvline(x=risk_threshold, color='black', linestyle='--', linewidth=1.5, label=f'Threshold ({risk_threshold:.0%})')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.legend()
            ax.set_title(f'Risk Score: {prob:.1%}')
            st.pyplot(fig)
            plt.close()
 
            # Advice
            if prob >= risk_threshold:
                st.error(f"⚠️ HIGH RISK ORDER — Proactive action recommended: Consider expedited shipping or customer notification.")
            elif prob >= 0.4:
                st.warning("🟡 MEDIUM RISK — Monitor this order closely. Consider backup shipping route.")
            else:
                st.success("✅ LOW RISK — This order is unlikely to be delayed.")
 
 
# ============================================================
# TAB 3: REGION & MODE ANALYSIS
# ============================================================
with tab3:
    st.header("Risk Analysis by Region and Shipping Mode")
 
    if df_raw is None:
        st.error("Dataset not found.")
    else:
        df = df_raw.copy()
 
        col1, col2 = st.columns(2)
 
        # Region risk heatmap
        with col1:
            st.subheader("Risk Rate by Market")
            if 'Market' in df.columns and 'Late_delivery_risk' in df.columns:
                market_risk = (
                    df.groupby('Market')['Late_delivery_risk']
                    .mean()
                    .reset_index()
                    .rename(columns={'Late_delivery_risk': 'Risk Rate'})
                    .sort_values('Risk Rate', ascending=False)
                )
                market_risk['Risk Rate %'] = (market_risk['Risk Rate'] * 100).round(1)
 
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['#e74c3c' if r > 0.6 else '#f39c12' if r > 0.4 else '#2ecc71'
                          for r in market_risk['Risk Rate']]
                ax.bar(market_risk['Market'], market_risk['Risk Rate %'], color=colors)
                ax.set_ylabel('Late Delivery Rate (%)')
                ax.set_title('Risk Rate by Market Region')
                ax.set_xticklabels(market_risk['Market'], rotation=30, ha='right')
                for i, v in enumerate(market_risk['Risk Rate %']):
                    ax.text(i, v + 0.5, f'{v}%', ha='center', fontsize=9)
                st.pyplot(fig)
                plt.close()
 
                st.dataframe(market_risk[['Market', 'Risk Rate %']], use_container_width=True)
 
        # Shipping mode comparison
        with col2:
            st.subheader("Risk Rate by Shipping Mode")
            if 'Shipping Mode' in df.columns and 'Late_delivery_risk' in df.columns:
                mode_risk = (
                    df.groupby('Shipping Mode')['Late_delivery_risk']
                    .agg(['mean', 'count'])
                    .reset_index()
                    .rename(columns={'mean': 'Risk Rate', 'count': 'Total Orders'})
                )
                mode_risk['Risk Rate %'] = (mode_risk['Risk Rate'] * 100).round(1)
 
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['#e74c3c' if r > 0.6 else '#f39c12' if r > 0.4 else '#2ecc71'
                          for r in mode_risk['Risk Rate']]
                ax.bar(mode_risk['Shipping Mode'], mode_risk['Risk Rate %'], color=colors)
                ax.set_ylabel('Late Delivery Rate (%)')
                ax.set_title('Risk Rate by Shipping Mode')
                ax.set_xticklabels(mode_risk['Shipping Mode'], rotation=15, ha='right')
                for i, v in enumerate(mode_risk['Risk Rate %']):
                    ax.text(i, v + 0.5, f'{v}%', ha='center', fontsize=9)
                st.pyplot(fig)
                plt.close()
 
                st.dataframe(mode_risk[['Shipping Mode', 'Risk Rate %', 'Total Orders']],
                             use_container_width=True)
 
        # Heatmap: Market × Shipping Mode
        st.subheader("Heatmap: Risk by Market × Shipping Mode")
        if all(c in df.columns for c in ['Market', 'Shipping Mode', 'Late_delivery_risk']):
            pivot = df.pivot_table(
                values='Late_delivery_risk',
                index='Market',
                columns='Shipping Mode',
                aggfunc='mean'
            ) * 100
 
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                        ax=ax, linewidths=0.5)
            ax.set_title('Late Delivery Risk % — Market × Shipping Mode')
            st.pyplot(fig)
            plt.close()
 
 
# ============================================================
# TAB 4: ACTION PANEL
# ============================================================
with tab4:
    st.header("⚡ Operations Action Panel")
    st.markdown("Orders that need immediate attention based on risk prediction.")
 
    if df_raw is None or model is None:
        st.error("Dataset or model not available. Please set up both first.")
    else:
        df = df_raw.copy()
 
        # Apply sidebar filters
        if selected_mode != 'All' and 'Shipping Mode' in df.columns:
            df = df[df['Shipping Mode'] == selected_mode]
        if selected_market != 'All' and 'Market' in df.columns:
            df = df[df['Market'] == selected_market]
        if selected_segment != 'All' and 'Customer Segment' in df.columns:
            df = df[df['Customer Segment'] == selected_segment]
 
        # Take a sample for prediction (to keep it fast)
        sample_size = min(500, len(df))
        df_sample = df.sample(n=sample_size, random_state=42).copy()
 
        # Prepare features
        useless = ['Customer Fname', 'Customer Lname', 'Customer Street',
                   'Customer Zipcode', 'Customer Id', 'Order Customer Id',
                   'Product Name', 'Delivery Status',
                   'order date (DateOrders)', 'shipping date (DateOrders)',
                   'Late_delivery_risk']
        drop_cols = [c for c in useless if c in df_sample.columns]
        X_sample = df_sample.drop(columns=drop_cols, errors='ignore').copy()
 
        # Feature engineering
        if ('Days for shipping (real)' in X_sample.columns and
                'Days for shipment (scheduled)' in X_sample.columns):
            X_sample['shipping_delay_gap'] = (
                X_sample['Days for shipping (real)'] -
                X_sample['Days for shipment (scheduled)']
            )
            X_sample['shipping_pressure_ratio'] = (
                X_sample['Days for shipment (scheduled)'] /
                (X_sample['Days for shipping (real)'] + 1)
            )
        if ('Order Item Quantity' in X_sample.columns and
                'Order Item Discount Rate' in X_sample.columns):
            X_sample['order_complexity'] = (
                X_sample['Order Item Quantity'] * X_sample['Order Item Discount Rate']
            )
        if 'Order Item Profit Ratio' in X_sample.columns:
            X_sample['low_profit_flag'] = (X_sample['Order Item Profit Ratio'] < 0).astype(int)
        if 'Order Item Discount Rate' in X_sample.columns:
            X_sample['high_discount_flag'] = (X_sample['Order Item Discount Rate'] > 0.2).astype(int)
 
        # Encode categoricals
        le = LabelEncoder()
        for col in X_sample.select_dtypes(include='object').columns:
            X_sample[col] = le.fit_transform(X_sample[col].astype(str))
 
        # Align to training features
        for feat in feature_names:
            if feat not in X_sample.columns:
                X_sample[feat] = 0
        X_sample = X_sample[feature_names]
        X_sample.fillna(0, inplace=True)
 
        # Predict
        X_scaled_sample = scaler.transform(X_sample)
        probs = model.predict_proba(X_scaled_sample)[:, 1]
 
        df_sample = df_sample.reset_index(drop=True)
        df_sample['Risk Probability'] = probs
        df_sample['Risk Category'] = df_sample['Risk Probability'].apply(get_risk_category)
 
        # Filter high risk only
        high_risk = df_sample[df_sample['Risk Probability'] >= risk_threshold].copy()
        high_risk = high_risk.sort_values('Risk Probability', ascending=False)
 
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Orders Analyzed", f"{sample_size:,}")
        with col2:
            st.metric("High Risk Orders", f"{len(high_risk):,}", f"{len(high_risk)/sample_size*100:.1f}%")
        with col3:
            avg_prob = probs.mean()
            st.metric("Average Risk Probability", f"{avg_prob:.1%}")
 
        st.divider()
 
        # Display high-risk orders table
        st.subheader(f"🔴 High-Risk Orders (Threshold ≥ {risk_threshold:.0%})")
 
        display_cols = ['Risk Probability', 'Risk Category']
        for col in ['Order Country', 'Market', 'Shipping Mode', 'Customer Segment',
                    'Days for shipping (real)', 'Days for shipment (scheduled)',
                    'Order Item Product Price', 'Sales']:
            if col in high_risk.columns:
                display_cols.append(col)
 
        if len(high_risk) > 0:
            display_df = high_risk[display_cols].head(50).copy()
            display_df['Risk Probability'] = display_df['Risk Probability'].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_df, use_container_width=True, height=400)
 
            # Download button
            csv = high_risk[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download High-Risk Orders CSV",
                data=csv,
                file_name="high_risk_orders.csv",
                mime="text/csv"
            )
        else:
            st.success("No high-risk orders found with current filters and threshold.")
 
        # Risk distribution histogram
        st.subheader("Risk Probability Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(probs, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(x=risk_threshold, color='red', linestyle='--',
                   linewidth=2, label=f'High-Risk Threshold ({risk_threshold:.0%})')
        ax.axvline(x=0.4, color='orange', linestyle='--',
                   linewidth=1.5, label='Medium-Risk Threshold (40%)')
        ax.set_xlabel('Late Delivery Probability')
        ax.set_ylabel('Number of Orders')
        ax.set_title('Distribution of Predicted Risk Scores')
        ax.legend()
        st.pyplot(fig)
        plt.close()
 
 
# Footer
st.divider()
st.caption("APL Logistics — Late Delivery Risk Prediction System | Powered by Machine Learning")
