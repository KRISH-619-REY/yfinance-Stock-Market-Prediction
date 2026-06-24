import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from tensorflow.keras.models import load_model

TICKER = "^NSEI"
TIME_STEP = 60

st.set_page_config(
    page_title="NIFTY 50 Trend Predictor",
    page_icon="📈",
    layout="centered"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00C9A7, #845EC2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin-top: -10px;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background-color: #1A1F2E;
        border: 1px solid #2D3344;
        border-radius: 12px;
        padding: 15px 20px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #9CA3AF;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #00C9A7, #845EC2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 0;
        font-weight: 600;
        font-size: 1rem;
        transition: transform 0.15s ease;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        color: white;
    }
    .trend-box {
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .trend-up {
        background-color: rgba(0, 201, 167, 0.15);
        border: 1px solid #00C9A7;
        color: #00C9A7;
    }
    .trend-down {
        background-color: rgba(255, 90, 95, 0.15);
        border: 1px solid #FF5A5F;
        color: #FF5A5F;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<p class="main-title">📈 NIFTY 50 Trend Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered next-day closing trend prediction using LSTM</p>', unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = load_model("model/lstm_model.keras")
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

predict_clicked = st.button("🔮 Predict Next Day Trend")

if predict_clicked:
    with st.spinner("Fetching live market data and running prediction..."):

        model, scaler = load_artifacts()

        df = yf.download("^NSEI", period="6mo", auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()

        # --- DEBUG: check why data might be lagging ---
        st.write("Last few rows of fetched data:")
        st.write(df.tail())
        st.write("Last index timestamp:", df.index[-1])
        st.write("Current system time:", pd.Timestamp.now())
        # --- END DEBUG ---

        close = df[['Close']].values
        scaled = scaler.transform(close)
        last_seq = scaled[-60:].reshape(1, 60, 1)

        pred_scaled = model.predict(last_seq, verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        last_price = close[-1][0]

        is_up = pred_price > last_price
        trend_label = "UP 📈" if is_up else "DOWN 📉"
        change_pct = ((pred_price - last_price) / last_price) * 100

        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Last Close", f"₹{last_price:,.2f}")
        with col2:
            st.metric(
                "Predicted Close",
                f"₹{pred_price:,.2f}",
                delta=f"{change_pct:+.2f}%"
            )
        with col3:
            st.metric("Change", f"{change_pct:+.2f}%")

        trend_class = "trend-up" if is_up else "trend-down"
        st.markdown(
            f'<div class="trend-box {trend_class}">Predicted Trend: {trend_label}</div>',
            unsafe_allow_html=True
        )

        st.markdown("##### 📊 Last 6 Months — NIFTY 50 Closing Price")
        st.line_chart(df[['Close']], height=300)

        st.caption(f"Data as of {df.index[-1].strftime('%d %b %Y')} • Source: Yahoo Finance")

else:
    st.info("👆 Click the button above to fetch the latest NIFTY 50 data and generate a prediction.")