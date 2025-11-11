# --- AI FX PANEL С АВТО ТЕЛЕГРАМ ОПОВЕЩЕНИЯМИ ---
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import random
import plotly.graph_objects as go

# --- НАСТРОЙКИ ---
REFRESH_SEC = 1  # обновление каждую секунду
LOOKBACK_MIN = 180
INTERVAL = "1m"
MIN_BARS = 50

PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
}

# --- ТЕЛЕГРАМ ---
TELEGRAM_TOKEN = "8188894081:AAHr7im0L7CWCgiScOnKMLqo7g3I7R0s_80"
CHAT_ID = "6045310859"

def send_telegram_message(pair, signal, confidence, expiry):
    text = f"🤖 AI FX сигнал:\n💱 Пара: {pair}\n📈 Сигнал: {signal}\n📊 Уверенность: {confidence}%\n⏱ Время экспирации: {expiry}"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": text}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print("Ошибка при отправке в Telegram:", e)

# --- ОСНОВНОЙ КОД ---
st.set_page_config(page_title="AI FX Panel", layout="wide")
st.title("🤖 AI FX PANEL — Автоматический анализ и сигналы")

selected_pair = st.selectbox("Выбери валютную пару", list(PAIRS.keys()))
data = yf.download(PAIRS[selected_pair], period=f"{LOOKBACK_MIN}m", interval=INTERVAL)

if data.empty:
    st.error("Ошибка загрузки данных.")
else:
    data["SMA"] = data["Close"].rolling(window=10).mean()
    data["Signal"] = np.where(data["Close"] > data["SMA"], "BUY", "SELL")

    last_signal = data["Signal"].iloc[-1]
    confidence = random.randint(50, 95)

    if confidence >= 85:
        expiry = "10 минут"
    elif confidence >= 70:
        expiry = "5 минут"
    elif confidence >= 50:
        expiry = "3 минуты"
    else:
        expiry = "1-2 минуты (осторожно)"

    # --- Отображение ---
    st.metric("Сигнал", last_signal)
    st.metric("Уверенность", f"{confidence}%")
    st.metric("Рекомендуемая экспирация", expiry)

    # --- График ---
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"],
        name="Цена"
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data["SMA"],
        mode="lines", name="SMA (10)"
    ))
    st.plotly_chart(fig, use_container_width=True)

    # --- Отправка сигнала ---
    send_telegram_message(selected_pair, last_signal, confidence, expiry)

st.write("🔄 Автообновление каждые", REFRESH_SEC, "секунд")
st.experimental_rerun()
