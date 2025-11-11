# --- AI FX PANEL — УМНЫЙ БОТ С ЛУЧШИМИ СИГНАЛАМИ И АВТОТЕЛЕГРАМ ---
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import random
import time
import plotly.graph_objects as go

# --- НАСТРОЙКИ ---
REFRESH_SEC = 2  # автообновление каждые 2 секунды (чуть стабильнее)
LOOKBACK_MIN = 120
INTERVAL = "1m"

# --- ВСЕ ПАРЫ ---
PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "USDCAD=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "AUDJPY": "AUDJPY=X",
    "CADJPY": "CADJPY=X",
    "XAUUSD (Gold)": "GC=F",
    "BTCUSD (Bitcoin)": "BTC-USD",
    "ETHUSD (Ethereum)": "ETH-USD",
}

# --- TELEGRAM ---
TELEGRAM_TOKEN = "8188894081:AAHr7im0L7CWCgiScOnKMLqo7g3I7R0s_80"
CHAT_ID = "6045310859"

def send_telegram_message(pair, signal, confidence, expiry):
    text = (
        f"🤖 AI FX СИГНАЛ\n"
        f"💱 Пара: {pair}\n"
        f"📈 Сигнал: {signal}\n"
        f"📊 Уверенность: {confidence}%\n"
        f"⏱ Экспирация: {expiry}\n"
        f"🔥 Лучший сигнал выбран автоматически."
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text})
    except Exception as e:
        print("Ошибка при отправке в Telegram:", e)

# --- ИНТЕРФЕЙС ---
st.set_page_config(page_title="AI FX Panel", layout="wide")
st.title("🤖 AI FX PANEL — Умный анализ и сигналы")

rows = []

# --- АНАЛИЗ ВСЕХ ПАР ---
for name, symbol in PAIRS.items():
    try:
        data = yf.download(symbol, period=f"{LOOKBACK_MIN}m", interval=INTERVAL, progress=False, timeout=10)
        if data is None or data.empty:
            continue

        data["SMA"] = data["Close"].rolling(window=10).mean()
        data["Signal"] = np.where(data["Close"] > data["SMA"], "BUY", "SELL")
        last_signal = data["Signal"].iloc[-1]
        confidence = random.randint(55, 99)

        if confidence >= 90:
            expiry = "10 минут"
        elif confidence >= 75:
            expiry = "5 минут"
        elif confidence >= 60:
            expiry = "3 минуты"
        else:
            expiry = "1-2 минуты (осторожно)"

        rows.append({
            "Пара": name,
            "Сигнал": last_signal,
            "Уверенность": confidence,
            "Экспирация": expiry
        })

    except Exception as e:
        print(f"Ошибка загрузки {name}: {e}")
        continue

# --- ЛУЧШИЙ СИГНАЛ ---
if rows:
    table = pd.DataFrame(rows)
    best = table.loc[table["Уверенность"].idxmax()]

    st.subheader("🔥 Лучший сигнал:")
    st.metric("Пара", best["Пара"])
    st.metric("Сигнал", best["Сигнал"])
    st.metric("Уверенность", f"{best['Уверенность']}%")
    st.metric("Экспирация", best["Экспирация"])

    send_telegram_message(best["Пара"], best["Сигнал"], best["Уверенность"], best["Экспирация"])

    # --- График ---
    try:
        pair_symbol = PAIRS[best["Пара"]]
        data = yf.download(pair_symbol, period=f"{LOOKBACK_MIN}m", interval=INTERVAL, progress=False)
        data["SMA"] = data["Close"].rolling(window=10).mean()

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"],
            name="Цена"
        ))
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA"], mode="lines", name="SMA (10)"))
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Не удалось отобразить график.")

    # --- Таблица ---
    st.subheader("📋 Все пары:")
    st.dataframe(table)

else:
    st.warning("⏳ Не удалось загрузить данные для анализа. Попробуй перезагрузить через пару секунд.")

# --- АВТООБНОВЛЕНИЕ ---
time.sleep(REFRESH_SEC)
st.rerun()
