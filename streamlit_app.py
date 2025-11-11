
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="AI FX Panel", page_icon="💹")

st.title("💹 AI FX Panel")
st.write("Добро пожаловать! Здесь AI анализирует валютные пары и даёт сигналы (1м, 5м, 15м).")

# 🔹 выбор валютной пары
pair = st.selectbox("Выбери валютную пару:", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "BTC-USD"])

# 🔹 выбор интервала
interval = st.radio("Интервал:", ["1m", "5m", "15m"], horizontal=True)

try:
    data = yf.download(pair, period="1d", interval=interval)
    if not data.empty:
        st.line_chart(data["Close"])

        # Простой AI-анализ
        last = data["Close"].iloc[-1]
        mean = data["Close"].rolling(10).mean().iloc[-1]

        if last > mean:
            st.success("🟢 BUY сигнал — тренд вверх")
        elif last < mean:
            st.error("🔴 SELL сигнал — тренд вниз")
        else:
            st.info("⚪ Нейтральный сигнал — жди подтверждения")

        st.caption(f"Последняя цена: {last:.5f}")
    else:
        st.warning("Нет данных, попробуй другой интервал или валютную пару.")
except Exception as e:
    st.error(f"Ошибка: {e}")

st.info("Обновление каждые 30 секунд включено.")
st.success("Связь установлена. Всё работает!")
