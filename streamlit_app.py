import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Настройки страницы
st.set_page_config(page_title="AI FX Panel", page_icon="💹", layout="wide")

st.title("💹 AI FX Panel — Живое обновление")
st.write("AI анализирует валютные пары и обновляет данные каждую секунду в реальном времени.")

# Выбор валютной пары
pair = st.selectbox("Выбери валютную пару:", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"], index=0)

# Выбор интервала
interval = st.radio("Интервал:", ["1m", "5m", "15m"], horizontal=True)

# Контейнер для автообновления
placeholder = st.empty()

# 🔁 Автообновление каждую секунду
while True:
    with placeholder.container():
        try:
            data = yf.download(pair, period="1d", interval=interval)
            if not data.empty:
                st.line_chart(data["Close"])

                last = float(data["Close"].iloc[-1])
                mean = float(data["Close"].rolling(10).mean().iloc[-1])
                diff = last - mean
                confidence = min(99, round(abs(diff / mean) * 1000, 2))

                if last > mean:
                    st.success(f"🟢 BUY сигнал — тренд вверх\n📈 Вероятность роста: {confidence}%")
                elif last < mean:
                    st.error(f"🔴 SELL сигнал — тренд вниз\n📉 Вероятность падения: {confidence}%")
                else:
                    st.info("⚪ Нейтральный сигнал — жди подтверждения")

                st.caption(f"Текущее значение: {last:.5f} | Среднее: {mean:.5f}")
                st.caption(f"⏱ Последнее обновление: {datetime.now().strftime('%H:%M:%S')}")
            else:
                st.warning("Нет данных. Попробуй другую валютную пару.")
        except Exception as e:
            st.error(f"Ошибка: {e}")

    # Обновление каждую секунду
    time.sleep(1)
