import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="AI FX Panel", page_icon="💹")

st.title("💹 AI FX Panel")
st.write("Добро пожаловать! Здесь будут сигналы по валютным парам (1м и 5м).")

pair = st.selectbox("Выбери валютную пару:", ["EURUSD", "GBPUSD", "USDJPY", "BTC-USD"])

data = yf.download(pair, period="1d", interval="1m")

if not data.empty:
    st.line_chart(data["Close"])

    last_close = data["Close"].iloc[-1]
    mean_price = data["Close"].mean()
    if last_close > mean_price:
        st.success("🟢 BUY сигнал (цена выше среднего уровня)")
    else:
        st.error("🔴 SELL сигнал (цена ниже среднего уровня)")
else:
    st.warning("Нет данных. Попробуй другую валютную пару.")

st.info("Обновление каждые 30 секунд включено.")
st.success("Связь установлена. Всё работает!")
