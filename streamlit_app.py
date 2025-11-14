import streamlit as st
import pandas as pd
import yfinance as yf
import time

st.title("AI FX Panel — Base System v1.0")

st.write("Основа приложения загружена. Сейчас проверим данные валютных пар.")

# автообновление
REFRESH = 1  
st.write(f"Авто-обновление каждые {REFRESH} сек")

# список валютных пар
PAIRS = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "NZDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
]

@st.cache_data(ttl=REFRESH)
def load(pair):
    return yf.download(pair, period="1d", interval="1m")

# таблица
data_list = []
for p in PAIRS:
    try:
        df = load(p)
        if len(df) > 0:
            last = df.iloc[-1]
            data_list.append([p, last["Close"]])
        else:
            data_list.append([p, "нет данных"])
    except:
        data_list.append([p, "ошибка"])

table = pd.DataFrame(data_list, columns=["Пара", "Цена"])

st.dataframe(table, use_container_width=True)

# автообновление
time.sleep(REFRESH)
st.rerun()
