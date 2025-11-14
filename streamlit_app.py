import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import os
import time

# ========= TELEGRAM ==========
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def tg(msg):
    try:
        requests.get(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            params={"chat_id": CHAT_ID, "text": msg}
        )
    except:
        pass

# ========= INDICATORS ==========
def rsi(series, period=14):
    if len(series) < period + 5:
        return None
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.ewm(span=period).mean() / loss.ewm(span=period).mean()
    return 100 - (100 / (1 + rs))

def ema(series, n):
    if len(series) < n + 5:
        return None
    return series.ewm(span=n, adjust=False).mean()

def macd(series):
    if len(series) < 50:
        return None, None
    fast = ema(series, 12)
    slow = ema(series, 26)
    signal = ema(fast - slow, 9)
    return fast - slow, signal

# ========= SAFE CLOSE ==========
def safe_close(df):
    if df is None or "Close" not in df:
        return None
    c = pd.to_numeric(df["Close"], errors="coerce")
    c = c.dropna()
    if len(c) < 50:
        return None
    return c

# ========= GET DATA ==========
def get(symbol, tf):
    interval = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m"}[tf]
    try:
        df = yf.download(symbol, interval=interval, period="1d")
        if df is None or len(df) == 0:
            return None
        return df.tail(200)
    except:
        return None

# ========= SIGNAL LOGIC ==========
def signal(df):
    close = safe_close(df)
    if close is None:
        return None, 0

    r = rsi(close)
    if r is None:
        return None, 0
    r = float(r.iloc[-1])

    m, s = macd(close)
    if m is None:
        return None, 0
    mcd = float((m - s).iloc[-1])

    ema50 = ema(close, 50)
    ema200 = ema(close, 200)

    if ema50 is None or ema200 is None:
        return None, 0

    trend = "UP" if ema50.iloc[-1] > ema200.iloc[-1] else "DOWN"

    # BUY
    if r < 30 and mcd > 0 and trend == "UP":
        return "BUY", 85

    # SELL
    if r > 70 and mcd < 0 and trend == "DOWN":
        return "SELL", 85

    return None, 0

# ========= MULTI-TF ==========
def multi(symbol):
    dfs = {}
    for tf in ["M1", "M5", "M15", "M30"]:
        df = get(symbol, tf)
        if df is None:
            return None, None
        dfs[tf] = df

    res = {}
    for tf in dfs:
        s, _ = signal(dfs[tf])
        res[tf] = s or "-"

    final = "-"
    if res["M1"] == res["M5"] == res["M15"] == res["M30"] and res["M1"] != "-":
        final = res["M1"]

    return final, res


# ========= STREAMLIT UI + AUTO REFRESH ==========
st.set_page_config(page_title="AI FX v102.1 Stable", layout="centered")

st.title("AI FX v102.1 — AUTO MODE 🔥")
st.write("⏱ Авто-обновление: **1 секунда**")

symbol = st.text_input("Валютная пара (пример: EURUSD=X)", "EURUSD=X")

placeholder = st.empty()

last_signal = None

while True:
    with placeholder.container():
        st.write("🔄 Сканирую…")

        final, res = multi(symbol)

        if final is None:
            st.error("❌ Нет данных!")
        elif final == "-":
            st.warning("⚪ Сигнал не найден")
        else:
            st.success(f"💥 Сигнал: **{final}**")

            # === отправка только при изменении сигнала ===
            if final != last_signal:
                msg = f"""
📡 AI FX v102.1 AUTO
Пара: {symbol}

M1: {res['M1']}
M5: {res['M5']}
M15: {res['M15']}
M30: {res['M30']}

🎯 Итоговый сигнал: {final}
⏳ Экспирация: 2 минуты
                """
                tg(msg)
                last_signal = final

        time.sleep(1)
