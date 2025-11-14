import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import os

# ===================== TELEGRAM ======================
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

# ===================== INDICATORS ======================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.ewm(span=period).mean() / loss.ewm(span=period).mean()
    return 100 - (100 / (1 + rs))

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def macd(series):
    fast = ema(series, 12)
    slow = ema(series, 26)
    signal = ema(fast - slow, 9)
    return fast - slow, signal

# ===================== SAFE CLOSE ======================
def safe_close(df):
    close = pd.to_numeric(df["Close"], errors="coerce")
    close = close.dropna()
    if len(close) < 50:
        return None
    return close

# ===================== GET DATA ======================
def get(symbol, tf):
    interval = {"M1":"1m", "M5":"5m", "M15":"15m", "M30":"30m"}[tf]
    try:
        df = yf.download(symbol, interval=interval, period="1d")
        if df is None or len(df)==0:
            return None
        df = df.tail(200)
        return df
    except:
        return None

# ===================== SIGNAL LOGIC ======================
def signal(df):
    close = safe_close(df)
    if close is None:
        return None, 0

    r = rsi(close).iloc[-1]
    m, s = macd(close)
    mcd = (m - s).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    ema200 = ema(close, 200).iloc[-1]
    last = close.iloc[-1]
    prev = close.iloc[-2]

    trend = "UP" if ema50 > ema200 else "DOWN"

    # BUY
    if r < 30 and mcd > 0 and last > prev and trend == "UP":
        return "BUY", 85

    # SELL
    if r > 70 and mcd < 0 and last < prev and trend == "DOWN":
        return "SELL", 85

    return None, 0

# ===================== MULTI TF ======================
def multi(symbol):
    dfs = {}
    for tf in ["M1","M5","M15","M30"]:
        df = get(symbol, tf)
        if df is None:
            return None
        dfs[tf] = df

    res = {}
    for tf in dfs:
        s, c = signal(dfs[tf])
        res[tf] = s or "-"

    # если М1 М5 М15 М30 совпадают
    final = "-"
    if res["M1"] == res["M5"] == res["M15"] == res["M30"] and res["M1"] != "-":
        final = res["M1"]

    return final, res

# ===================== UI ======================
st.title("AI FX v102 — SIMPLE & STABLE 🔥")
st.write("Multi-Timeframe (M1, M5, M15, M30) — стабильные сигналы")

symbol = st.text_input("Введите валюту (пример: EURUSD=X, GBPUSD=X, USDJPY=X)", "EURUSD=X")
btn = st.button("Сканировать")

if btn:
    st.write("Сканирую…")
    final, res = multi(symbol)

    if final == "-" or final is None:
        st.warning("Нет чёткого сигнала")
    else:
        st.success(f"Сигнал: **{final}**")

        msg = f"""
📡 AI FX v102
Пара: {symbol}
Multi-TF:
M1 = {res['M1']}
M5 = {res['M5']}
M15 = {res['M15']}
M30 = {res['M30']}

🎯 Итоговый сигнал: {final}
🕑 Экспирация: 2 минуты
        """

        tg(msg)
        st.write("📩 Отправлено в Telegram!")
