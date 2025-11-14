import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime

# ========= SAFE HELPERS ========= #
def safe_close(df):
    """Гарантированно возвращает чистый Series с числами."""
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    close = pd.to_numeric(df["Close"], errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan).dropna()

    return close


# ========= INDICATORS ========= #
def rsi(series, period=14):
    series = safe_close(pd.DataFrame({"Close": series}))
    if len(series) < period + 1:
        return pd.Series([50])  # нейтральное значение

    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    ema_up = up.ewm(span=period).mean()
    ema_down = down.ewm(span=period).mean()

    rs = ema_up / (ema_down + 1e-9)
    return 100 - (100 / (1 + rs))


def adx(series, period=14):
    """Простая безопасная замена ADX."""
    series = safe_close(pd.DataFrame({"Close": series}))
    if len(series) < period + 2:
        return pd.Series([20])  # нейтральное значение

    change = series.diff().abs()
    adx_raw = change.rolling(period).mean()
    return adx_raw / (adx_raw.max() + 1e-9) * 40  # от 0 до 40


def macd(series):
    series = safe_close(pd.DataFrame({"Close": series}))
    if len(series) < 35:
        return pd.Series([0])

    fast = series.ewm(span=12).mean()
    slow = series.ewm(span=26).mean()
    return fast - slow


# ========= MULTI-TIMEFRAME BLOCK ========= #
def download_tf(symbol, tf="5m"):
    try:
        df = yf.download(symbol, period="2d", interval=tf)
        return df
    except:
        return pd.DataFrame()


def tf_direction(df):
    close = safe_close(df)
    if len(close) < 3:
        return "FLAT"

    return "BUY" if close.iloc[-1] > close.iloc[-3] else "SELL"


def combine_signal(m1, m5, m15, m30):
    arr = [m1, m5, m15, m30]
    if arr.count("BUY") >= 3:
        return "BUY"
    if arr.count("SELL") >= 3:
        return "SELL"
    return "FLAT"


# ========= MAIN SCORE ========= #
def compute_signal(sym):
    # Загружаем все ТФ
    df1 = download_tf(sym, "1m")
    df5 = download_tf(sym, "5m")
    df15 = download_tf(sym, "15m")
    df30 = download_tf(sym, "30m")

    # Направления
    d1 = tf_direction(df1)
    d5 = tf_direction(df5)
    d15 = tf_direction(df15)
    d30 = tf_direction(df30)

    # Индикаторы
    close = safe_close(df5)
    r = float(rsi(close).iloc[-1])
    a = float(adx(close).iloc[-1])
    m = float(macd(close).iloc[-1])

    # Итоговое направление
    main = combine_signal(d1, d5, d15, d30)

    # Уверенность
    conf = 50
    if main != "FLAT":
        conf += 10
    if abs(m) > 0.01:
        conf += 10
    if a > 10:
        conf += 10
    if 45 < r < 55:
        conf -= 10

    conf = max(1, min(conf, 99))

    return main, conf, r, a, m, (d1, d5, d15, d30)


# ========= TELEGRAM SENDER ========= #
def send_telegram(msg):
    import os
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    CHAT = os.getenv("CHAT_ID")

    if not TOKEN or not CHAT:
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT, "text": msg})


# ========= STREAMLIT UI ========= #
st.title("AI FX v102.1 — MAX-FILTER SAFE")

symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", "BTC-USD"]

pair = st.selectbox("Валютная пара:", symbols)
thr = st.slider("Минимальная уверенность (%)", 50, 95, 60)
pause = st.number_input("Пауза между сигналами (сек)", 20, 600, 60)

if st.button("Сканировать"):
    sig, conf, r, a, m, tf = compute_signal(pair)

    st.write("Сигнал:", sig)
    st.write("Уверенность:", conf, "%")
    st.write("RSI:", r)
    st.write("ADX:", a)
    st.write("MACD:", m)
    st.write("TF:", tf)

    if conf >= thr and sig != "FLAT":
        msg = f"""
📊 AI FX SIGNAL v102.1
Пара: {pair}
Сигнал: {sig}

🧩 Multi-TF: M1={tf[0]} | M5={tf[1]} | M15={tf[2]} | M30={tf[3]}
💪 Уверенность: {conf}%
📈 RSI: {r:.1f}
📉 ADX: {a:.1f}
📊 MACD: {m:.5f}
⏰ {datetime.utcnow().strftime("%H:%M:%S")} UTC
"""
        send_telegram(msg)
        st.success("Отправлено!")
