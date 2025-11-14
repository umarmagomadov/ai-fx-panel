# ===================== AI FX Bot v4.1 PRO =====================
# M1 + M5 + M15 + M30 + Telegram

import time, json, random, os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st

TELEGRAM_TOKEN = st.secrets.get(
    "TELEGRAM_TOKEN",
    os.getenv("TELEGRAM_TOKEN", "")
)
CHAT_ID = st.secrets.get(
    "CHAT_ID",
    os.getenv("CHAT_ID", "")
)

REFRESH_SEC        = 1
ONLY_NEW           = True
MIN_SEND_GAP_S     = 60
BASE_CONF_THRESHOLD = 70

MODES = {
    "Safe 85%":   85,
    "Normal 90%": 90,
    "Hard 95%":   95,
    "Ultra 99%":  99,
}

TF_M1  = ("1m",  "1d")
TF_M5  = ("5m",  "5d")
TF_M15 = ("15m", "5d")
TF_M30 = ("30m", "10d")

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
    "CADJPY": "CADJPY=X",
    "CHFJPY": "CHFJPY=X",
    "EURAUD": "EURAUD=X",
    "EURNZD": "EURNZD=X",
    "GBPAUD": "GBPAUD=X",
    "GBPNZD": "GBPNZD=X",

    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "XAUUSD": "XAUUSD=X",
    "USOIL":   "BZ=F",
}

@st.cache_data(show_spinner=False)
def load_history(symbol, interval, period):
    try:
        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return pd.DataFrame()
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        return df.dropna()
    except:
        return pd.DataFrame()

def get_or_fake(symbol, tf):
    interval, period = tf
    df = load_history(symbol, interval, period)
    if df.empty:
        now = datetime.now(timezone.utc)
        return pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.0],
                "low":  [1.0],
                "close": [1.0],
                "volume": [0],
            },
            index=[now],
        )
    return df

def calc_rsi(series, period=14):
    if series is None or len(series) < period + 1:
        return pd.Series([50.0] * len(series), index=series.index)

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)

    return rsi

def calc_macd(series):
    if len(series) < 35:
        s = pd.Series([0.0] * len(series), index=series.index)
        return s, s, s

    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calc_adx(df, period=14):
    if df.empty or len(df) < period + 2:
        return pd.Series([20.0] * len(df), index=df.index)

    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace(np.nan, 0)
    adx = dx.rolling(period).mean().fillna(20)

    return adx

def analyze_tf(df):
    close = df["close"]

    rsi = calc_rsi(close)
    macd, sig, hist = calc_macd(close)
    adx = calc_adx(df)

    last = df.index[-1]

    r = float(rsi.iloc[-1])
    m = float(macd.iloc[-1])
    s = float(sig.iloc[-1])
    h = float(hist.iloc[-1])
    a = float(adx.iloc[-1])

    if r > 60 and m > s and h > 0:
        signal = "BUY"
    elif r < 40 and m < s and h < 0:
        signal = "SELL"
    else:
        signal = "FLAT"

    regime = "trend" if a >= 25 else "flat"

    return {
        "time": last,
        "signal": signal,
        "RSI": r,
        "MACD": m,
        "MACD_sig": s,
        "MACD_hist": h,
        "ADX": a,
        "Regime": regime,
    }

def combine_multi_tf(m1, m5, m15, m30):
    signals = [m1["signal"], m5["signal"], m15["signal"], m30["signal"]]
    buy_votes = signals.count("BUY")
    sell_votes = signals.count("SELL")

    if buy_votes == 0 and sell_votes == 0:
        final_signal = "FLAT"
    elif buy_votes > sell_votes:
        final_signal = "BUY"
    elif sell_votes > buy_votes:
        final_signal = "SELL"
    else:
        final_signal = "FLAT"

    conf = 50
    conf += 10 * max(buy_votes, sell_votes)

    regimes = [m5["Regime"], m15["Regime"], m30["Regime"]]
    trend_votes = regimes.count("trend")
    if trend_votes >= 2:
        conf += 10

    avg_rsi = (m5["RSI"] + m15["RSI"] + m30["RSI"]) / 3
    if final_signal == "BUY" and avg_rsi < 55:
        conf -= 10
    if final_signal == "SELL" and avg_rsi > 45:
        conf -= 10

    conf = int(max(0, min(100, conf)))

    if conf >= 90:
        trade_class = "A"
    elif conf >= 80:
        trade_class = "B"
    else:
        trade_class = "C"

    regime = "trend" if trend_votes >= 2 else "flat"
    phase = "start" if avg_rsi < 45 or avg_rsi > 55 else "mid"

    info = {
        "M1": m1["signal"],
        "M5": m5["signal"],
        "M15": m15["signal"],
        "M30": m30["signal"],
        "Regime": regime,
        "Phase": phase,
        "RSI": avg_rsi,
        "ADX30": m30["ADX"],
    }

    return final_signal, conf, trade_class, info

def choose_expiry(conf, regime=None, phase=None):
    if conf >= 95:
        base = 2
    elif conf >= 90:
        base = 3
    elif conf >= 85:
        base = 4
    elif conf >= 80:
        base = 5
    else:
        return 0

    if regime == "trend":
        base += 2
    elif regime == "flat":
        base -= 1

    if phase == "start":
        base += 1
    elif phase == "end":
        base -= 1

    return int(max(1, min(30, base)))

def send_telegram(pair_name, pair_code, signal, conf, expiry, mtype, info):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return

    text = (
        "AI FX Bot v4.1 PRO\n"
        "Pair: " + pair_name + "\n"
        "Signal: " + signal + "\n"
        "Confidence: " + str(conf) + "%\n"
        "Expiry: " + str(expiry) + " min\n"
        "Regime: " + info["Regime"] + "\n"
        "Phase: " + info["Phase"] + "\n"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except:
        pass

st.set_page_config(page_title="AI FX Bot", layout="wide")
st.title("AI FX Bot v4.1 PRO")

col1, col2 = st.columns(2)
with col1:
    mode_name = st.selectbox("Сложность сигнала", list(MODES.keys()), index=0)
with col2:
    min_conf_slider = st.slider("Минимальная уверенность", 50, 99, 85)

gap_input = st.number_input("Пауза между сигналами", 10, 3600, MIN_SEND_GAP_S)
MIN_SEND_GAP_S = int(gap_input)

work_threshold = max(MODES[mode_name], min_conf_slider)
st.markdown("Текущий порог: " + str(work_threshold) + "%")

if "last_sent" not in st.session_state:
    st.session_state.last_sent = {}

rows = []

for name, symbol in PAIRS.items():
    df1 = get_or_fake(symbol, TF_M1)
    df5 = get_or_fake(symbol, TF_M5)
    df15 = get_or_fake(symbol, TF_M15)
    df30 = get_or_fake(symbol, TF_M30)

    a1 = analyze_tf(df1)
    a5 = analyze_tf(df5)
    a15 = analyze_tf(df15)
    a30 = analyze_tf(df30)

    sig, conf, cls, info = combine_multi_tf(a1, a5, a15, a30)

    if "BTC" in name or "ETH" in name or "OIL" in name:
        mtype = "OTC"
    else:
        mtype = "EXCHANGE"

    expiry = choose_expiry(conf, info["Regime"], info["Phase"])

    rows.append([
        name, mtype, sig, conf, cls, expiry,
        f"M1={info['M1']} M5={info['M5']} M15={info['M15']} M30={info['M30']}",
        round(info["ADX30"], 2)
    ])

    now_ts = time.time()
    last_ts = st.session_state.last_sent.get(name, 0)

    if sig in ("BUY", "SELL") and conf >= work_threshold and expiry > 0:
        if now_ts - last_ts >= MIN_SEND_GAP_S:
            send_telegram(name, name, sig, conf, expiry, mtype, info)
            st.session_state.last_sent[name] = now_ts

df = pd.DataFrame(rows,
    columns=[
        "Pair","Type","Signal","Confidence","Class",
        "Expiry","MultiTF","ADX30"
    ]
)

st.dataframe(df, use_container_width=True)
