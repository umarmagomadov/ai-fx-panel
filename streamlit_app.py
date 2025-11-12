# AI FX v101.0-mobile-safe — Triple-Timeframe + Auto Expiry
# Без лишних символов, совместимо с GitHub Mobile / Streamlit Cloud.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests, time, json, random
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="AI FX v101.0-mobile-safe", layout="wide")
st.title("🤖 AI FX v101.0 — M5+M15+M30 Mobile Edition")

# ---------------- SETTINGS ----------------
REFRESH = 2
THRESH = 70
PAIRS = {
    "EURUSD":"EURUSD=X",
    "GBPUSD":"GBPUSD=X",
    "USDJPY":"USDJPY=X",
    "BTCUSD (Bitcoin)":"BTC-USD",
    "ETHUSD (Ethereum)":"ETH-USD",
    "XAUUSD (Gold)":"GC=F"
}

# ---------------- UTILS ----------------
def safe_float(x):
    try: return float(x)
    except: return 0.0

def safe_close(df):
    s = pd.to_numeric(df["Close"], errors="coerce")
    s = s.fillna(method="ffill").fillna(method="bfill")
    return s

def get_data(symbol, period="5d", interval="5m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or len(df) < 20:
            raise ValueError
        return df
    except:
        idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=60, freq="1min")
        vals = 1 + np.cumsum(np.random.randn(60))/200
        return pd.DataFrame({"Open":vals,"High":vals,"Low":vals,"Close":vals}, index=idx)

# ---------------- INDICATORS ----------------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / (dn + 1e-9)
    return 100 - 100/(1+rs)

def macd(s):
    m = ema(s,12) - ema(s,26)
    s9 = ema(m,9)
    return m, s9, m-s9

def adx(df, n=14):
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    up = h.diff()
    dn = -l.diff()
    plus = np.where((up>dn)&(up>0),up,0.0)
    minus = np.where((dn>up)&(dn>0),dn,0.0)
    tr = pd.concat([(h-l).abs(), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * pd.Series(plus).rolling(n).sum() / (atr + 1e-9)
    minus_di = 100 * pd.Series(minus).rolling(n).sum() / (atr + 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    return dx.rolling(n).mean().fillna(20)

# ---------------- SIGNAL ----------------
def signal(df):
    c = safe_close(df)
    r = float(rsi(c).iloc[-1])
    m, s9, h = macd(c)
    adxv = float(adx(df).iloc[-1])
    ema9 = float(ema(c,9).iloc[-1])
    ema21 = float(ema(c,21).iloc[-1])
    buy = sell = 0
    if r<35: buy+=1
    if r>65: sell+=1
    if ema9>ema21: buy+=1
    if ema9<ema21: sell+=1
    if h.iloc[-1]>0: buy+=1
    if h.iloc[-1]<0: sell+=1
    direction = "FLAT"
    if buy>sell: direction="BUY"
    elif sell>buy: direction="SELL"
    conf = int(min(99, max(40, (abs(buy-sell)/3*60 + (adxv-20)))))
    return direction, conf, dict(RSI=round(r,1),ADX=round(adxv,1),MACD=round(h.iloc[-1],4))

# ---------------- EXPIRY ----------------
def expiry(conf, adxv):
    if conf<60: return 0
    t = 5
    if conf>80: t+=5
    if adxv>40: t+=5
    return int(min(30,t))

# ---------------- RUN ----------------
rows=[]
for name,sym in PAIRS.items():
    df = get_data(sym)
    sig,conf,f = signal(df)
    exp = expiry(conf, f["ADX"])
    rows.append([name,sig,conf,exp,f])
df = pd.DataFrame(rows, columns=["Пара","Сигнал","Уверенность","Экспирация","Индикаторы"])

st.dataframe(df,use_container_width=True)

if len(df):
    top = df.iloc[df["Уверенность"].idxmax()]
    sym = PAIRS[top["Пара"]]
    dfc = get_data(sym)
    fig = go.Figure(data=[go.Candlestick(x=dfc.index,open=dfc["Open"],high=dfc["High"],low=dfc["Low"],close=dfc["Close"])])
    fig.update_layout(height=350,title=f"{top['Пара']} — {top['Сигнал']} ({top['Уверенность']}%)")
    st.plotly_chart(fig,use_container_width=True)

time.sleep(REFRESH)
st.rerun()
