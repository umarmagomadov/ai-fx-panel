# 🤖 AI FX Signal Bot v100.7-final — Triple-Timeframe Smart Mode + Pocket Copy
# Исправлены все ошибки TypeError при обработке мультииндексных данных из yfinance

import time, json, random, os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
import plotly.graph_objects as go

# ================== SECRETS ==================
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", os.getenv("TELEGRAM_TOKEN", ""))
CHAT_ID        = st.secrets.get("CHAT_ID",        os.getenv("CHAT_ID", ""))

# ================== SETTINGS =================
REFRESH_SEC     = 1
ONLY_NEW        = True
MIN_SEND_GAP_S  = 60
CONF_THRESHOLD  = 70

TF_MAIN  = ("5m",  "2d")
TF_MID   = ("15m", "5d")
TF_TREND = ("30m", "10d")

PAIRS = {
    "EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"USDJPY=X","AUDUSD":"AUDUSD=X","NZDUSD":"NZDUSD=X",
    "BTCUSD (Bitcoin)":"BTC-USD","ETHUSD (Ethereum)":"ETH-USD","XAUUSD (Gold)":"GC=F","BRENT (Oil)":"BZ=F",
    "WTI Crude Oil OTC":"WTI-OTC","GBP/USD OTC":"GBPUSD-OTC"
}

# ================== INDICATORS ===============
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(close, period=14):
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, signal)
    return m, s, m - s

def adx(df, n=14):
    if df is None or len(df) < n+2 or "High" not in df or "Low" not in df or "Close" not in df:
        return pd.Series([25]*len(df), index=df.index)
    h, l, c = df["High"], df["Low"], df["Close"]
    up_move = h.diff(); dn_move = -l.diff()
    plus_dm  = up_move.where((up_move>0)&(up_move>dn_move), 0.0).fillna(0)
    minus_dm = dn_move.where((dn_move>0)&(dn_move>up_move), 0.0).fillna(0)
    tr = pd.concat([(h-l).abs(), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di  = 100 * (plus_dm.rolling(n).sum() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(n).sum() / (atr + 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9))
    return dx.rolling(n).mean().fillna(25)

def bbands(close, n=20, k=2):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    up, lo = ma + k*sd, ma - k*sd
    width = (up - lo) / (ma + 1e-9) * 100
    return up, ma, lo, width

# ================== SAFE CLOSE ==================
def _safe_close_series(df: pd.DataFrame) -> pd.Series:
    """Безопасно извлекает колонку Close из любых форматов DataFrame"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series([], dtype="float64")

    # Если мультииндекс — убираем уровень
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close" in df.columns.get_level_values(0)):
            df = df["Close"]
            if isinstance(df, pd.DataFrame):
                # Берем первый столбец
                df = df.iloc[:,0]
        elif "close" in df.columns.get_level_values(0):
            df = df["close"]
            if isinstance(df, pd.DataFrame):
                df = df.iloc[:,0]

    # Если осталась колонка Close
    if isinstance(df, pd.DataFrame) and "Close" in df.columns:
        s = df["Close"]
    elif isinstance(df, pd.DataFrame) and "close" in df.columns:
        s = df["close"]
    elif isinstance(df, pd.Series):
        s = df
    else:
        # fallback: последняя числовая колонка
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            return pd.Series([], dtype="float64")
        s = num.iloc[:, -1]

    s = pd.to_numeric(s, errors="coerce").fillna(method="ffill")
    return s.astype("float64")

# ================== DATA =====================
@st.cache_data(show_spinner=False, ttl=60)
def load_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 50:
            return pd.DataFrame()
        return df.tail(600)
    except Exception:
        return pd.DataFrame()

def is_otc(name): return "OTC" in name.upper()

def pocket_code(name, symbol):
    if "=X" in symbol: return symbol.replace("=X","").upper().replace("USD","/USD")
    if "-USD" in symbol: return symbol.replace("-USD","/USD")
    if symbol == "GC=F": return "XAU/USD"
    if symbol == "BZ=F": return "BRENT/USD"
    return name.upper().replace(" ","_")

# ================== СИГНАЛЫ ==================
def score_single(df):
    if df is None or df.empty:
        return "FLAT", 0, {}

    close = _safe_close_series(df)
    if close.empty or close.size < 30:
        return "FLAT", 0, {}

    rsv = float(rsi(close).iloc[-1])
    ema9, ema21 = float(ema(close,9).iloc[-1]), float(ema(close,21).iloc[-1])
    macd_line, macd_sig, macd_hist = macd(close)
    m_hist = float(macd_hist.iloc[-1])
    up, mid, lo, width = bbands(close)
    bb_pos = float((close.iloc[-1]-mid.iloc[-1])/((up.iloc[-1]-lo.iloc[-1])+1e-9))
    adx_v = float(adx(df).iloc[-1])

    votes_buy = votes_sell = 0
    if rsv < 35: votes_buy+=1
    if rsv > 65: votes_sell+=1
    if ema9 > ema21: votes_buy+=1
    if ema9 < ema21: votes_sell+=1
    if m_hist > 0: votes_buy+=1
    if m_hist < 0: votes_sell+=1
    if bb_pos < -0.25: votes_buy+=1
    if bb_pos >  0.25: votes_sell+=1

    if votes_buy == votes_sell: sig = "FLAT"
    elif votes_buy > votes_sell: sig = "BUY"
    else: sig = "SELL"

    trend_boost = min(max((adx_v - 18)/25,0),1)
    raw = abs(votes_buy - votes_sell)/4.0
    conf = int(100*(0.55*raw + 0.45*trend_boost))
    feats = {"RSI":round(rsv,1),"ADX":round(adx_v,1),"MACD":round(m_hist,5)}
    return sig, conf, feats

def tf_dir(df):
    if df is None or df.empty:
        return "FLAT"
    c = _safe_close_series(df)
    if c.empty or c.size < 30:
        return "FLAT"
    macd_line, macd_sig, macd_hist = macd(c)
    rsv = float(rsi(c).iloc[-1])
    mh = float(macd_hist.iloc[-1])
    if mh > 0 and rsv > 50: return "BUY"
    if mh < 0 and rsv < 50: return "SELL"
    return "FLAT"

def score_multi(symbol):
    df5  = load_data(symbol, TF_MAIN[1],  TF_MAIN[0])
    df15 = load_data(symbol, TF_MID[1],   TF_MID[0])
    df30 = load_data(symbol, TF_TREND[1], TF_TREND[0])
    s5,c5,f5 = score_single(df5)
    d5,d15,d30 = tf_dir(df5), tf_dir(df15), tf_dir(df30)
    conf = c5
    if d5==d15==d30 and d5!="FLAT": conf+=15
    elif (d5==d15) or (d5==d30): conf+=7
    else: conf-=10
    conf = max(0,min(100,conf))
    return s5, conf, f5, {"M5":d5,"M15":d15,"M30":d30}

def expiry(conf, adx):
    if conf<60: return 5
    if conf<70: return 8
    if conf<80: return 12
    if conf<90: return 20
    return 30

# ================== TELEGRAM =================
def send_tg(name, symbol, signal, conf, exp, feats, mtf):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    arrow = "⬆️" if signal=="BUY" else "⬇️" if signal=="SELL" else "➖"
    text = (
        f"🤖 AI FX СИГНАЛ v100.7-final\n"
        f"💱 Пара: {name}\n"
        f"📌 Код для Pocket Option: `{pocket_code(name,symbol)}`\n"
        f"📈 Сигнал: {arrow} {signal}\n"
        f"💪 Уверенность: {conf}%\n"
        f"⏱ Экспирация: {exp} мин\n"
        f"📊 RSI {feats['RSI']} | ADX {feats['ADX']} | MACD {feats['MACD']}\n"
        f"🕒 Multi-TF: M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}\n"
        f"⏰ {datetime.utcnow().strftime('%H:%M:%S')} UTC"
    )
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id":CHAT_ID,"text":text,"parse_mode":"Markdown"})
    except: pass

# ================== UI =======================
st.set_page_config(page_title="AI FX v100.7-final — M5+M15+M30", layout="wide")
st.title("🤖 AI FX Signal Bot v100.7-final — Triple-Timeframe + Pocket Copy")

thr = st.slider("Порог уверенности (Telegram)", 50, 95, CONF_THRESHOLD)
rows=[]

if "sent" not in st.session_state: st.session_state.sent={}

for name,symbol in PAIRS.items():
    sig,conf,feats,mtf = score_multi(symbol)
    otc = is_otc(name)
    exp = expiry(conf, feats.get("ADX",30))
    pocket = pocket_code(name,symbol)
    rows.append([name,"OTC" if otc else "Биржевая",sig,conf,exp,pocket,
                 f"M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}",
                 f"RSI {feats['RSI']} | ADX {feats['ADX']} | MACD {feats['MACD']}"])

    if sig in ["BUY","SELL"] and conf>=thr:
        prev = st.session_state.sent.get(name,{})
        recent = time.time()-prev.get("ts",0)<MIN_SEND_GAP_S
        if not recent or conf>prev.get("conf",0):
            send_tg(name,symbol,sig,conf,exp,feats,mtf)
            st.session_state.sent[name]={"ts":time.time(),"conf":conf}

df = pd.DataFrame(rows,columns=["Пара","Тип","Сигнал","Уверенность","Экспирация","Pocket код","Multi-TF","Индикаторы"])
df = df.sort_values("Уверенность",ascending=False)
st.dataframe(df,use_container_width=True)

if len(df):
    top = df.iloc[0]
    st.subheader("📋 Копировать для Pocket Option:")
    st.text_input("Tap to copy:", value=top["Pocket код"], key="copy_top")

time.sleep(REFRESH_SEC)
st.rerun()
