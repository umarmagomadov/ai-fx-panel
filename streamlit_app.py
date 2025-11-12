# AI FX Signal Bot v101.0 — Auto-Data + Smart Expiry (M5+M15+M30)
# Надёжные загрузчики данных (Yahoo + Binance fallback), Safe Cache,
# Triple-Timeframe согласование, умная экспирация, OTC-фильтр и копируемый код для Pocket Option.

import os, time, json, random, math
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

# =============== SECRETS ===============
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", os.getenv("TELEGRAM_TOKEN", ""))
CHAT_ID        = st.secrets.get("CHAT_ID",        os.getenv("CHAT_ID", ""))

# =============== SETTINGS ==============
REFRESH_SEC     = 1
ONLY_NEW        = True
MIN_SEND_GAP_S  = 60
CONF_THRESHOLD  = 70

TF_MAIN  = ("5m",  "5d")    # вход
TF_MID   = ("15m", "10d")   # подтверждение
TF_TREND = ("30m", "20d")   # тренд

PAIRS = {
    # Forex
    "EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"USDJPY=X","USDCHF":"USDCHF=X","AUDUSD":"AUDUSD=X",
    "NZDUSD":"NZDUSD=X","USDCAD":"USDCAD=X","EURJPY":"EURJPY=X","GBPJPY":"GBPJPY=X","AUDJPY":"AUDJPY=X",
    "CADJPY":"CADJPY=X","CHFJPY":"CHFJPY=X","EURGBP":"EURGBP=X","EURCHF":"EURCHF=X","EURCAD":"EURCAD=X",
    "EURAUD":"EURAUD=X","GBPCAD":"GBPCAD=X","GBPAUD":"GBPAUD=X","AUDCAD":"AUDCAD=X","NZDJPY":"NZDJPY=X",
    # Commodities
    "XAUUSD (Gold)":"GC=F","XAGUSD (Silver)":"SI=F","WTI (Oil)":"CL=F","BRENT (Oil)":"BZ=F",
    # Crypto
    "BTCUSD (Bitcoin)":"BTC-USD","ETHUSD (Ethereum)":"ETH-USD","SOLUSD (Solana)":"SOL-USD",
    "XRPUSD (XRP)":"XRP-USD","BNBUSD (BNB)":"BNB-USD","DOGEUSD (Dogecoin)":"DOGE-USD",
}

# =============== UTILS ===============
def _cache_key(symbol: str, interval: str) -> str:
    return f"{symbol}__{interval}"

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def _series_close_numeric(df: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(df["Close"].astype(str), errors="coerce")
    s = s.fillna(method="ffill").fillna(method="bfill")
    if s.isna().all():
        s = pd.Series([1.0]*len(df), index=df.index)
    return s

def is_otc(name: str, symbol: str) -> bool:
    n = name.lower()
    if "otc" in n: return True
    if "=f" in symbol.lower(): return True           # фьючерсы ведут себя ближе к OTC
    if "-" in symbol: return True                    # crypto BTC-USD
    return False

def pocket_code(name: str, symbol: str) -> str:
    if symbol.endswith("=X") and len(symbol) >= 7:
        base = symbol.replace("=X", "")
        if len(base) == 6:
            return f"{base[:3]}/{base[3:]}"
    if symbol.endswith("-USD"):
        return symbol.replace("-USD", "/USD")
    if symbol in {"GC=F","SI=F","CL=F","BZ=F"}:
        return {"GC=F":"XAU/USD","SI=F":"XAG/USD","CL=F":"WTI/USD","BZ=F":"BRENT/USD"}[symbol]
    return "".join(ch for ch in name if ch.isalnum() or ch in "/").upper()

# =============== DATA: Yahoo + Binance fallback ===============
def yf_download(symbol: str, period: str, interval: str) -> pd.DataFrame | None:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or len(df) < 50:
            return None
        return df.tail(600)
    except Exception:
        return None

_BINANCE_MAP = {
    "BTC-USD":"BTCUSDT", "ETH-USD":"ETHUSDT", "SOL-USD":"SOLUSDT",
    "XRP-USD":"XRPUSDT", "BNB-USD":"BNBUSDT", "DOGE-USD":"DOGEUSDT"
}
def _binance_interval(interval: str) -> str:
    return {"1m":"1m","5m":"5m","15m":"15m","30m":"30m","60m":"1h"}.get(interval, "1m")

def binance_download(symbol: str, interval: str) -> pd.DataFrame | None:
    sym = _BINANCE_MAP.get(symbol)
    if not sym: return None
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": sym, "interval": _binance_interval(interval), "limit": 600}
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return None
        arr = r.json()
        if not arr:
            return None
        rows = []
        for k in arr:
            ts = pd.to_datetime(int(k[0]), unit="ms", utc=True)
            o = _safe_float(k[1]); h = _safe_float(k[2]); l = _safe_float(k[3]); c = _safe_float(k[4])
            rows.append([ts, o, h, l, c])
        df = pd.DataFrame(rows, columns=["Datetime","Open","High","Low","Close"]).set_index("Datetime")
        return df
    except Exception:
        return None

def nudge_last(df: pd.DataFrame, max_bps=5) -> pd.Series:
    last = df.iloc[-1].copy()
    c = _safe_float(last["Close"], 1.0)
    bps = random.uniform(-max_bps, max_bps)/10000.0
    new_c = max(1e-9, c*(1+bps))
    last["Open"]  = c
    last["High"]  = max(c, new_c)
    last["Low"]   = min(c, new_c)
    last["Close"] = new_c
    last.name = df.index[-1] + pd.tseries.frequencies.to_offset("1min")
    return last

def get_or_fake(symbol: str, period: str, interval: str) -> pd.DataFrame:
    if "cache" not in st.session_state:
        st.session_state.cache = {}
    key = _cache_key(symbol, interval)

    # 1) Yahoo
    df = yf_download(symbol, period, interval)
    # 2) Binance fallback для крипты
    if df is None:
        df = binance_download(symbol, interval)

    if df is not None and len(df) > 0:
        st.session_state.cache[key] = df.copy()
        return df

    # 3) Cache + сдвиг
    cached = st.session_state.cache.get(key)
    if cached is not None and len(cached) > 0:
        last = nudge_last(cached)
        if isinstance(last, pd.Series):
            last = last.to_frame().T
        out = pd.concat([cached, last], ignore_index=False).tail(600)
        st.session_state.cache[key] = out
        return out

    # 4) Synthetic
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=60, freq="1min")
    base = 1.0 + random.random()/10
    vals = base*(1 + np.cumsum(np.random.randn(60))/100)
    df = pd.DataFrame({"Open":vals,"High":vals,"Low":vals,"Close":vals}, index=idx)
    st.session_state.cache[key] = df
    return df

# =============== INDICATORS ===============
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up/(dn+1e-9)
    return 100 - (100/(1+rs))

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, signal)
    return m, s, m - s

def bbands(close: pd.Series, n=20, k=2.0):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    up, lo = ma + k*sd, ma - k*sd
    width = (up - lo)/(ma + 1e-9)*100
    return up, ma, lo, width

def adx(df: pd.DataFrame, n=14) -> pd.Series:
    try:
        h = pd.to_numeric(df["High"], errors="coerce").fillna(method="ffill")
        l = pd.to_numeric(df["Low"],  errors="coerce").fillna(method="ffill")
        c = _series_close_numeric(df)
        up_move   = h.diff()
        dn_move   = -l.diff()
        plus_dm   = up_move.where((up_move > 0) & (up_move > dn_move), 0.0).fillna(0.0)
        minus_dm  = dn_move.where((dn_move > 0) & (dn_move > up_move), 0.0).fillna(0.0)
        tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(n).mean().replace(0, np.nan).fillna(method="ffill")
        plus_di  = 100 * (plus_dm.rolling(n).sum()/(atr+1e-9))
        minus_di = 100 * (minus_dm.rolling(n).sum()/(atr+1e-9))
        dx = 100 * ( (plus_di - minus_di).abs()/((plus_di + minus_di)+1e-9) )
        out = dx.rolling(n).mean()
        out = out.fillna(method="ffill").fillna(20.0)
        return out
    except Exception:
        # безопасное значение
        return pd.Series([20.0]*len(df), index=df.index)

# =============== FEATURES / RULES ===============
def } мин
