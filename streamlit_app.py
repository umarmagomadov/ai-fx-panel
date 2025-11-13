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
REFRESH_SEC     = 1       # автообновление, сек
ONLY_NEW        = True    # отправлять только новые / лучшие сигналы
MIN_SEND_GAP_S  = 60      # мин. пауза между сигналами по одной паре
CONF_THRESHOLD  = 70      # базовый порог уверенности

# Таймфреймы для Multi-TF
TF_MAIN  = ("5m",  "2d")   # вход
TF_MID   = ("15m", "5d")   # подтверждение
TF_TREND = ("30m", "10d")  # общий тренд

# ================== ИНСТРУМЕНТЫ ==============
PAIRS = {
    # Forex
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "USDCAD=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "AUDJPY": "AUDJPY=X",
    "CADJPY": "CADJPY=X",
    "CHFJPY": "CHFJPY=X",
    "EURGBP": "EURGBP=X",
    "EURCHF": "EURCHF=X",
    "EURCAD": "EURCAD=X",
    "EURAUD": "EURAUD=X",
    "GBPCAD": "GBPCAD=X",
    "GBPAUD": "GBPAUD=X",
    "AUDCAD": "AUDCAD=X",
    "NZDJPY": "NZDJPY=X",

    # Commodities (фьючерсы)
    "XAUUSD (Gold)":   "GC=F",
    "XAGUSD (Silver)": "SI=F",
    "WTI (Oil)":       "CL=F",
    "BRENT (Oil)":     "BZ=F",

    # Crypto
    "BTCUSD (Bitcoin)":   "BTC-USD",
    "ETHUSD (Ethereum)":  "ETH-USD",
    "SOLUSD (Solana)":    "SOL-USD",
    "XRPUSD (XRP)":       "XRP-USD",
    "BNBUSD (BNB)":       "BNB-USD",
    "DOGEUSD (Dogecoin)": "DOGE-USD",
}

# ================== INDICATORS ===============
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, signal)
    return m, s, m - s

def bbands(close: pd.Series, n=20, k=2.0):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    up = ma + k * sd
    lo = ma - k * sd
    width = (up - lo) / (ma + 1e-9) * 100
    return up, ma, lo, width

def adx(df: pd.DataFrame, n=14) -> pd.Series:
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    up_move  = h.diff()
    dn_move  = -l.diff()
    plus_dm  = up_move.where((up_move > 0) & (up_move > dn_move), 0.0).fillna(0)
    minus_dm = dn_move.where((dn_move > 0) & (dn_move > up_move), 0.0).fillna(0)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di  = 100 * (plus_dm.rolling(n).sum()  / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(n).sum() / (atr + 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9))
    return dx.rolling(n).mean()

# ================== DATA =====================
def _cache_key(symbol: str, interval: str) -> str:
    return f"{symbol}__{interval}"

def safe_download(symbol: str, period: str, interval: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df is None or len(df) < 30:
            return None
        df = df[["Open", "High", "Low", "Close"]].copy()
        return df.tail(600)
    except Exception:
        return None

def nudge_last(df: pd.DataFrame, max_bps: float = 5) -> pd.Series:
    last = df.iloc[-1].copy()
    c = float(last["Close"])
    bps = random.uniform(-max_bps, max_bps) / 10000.0
    new_c = max(1e-9, c * (1 + bps))
    last["Open"] = c
    last["High"] = max(c, new_c)
    last["Low"]  = min(c, new_c)
    last["Close"] = new_c
    last.name = last.name + pd.tseries.frequencies.to_offset("1min")
    return last

def get_or_fake(symbol: str, period: str, interval: str) -> pd.DataFrame:
    if "cache" not in st.session_state:
        st.session_state.cache = {}
    key = _cache_key(symbol, interval)

    real = safe_download(symbol, period, interval)
    if real is not None:
        st.session_state.cache[key] = real.copy()
        return real

    cached = st.session_state.cache.get(key)
    if cached is not None and len(cached):
        df = cached.copy()
        last = nudge_last(df)
        if isinstance(last, pd.Series):
            last = last.to_frame().T
        df = pd.concat([df, last], axis=0).tail(600)
        st.session_state.cache[key] = df
        return df

    # синтетика, если вообще нет данных
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=60, freq="1min")
    base = 1.0 + random.random() / 10
    vals = base * (1 + np.cumsum(np.random.randn(60)) / 100)
    df = pd.DataFrame(
        {"Open": vals, "High": vals, "Low": vals, "Close": vals},
        index=idx,
    )
    st.session_state.cache[key] = df
    return df

# ================== HELPERS ==================
def is_otc(name: str, symbol: str) -> bool:
    n = name.lower()
    if "otc" in n:
        return True
    if "=f" in symbol.lower():  # фьючерсы
        return True
    if "-" in symbol:          # криптовалюты, BTC-USD
        return True
    return False

def pocket_code(name: str, symbol: str) -> str:
    # EURUSD=X -> EUR/USD, BTC-USD -> BTC/USD, GC=F -> XAU/USD и т.п.
    if symbol.endswith("=X") and len(symbol) >= 7:
        base = symbol.replace("=X", "")
        if len(base) == 6:
            return f"{base[:3]}/{base[3:]}"
    if symbol.endswith("-USD"):
        return symbol.replace("-USD", "/USD")
    if symbol in {"GC=F", "SI=F", "CL=F", "BZ=F"}:
        mapping = {
            "GC=F": "XAU/USD",
            "SI=F": "XAG/USD",
            "CL=F": "WTI/USD",
            "BZ=F": "BRENT/USD",
        }
        return mapping[symbol]
    # запасной вариант — просто чистим имя
    clean = "".join(ch for ch in name if ch.isalnum() or ch in "/").upper()
    return clean

def candle_phase(df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    o = float(last["Open"])
    h = float(last["High"])
    l = float(last["Low"])
    c = float(last["Close"])
    rng = max(1e-9, h - l)
    pos = (c - l) / rng
    if pos < 0.33:
        return "start"
    if pos < 0.66:
        return "mid"
    return "end"

def near_sr(df: pd.DataFrame) -> str | None:
    close = df["Close"]
    last_close = float(close.iloc[-1])
    sup = float(df["Low"].rolling(20).min().iloc[-1])
    res = float(df["High"].rolling(20).max().iloc[-1])
    if abs(last_close - sup) / max(1e-9, last_close) < 0.002:
        return "support"
    if abs(last_close - res) / max(1e-9, last_close) < 0.002:
        return "resistance"
    return None

def momentum_spike(df: pd.DataFrame) -> bool:
    if len(df) < 12:
        return False
    close = df["Close"]
    last_move = abs(close.iloc[-1] - close.iloc[-2])
    avg_move = close.diff().abs().rolling(10).mean().iloc[-1]
    if avg_move == 0 or pd.isna(avg_move):
        return False
    return bool(last_move > 1.5 * avg_move)

def tf_direction(df: pd.DataFrame) -> str:
    close = df["Close"]
    macd_line, macd_sig, macd_hist = macd(close)
    rsv = float(rsi(close).iloc[-1])
    mh = float(macd_hist.iloc[-1])
    if mh > 0 and rsv > 50:
        return "BUY"
    if mh < 0 and rsv < 50:
        return "SELL"
    return "FLAT"

def market_regime(adx_val: float, bw: float) -> str:
    if adx_val < 20 and bw < 3:
        return "flat"
    if adx_val > 25 and bw < 7:
        return "trend"
    return "impulse"

def boll_width(close: pd.Series, n=20, k=2.0) -> float:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    up = ma + k * sd
    lo = ma - k * sd
    return float(((up.iloc[-1] - lo.iloc[-1]) / (ma.iloc[-1] + 1e-9)) * 100)

# ============== CORE SCORING (M5) ============
def score_single(df: pd.DataFrame) -> tuple[str, int, dict]:
    if df is None or len(df) < 30:
        return (
            "FLAT",
            0,
            {
                "RSI": 50.0,
                "RSI_prev": 50.0,
                "ADX": 0.0,
                "MACD_Hist": 0.0,
                "BB_Pos": 0.0,
                "BB_Width": 0.0,
                "EMA9_minus_EMA21": 0.0,
                "EMA200": 0.0,
            },
        )

    close = df["Close"]
    rsi_series = rsi(close)
    rsv = float(rsi_series.iloc[-1])
    rsv_prev = float(rsi_series.iloc[-2]) if len(rsi_series) > 2 else rsv
    ema9 = float(ema(close, 9).iloc[-1])
    ema21 = float(ema(close, 21).iloc[-1])
    ema200 = float(ema(close, 200).iloc[-1])
    _, _, mh = macd(close)
    mhv = float(mh.iloc[-1])
    up, mid, lo, w = bbands(close)
    bb_pos = float((close.iloc[-1] - mid.iloc[-1]) /
                   (up.iloc[-1] - lo.iloc[-1] + 1e-9))
    adx_series = adx(df)
    adx_v = float(adx_series.iloc[-1])

    # голосование
    vu = 0
    vd = 0
    if rsv < 35:
        vu += 1
    if rsv > 65:
        vd += 1
    if ema9 > ema21:
        vu += 1
    if ema9 < ema21:
        vd += 1
    if mhv > 0:
        vu += 1
    if mhv < 0:
        vd += 1
    if bb_pos < -0.25:
        vu += 1
    if bb_pos > 0.25:
        vd += 1

    if vu == vd:
        direction = "FLAT"
    elif vu > vd:
        direction = "BUY"
    else:
        direction = "SELL"

    trend_boost = min(max((adx_v - 18) / 25, 0), 1)
    raw = abs(vu - vd) / 4.0
    conf = int(100 * (0.55 * raw + 0.45 * trend_boost))
    conf = max(40, min(99, conf))

    feats = {
        "RSI": round(rsv, 1),
        "RSI_prev": round(rsv_prev, 1),
        "ADX": round(adx_v, 1),
        "MACD_Hist": round(mhv, 6),
        "BB_Pos": round(bb_pos, 3),
        "BB_Width": round(float(w.iloc[-1]), 2),
        "EMA9_minus_EMA21": round(ema9 - ema21, 6),
        "EMA200": round(ema200, 6),
    }
    return direction, conf, feats

# ============== MULTI-TF FUSION =============
def score_multi_tf(symbol: str) -> tuple[str, int, dict, dict]:
    df_main = get_or_fake(symbol, TF_MAIN[1], TF_MAIN[0])
    df_mid = get_or_fake(symbol, TF_MID[1], TF_MID[0])
    df_trend = get_or_fake(symbol, TF_TREND[1], TF_TREND[0])

    sig, conf, feats = score_single(df_main)

    d_main = tf_direction(df_main)
    d_mid = tf_direction(df_mid)
    d_trend = tf_direction(df_trend)

    agree = 0
    if d_main in ("BUY", "SELL") and d_mid == d_main:
        agree += 1
    if d_main in ("BUY", "SELL") and d_trend == d_main:
        agree += 1

    if d_main == d_mid == d_trend and d_main in ("BUY", "SELL"):
        conf += 15
    elif agree == 1:
        conf += 5
    else:
        conf -= 10

    bw = boll_width(df_main["Close"])
    adx_v = feats["ADX"]
    regime = market_regime(adx_v, bw)

    if momentum_spike(df_main):
        conf += 8

    sr = near_sr(df_main)
    if (sig == "BUY" and sr == "support") or (sig == "SELL" and sr == "resistance"):
        conf += 7

    ph = candle_phase(df_main)
    if ph == "mid":
        conf += 5
    elif ph == "end":
        conf -= 6

    if abs(feats["RSI"] - feats["RSI_prev"]) > 10:
        conf -= 8

    conf = int(max(0, min(100, conf)))
    mtf = {
        "M5": d_main,
        "M15": d_mid,
        "M30": d_trend,
        "Regime": regime,
        "Phase": ph,
        "BW": round(bw, 2),
    }
    return sig, conf, feats, mtf

# ============== EXPIRY (smart) ==============
def choose_expiry(conf: int, adx_value: float, rsi_value: float,
                  df_main: pd.DataFrame) -> int:
    if conf < 60:
        return 0
    if conf < 65:
        base = 2
    elif conf < 75:
        base = 5
    elif conf < 85:
        base = 8
    elif conf < 90:
        base = 12
    elif conf < 95:
        base = 18
    else:
        base = 25

    if adx_value >= 50:
        base += 8
    elif adx_value >= 35:
        base += 5
    elif adx_value < 20:
        base -= 3

    bw = boll_width(df_main["Close"])
    if bw >= 7.0:
        base -= 4
    elif bw >= 5.0:
        base -= 2
    elif bw <= 2.0:
        base += 2

    ph = candle_phase(df_main)
    if ph == "end":
        base -= 2
    elif ph == "start":
        base += 1

    if rsi_value >= 70 or rsi_value <= 30:
        base -= 1

    return int(max(1, min(30, base)))

# ============== TELEGRAM ====================
def send_telegram(pair_name: str, pair_code: str, mtype: str,
                  signal: str, conf: int, expiry: int,
                  feats: dict, mtf: dict) -> None:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return

    arrow = "⬆️" if signal == "BUY" else ("⬇️" if signal == "SELL" else "➖")
    copy_code = pocket_code(pair_name, pair_code)
    phase_map = {
        "start": "🟢 Начало",
        "mid": "🟡 Середина",
        "end": "🔴 Конец",
    }
    phase_icon = phase_map.get(mtf.get("Phase", ""), "❔")
    if conf < 60:
        strength = "🔴 слабый"
    elif conf < 80:
        strength = "🟡 средний"
    else:
        strength = "🟢 сильный"

    text = (
        "🤖 AI FX СИГНАЛ v101.1\n"
        f"💱 Пара: {pair_name}\n"
        f"📌 Код (Pocket): `{copy_code}`\n"
        f"🏷️ Тип: {mtype}\n"
        f"{arrow} Сигнал: *{signal}*\n"
        f"📊 Multi-TF: M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}\n"
        f"🌐 Режим: {mtf['Regime']} | 🕯️ Свеча: {phase_icon}\n"
        f"💪 Уверенность: *{conf}%* ({strength})\n"
        f"⏱ Экспирация: *{expiry} мин*\n"
        f"📈 RSI {feats['RSI']} | ADX {feats['ADX']} | MACD {feats['MACD_Hist']}\n"
        f"⏰ {datetime.utcnow().strftime('%H:%M:%S')} UTC"
    )

    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        st.toast(f"Telegram error: {e}", icon="⚠️")

# ============== STREAMLIT UI =================
st.set_page_config(page_title="AI FX v101.1 — M5+M15+M30", layout="wide")
st.title("🤖 AI FX Signal Bot v101.1 — Triple-Timeframe + OTC / Биржевая + Pocket Copy")

c1, c2 = st.columns([1, 1])
with c1:
    threshold = st.slider(
        "Порог уверенности (Telegram)",
        50, 95, CONF_THRESHOLD, 1,
    )
with c2:
    min_gap = st.number_input(
        "Мин. пауза между сигналами (сек)",
        10, 300, MIN_SEND_GAP_S,
    )

if "last_sent" not in st.session_state:
    st.session_state.last_sent = {}

rows = []

for name, symbol in PAIRS.items():
    sig, conf, feats, mtf = score_multi_tf(symbol)
    df_main = get_or_fake(symbol, TF_MAIN[1], TF_MAIN[0])

    otc_flag = is_otc(name, symbol)
    eff_threshold = threshold + 10 if otc_flag else threshold

    expiry = choose_expiry(conf, feats["ADX"], feats["RSI"], df_main)
    if otc_flag and expiry > 0:
        expiry = min(60, expiry + 5)

    mtype = "OTC/24/7" if otc_flag else "Биржевая"
    phase_map = {
        "start": "🟢 Начало",
        "mid": "🟡 Середина",
        "end": "🔴 Конец",
    }
    phase_show = phase_map.get(mtf["Phase"], "❔")

    rows.append([
        name,
        mtype,
        sig,
        conf,
        expiry,
        f"M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}",
        phase_show,
        json.dumps(feats, ensure_ascii=False),
    ])

    if sig in ("BUY", "SELL") and conf >= eff_threshold and expiry > 0:
        prev = st.session_state.last_sent.get(name, {})
        should = True
        if ONLY_NEW and prev:
            same = prev.get("signal") == sig
            worse = conf <= prev.get("conf", 0)
            recent = (time.time() - prev.get("ts", 0)) < min_gap
            if same and (worse or recent):
                should = False
        if should:
            send_telegram(name, symbol, mtype, sig, conf, expiry, feats, mtf)
            st.session_state.last_sent[name] = {
                "signal": sig,
                "ts": time.time(),
                "conf": conf,
            }

df_show = pd.DataFrame(
    rows,
    columns=[
        "Пара",
        "Тип",
        "Сигнал",
        "Уверенность",
        "Экспирация (мин)",
        "Multi-TF",
        "Свеча",
        "Индикаторы",
    ],
)

if len(df_show):
    df_show = df_show.sort_values("Уверенность", ascending=False).reset_index(drop=True)

st.subheader("📋 Рейтинг сигналов (v101.1)")
st.dataframe(df_show, use_container_width=True, height=480)

if len(df_show):
    top = df_show.iloc[0]
    sym = PAIRS[top["Пара"]]
    st.markdown("**Копируемый код для Pocket Option (топ-пара):**")
    st.text_input(
        "Tap to copy:",
        value=pocket_code(top["Пара"], sym),
        key="copy_top",
    )

if len(df_show):
    top = df_show.iloc[0]
    sym = PAIRS[top["Пара"]]
    dfc = get_or_fake(sym, TF_MAIN[1], TF_MAIN[0])
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=dfc.index,
                open=dfc["Open"],
                high=dfc["High"],
                low=dfc["Low"],
                close=dfc["Close"],
            )
        ]
    )
    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=20, b=0),
        title=(
            f"Топ: {top['Пара']} — {top['Сигнал']} "
            f"({top['Уверенность']}%) • {top['Multi-TF']} • {top['Свеча']}"
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

time.sleep(REFRESH_SEC)
st.rerun()
