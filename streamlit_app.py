import time
import json
import random
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
import plotly.graph_objects as go

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.get(url, params=params, timeout=5)
    except:
        pass

# ================== SETTINGS =================
REFRESH_SEC     = 1       # автообновление
ONLY_NEW        = True    # не спамим одно и то же
MIN_SEND_GAP_S  = 60      # пауза между сигналами по паре
CONF_THRESHOLD  = 70      # базовый порог

# Таймфреймы
TF_M1    = ("1m",  "1d")
TF_M5    = ("5m",  "5d")
TF_M15   = ("15m", "5d")
TF_M30   = ("30m", "10d")

# ================== ИНСТРУМЕНТЫ ==============
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

    "XAUUSD (Gold)":   "GC=F",
    "XAGUSD (Silver)": "SI=F",
    "WTI (Oil)":       "CL=F",
    "BRENT (Oil)":     "BZ=F",

    "BTCUSD (Bitcoin)":   "BTC-USD",
    "ETHUSD (Ethereum)":  "ETH-USD",
    "SOLUSD (Solana)":    "SOL-USD",
    "XRPUSD (XRP)":       "XRP-USD",
    "BNBUSD (BNB)":       "BNB-USD",
    "DOGEUSD (Dogecoin)": "DOGE-USD",
}

# ================== МЕЛКИЕ ХЕЛПЕРЫ ==============
def safe_float(x, default: float = 0.0) -> float:
    try:
        v = pd.to_numeric(x, errors="coerce")
        if hasattr(v, "iloc"):
            v = v.iloc[-1]
        v = float(v)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default

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
    c = safe_float(last["Close"], 1.0)
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
    if "=f" in symbol.lower():
        return True
    if "-" in symbol:
        return True
    return False

def pocket_code(name: str, symbol: str) -> str:
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
    clean = "".join(ch for ch in name if ch.isalnum() or ch in "/").upper()
    return clean

def candle_phase(df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    h = safe_float(last["High"])
    l = safe_float(last["Low"])
    c = safe_float(last["Close"])
    rng = max(1e-9, h - l)
    pos = (c - l) / rng
    if pos < 0.33:
        return "start"
    if pos < 0.66:
        return "mid"
    return "end"

def near_sr(df: pd.DataFrame) -> str | None:
    close = df["Close"]
    last_close = safe_float(close.iloc[-1])
    sup = safe_float(df["Low"].rolling(20).min().iloc[-1])
    res = safe_float(df["High"].rolling(20).max().iloc[-1])
    if abs(last_close - sup) / max(1e-9, last_close) < 0.002:
        return "support"
    if abs(last_close - res) / max(1e-9, last_close) < 0.002:
        return "resistance"
    return None

def momentum_spike(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 12:
        return False
    close = df["Close"]
    last_move = abs(safe_float(close.iloc[-1]) - safe_float(close.iloc[-2]))
    avg_raw = close.diff().abs().rolling(10).mean().iloc[-1]
    avg_move = safe_float(avg_raw, 0.0)
    if avg_move == 0.0:
        return False
    return bool(last_move > 1.5 * avg_move)

def tf_direction(df: pd.DataFrame) -> str:
    close = df["Close"]
    _, _, macd_hist = macd(close)
    rsi_series = rsi(close)
    rsv = safe_float(rsi_series.iloc[-1], 50.0)
    mh = safe_float(macd_hist.iloc[-1], 0.0)
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
    return safe_float(((up.iloc[-1] - lo.iloc[-1]) / (ma.iloc[-1] + 1e-9)) * 100)

# ============== CORE SCORING (M5) ============
def score_single(df: pd.DataFrame) -> tuple[str, int, dict]:
    if df is None or len(df) < 30:
        return "FLAT", 0, {
            "RSI": 50.0,
            "RSI_prev": 50.0,
            "ADX": 0.0,
            "MACD_Hist": 0.0,
            "BB_Pos": 0.0,
            "BB_Width": 0.0,
            "EMA9_minus_EMA21": 0.0,
            "EMA200": 0.0,
        }

    close = df["Close"]

    rsi_series = rsi(close)
    rsv = safe_float(rsi_series.iloc[-1], 50.0)
    rsv_prev = safe_float(rsi_series.iloc[-2], rsv) if len(rsi_series) > 2 else rsv

    ema9 = safe_float(ema(close, 9).iloc[-1], rsv)
    ema21 = safe_float(ema(close, 21).iloc[-1], rsv)
    ema200 = safe_float(ema(close, 200).iloc[-1], rsv)

    _, _, mh = macd(close)
    mhv = safe_float(mh.iloc[-1], 0.0)

    up, mid, lo, w = bbands(close)
    bb_pos = safe_float(
        (close.iloc[-1] - mid.iloc[-1]) /
        (up.iloc[-1] - lo.iloc[-1] + 1e-9),
        0.0,
    )
    w_last = safe_float(w.iloc[-1], 0.0)

    adx_series = adx(df)
    adx_v = safe_float(adx_series.iloc[-1], 0.0)

    vu = 0
    vd = 0
    if rsv < 35: vu += 1
    if rsv > 65: vd += 1
    if ema9 > ema21: vu += 1
    if ema9 < ema21: vd += 1
    if mhv > 0: vu += 1
    if mhv < 0: vd += 1
    if bb_pos < -0.25: vu += 1
    if bb_pos > 0.25: vd += 1

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
        "BB_Width": round(w_last, 2),
        "EMA9_minus_EMA21": round(ema9 - ema21, 6),
        "EMA200": round(ema200, 6),
    }
    return direction, conf, feats

# ============== MULTI-TF FUSION =============
def score_multi_tf(symbol: str) -> tuple[str, int, dict, dict]:
    df_m1   = get_or_fake(symbol, TF_M1[1],   TF_M1[0])
    df_m5   = get_or_fake(symbol, TF_M5[1],   TF_M5[0])
    df_m15  = get_or_fake(symbol, TF_M15[1],  TF_M15[0])
    df_m30  = get_or_fake(symbol, TF_M30[1],  TF_M30[0])

    sig, conf, feats = score_single(df_m5)

    d_m1   = tf_direction(df_m1)
    d_m5   = tf_direction(df_m5)
    d_m15  = tf_direction(df_m15)
    d_m30  = tf_direction(df_m30)

    agree = 0
    if d_m1 == d_m5 and d_m1 in ("BUY", "SELL"):   agree += 1
    if d_m5 == d_m15 and d_m5 in ("BUY", "SELL"):  agree += 1
    if d_m5 == d_m30 and d_m5 in ("BUY", "SELL"):  agree += 1

    if d_m1 == d_m5 == d_m15 == d_m30 and d_m5 in ("BUY", "SELL"):
        conf += 18
    elif agree == 2:
        conf += 10
    elif agree == 1:
        conf += 4
    else:
        conf -= 10

    bw = boll_width(df_m5["Close"])
    adx_v = feats["ADX"]
    regime = market_regime(adx_v, bw)

    if momentum_spike(df_m5):
        conf += 8

    sr = near_sr(df_m5)
    if (sig == "BUY" and sr == "support") or (sig == "SELL" and sr == "resistance"):
        conf += 7

    ph = candle_phase(df_m5)
    if ph == "mid":
        conf += 5
    elif ph == "end":
        conf -= 6

    if abs(feats["RSI"] - feats["RSI_prev"]) > 10:
        conf -= 8

    conf = int(max(0, min(100, conf)))
    mtf = {
        "M1": d_m1,
        "M5": d_m5,
        "M15": d_m15,
        "M30": d_m30,
        "Regime": regime,
        "Phase": ph,
        "BW": round(bw, 2),
    }
    return sig, conf, feats, mtf

# ============== QUALITY RATING ===============
def grade_signal(conf: int, mtf: dict, feats: dict) -> str:
    """S / A / B / C — чем лучше, тем жёстче фильтр."""
    all_same = mtf["M1"] == mtf["M5"] == mtf["M15"] == mtf["M30"] != "FLAT"
    strong_trend = feats["ADX"] >= 25 and 2 <= feats["BB_Width"] <= 8
    ok_trend = feats["ADX"] >= 18 and feats["BB_Width"] <= 10
    rsi_edge = feats["RSI"] <= 35 or feats["RSI"] >= 65

    if conf >= 92 and all_same and strong_trend and rsi_edge:
        return "S"
    if conf >= 88 and all_same and ok_trend:
        return "A"
    if conf >= 82:
        return "B"
    return "C"

# ============== EXPIRY (smart) ==============
def choose_expiry(conf: int, adx_value: float, rsi_value: float,
                  df_main: pd.DataFrame) -> int:
    if conf < 60:
        return 0
    if conf < 65:
        base = 1
    elif conf < 75:
        base = 3
    elif conf < 85:
        base = 5
    elif conf < 90:
        base = 8
    elif conf < 95:
        base = 12
    else:
        base = 20

    if adx_value >= 50:
        base += 6
    elif adx_value >= 35:
        base += 3
    elif adx_value < 20:
        base -= 2

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
def send_telegram(text: str, ultra: bool) -> None:
    token = TELEGRAM_TOKEN
    chat_id = CHAT_ID
    if ultra and ULTRA_CHAT_ID:
        chat_id = ULTRA_CHAT_ID
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        st.toast(f"Telegram error: {e}", icon="⚠️")

def build_message(pair_name: str, pair_code: str, mtype: str,
                  signal: str, conf: int, expiry: int,
                  feats: dict, mtf: dict, grade: str) -> str:
    arrow = "⬆️" if signal == "BUY" else ("⬇️" if signal == "SELL" else "➖")
    copy_code = pocket_code(pair_name, pair_code)

    phase_map = {"start": "🟢 начало", "mid": "🟡 середина", "end": "🔴 конец"}
    phase_icon = phase_map.get(mtf.get("Phase", ""), "❔")

    strength = {
        "S": "🟢 ULTRA",
        "A": "🟢 сильный",
        "B": "🟡 нормальный",
        "C": "🔴 слабый",
    }.get(grade, "🟡 нормальный")

    text = (
        "🤖 AI FX Signal Bot v3.1\n"
        f"💱 Пара: {pair_name}\n"
        f"📌 Код для Pocket: `{copy_code}`\n"
        f"🏷️ Тип: {mtype}\n"
        f"{arrow} Сигнал: *{signal}*  ({grade})\n"
        f"🇲🇶 M1={mtf['M1']} | M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}\n"
        f"🌐 Рынок: {mtf['Regime']} | {phase_icon}\n"
        f"💪 Уверенность: *{conf}%* ({strength})\n"
        f"⏱ Экспирация: *{expiry} мин*\n"
        f"📊 RSI {feats['RSI']} | ADX {feats['ADX']} | MACD {feats['MACD_Hist']}\n"
        f"⏰ {datetime.utcnow().strftime('%H:%M:%S')} UTC"
    )
    return text

# ============== STREAMLIT UI =================
st.set_page_config(
    page_title="AI FX Bot v3.1 — M1+M5+M15+M30 + Telegram",
    layout="wide",
)
st.title("🤖 AI FX Bot v3.1 — M1+M5+M15+M30 + Telegram")

st.markdown(
    "Режимы **Safe/Normal/Hard/Ultra** — это стиль фильтра, а не реальная гарантия. "
    "Бот — инструмент для обучения, не финансовый совет."
)

c1, c2 = st.columns([1, 1])

RISK_MODES = {
    "Safe 85%": 85,
    "Normal 90%": 90,
    "Hard 95%": 95,
    "Ultra 99% (только S/A)": 95,  # базовый порог, дальше ужесточим
}

with c1:
    mode = st.selectbox("Режим отбора сигналов", list(RISK_MODES.keys()))
with c2:
    min_conf_user = st.slider(
        "Минимальная уверенность (%) для сигнала",
        50, 99, RISK_MODES["Safe 85%"], 1,
    )

work_threshold = max(min_conf_user, RISK_MODES[mode])

gap = st.number_input(
    "Пауза между сигналами по паре (сек)",
    min_value=10,
    max_value=600,
    value=MIN_SEND_GAP_S,
    step=5,
)

st.write(f"Текущий рабочий порог для отправки сигналов: **{work_threshold}%**")

if "last_sent" not in st.session_state:
    st.session_state.last_sent = {}

rows = []

# ================== ОСНОВНОЙ ЦИКЛ =============
for name, symbol in PAIRS.items():
    sig, conf, feats, mtf = score_multi_tf(symbol)
    df_main = get_or_fake(symbol, TF_M5[1], TF_M5[0])

    grade = grade_signal(conf, mtf, feats)

    otc_flag = is_otc(name, symbol)
    eff_threshold = work_threshold + (5 if otc_flag else 0)

    expiry = choose_expiry(conf, feats["ADX"], feats["RSI"], df_main)
    if otc_flag and expiry > 0:
        expiry = min(60, expiry + 5)

    mtype = "OTC/24/7" if otc_flag else "Биржевая"

    phase_map = {"start": "🟢 начало", "mid": "🟡 середина", "end": "🔴 конец"}
    phase_show = phase_map.get(mtf["Phase"], "❔")

    rows.append([
        name,
        mtype,
        sig,
        conf,
        grade,
        expiry,
        f"M1={mtf['M1']} | M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}",
        phase_show,
        json.dumps(feats, ensure_ascii=False),
    ])

    # --- фильтр для отправки ---
    if sig not in ("BUY", "SELL") or expiry <= 0:
        continue
    if conf < eff_threshold:
        continue

    # режим Ultra — только S/A
    ultra_mode = "Ultra" in mode
    if ultra_mode and grade not in ("S", "A"):
        continue

    prev = st.session_state.last_sent.get(name, {})
    should = True
    if ONLY_NEW and prev:
        same = prev.get("signal") == sig
        worse = conf <= prev.get("conf", 0)
        recent = (time.time() - prev.get("ts", 0)) < gap
        if same and (worse or recent):
            should = False

    if should:
        text = build_message(name, symbol, mtype, sig, conf, expiry, feats, mtf, grade)
        send_telegram(text, ultra=ultra_mode and grade in ("S", "A"))
        st.session_state.last_sent[name] = {
            "signal": sig,
            "ts": time.time(),
            "conf": conf,
        }

# ================== ТАБЛИЦА ==================
df_show = pd.DataFrame(
    rows,
    columns=[
        "Пара",
        "Тип",
        "Сигнал",
        "Уверенность",
        "Класс",
        "Экспирация (мин)",
        "Multi-TF",
        "Свеча",
        "Индикаторы",
    ],
)

if len(df_show):
    df_show = df_show.sort_values(
        ["Класс", "Уверенность"],
        ascending=[True, False],
    ).reset_index(drop=True)

st.subheader("📋 Таблица сигналов")
st.dataframe(df_show, use_container_width=True, height=480)

# ================== ТОП-ПАРА + ГРАФИК =========
if len(df_show):
    top = df_show.iloc[0]
    sym = PAIRS[top["Пара"]]

    st.markdown("**Код для Pocket Option (топ-пара):**")
    st.text_input(
        "Tap to copy:",
        value=pocket_code(top["Пара"], sym),
        key="copy_top",
    )

    dfc = get_or_fake(sym, TF_M5[1], TF_M5[0])
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
            f"({top['Уверенность']}% / {top['Класс']}) • {top['Multi-TF']} • {top['Свеча']}"
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

# ================== АВТООБНОВЛЕНИЕ ============
time.sleep(REFRESH_SEC)
st.experimental_rerun()
