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

# ================== SECRETS ==================
# Токен и чат берем из секретов / переменных окружения
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", os.getenv("TELEGRAM_TOKEN", ""))
CHAT_ID        = st.secrets.get("CHAT_ID",        os.getenv("CHAT_ID", ""))

# ================== SETTINGS =================
REFRESH_SEC     = 1       # автообновление, сек
ONLY_NEW        = True    # отправлять только новые / лучшие сигналы
MIN_SEND_GAP_S  = 60      # мин. пауза между сигналами по одной паре
BASE_THRESHOLD  = 70      # базовый порог уверенности

# Таймфреймы для Multi-TF (M1+M5+M15+M30)
TF_1M  = ("1m",  "2d")    # вход
TF_5M  = ("5m",  "5d")    # подтверждение
TF_15M = ("15m", "10d")   # среднесрочный
TF_30M = ("30m", "30d")   # общий тренд

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

# ================== МЕЛКИЕ ХЕЛПЕРЫ ==============
def safe_float(x, default: float = 0.0) -> float:
    """Безопасно конвертирует всё во float. NaN / ошибка -> default."""
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
    """Подделываем последнюю свечу, если реальных данных нет (демо)."""
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
    """Скачиваем или генерируем данные (для демо/ошибок)."""
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
    if "-" in symbol:          # крипта BTC-USD
        return True
    return False

def pocket_code(name: str, symbol: str) -> str:
    """Код вида EUR/USD, BTC/USD и т.п. удобный для Pocket Option."""
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
    o = safe_float(last["Open"])
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
    """Близко ли цена к простым уровням поддержки/сопротивления."""
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
    """Импульс по последней свече (без падения при нулевой волатильности)."""
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
    macd_line, macd_sig, macd_hist = macd(close)
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

# ============== CORE SCORING для одного TF ============
def score_single(df: pd.DataFrame) -> tuple[str, int, dict]:
    """RSI + EMA + MACD + BB + ADX -> направление + уверенность."""
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

    # RSI
    rsi_series = rsi(close)
    rsv = safe_float(rsi_series.iloc[-1], 50.0)
    rsv_prev = safe_float(rsi_series.iloc[-2], rsv) if len(rsi_series) > 2 else rsv

    # EMA
    ema9 = safe_float(ema(close, 9).iloc[-1], rsv)
    ema21 = safe_float(ema(close, 21).iloc[-1], rsv)
    ema200 = safe_float(ema(close, 200).iloc[-1], rsv)

    # MACD
    _, _, mh = macd(close)
    mhv = safe_float(mh.iloc[-1], 0.0)

    # Bollinger
    up, mid, lo, w = bbands(close)
    bb_pos = safe_float(
        (close.iloc[-1] - mid.iloc[-1]) /
        (up.iloc[-1] - lo.iloc[-1] + 1e-9),
        0.0,
    )
    w_last = safe_float(w.iloc[-1], 0.0)

    # ADX
    adx_series = adx(df)
    adx_v = safe_float(adx_series.iloc[-1], 0.0)

    # Голосование
    vu = 0
    vd = 0
    # RSI
    if rsv < 35:
        vu += 1
    if rsv > 65:
        vd += 1
    # EMA 9/21
    if ema9 > ema21:
        vu += 1
    if ema9 < ema21:
        vd += 1
    # MACD
    if mhv > 0:
        vu += 1
    if mhv < 0:
        vd += 1
    # Bollinger
    if bb_pos < -0.25:
        vu += 1
    if bb_pos > 0.25:
        vd += 1

    # Направление
    if vu == vd:
        direction = "FLAT"
    elif vu > vd:
        direction = "BUY"
    else:
        direction = "SELL"

    # База уверенности
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

# ============== MULTI-TF (M1+M5+M15+M30) ============
def score_multi_tf(symbol: str) -> tuple[str, int, dict, dict]:
    # Данные по всем TF
    df_1m  = get_or_fake(symbol, TF_1M[1],  TF_1M[0])
    df_5m  = get_or_fake(symbol, TF_5M[1],  TF_5M[0])
    df_15m = get_or_fake(symbol, TF_15M[1], TF_15M[0])
    df_30m = get_or_fake(symbol, TF_30M[1], TF_30M[0])

    # Основной анализ по M1
    sig_1m, conf_1m, feats = score_single(df_1m)

    # Направления по всем TF (MACD+RSI)
    d_1m  = tf_direction(df_1m)
    d_5m  = tf_direction(df_5m)
    d_15m = tf_direction(df_15m)
    d_30m = tf_direction(df_30m)

    # Базовый сигнал = M1
    base_sig = sig_1m if sig_1m != "FLAT" else d_1m
    if base_sig == "FLAT":
        base_sig = d_5m

    if base_sig not in ("BUY", "SELL"):
        return "FLAT", 0, feats, {
            "M1": d_1m,
            "M5": d_5m,
            "M15": d_15m,
            "M30": d_30m,
            "Regime": "flat",
            "Phase": candle_phase(df_1m),
            "BW": 0.0,
        }

    conf = conf_1m

    # ---------- Стратегия 1: Triple / Quad TF Match ----------
    dirs = [d_1m, d_5m, d_15m, d_30m]
    same_count = sum(1 for d in dirs if d == base_sig)
    if same_count == 4:
        conf += 25   # все TF в одну сторону
    elif same_count == 3:
        conf += 18
    elif same_count == 2:
        conf += 8
    else:
        conf -= 10   # M1 одиночка

    # ---------- Стратегия 2: RSI Reverse / Extreme ----------
    rsi_val = feats["RSI"]
    if base_sig == "BUY" and rsi_val < 30:
        conf += 10
    if base_sig == "SELL" and rsi_val > 70:
        conf += 10
    if 45 <= rsi_val <= 55:
        conf -= 5  # середина диапазона = слабый сигнал

    # ---------- Стратегия 3: Momentum Spike ----------
    if momentum_spike(df_1m):
        conf += 7

    # ---------- Стратегия 4: S/R Bounce ----------
    sr = near_sr(df_1m)
    if (base_sig == "BUY" and sr == "support") or (base_sig == "SELL" and sr == "resistance"):
        conf += 8

    # ---------- Стратегия 5: Regime + Bollinger ----------
    bw = boll_width(df_1m["Close"])
    adx_val = feats["ADX"]
    regime = market_regime(adx_val, bw)

    if regime == "trend":
        conf += 8
    elif regime == "flat":
        conf -= 5

    # ---------- Фаза свечи на M1 ----------
    phase = candle_phase(df_1m)
    if phase == "end":
        conf -= 5
    elif phase == "start":
        conf += 2

    # ---------- Анти-скачки RSI ----------
    if abs(feats["RSI"] - feats["RSI_prev"]) > 12:
        conf -= 6

    conf = int(max(0, min(100, conf)))

    mtf = {
        "M1": d_1m,
        "M5": d_5m,
        "M15": d_15m,
        "M30": d_30m,
        "Regime": regime,
        "Phase": phase,
        "BW": round(bw, 2),
    }
    return base_sig, conf, feats, mtf

# ============== EXPIRY (умный выбор МИНУТ) ============
def choose_expiry(conf: int, adx_value: float, rsi_value: float,
                  df_1m: pd.DataFrame) -> int:
    if conf < 60:
        return 0  # пропускаем слабое

    if conf < 70:
        base = 1
    elif conf < 80:
        base = 3
    elif conf < 88:
        base = 5
    elif conf < 94:
        base = 8
    elif conf < 98:
        base = 12
    else:
        base = 18  # Ultra 99

    # ADX (тренд)
    if adx_value >= 50:
        base += 6
    elif adx_value >= 35:
        base += 3
    elif adx_value < 20:
        base -= 2

    # Волатильность по Bollinger
    bw = boll_width(df_1m["Close"])
    if bw >= 7.0:
        base -= 3
    elif bw >= 5.0:
        base -= 1
    elif bw <= 2.0:
        base += 2

    # Фаза свечи
    ph = candle_phase(df_1m)
    if ph == "end":
        base -= 2
    elif ph == "start":
        base += 1

    # RSI на экстремумах -> чуть меньше
    if rsi_value >= 75 or rsi_value <= 25:
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
        "start": "🟢 начало",
        "mid": "🟡 середина",
        "end": "🔴 конец",
    }
    phase_icon = phase_map.get(mtf.get("Phase", ""), "❔")
    if conf < 60:
        strength = "🔴 слабый"
    elif conf < 80:
        strength = "🟡 нормальный"
    else:
        strength = "🟢 сильный"

    text = (
        "🤖 AI FX Signal Bot v3.0\n"
        f"💱 Пара: {pair_name}\n"
        f"📌 Код для Pocket: {copy_code}\n"
        f"📋 Тип: {mtype}\n"
        f"{arrow} Сигнал: *{signal}*\n"
        f"📊 TF: M1={mtf['M1']} | M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}\n"
        f"🌐 Рынок: {mtf['Regime']} | 🕯 Свеча: {phase_icon}\n"
        f"💪 Уверенность: *{conf}%* ({strength})\n"
        f"⏱ Экспирация: *{expiry} мин*\n"
        f"📈 RSI {feats['RSI']} | ADX {feats['ADX']} | MACD {feats['MACD_Hist']}\n"
        f"⏰ {datetime.utcnow().strftime('%H:%M:%S')} UTC\n\n"
        "⚠️ Это обучающие сигналы. Торгуй только с головой."
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
st.set_page_config(page_title="AI FX Bot v3.0 — M1+M5+M15+M30 + Telegram", layout="wide")
st.title("🤖 AI FX Bot v3.0 — M1+M5+M15+M30 + Telegram")

c1, c2, c3 = st.columns([1.2, 1, 1])

with c1:
    mode = st.selectbox(
        "Режим отбора сигналов",
        ["Safe 85%", "Normal 90%", "Pro 95%", "Ultra 99%"],
        index=3,
    )

with c2:
    slider_thr = st.slider(
        "Минимальная уверенность (%) для сигнала",
        50, 99, 95, 1,
    )

with c3:
    min_gap = st.number_input(
        "Пауза между сигналами по паре (сек)",
        10, 300, MIN_SEND_GAP_S, 5,
    )

mode_base = {
    "Safe 85%": 85,
    "Normal 90%": 90,
    "Pro 95%": 95,
    "Ultra 99%": 99,
}[mode]

work_threshold = max(mode_base, slider_thr)

st.write(f"**Текущий рабочий порог для отправки сигналов: {work_threshold}%**")

if "last_sent" not in st.session_state:
    st.session_state.last_sent = {}

rows = []

# ============== ОСНОВНОЙ ЦИКЛ ПО ПАРАМ ==============
for name, symbol in PAIRS.items():
    sig, conf, feats, mtf = score_multi_tf(symbol)
    df_1m = get_or_fake(symbol, TF_1M[1], TF_1M[0])

    otc_flag = is_otc(name, symbol)
    eff_threshold = work_threshold + 5 if otc_flag else work_threshold

    expiry = choose_expiry(conf, feats["ADX"], feats["RSI"], df_1m)
    if otc_flag and expiry > 0:
        expiry = min(60, expiry + 3)

    mtype = "OTC/24/7" if otc_flag else "Биржевая"
    phase_map = {
        "start": "🟢 начало",
        "mid": "🟡 середина",
        "end": "🔴 конец",
    }
    phase_show = phase_map.get(mtf["Phase"], "❔")

    rows.append([
        name,
        mtype,
        sig,
        conf,
        expiry,
        f"M1={mtf['M1']} | M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}",
        phase_show,
        json.dumps(feats, ensure_ascii=False),
    ])

    # ----- Отправка в Telegram -----
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

# ============== ТАБЛИЦА =====================
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

st.subheader
