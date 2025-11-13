# streamlit_app.py
# ============================================
#  🤖 AI FX PO Bot v102 — Multi-Strategy
#  M5+M15+M30 • Trend • Countertrend • PO-style
#  Режимы 85/90/95/99 + Telegram + автообновление 1 сек
# ============================================

import os
import time
import json
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# =============== SECRETS =====================
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", os.getenv("TELEGRAM_TOKEN", ""))
CHAT_ID        = st.secrets.get("CHAT_ID",        os.getenv("CHAT_ID", ""))

# =============== SETTINGS ====================
REFRESH_SEC     = 1           # автообновление, сек
ONLY_NEW        = True        # отправлять только новые / лучшие сигналы
MIN_SEND_GAP_S  = 60          # дефолтная пауза между сигналами по одной паре
BASE_THRESHOLD  = 70          # базовый порог уверенности

TF_MAIN  = ("5m",  "2d")      # вход
TF_MID   = ("15m", "5d")      # подтверждение
TF_TREND = ("30m", "10d")     # общий тренд

# =============== INSTRUMENTS =================
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

    # Commodities
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

# =============== HELPERS =====================
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
    df = pd.DataFrame({"Open": vals, "High": vals, "Low": vals, "Close": vals}, index=idx)
    st.session_state.cache[key] = df
    return df

# --------- extra helpers ----------
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

def boll_width(close: pd.Series, n=20, k=2.0) -> float:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    up = ma + k * sd
    lo = ma - k * sd
    return safe_float(((up.iloc[-1] - lo.iloc[-1]) / (ma.iloc[-1] + 1e-9)) * 100)

def market_regime(adx_val: float, bw: float) -> str:
    if adx_val < 20 and bw < 3:
        return "flat"
    if adx_val > 25 and bw < 7:
        return "trend"
    return "impulse"

# =========== BASE M5 SCORE ===================
def score_single(df: pd.DataFrame) -> tuple[str, int, dict]:
    if df is None or len(df) < 30:
        return ("FLAT", 0, {
            "RSI": 50.0, "RSI_prev": 50.0, "ADX": 0.0,
            "MACD_Hist": 0.0, "BB_Pos": 0.0, "BB_Width": 0.0,
            "EMA9_minus_EMA21": 0.0, "EMA200": 0.0
        })

    close = df["Close"]
    rsi_series = rsi(close)
    rsv = safe_float(rsi_series.iloc[-1], 50.0)
    rsv_prev = safe_float(rsi_series.iloc[-2], rsv) if len(rsi_series) > 2 else rsv

    ema9   = safe_float(ema(close, 9).iloc[-1], rsv)
    ema21  = safe_float(ema(close, 21).iloc[-1], rsv)
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

    vu = vd = 0
    if rsv < 35:  vu += 1
    if rsv > 65:  vd += 1
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

# =========== MULTI-TF SCORE ==================
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

def score_multi_tf(symbol: str) -> tuple[str, int, dict, dict, pd.DataFrame]:
    df_main  = get_or_fake(symbol, TF_MAIN[1],  TF_MAIN[0])
    df_mid   = get_or_fake(symbol, TF_MID[1],   TF_MID[0])
    df_trend = get_or_fake(symbol, TF_TREND[1], TF_TREND[0])

    sig, conf, feats = score_single(df_main)

    d_main  = tf_direction(df_main)
    d_mid   = tf_direction(df_mid)
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
    return sig, conf, feats, mtf, df_main

# =========== PO-STYLE STRATEGY (5 CANDLES) ===
def candle_color_size(df: pd.DataFrame, n: int = 5):
    last = df.tail(n)
    bodies = (last["Close"] - last["Open"]).abs()
    med = bodies.median()
    info = []
    for _, row in last.iterrows():
        body = safe_float(row["Close"] - row["Open"])
        color = "G" if body > 0 else "R"
        size = "B" if abs(body) > med else "S"
        info.append((color, size))
    return info  # [(G,B), (R,S), ...]

def strategy_po_pattern(df: pd.DataFrame) -> tuple[str, int]:
    """
    Эмуляция PO-ботов: смотрим последние 5 свечей.
    Возвращаем направление и «уверенность» 0-30.
    """
    if df is None or len(df) < 10:
        return "FLAT", 0

    pattern = candle_color_size(df, 5)
    greens = sum(1 for c, _ in pattern if c == "G")
    reds   = sum(1 for c, _ in pattern if c == "R")

    direction = "FLAT"
    score = 0

    # Примеры правил:
    if greens >= 4 and pattern[-1][0] == "G":
        direction = "SELL"      # ищем откат
        score = 18
    if reds >= 4 and pattern[-1][0] == "R":
        direction = "BUY"
        score = 18

    # паттерн разворота: G,G,R,R,R
    colors = "".join(c for c, _ in pattern)
    if colors.endswith("GGGRR"):
        direction = "SELL"
        score = 22
    if colors.endswith("RRRGG"):
        direction = "BUY"
        score = 22

    # сильная большая последняя свеча по тренду
    last_color, last_size = pattern[-1]
    if last_size == "B":
        if last_color == "G":
            direction = "BUY"
        else:
            direction = "SELL"
        score = max(score, 15)

    return direction, score

# =========== TREND STRATEGY ==================
def strategy_trend(df: pd.DataFrame, feats: dict, mtf: dict) -> tuple[str, int]:
    if df is None or len(df) < 50:
        return "FLAT", 0

    close = df["Close"]
    ema50 = safe_float(ema(close, 50).iloc[-1], feats["EMA200"])
    ema200 = feats["EMA200"]
    rsi_val = feats["RSI"]
    adx_val = feats["ADX"]

    dir_ = "FLAT"
    score = 0

    if ema50 > ema200 and rsi_val > 55 and mtf["M15"] == "BUY" and adx_val > 20:
        dir_ = "BUY"
        score = 20
        if mtf["M30"] == "BUY" and adx_val > 25:
            score = 28

    if ema50 < ema200 and rsi_val < 45 and mtf["M15"] == "SELL" and adx_val > 20:
        dir_ = "SELL"
        score = 20
        if mtf["M30"] == "SELL" and adx_val > 25:
            score = 28

    return dir_, score

# =========== COUNTER-TREND STRATEGY ==========
def strategy_countertrend(df: pd.DataFrame, feats: dict) -> tuple[str, int]:
    if df is None or len(df) < 40:
        return "FLAT", 0

    rsi_val = feats["RSI"]
    bb_pos  = feats["BB_Pos"]
    sr = near_sr(df)

    dir_ = "FLAT"
    score = 0

    # Перекупленность
    if rsi_val > 70 and bb_pos > 0.4:
        dir_ = "SELL"
        score = 18
        if sr == "resistance":
            score = 24

    # Перепроданность
    if rsi_val < 30 and bb_pos < -0.4:
        dir_ = "BUY"
        score = 18
        if sr == "support":
            score = 24

    return dir_, score

# =========== COMBINED STRATEGY ===============
def combine_strategies(base_sig: str, base_conf: int,
                       po_sig: str, po_score: int,
                       tr_sig: str, tr_score: int,
                       ct_sig: str, ct_score: int) -> tuple[str, int, dict]:
    """
    Объединяем 4 стратегии:
    - базовая multi-TF
    - PO-style
    - трендовая
    - контртренд
    """
    votes = {"BUY": 0.0, "SELL": 0.0, "FLAT": 0.0}
    weights = {
        "base": 1.0,
        "po": 0.6,
        "trend": 0.9,
        "ctr": 0.7,
    }

    votes[base_sig] += weights["base"] * (base_conf / 100)
    votes[po_sig]   += weights["po"]   * (po_score / 30 if po_score else 0)
    votes[tr_sig]   += weights["trend"]* (tr_score / 30 if tr_score else 0)
    votes[ct_sig]   += weights["ctr"]  * (ct_score / 30 if ct_score else 0)

    final_sig = max(votes, key=votes.get)
    raw_power = votes[final_sig]

    # Переводим в 0-100
    final_conf = int(40 + 60 * min(1.0, raw_power))
    details = {
        "base": base_conf,
        "po": po_score,
        "trend": tr_score,
        "ctr": ct_score,
        "votes": votes,
    }
    return final_sig, final_conf, details

# =========== EXPIRY ==========================
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

# =========== TELEGRAM ========================
def send_telegram(pair_name: str, pair_code: str, mtype: str,
                  signal: str, conf: int, expiry: int,
                  feats: dict, mtf: dict, strat: dict) -> None:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return

    arrow = "⬆️" if signal == "BUY" else ("⬇️" if signal == "SELL" else "➖")
    copy_code = pocket_code(pair_name, pair_code)
    phase_map = {"start": "🟢 Начало", "mid": "🟡 Середина", "end": "🔴 Конец"}
    phase_icon = phase_map.get(mtf.get("Phase", ""), "❔")
    if conf < 60:
        strength = "🔴 слабый"
    elif conf < 80:
        strength = "🟡 средний"
    else:
        strength = "🟢 сильный"

    text = (
        "🤖 AI FX PO Bot v102\n"
        f"💱 Пара: {pair_name}\n"
        f"📌 Код (Pocket): `{copy_code}`\n"
        f"🏷️ Тип: {mtype}\n"
        f"{arrow} Сигнал: *{signal}*\n"
        f"📊 Multi-TF: M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}\n"
        f"🌐 Режим: {mtf['Regime']} | 🕯️ Свеча: {phase_icon}\n"
        f"💪 Уверенность: *{conf}%* ({strength})\n"
        f"⏱ Экспирация: *{expiry} мин*\n"
        f"🧠 Стратегии: base={strat['base']} po={strat['po']} "
        f"trend={strat['trend']} ctr={strat['ctr']}\n"
        f"📈 RSI {feats['RSI']} | ADX {feats['ADX']} | MACD {feats['MACD_Hist']}\n"
        f"⚠️ Только обучение, не финансовый совет.\n"
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

# =========== STREAMLIT UI ====================
st.set_page_config(page_title="AI FX PO Bot v102", layout="wide")

st.title("🤖 AI FX PO Bot v102 — 5 стратегий + Telegram + Pocket")

st.markdown(
    "⚠️ **99% гарантий не существует.** Режимы 85/90/95/99 здесь — "
    "это стиль фильтрации (осторожный/агрессивный), а не реальная точность. "
    "Используй бота как обучающий инструмент."
)

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    risk_mode = st.selectbox(
        "Режим расчёта (стиль)",
        ["85%", "90%", "95%", "99%"],
        index=1,
    )
with c2:
    threshold_ui = st.slider(
        "Базовый порог уверенности (до учёта режима)",
        50, 95, BASE_THRESHOLD, 1,
    )
with c3:
    min_gap = st.number_input(
        "Мин. пауза между сигналами (сек, на пару)",
        10, 300, MIN_SEND_GAP_S,
    )

# корректировка порога по режиму
mode_add = {"85%": -5, "90%": 0, "95%": +5, "99%": +10}[risk_mode]
eff_base_threshold = threshold_ui + mode_add

if "last_sent" not in st.session_sta
