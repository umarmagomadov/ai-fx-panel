# ================== IMPORTS ==================
import os
import time
import json
import random
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# ================== SECRETS ==================
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", os.getenv("TELEGRAM_TOKEN", ""))
CHAT_ID        = st.secrets.get("CHAT_ID",        os.getenv("CHAT_ID", ""))

# ================== БАЗОВЫЕ НАСТРОЙКИ ==================
REFRESH_SEC    = 1         # автообновление (секунды)
ONLY_NEW       = True      # отправлять только новые / лучшие сигналы
MIN_SEND_GAP_S = 60        # мин. пауза между сигналами по одной паре

# Таймфреймы
TF_MAIN  = ("5m",  "2d")   # вход
TF_MID   = ("15m", "5d")   # подтверждение
TF_TREND = ("30m", "10d")  # глобальный тренд

# ================== СПИСОК ИНСТРУМЕНТОВ ==================
PAIRS: Dict[str, str] = {
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

    # Commodities (фьючи)
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

# ================== ХЕЛПЕРЫ ==================
def safe_float(x, default: float = 0.0) -> float:
    """Безопасно превращает что угодно в float."""
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


def is_otc(name: str, symbol: str) -> bool:
    """Грубое разделение OTC / биржевые."""
    n = name.lower()
    if "otc" in n:
        return True
    if "=f" in symbol.lower():   # фьючерсы
        return True
    if "-" in symbol:           # крипта BTC-USD и т.п.
        return True
    return False


def pocket_code(name: str, symbol: str) -> str:
    """Код, который удобно копировать в Pocket Option."""
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
    """Фаза последней свечи: start/mid/end (для более умной экспирации)."""
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


def near_sr(df: pd.DataFrame) -> Optional[str]:
    """Близко ли цена к поддержке/сопротивлению."""
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
    """Импульс – резкое движение против среднего."""
    if df is None or len(df) < 12:
        return False
    close = df["Close"]
    last_move = abs(safe_float(close.iloc[-1]) - safe_float(close.iloc[-2]))
    avg_raw = close.diff().abs().rolling(10).mean().iloc[-1]
    avg_move = safe_float(avg_raw, 0.0)
    if avg_move == 0.0:
        return False
    return bool(last_move > 1.5 * avg_move)

# ================== ИНДИКАТОРЫ ==================
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
    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()],
        axis=1
    ).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di  = 100 * (plus_dm.rolling(n).sum()  / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(n).sum() / (atr + 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9))
    return dx.rolling(n).mean()


def boll_width(close: pd.Series, n=20, k=2.0) -> float:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    up = ma + k * sd
    lo = ma - k * sd
    return safe_float((up.iloc[-1] - lo.iloc[-1]) / (ma.iloc[-1] + 1e-9) * 100)

# ================== ДАННЫЕ (YFINANCE + КЭШ) ==================
@st.cache_data(ttl=60, show_spinner=False)
def safe_download(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
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
    """Лёгкая рандомизация последней свечи, если данных нет (демо-режим)."""
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
    """Кэш + подделка свечей, если оффлайн / мало истории."""
    if "cache" not in st.session_state:
        st.session_state.cache = {}

    key = f"{symbol}__{interval}"
    real = safe_download(symbol, period, interval)
    if real is not None:
        st.session_state.cache[key] = real.copy()
        return real

    cached = st.session_state.cache.get(key)
    if cached is not None and len(cached):
        df = cached.copy()
        last = nudge_last(df)
        df = pd.concat([df, last.to_frame().T], axis=0).tail(600)
        st.session_state.cache[key] = df
        return df

    # Полностью синтетика, если ничего нет
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=60, freq="1min")
    base = 1.0 + random.random() / 10
    vals = base * (1 + np.cumsum(np.random.randn(60)) / 100)
    df = pd.DataFrame(
        {"Open": vals, "High": vals, "Low": vals, "Close": vals},
        index=idx,
    )
    st.session_state.cache[key] = df
    return df

# ================== СТРАТЕГИИ / СКОРИНГ M5 ==================
def score_single(df: pd.DataFrame) -> Tuple[str, int, Dict]:
    """
    Основной сигнал на M5.
    Внутри несколько стратегий:
      1) Тренд по EMA (EMA9 vs EMA21, EMA200)
      2) Осциллятор RSI (перекуп/перепродажа + разворот)
      3) MACD (разворот/продолжение)
      4) Bollinger (отскок от границ)
    Всё голосует и даёт направление + уверенность.
    """
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

    # RSI
    rsi_series = rsi(close)
    rsv = safe_float(rsi_series.iloc[-1], 50.0)
    rsv_prev = safe_float(
        rsi_series.iloc[-2], rsv
    ) if len(rsi_series) > 2 else rsv

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
        (close.iloc[-1] - mid.iloc[-1])
        / (up.iloc[-1] - lo.iloc[-1] + 1e-9),
        0.0,
    )
    w_last = safe_float(w.iloc[-1], 0.0)

    # ADX
    adx_series = adx(df)
    adx_v = safe_float(adx_series.iloc[-1], 0.0)

    # ==== голосование стратегий ====
    vu = 0  # голоса за BUY
    vd = 0  # голоса за SELL

    # Стратегия 1: RSI
    if rsv < 35:
        vu += 1
    if rsv > 65:
        vd += 1

    # Стратегия 2: EMA-тренд
    if ema9 > ema21:
        vu += 1
    if ema9 < ema21:
        vd += 1

    # Стратегия 3: MACD
    if mhv > 0:
        vu += 1
    if mhv < 0:
        vd += 1

    # Стратегия 4: Bollinger отскок
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

    # Уверенность = разница голосов + сила тренда по ADX
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

# ============== MULTI-TF СЛИЯНИЕ ==============
def tf_direction(df: pd.DataFrame) -> str:
    close = df["Close"]
    m, s, h = macd(close)
    r = rsi(close)
    rsv = safe_float(r.iloc[-1], 50.0)
    mh = safe_float(h.iloc[-1], 0.0)
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


def detect_strategies(df: pd.DataFrame, feats: Dict, mtf: Dict) -> List[str]:
    """
    Возвращает список сработавших стратегий (для описания / Telegram).
    """
    strategies = []
    rsi_v = feats["RSI"]
    bb_pos = feats["BB_Pos"]
    adx_v = feats["ADX"]

    # 1) Тренд-фоллоуинг
    if mtf["Regime"] == "trend" and mtf["M5"] == mtf["M15"] == mtf["M30"]:
        strategies.append("Тренд по всем TF")

    # 2) Разворот от Боллинджера
    if rsi_v < 35 and bb_pos < -0.3:
        strategies.append("Отскок снизу Bollinger (перепроданность)")
    if rsi_v > 65 and bb_pos > 0.3:
        strategies.append("Отскок сверху Bollinger (перекупленность)")

    # 3) Импульс
    if momentum_spike(df):
        strategies.append("Сильный импульс")

    # 4) Уровень поддержки / сопротивления
    sr = near_sr(df)
    if sr == "support":
        strategies.append("Сильная поддержка рядом")
    elif sr == "resistance":
        strategies.append("Сильное сопротивление рядом")

    # 5) Сильный тренд по ADX
    if adx_v >= 30:
        strategies.append("Сильный тренд по ADX")

    return strategies


def score_multi_tf(symbol: str) -> Tuple[str, int, Dict, Dict, List[str], pd.DataFrame]:
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

    # Импульс
    if momentum_spike(df_main):
        conf += 8

    # Уровни
    sr = near_sr(df_main)
    if (sig == "BUY" and sr == "support") or (sig == "SELL" and sr == "resistance"):
        conf += 7

    # Фаза свечи
    ph = candle_phase(df_main)
    if ph == "mid":
        conf += 5
    elif ph == "end":
        conf -= 6

    # Сильный резкий скачок RSI – осторожнее
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

    strategies = detect_strategies(df_main, feats, mtf)
    return sig, conf, feats, mtf, strategies, df_main

# ============== SMART EXPIRY ==================
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

# ============== TELEGRAM ======================
def send_telegram(pair_name: str, pair_code: str, mtype: str,
                  signal: str, conf: int, expiry: int,
                  feats: Dict, mtf: Dict, strategies: List[str]) -> None:
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

    strat_text = " · ".join(strategies) if strategies else "Базовый AI-сигнал"

    text = (
        "🤖 *AI FX Signal Bot v102*\n"
        f"💱 Пара: *{pair_name}*\n"
        f"📌 Код (Pocket): `{copy_code}`\n"
        f"🏷️ Тип: {mtype}\n"
        f"{arrow} Сигнал: *{signal}*\n"
        f"📊 Multi-TF: M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}\n"
        f"🌐 Режим: {mtf['Regime']} | 🕯 Свеча: {phase_icon}\n"
        f"💪 Уверенность: *{conf}%* ({strength})\n"
        f"⏱ Экспирация: *{expiry} мин*\n"
        f"🧠 Стратегии: {strat_text}\n"
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

# ============== STREAMLIT UI ==================
st.set_page_config(
    page_title="AI FX Signal Bot v102 — Triple TF",
    layout="wide"
)

st.title("🤖 AI FX Signal Bot v102 — M5+M15+M30 + Pocket Option")

st.markdown(
    "Режимы 85/90/95/99% здесь — это стиль отбора сигналов (строгость фильтра), "
    "а не реальная гарантия. Используй бота как обучающий инструмент + свою голову. 😉"
)

col_mode, col_gap = st.columns([1, 1])

with col_mode:
    mode_label = st.selectbox(
        "Режим отбора сигналов",
        [
            "85% — спокойный (больше сигналов)",
            "90% — стандарт",
            "95% — жёсткий отбор",
            "99% — только топ-сигналы",
        ],
    )

mode_num = 85
if mode_label.startswith("90"):
    mode_num = 90
elif mode_label.startswith("95"):
    mode_num = 95
elif mode_label.startswith("99"):
    mode_num = 99

# Базовый порог уверенности в зависимости от режима
BASE_THRESHOLDS = {85: 60, 90: 70, 95: 80, 99: 88}
threshold = BASE_THRESHOLDS[mode_num]

with col_gap:
    min_gap = st.slider(
        "Мин. пауза между сигналами по одной паре (сек)",
        10, 300, MIN_SEND_GAP_S, 5,
    )

if "last_sent" not in st.session_state:
    st.session_state.last_sent = {}

rows = []

# ================== ГЛАВНЫЙ ЦИКЛ ПО ПАРАМ ==================
for name, symbol in PAIRS.items():
    sig, conf, feats, mtf, strategies, df_main = score_multi_tf(symbol)

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
    strat_text_short = " · ".join(strategies) if strategies else "AI-сигнал"

    rows.append([
        name,
        mtype,
        sig,
        conf,
        expiry,
        f"M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}",
        phase_show,
        strat_text_short,
        json.dumps(feats, ensure_ascii=False),
    ])

    # ====== Telegram логика ======
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
            send_telegram(name, symbol, mtype, sig, conf, expiry, feats, mtf, strategies)
            st.session_state.last_sent[name] = {
                "signal": sig,
                "ts": time.time(),
        
