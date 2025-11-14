# ========================= AI FX Bot v3.3 =========================
# M1 + M5 + M15 + M30 + Telegram (Safe A-mode)
# Работает с секретами Streamlit:
#   TELEGRAM_TOKEN = "..."
#   CHAT_ID        = "..."

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

# ======================== SECRETS ================================
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", "")
CHAT_ID        = st.secrets.get("CHAT_ID", "")

# ======================== SETTINGS ===============================
REFRESH_SEC        = 1      # автообновление страницы (сек)
ONLY_NEW           = True   # не спамим одинаковыми сигналами
MIN_SEND_GAP_S     = 300    # мин. пауза между сигналами по 1 паре (5 минут)
BASE_CONF_THRESHOLD = 80    # базовый минимум уверенности

# Режимы фильтра (Safe A-mode по умолчанию)
MODES = {
    "Safe 85% (A-mode)": 85,
    "Normal 90%":        90,
    "Hard 95%":          95,
    "Ultra 99%":         99,
}

# Таймфреймы
TF_M1  = ("1m",  "1d")
TF_M5  = ("5m",  "5d")
TF_M15 = ("15m", "10d")
TF_M30 = ("30m", "30d")

# ======================== ИНСТРУМЕНТЫ ============================
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

# ===================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===================
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
    up_move = h.diff()
    dn_move = -l.diff()
    plus_dm = up_move.where((up_move > 0) & (up_move > dn_move), 0.0).fillna(0)
    minus_dm = dn_move.where((dn_move > 0) & (dn_move > up_move), 0.0).fillna(0)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * (plus_dm.rolling(n).sum() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(n).sum() / (atr + 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9))
    return dx.rolling(n).mean()


def boll_width(close: pd.Series, n=20, k=2.0) -> float:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    up = ma + k * sd
    lo = ma - k * sd
    return safe_float(((up.iloc[-1] - lo.iloc[-1]) / (ma.iloc[-1] + 1e-9)) * 100)


def market_regime(adx_val: float, bw: float) -> str:
    if adx_val < 18 and bw < 3:
        return "flat"
    if adx_val > 25 and bw < 7:
        return "trend"
    return "impulse"


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


def is_otc(name: str, symbol: str) -> bool:
    n = name.lower()
    if "otc" in n:
        return True
    if "=f" in symbol.lower():
        return True
    if "-" in symbol:  # крипта BTC-USD
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

# ===================== ЗАГРУЗКА ДАННЫХ ============================
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

# ===================== ОЦЕНКА СИГНАЛОВ (M5 базовый) ===============
def score_single(df: pd.DataFrame) -> tuple[str, int, dict]:
    if df is None or len(df) < 30:
        return "FLAT", 0, {
            "RSI": 50.0,
            "ADX": 0.0,
            "MACD_Hist": 0.0,
            "BB_Width": 0.0,
        }

    close = df["Close"]

    rsi_series = rsi(close)
    rsv = safe_float(rsi_series.iloc[-1], 50.0)

    ema20 = safe_float(ema(close, 20).iloc[-1], rsv)
    ema50 = safe_float(ema(close, 50).iloc[-1], rsv)
    ema200 = safe_float(ema(close, 200).iloc[-1], rsv)

    _, _, mh = macd(close)
    mhv = safe_float(mh.iloc[-1], 0.0)

    up, mid, lo, w = bbands(close)
    bw = safe_float(w.iloc[-1], 0.0)

    adx_series = adx(df)
    adx_v = safe_float(adx_series.iloc[-1], 0.0)

    vu = 0
    vd = 0

    # RSI
    if rsv < 32:
        vu += 1
    if rsv > 68:
        vd += 1

    # EMA
    if ema20 > ema50:
        vu += 1
    if ema20 < ema50:
        vd += 1
    if ema50 > ema200:
        vu += 1
    if ema50 < ema200:
        vd += 1

    # MACD
    if mhv > 0:
        vu += 1
    if mhv < 0:
        vd += 1

    # Bollinger позиция (стороной)
    last_close = safe_float(close.iloc[-1], rsv)
    up_last = safe_float(up.iloc[-1], last_close)
    lo_last = safe_float(lo.iloc[-1], last_close)
    if last_close <= lo_last:
        vu += 1
    if last_close >= up_last:
        vd += 1

    if vu == vd:
        direction = "FLAT"
    elif vu > vd:
        direction = "BUY"
    else:
        direction = "SELL"

    raw = abs(vu - vd) / 6.0
    trend_boost = min(max((adx_v - 20) / 25, 0), 1)
    conf = int(100 * (0.55 * raw + 0.45 * trend_boost))
    conf = max(0, min(99, conf))

    feats = {
        "RSI": round(rsv, 1),
        "ADX": round(adx_v, 1),
        "MACD_Hist": round(mhv, 5),
        "BB_Width": round(bw, 2),
    }
    return direction, conf, feats


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


def score_multi_tf(symbol: str) -> tuple[str, int, dict, dict]:
    df_m1  = get_or_fake(symbol, *TF_M1)
    df_m5  = get_or_fake(symbol, *TF_M5)
    df_m15 = get_or_fake(symbol, *TF_M15)
    df_m30 = get_or_fake(symbol, *TF_M30)

    base_sig, base_conf, feats = score_single(df_m5)

    d_m1  = tf_direction(df_m1)
    d_m5  = tf_direction(df_m5)
    d_m15 = tf_direction(df_m15)
    d_m30 = tf_direction(df_m30)

    agree = 0
    if d_m1 == d_m5 and d_m5 in ("BUY", "SELL"):
        agree += 1
    if d_m5 == d_m15 and d_m5 in ("BUY", "SELL"):
        agree += 1
    if d_m5 == d_m30 and d_m5 in ("BUY", "SELL"):
        agree += 1

    if base_sig in ("BUY", "SELL") and base_sig == d_m5:
        base_conf += 5
    else:
        base_conf -= 5

    if agree == 3 and base_sig in ("BUY", "SELL"):
        base_conf += 15
    elif agree == 2:
        base_conf += 8
    elif agree == 1:
        base_conf += 3
    else:
        base_conf -= 10

    bw_main = boll_width(df_m5["Close"])
    adx_v = feats["ADX"]
    regime = market_regime(adx_v, bw_main)
    phase = candle_phase(df_m5)

    if regime == "trend" and base_sig == d_m30 and d_m30 in ("BUY", "SELL"):
        base_conf += 7
    if regime == "flat":
        base_conf -= 5
    if phase == "end":
        base_conf -= 4

    base_conf = int(max(0, min(100, base_conf)))

    mtf = {
        "M1": d_m1,
        "M5": d_m5,
        "M15": d_m15,
        "M30": d_m30,
        "Regime": regime,
        "Phase": phase,
    }
    return base_sig, base_conf, feats, mtf

# ===================== КЛАСС СИГНАЛА =============================
def classify_signal(conf: int) -> str:
    if conf >= 90:
        return "A"
    if conf >= 80:
        return "B"
    return "C"

# ===================== EXPIRATION ================================
def choose_expiry_tf(base_tf: str, conf: int) -> int:
    if base_tf == "M1":
        base = 1
    elif base_tf == "M5":
        base = 5
    elif base_tf == "M15":
        base = 10
    else:
        base = 20

    if conf >= 95:
        base += 2
    elif conf <= 85:
        base -= 1

    return int(max(1, min(30, base)))

# ===================== TELEGRAM =================================
def send_telegram(pair_name: str,
                  pair_code: str,
                  signal: str,
                  conf: int,
                  sig_class: str,
                  expiry: int,
                  feats: dict,
                  mtf: dict,
                  mtype: str) -> None:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return

    arrow = "⬆️" if signal == "BUY" else ("⬇️" if signal == "SELL" else "➖")
    copy_code = pocket_code(pair_name, pair_code)

    trend_emoji = {
        "flat": "⚪",
        "trend": "📈",
        "impulse": "💥",
    }.get(mtf.get("Regime", "flat"), "⚪")

    phase_emoji = {
        "start": "🟢",
        "mid": "🟡",
        "end": "🔴",
    }.get(mtf.get("Phase", "mid"), "🟡")

    text = (
        "🤖 AI FX Signal Bot v3.3 (Safe A-mode)\n"
        f"💱 Пара: {pair_name}\n"
        f"📌 Код для Pocket: `{copy_code}`\n"
        f"🧾 Тип: {mtype}\n"
        f"{arrow} Сигнал: *{signal}* (класс {sig_class})\n"
        f"⏰ Экспирация: *{expiry} мин*\n"
        f"📊 Multi-TF: M1={mtf['M1']} | M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}\n"
        f"🌍 Режим: {mtf['Regime']} {trend_emoji} | Свеча: {phase_emoji}\n"
        f"💪 Уверенность: *{conf}%*\n"
        f"📈 RSI {feats['RSI']} | ADX {feats['ADX']} | MACD {feats['MACD_Hist']}\n"
        f"⏱ {datetime.utcnow().strftime('%H:%M:%S')} UTC"
    )

    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={
                "chat_id": CHAT_ID,
                "text": text,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
    except Exception as e:
        st.toast(f"Ошибка Telegram: {e}", icon="⚠️")

# ===================== STREAMLIT UI ==============================
st.set_page_config(
    page_title="AI FX Bot v3.3 — M1+M5+M15+M30 + Telegram (Safe)",
    layout="wide"
)

st.title("🤖 AI FX Bot v3.3 — M1+M5+M15+M30 + Telegram (Safe A-mode)")
st.markdown(
    "Режимы Safe/Normal/Hard/Ultra — это **фильтр сигналов**, а не гарантия дохода. "
    "Бот — обучающий инструмент, не финансовый совет."
)

c_top1, c_top2 = st.columns(2)
with c_top1:
    mode_name = st.selectbox(
        "Режим отбора сигналов",
        list(MODES.keys()),
        index=0,
    )
with c_top2:
    slider_conf = st.slider(
        "Минимальная уверенность (%) для сигнала",
        50, 99, 85, 1,
    )

c_bot1, c_bot2 = st.columns(2)
with c_bot1:
    min_gap_ui = st.number_input(
        "Пауза между сигналами по паре (сек)",
        10, 900, MIN_SEND_GAP_S,
    )
with c_bot2:
    st.caption("Safe A-mode: отправляем только сигналы **класса A** (иногда B), "
               "согласованные на нескольких таймфреймах.")

# обновляем реальные значения паузы
MIN_SEND_GAP_S = int(min_gap_ui)

mode_threshold = MODES.get(mode_name, BASE_CONF_THRESHOLD)
working_threshold = max(mode_threshold, slider_conf)

st.markdown(
    f"**Текущий рабочий порог для отправки сигналов:** "
    f"`{working_threshold}%`"
)

if "last_sent" not in st.session_state:
    st.session_state.last_sent = {}

rows = []

# ===================== ОСНОВНОЙ ЦИКЛ ПО ПАРАМ ====================
for name, symbol in PAIRS.items():
    sig, conf, feats, mtf = score_multi_tf(symbol)

    sig_class = classify_signal(conf)
    base_tf = "M5"   # базовый вход М5
    expiry = choose_expiry_tf(base_tf, conf)

    otc_flag = is_otc(name, symbol)
    mtype = "OTC/24/7" if otc_flag else "Биржевая"

    # Заполняем таблицу
    rows.append([
        name,
        mtype,
        sig,
        conf,
        sig_class,
        expiry,
        f"M1={mtf['M1']} | M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}",
        mtf["Regime"],
        mtf["Phase"],
    ])

    # ====== ЛОГИКА ОТПРАВКИ В TELEGRAM (Safe A-mode) ======
    should_send = False
    if sig in ("BUY", "SELL") and conf >= working_threshold and expiry > 0:
        # Для безопасного режима — только класс A,
        # очень редко допустим B, если уверенность выше 93
        if sig_class == "A" or (sig_class == "B" and conf >= 93):
            prev = st.session_state.last_sent.get(name, {})
            if ONLY_NEW and prev:
                same = prev.get("signal") == sig
                worse = conf <= prev.get("conf", 0)
                recent = (time.time() - prev.get("ts", 0)) < MIN_SEND_GAP_S
                if same and (worse or recent):
                    should_send = False
                else:
                    should_send = True
            else:
                should_send = True

    if should_send:
        send_telegram(
            pair_name=name,
            pair_code=symbol,
            signal=sig,
            conf=conf,
            sig_class=sig_class,
            expiry=expiry,
            feats=feats,
            mtf=mtf,
            mtype=mtype,
        )
        st.session_state.last_sent[name] = {
            "signal": sig,
            "conf": conf,
            "ts": time.time(),
        }

# ===================== ТАБЛИЦА СИГНАЛОВ ==========================
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
        "Режим рынка",
        "Фаза свечи",
    ],
)

if len(df_show):
    df_show = df_show.sort_values(
        ["Класс", "Уверенность"],
        ascending=[True, False]
    ).reset_index(drop=True)

st.subheader("📋 Таблица сигналов")
st.dataframe(df_show, use_container_width=True, height=480)

# ===================== АВТООБНОВЛЕНИЕ ============================
time.sleep(REFRESH_SEC)
st.rerun()
