# ===================== AI FX Bot v3.1 =====================
# M1 + M5 + M15 + M30 + Telegram
# Работает с секретами:
# TELEGRAM_TOKEN = "..."
# CHAT_ID        = "..."

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


# ==================== SECRETS ====================

TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", "")
CHAT_ID = st.secrets.get("CHAT_ID", "")

# ==================== SETTINGS ====================

REFRESH_SEC = 1          # автообновление, сек
ONLY_NEW = True          # не спамим одно и то же
MIN_SEND_GAP_S = 60      # пауза между сигналами по одной паре
BASE_CONF_THRESHOLD = 70 # базовый минимум

# Режимы фильтра
MODES = {
    "Safe 85%": 85,
    "Normal 90%": 90,
    "Hard 95%": 95,
    "Ultra 99%": 99,
}

# Таймфреймы
TF_M1 = ("1m", "1d")
TF_M5 = ("5m", "5d")
TF_M15 = ("15m", "5d")
TF_M30 = ("30m", "10d")

# ==================== ИНСТРУМЕНТЫ ====================

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
    "CADJPY": "CADJPY=X",
    "CHFJPY": "CHFJPY=X",
    "NZDJPY": "NZDJPY=X",
    "EURGBP": "EURGBP=X",
    "EURCHF": "EURCHF=X",

    # Metals / Commodities (как OTC)
    "XAUUSD (Gold)": "GC=F",
    "XAGUSD (Silver)": "SI=F",
    "WTI (Oil)": "CL=F",
    "BRENT (Oil)": "BZ=F",

    # Crypto (как OTC/24/7)
    "BTCUSD (Bitcoin)": "BTC-USD",
    "ETHUSD (Ethereum)": "ETH-USD",
    "XRPUSD (XRP)": "XRP-USD",
}

# ==================== ХЕЛПЕРЫ ====================

def safe_float(x, default: float = 0.0) -> float:
    """Безопасное преобразование к float."""
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


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


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


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    up_move = h.diff()
    dn_move = -l.diff()

    plus_dm = up_move.where((up_move > 0) & (up_move > dn_move), 0.0).fillna(0)
    minus_dm = dn_move.where((dn_move > 0) & (dn_move > up_move), 0.0).fillna(0)

    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(n).mean()
    plus_di = 100 * (plus_dm.rolling(n).sum() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(n).sum() / (atr + 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() /
                ((plus_di + minus_di) + 1e-9))

    return dx.rolling(n).mean()


def is_otc(name: str, symbol: str) -> bool:
    """Фьючерсы и крипта считаем как OTC/24/7."""
    s = symbol.lower()
    n = name.lower()
    if "btc" in n or "eth" in n or "xrp" in n:
        return True
    if s.endswith("-usd"):
        return True
    if s.endswith("=f"):
        return True
    return False


def pocket_code(name: str, symbol: str) -> str:
    """Генерация кода инструмента для Pocket Option."""
    if symbol.endswith("=X") and len(symbol) >= 7:
        base = symbol.replace("=X", "")
        if len(base) == 6:
            return f"{base[:3]}/{base[3:]}"
    if symbol.endswith("-USD"):
        return symbol.replace("-USD", "/USD")
    if symbol == "GC=F":
        return "XAU/USD"
    if symbol == "SI=F":
        return "XAG/USD"
    if symbol == "CL=F":
        return "WTI/USD"
    if symbol == "BZ=F":
        return "BRENT/USD"

    clean = "".join(ch for ch in name if ch.isalnum() or ch in "/").upper()
    return clean


def market_regime(adx_val: float, bb_width: float) -> str:
    if adx_val < 18 and bb_width < 3:
        return "flat"
    if adx_val > 25 and bb_width < 7:
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


def boll_width(close: pd.Series, n=20, k=2.0) -> float:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    up = ma + k * sd
    lo = ma - k * sd
    return safe_float((up.iloc[-1] - lo.iloc[-1]) /
                      (ma.iloc[-1] + 1e-9) * 100)


# ==================== ЗАГРУЗКА ДАННЫХ ====================

def load_tf(symbol: str, tf: tuple[str, str]) -> pd.DataFrame | None:
    interval, period = tf
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df is None or len(df) < 40:
            return None
        return df[["Open", "High", "Low", "Close"]].copy()
    except Exception:
        return None


# ==================== СКОПРИНГ СИГНАЛА ====================

def score_single_tf(df: pd.DataFrame) -> tuple[str, int, dict]:
    """Один ТФ (M5) — направление + уверенность."""
    if df is None or len(df) < 40:
        return "FLAT", 0, {
            "RSI": 50.0,
            "ADX": 0.0,
            "MACD": 0.0,
            "BB_Width": 0.0,
        }

    close = df["Close"]

    rsi_series = rsi(close)
    rsv = safe_float(rsi_series.iloc[-1], 50.0)

    ema9 = safe_float(ema(close, 9).iloc[-1], 0.0)
    ema21 = safe_float(ema(close, 21).iloc[-1], 0.0)
    ema200 = safe_float(ema(close, 200).iloc[-1], 0.0)

    _, _, macd_hist = macd(close)
    macd_val = safe_float(macd_hist.iloc[-1], 0.0)

    up, mid, lo, w = bbands(close)
    w_last = safe_float(w.iloc[-1], 0.0)

    adx_series = adx(df)
    adx_val = safe_float(adx_series.iloc[-1], 0.0)

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

    if macd_val > 0:
        vu += 1
    if macd_val < 0:
        vd += 1

    if close.iloc[-1] < lo.iloc[-1]:
        vu += 1
    if close.iloc[-1] > up.iloc[-1]:
        vd += 1

    if vu == vd:
        direction = "FLAT"
    elif vu > vd:
        direction = "BUY"
    else:
        direction = "SELL"

    trend_boost = min(max((adx_val - 18) / 25, 0), 1)
    raw = abs(vu - vd) / 4.0
    conf = int(100 * (0.55 * raw + 0.45 * trend_boost))
    conf = max(40, min(99, conf))

    feats = {
        "RSI": round(rsv, 1),
        "ADX": round(adx_val, 1),
        "MACD": round(macd_val, 5),
        "BB_Width": round(w_last, 2),
        "EMA200": round(ema200, 5),
    }

    return direction, conf, feats


def tf_direction(df: pd.DataFrame) -> str:
    """Направление по ТФ (для M1/M15/M30)."""
    if df is None or len(df) < 40:
        return "FLAT"
    close = df["Close"]
    macd_line, macd_sig, macd_hist = macd(close)
    rsi_series = rsi(close)
    mh = safe_float(macd_hist.iloc[-1], 0.0)
    rsv = safe_float(rsi_series.iloc[-1], 50.0)
    if mh > 0 and rsv > 50:
        return "BUY"
    if mh < 0 and rsv < 50:
        return "SELL"
    return "FLAT"


def choose_expiry(conf: int, adx_val: float, rsi_val: float,
                  bw: float) -> int:
    """Смарт-экспирация (минуты) под Pocket."""
    if conf < 60:
        return 0
    if conf < 70:
        base = 1
    elif conf < 80:
        base = 3
    elif conf < 90:
        base = 5
    elif conf < 95:
        base = 8
    else:
        base = 12

    if adx_val >= 40:
        base += 4
    elif adx_val >= 30:
        base += 2
    elif adx_val < 18:
        base -= 1

    if bw > 8:
        base -= 2
    elif bw < 3:
        base += 1

    if rsi_val >= 70 or rsi_val <= 30:
        base -= 1

    return int(max(1, min(30, base)))


def multi_tf_score(symbol: str) -> tuple[str, int, dict, dict, int]:
    """
    Возвращает:
      signal, conf, feats, mtf_info, expiry
    """
    df_m1 = load_tf(symbol, TF_M1)
    df_m5 = load_tf(symbol, TF_M5)
    df_m15 = load_tf(symbol, TF_M15)
    df_m30 = load_tf(symbol, TF_M30)

    sig_m5, conf_m5, feats = score_single_tf(df_m5)

    d_m1 = tf_direction(df_m1)
    d_m5 = sig_m5
    d_m15 = tf_direction(df_m15)
    d_m30 = tf_direction(df_m30)

    conf = conf_m5

    agree = 0
    if d_m1 == d_m5 and d_m1 in ("BUY", "SELL"):
        agree += 1
    if d_m15 == d_m5 and d_m15 in ("BUY", "SELL"):
        agree += 1
    if d_m30 == d_m5 and d_m30 in ("BUY", "SELL"):
        agree += 1

    if agree == 3 and d_m5 in ("BUY", "SELL"):
        conf += 15
    elif agree == 2:
        conf += 8
    elif agree == 1:
        conf += 3
    else:
        conf -= 10

    if df_m5 is not None and len(df_m5) > 40:
        bw = boll_width(df_m5["Close"])
        regime = market_regime(feats["ADX"], bw)
        phase = candle_phase(df_m5)
    else:
        bw = 0.0
        regime = "unknown"
        phase = "mid"

    if phase == "mid":
        conf += 3
    elif phase == "end":
        conf -= 5

    conf = int(max(0, min(100, conf)))

    expiry = choose_expiry(conf, feats["ADX"], feats["RSI"], bw)

    mtf = {
        "M1": d_m1,
        "M5": d_m5,
        "M15": d_m15,
        "M30": d_m30,
        "Regime": regime,
        "Phase": phase,
    }

    return d_m5, conf, feats, mtf, expiry


# ==================== TELEGRAM ====================

def send_telegram(pair_name: str,
                  pair_code: str,
                  mtype: str,
                  signal: str,
                  conf: int,
                  expiry: int,
                  feats: dict,
                  mtf: dict) -> None:
    """Отправка сигнала в Telegram."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return

    arrow = "⬆️" if signal == "BUY" else ("⬇️" if signal == "SELL" else "➖")

    phase_icon = {
        "start": "🟢 начало",
        "mid": "🟡 середина",
        "end": "🔴 конец",
    }.get(mtf.get("Phase", "mid"), "🟡 середина")

    if conf < 70:
        strength = "🟥 слабый"
    elif conf < 85:
        strength = "🟡 нормальный"
    else:
        strength = "🟩 сильный"

    multi_line = f"M1={mtf['M1']} | M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}"
    copy_code = pocket_code(pair_name, pair_code)

    text = (
        "🤖 AI FX Signal Bot v3.1\n"
        f"💱 Пара: {pair_name}\n"
        f"📌 Код для Pocket: {copy_code}\n"
        f"🧾 Тип: {mtype}\n"
        f"{arrow} Сигнал: *{signal}*\n"
        f"🇲🇫 ТФ: {multi_line}\n"
        f"🌐 Рынок: {mtf['Regime']} | {phase_icon}\n"
        f"💪 Уверенность: *{conf}%* ({strength})\n"
        f"⏱ Экспирация: {expiry} мин\n"
        f"📊 RSI {feats['RSI']} | ADX {feats['ADX']} | MACD {feats['MACD']}\n"
        f"⏰ {datetime.utcnow().strftime('%H:%M:%S')} UTC"
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
        st.toast(f"Telegram error: {e}", icon="⚠️")


# ==================== STREAMLIT UI ====================

st.set_page_config(
    page_title="AI FX Bot v3.1 — M1+M5+M15+M30 + Telegram",
    layout="wide",
)

st.title("AI FX Bot v3.1 — M1+M5+M15+M30 + Telegram")

st.markdown(
    "Режимы *Safe/Normal/Hard/Ultra* — это **фильтр**, а не реальная гарантия. "
    "Бот — инструмент для обучения, не финансовый совет."
)

if "last_sent" not in st.session_state:
    st.session_state.last_sent = {}

# ----- Контролы -----
mode_name = st.selectbox(
    "Режим отбора сигналов",
    list(MODES.keys()),
    index=0,
)

min_conf_ui = st.slider(
    "Минимальная уверенность (%) для сигнала",
    50,
    99,
    MODES[mode_name],
)

pause_s = st.number_input(
    "Пауза между сигналами по паре (сек)",
    min_value=10,
    max_value=600,
    value=MIN_SEND_GAP_S,
    step=5,
)

working_threshold = max(min_conf_ui, BASE_CONF_THRESHOLD)

st.write(
    f"Текущий рабочий порог для отправки сигналов: **{working_threshold}%**"
)

# ----- Основной цикл по парам -----
rows = []

for name, symbol in PAIRS.items():
    try:
        sig, conf, feats, mtf, expiry = multi_tf_score(symbol)
    except Exception:
        # если по паре совсем беда с данными
        sig, conf, feats, mtf, expiry = "FLAT", 0, {
            "RSI": 50.0,
            "ADX": 0.0,
            "MACD": 0.0,
            "BB_Width": 0.0,
        }, {
            "M1": "FLAT",
            "M5": "FLAT",
            "M15": "FLAT",
            "M30": "FLAT",
            "Regime": "unknown",
            "Phase": "mid",
        }, 0

    otc_flag = is_otc(name, symbol)
    mtype = "OTC/24/7" if otc_flag else "Биржевая"

    # Класс сигнала по уверенности
    if conf >= 90:
        klass = "A"
    elif conf >= 80:
        klass = "B"
    else:
        klass = "C"

    rows.append([
        name,
        mtype,
        sig,
        conf,
        klass,
        expiry,
        f"M1={mtf['M1']} | M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}",
        mtf["Regime"],
        mtf["Phase"],
        feats["RSI"],
        feats["ADX"],
        feats["MACD"],
    ])

    # ----- TELEGRAM ОТПРАВКА -----
    if (
        sig in ("BUY", "SELL")
        and conf >= working_threshold
        and expiry > 0
        and TELEGRAM_TOKEN
        and CHAT_ID
    ):
        prev = st.session_state.last_sent.get(name, {})
        should_send = True

        if ONLY_NEW and prev:
            same_direction = prev.get("signal") == sig
            worse_conf = conf <= prev.get("conf", 0)
            recent = (time.time() - prev.get("ts", 0)) < pause_s
            if same_direction and (worse_conf or recent):
                should_send = False

        if should_send:
            send_telegram(name, symbol, mtype, sig, conf,
                          expiry, feats, mtf)
            st.session_state.last_sent[name] = {
                "signal": sig,
                "conf": conf,
                "ts": time.time(),
            }

# ----- Таблица -----
df = pd.DataFrame(
    rows,
    columns=[
        "Пара",
        "Тип",
        "Сигнал",
        "Уверенность",
        "Класс",
        "Экспирация (мин)",
        "Multi-TF",
        "Режим",
        "Фаза свечи",
        "RSI",
        "ADX",
        "MACD",
    ],
)

if len(df):
    df = df.sort_values("Уверенность", ascending=False).reset_index(drop=True)

st.subheader("📋 Таблица сигналов")

st.dataframe(
    df,
    use_container_width=True,
    height=480,
)

# ----- График по топ-паре -----
if len(df):
    top = df.iloc[0]
    sym = PAIRS[top["Пара"]]
    df_chart = load_tf(sym, TF_M5)

    if df_chart is not None and len(df_chart):
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df_chart.index,
                    open=df_chart["Open"],
                    high=df_chart["High"],
                    low=df_chart["Low"],
                    close=df_chart["Close"],
                )
            ]
        )
        fig.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=20, b=0),
            title=(
                f"Топ-пара: {top['Пара']} — {top['Сигнал']} "
                f"({top['Уверенность']}%) • {top['Multi-TF']}"
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

# ----- Авто-обновление -----
time.sleep(REFRESH_SEC)
st.experimental_rerun()
