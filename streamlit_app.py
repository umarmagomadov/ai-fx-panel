# AI FX Signal Bot v103.0 — Triple-TF Safe Engine
# Автор: для Umar 🙂  | Язык: ru

import os
import time
import json
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
import plotly.graph_objects as go

# ======================= НАСТРОЙКИ =======================

VERSION = "v103.0"

# Авто-обновление по умолчанию (секунды)
DEFAULT_REFRESH_SEC = 5

# Порог уверенности по умолчанию (для Telegram)
DEFAULT_THRESHOLD = 70

# Минимальная пауза между сигналами по одной паре (сек)
DEFAULT_MIN_GAP = 60

# Только новые сигналы (не спамить одинаковыми)
ONLY_NEW = True

# --------- читаем токен / чат из Secrets / переменных окружения ----------
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", os.getenv("TELEGRAM_TOKEN", ""))
CHAT_ID        = st.secrets.get("CHAT_ID",        os.getenv("CHAT_ID", ""))

# ======================= ИНСТРУМЕНТЫ =======================

# Ключ — имя для отображения, значение — символ в yfinance
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
    "EURAUD": "EURAUD=X",
    "GBPAUD": "GBPAUD=X",

    # Metals & Oil (будут отображаться как OTC/24/7)
    "XAUUSD (Gold)": "GC=F",
    "XAGUSD (Silver)": "SI=F",
    "WTI Crude": "CL=F",
    "Brent Oil": "BZ=F",

    # Crypto (OTC/24/7)
    "BTCUSD (Bitcoin)": "BTC-USD",
    "ETHUSD (Ethereum)": "ETH-USD",
    "BNBUSD (BNB)": "BNB-USD",
    "XRPUSD (XRP)": "XRP-USD",
    "SOLUSD (Solana)": "SOL-USD",
}

# Таймфреймы: (interval, period)
TF_CONFIG = {
    "M5":  ("5m",  "2d"),
    "M15": ("15m", "5d"),
    "M30": ("30m", "10d"),
}

# =============== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==================

def is_otc(symbol_name: str, yf_symbol: str) -> bool:
    """Отличаем OTC/24/7 от биржевых."""
    name = symbol_name.lower()
    s = yf_symbol.lower()
    if "btc" in name or "eth" in name or "crypto" in name:
        return True
    if "-usd" in s:  # BTC-USD
        return True
    if s.endswith("=f"):  # фьючерсы GC=F, CL=F
        return True
    return False


def pocket_code(symbol_name: str, yf_symbol: str) -> str:
    """Код для Pocket Option (EURUSD=X → EUR/USD, BTC-USD → BTC/USD, GC=F → XAU/USD)."""
    s = yf_symbol

    if s.endswith("=X") and len(s) >= 7:
        base = s.replace("=X", "")
        if len(base) == 6:
            return f"{base[:3]}/{base[3:]}"
        return base

    if s.endswith("-USD"):
        return s.replace("-USD", "/USD").upper()

    futures_map = {
        "GC=F": "XAU/USD",
        "SI=F": "XAG/USD",
        "CL=F": "WTI/USD",
        "BZ=F": "BRENT/USD",
    }
    if s in futures_map:
        return futures_map[s]

    # запасной вариант — чистим имя
    return "".join(ch for ch in symbol_name if ch.isalnum() or ch in "/").upper()


# ====================== ЗАГРУЗКА ДАННЫХ ===================

@st.cache_data(ttl=60, show_spinner=False)
def load_ohlc(yf_symbol: str, period: str, interval: str) -> pd.DataFrame | None:
    """Безопасная загрузка OHLC через yfinance."""
    try:
        df = yf.download(
            yf_symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            return None
        df = df.dropna()
        if "Close" not in df.columns or len(df) < 50:
            return None
        return df.tail(500)
    except Exception:
        return None


# ==================== ИНДИКАТОРЫ ==========================

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = low.diff() * -1

    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0.0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.rolling(period).sum() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(period).sum() / (atr + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
    return dx.rolling(period).mean()


def bollinger_width(series: pd.Series, n: int = 20, k: float = 2.0) -> float:
    ma = series.rolling(n).mean()
    std = series.rolling(n).std()
    upper = ma + k * std
    lower = ma - k * std
    if len(ma.dropna()) == 0:
        return 0.0
    return float((upper.iloc[-1] - lower.iloc[-1]) / (ma.iloc[-1] + 1e-9) * 100)


# ==================== ОЦЕНКА НА ОДНОМ ТФ ==================

def score_single_tf(df: pd.DataFrame) -> dict:
    """
    Возвращает словарь с направлением, уверенностью и индикаторами.
    Если данных мало — FLAT, 0%.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return {
            "direction": "FLAT",
            "confidence": 0,
            "RSI": 0.0,
            "ADX": 0.0,
            "MACD_HIST": 0.0,
            "BW": 0.0,
        }

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 50:
        return {
            "direction": "FLAT",
            "confidence": 0,
            "RSI": 0.0,
            "ADX": 0.0,
            "MACD_HIST": 0.0,
            "BW": 0.0,
        }

    rsi_series = rsi(close)
    macd_line, macd_signal, macd_hist = macd(close)
    adx_series = adx(df)
    bw = bollinger_width(close)

    rsi_val = float(rsi_series.iloc[-1])
    macd_val = float(macd_hist.iloc[-1])
    adx_val = float(adx_series.iloc[-1]) if not adx_series.isna().all() else 0.0

    ema9 = float(ema(close, 9).iloc[-1])
    ema21 = float(ema(close, 21).iloc[-1])
    ema200 = float(ema(close, 200).iloc[-1])
    last_price = float(close.iloc[-1])

    up_votes = 0
    down_votes = 0

    # тренд относительно EMA200
    if last_price > ema200:
        up_votes += 1
    else:
        down_votes += 1

    # EMA9 vs EMA21
    if ema9 > ema21:
        up_votes += 1
    else:
        down_votes += 1

    # RSI
    if rsi_val > 60:
        up_votes += 1
    elif rsi_val < 40:
        down_votes += 1

    # MACD
    if macd_val > 0:
        up_votes += 1
    elif macd_val < 0:
        down_votes += 1

    if up_votes == down_votes:
        direction = "FLAT"
    elif up_votes > down_votes:
        direction = "BUY"
    else:
        direction = "SELL"

    # уверенность
    vote_diff = abs(up_votes - down_votes)
    base_conf = 40 + vote_diff * 10      # 1 голос разницы → 50, 2 → 60, 3 → 70…
    trend_boost = max(0.0, min((adx_val - 15) / 20.0 * 30, 25))  # чем больше ADX, тем выше
    conf = int(max(0, min(95, base_conf + trend_boost)))

    return {
        "direction": direction,
        "confidence": conf,
        "RSI": round(rsi_val, 1),
        "ADX": round(adx_val, 1),
        "MACD_HIST": round(macd_val, 6),
        "BW": round(bw, 2),
    }


# ==================== МУЛЬТИ-ТАЙМФРЕЙМ ====================

def multi_tf_analyze(yf_symbol: str) -> tuple[str, int, dict, dict]:
    """
    Возвращает:
    - финальное направление (BUY/SELL/FLAT)
    - уверенность (0–100)
    - словарь features (RSI, ADX, MACD, BW) по M5
    - словарь mtf_info: directions по М5/15/30 + режим
    """
    tf_results = {}
    tf_dirs = {}

    main_tf_name = "M5"

    for tf_name, (interval, period) in TF_CONFIG.items():
        df = load_ohlc(yf_symbol, period, interval)
        res = score_single_tf(df)
        tf_results[tf_name] = res
        tf_dirs[tf_name] = res["direction"]

    main_res = tf_results[main_tf_name]
    final_dir = main_res["direction"]
    final_conf = main_res["confidence"]

    # согласование таймфреймов
    agree = 0
    for other_tf in ("M15", "M30"):
        if tf_dirs.get(other_tf) == final_dir and final_dir in ("BUY", "SELL"):
            agree += 1

    if final_dir in ("BUY", "SELL"):
        if agree == 2:
            final_conf += 10
        elif agree == 1:
            final_conf += 5
        else:
            final_conf -= 10

    # режим рынка по ADX и ширине Боллинджера
    adx_val = main_res["ADX"]
    bw_val = main_res["BW"]
    if adx_val < 18 and bw_val < 3:
        regime = "flat"
    elif adx_val > 25 and bw_val < 7:
        regime = "trend"
    else:
        regime = "impulse"

    final_conf = int(max(0, min(99, final_conf)))

    mtf_info = {
        "M5": tf_dirs.get("M5", "FLAT"),
        "M15": tf_dirs.get("M15", "FLAT"),
        "M30": tf_dirs.get("M30", "FLAT"),
        "Regime": regime,
    }

    return final_dir, final_conf, main_res, mtf_info


# ================== ВЫБОР ЭКСПИРАЦИИ ======================

def choose_expiry(conf: int, adx_value: float) -> int:
    """Подбор экспирации в минутах на основе уверенности и ADX."""
    if conf < 55:
        return 0

    if conf < 65:
        base = 2
    elif conf < 75:
        base = 3
    elif conf < 85:
        base = 5
    elif conf < 92:
        base = 7
    else:
        base = 10

    # корректировка по ADX
    if adx_value >= 35:
        base += 1
    elif adx_value <= 15:
        base -= 1

    return int(max(1, min(20, base)))


# ================== TELEGRAM ==============================

def send_telegram_signal(
    pair_name: str,
    yf_symbol: str,
    mtype: str,
    direction: str,
    confidence: int,
    expiry_min: int,
    feats: dict,
    mtf_info: dict,
):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return

    arrow = "⬆️" if direction == "BUY" else ("⬇️" if direction == "SELL" else "➖")
    strength = "🔴 слабый" if confidence < 60 else ("🟡 средний" if confidence < 80 else "🟢 сильный")
    copy_code = pocket_code(pair_name, yf_symbol)

    text = (
        f"🤖 AI FX Сигнал {VERSION}\n"
        f"📊 Пара: {pair_name}\n"
        f"📌 Код для Pocket: {copy_code}\n"
        f"📄 Тип: {mtype}\n"
        f"{arrow} Сигнал: *{direction}*\n"
        f"📈 Multi-TF: M5={mtf_info['M5']} | M15={mtf_info['M15']} | M30={mtf_info['M30']}\n"
        f"🌍 Режим: {mtf_info['Regime']}\n"
        f"💪 Уверенность: *{confidence}%* ({strength})\n"
        f"⏱ Экспирация: *{expiry_min} мин*\n"
        f"📉 RSI {feats['RSI']} | ADX {feats['ADX']} | MACD {feats['MACD_HIST']}\n"
        f"⏰ {datetime.utcnow().strftime('%H:%M:%S')} UTC"
    )

    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        st.toast(f"Ошибка Telegram: {e}", icon="⚠️")


# ================== STREAMLIT UI ==========================

st.set_page_config(page_title=f"AI FX Bot {VERSION}", layout="wide")

st.title(f"🤖 AI FX Signal Bot {VERSION}")
st.caption("Тройной таймфрейм (M5+M15+M30), авто-обновление и Telegram сигналы.")

# Контролы
col_top1, col_top2, col_top3 = st.columns(3)
with col_top1:
    threshold = st.slider("Порог уверенности для сигнала (Telegram)", 50, 95, DEFAULT_THRESHOLD, 1)
with col_top2:
    min_gap = st.number_input("Мин. пауза между сигналами по паре (сек)", 10, 600, DEFAULT_MIN_GAP, 10)
with col_top3:
    refresh_sec = st.number_input("Авто-обновление, каждые (сек)", 1, 60, DEFAULT_REFRESH_SEC, 1)

if not TELEGRAM_TOKEN or not CHAT_ID:
    st.warning(
        "Telegram не настроен. Задай `TELEGRAM_TOKEN` и `CHAT_ID` в Secrets "
        "или переменных окружения, чтобы бот мог отправлять сигналы.",
        icon="⚠️",
    )

# память отправленных сигналов
if "last_sent" not in st.session_state:
    st.session_state.last_sent = {}

rows = []

# ================== ОСНОВНОЙ ЦИКЛ ПО ПАРАМ =================

for pair_name, yf_symbol in PAIRS.items():
    direction, conf, feats, mtf_info = multi_tf_analyze(yf_symbol)

    # тип инструмента
    is_otc_flag = is_otc(pair_name, yf_symbol)
    mtype = "OTC/24/7" if is_otc_flag else "Биржевая"

    expiry_min = choose_expiry(conf, feats["ADX"])
    # для OTC можно чуть больше
    if is_otc_flag and expiry_min > 0:
        expiry_min = min(30, expiry_min + 1)

    rows.append(
        [
            pair_name,
            mtype,
            direction,
            conf,
            expiry_min,
            f"M5={mtf_info['M5']} | M15={mtf_info['M15']} | M30={mtf_info['M30']}",
            mtf_info["Regime"],
            json.dumps(feats, ensure_ascii=False),
        ]
    )

    # проверяем условие отправки сигнала
    if (
        direction in ("BUY", "SELL")
        and conf >= threshold
        and expiry_min > 0
        and TELEGRAM_TOKEN
        and CHAT_ID
    ):
        prev = st.session_state.last_sent.get(pair_name, {})
        should_send = True

        if prev and ONLY_NEW:
            same_dir = prev.get("direction") == direction
            not_better = conf <= prev.get("conf", 0)
            too_soon = (time.time() - prev.get("ts", 0)) < min_gap
            if same_dir and (not_better or too_soon):
                should_send = False

        if should_send:
            send_telegram_signal(
                pair_name,
                yf_symbol,
                mtype,
                direction,
                conf,
                expiry_min,
                feats,
                mtf_info,
            )
            st.session_state.last_sent[pair_name] = {
                "direction": direction,
                "conf": conf,
                "ts": time.time(),
            }

# ================== ТАБЛИЦА СИГНАЛОВ ======================

df_show = pd.DataFrame(
    rows,
    columns=[
        "Пара",
        "Тип",
        "Сигнал",
        "Уверенность",
        "Экспирация (мин)",
        "Multi-TF",
        "Режим",
        "Индикаторы",
    ],
)

if not df_show.empty:
    df_show = df_show.sort_values("Уверенность", ascending=False).reset_index(drop=True)

st.subheader("📋 Рейтинг сигналов")
st.dataframe(df_show, use_container_width=True, height=480)

# ================== ГРАФИК ЛУЧШЕГО СИГНАЛА ================

if not df_show.empty:
    top_row = df_show.iloc[0]
    top_pair = top_row["Пара"]
    top_symbol = PAIRS[top_pair]
    top_dir = top_row["Сигнал"]
    top_conf = top_row["Уверенность"]

    df_chart = load_ohlc(top_symbol, TF_CONFIG["M5"][1], TF_CONFIG["M5"][0])
    if df_chart is not None and not df_chart.empty:
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
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            title=f"Топ-сигнал: {top_pair} — {top_dir} ({top_conf}%)",
        )
        st.subheader("📈 График лучшего сигнала (M5)")
        st.plotly_chart(fig, use_container_width=True)

# ================== АВТО-ОБНОВЛЕНИЕ ========================

st.caption(f"Версия {VERSION}. Авто-обновление: каждые {int(refresh_sec)} сек.")
time.sleep(int(refresh_sec))
st.rerun()
