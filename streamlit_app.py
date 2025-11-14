# ===================== AI FX Bot v4.1 PRO =====================
# M1 + M5 + M15 + M30 + Telegram
# Многотаймфреймовый анализ, мощный фильтр, умная экспирация.
# Бот — инструмент ДЛЯ ОБУЧЕНИЯ, не финсовет.

import time, json, random, os
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
import plotly.graph_objects as go

# ==================== SECRETS ====================

TELEGRAM_TOKEN = st.secrets.get(
    "TELEGRAM_TOKEN",
    os.getenv("TELEGRAM_TOKEN", "")
)
CHAT_ID = st.secrets.get(
    "CHAT_ID",
    os.getenv("CHAT_ID", "")
)

# ==================== SETTINGS ====================

REFRESH_SEC        = 1          # автообновление, сек (используем на стороне браузера)
ONLY_NEW           = True       # не спамим одно и то же
MIN_SEND_GAP_S     = 60         # пауза между сигналами по одной паре
BASE_CONF_THRESHOLD = 70        # базовый порог уверенности

# Режимы фильтра
MODES = {
    "Safe 85%":   85,
    "Normal 90%": 90,
    "Hard 95%":   95,
    "Ultra 99%":  99,
}

# Таймфреймы
TF_M1  = ("1m",  "1d")
TF_M5  = ("5m",  "5d")
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
    "EURAUD": "EURAUD=X",
    "EURNZD": "EURNZD=X",
    "GBPAUD": "GBPAUD=X",
    "GBPNZD": "GBPNZD=X",

    # Crypto / индексы (PO любит)
    "BTCUSD (Bitcoin)": "BTC-USD",
    "ETHUSD (Ethereum)": "ETH-USD",
    "XAUUSD (Gold)": "XAUUSD=X",
    "USOIL (Brent/WTI)": "BZ=F",
}

# ==================== УТИЛИТЫ ====================

@st.cache_data(show_spinner=False)
def load_history(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Загрузка истории через yfinance с кэшем."""
    try:
        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return pd.DataFrame()
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        return df.dropna()
    except Exception:
        return pd.DataFrame()


def get_or_fake(symbol: str, tf: tuple) -> pd.DataFrame:
    """Получаем реальный tf, если не вышло — делаем фейковый FLAT."""
    interval, period = tf
    df = load_history(symbol, interval, period)
    if df.empty:
        now = datetime.now(timezone.utc)
        return pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.0],
                "low":  [1.0],
                "close": [1.0],
                "volume": [0],
            },
            index=[now],
        )
    return df


# ==================== ИНДИКАТОРЫ ====================

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(series: pd.Series):
    fast = series.ewm(span=12, adjust=False).mean()
    slow = series.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = low.diff() * -1

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).sum() / (atr + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).sum() / (atr + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx = dx.rolling(period).mean()
    return adx


# ==================== ЛОГИКА СИГНАЛОВ ====================

def analyze_tf(df: pd.DataFrame) -> dict:
    """Сигнал по одному tf."""
    close = df["close"]
    rsi = calc_rsi(close)
    macd, sig, hist = calc_macd(close)
    adx = calc_adx(df)

    last = df.index[-1]
    rsi_v = rsi.iloc[-1]
    macd_v = macd.iloc[-1]
    sig_v = sig.iloc[-1]
    hist_v = hist.iloc[-1]
    adx_v = adx.iloc[-1]

    # Направление
    if rsi_v > 60 and macd_v > sig_v and hist_v > 0:
        signal = "BUY"
    elif rsi_v < 40 and macd_v < sig_v and hist_v < 0:
        signal = "SELL"
    else:
        signal = "FLAT"

    # Режим рынка по ADX
    if adx_v >= 25:
        regime = "trend"
    else:
        regime = "flat"

    return {
        "time": last,
        "signal": signal,
        "RSI": float(rsi_v),
        "MACD": float(macd_v),
        "MACD_sig": float(sig_v),
        "MACD_hist": float(hist_v),
        "ADX": float(adx_v),
        "Regime": regime,
    }


def combine_multi_tf(m1_info, m5_info, m15_info, m30_info):
    """Комбинируем 4 таймфрейма → один мощный сигнал + уверенность."""
    signals = [m1_info["signal"], m5_info["signal"],
               m15_info["signal"], m30_info["signal"]]

    # Переводим в голоса
    buy_votes = signals.count("BUY")
    sell_votes = signals.count("SELL")

    if buy_votes == 0 and sell_votes == 0:
        final_signal = "FLAT"
    elif buy_votes > sell_votes:
        final_signal = "BUY"
    elif sell_votes > buy_votes:
        final_signal = "SELL"
    else:
        final_signal = "FLAT"

    # Уверенность
    conf = 50  # база
    conf += 10 * max(buy_votes, sell_votes)  # чем больше совпадений, тем лучше

    # Усиливаем за тренд по старшим tf
    regimes = [m5_info["Regime"], m15_info["Regime"], m30_info["Regime"]]
    trend_votes = regimes.count("trend")
    if trend_votes >= 2:
        conf += 10

    # Проверяем RSI по старшим tf
    avg_rsi = (m5_info["RSI"] + m15_info["RSI"] + m30_info["RSI"]) / 3
    if final_signal == "BUY" and avg_rsi < 55:
        conf -= 10
    if final_signal == "SELL" and avg_rsi > 45:
        conf -= 10

    conf = int(max(0, min(100, conf)))

    # Класс сигнала
    if conf >= 90:
        trade_class = "A"
    elif conf >= 80:
        trade_class = "B"
    else:
        trade_class = "C"

    # Режим/фаза для инфо
    regime = "trend" if trend_votes >= 2 else "flat"
    phase = "start" if avg_rsi < 45 or avg_rsi > 55 else "mid"

    # Детальная инфа для таблицы/телеги
    info = {
        "M1":  m1_info["signal"],
        "M5":  m5_info["signal"],
        "M15": m15_info["signal"],
        "M30": m30_info["signal"],
        "Conf_M1":  conf,
        "Conf_M5":  conf,
        "Conf_M15": conf,
        "Conf_M30": conf,
        "Regime": regime,
        "Phase": phase,
        "BW": abs(m30_info["RSI"] - 50),  # грубый "bandwidth" тренда
        "ADX30": m30_info["ADX"],
    }

    return final_signal, conf, trade_class, info


# ==================== ЭКСПИРАЦИЯ ====================

def choose_expiry(conf: int, regime: str = None, phase: str = None) -> int:
    """
    Умная экспирация, БЕЗ KeyError.
    Никаких обращений к mtf['xxx'] внутри — только значения, которые передали.
    Возвращает минуты (1–30).
    """
    # 1) сильные сигналы → короткая экспирация (2–3 минуты)
    if conf >= 95:
        base = 2
    elif conf >= 90:
        base = 3
    elif conf >= 85:
        base = 4
    elif conf >= 80:
        base = 5
    else:
        base = 0  # слишком слабый, можно вообще не торговать

    # 2) тренд → держим дольше
    if regime == "trend":
        base += 2
    elif regime == "flat":
        base -= 1

    # 3) фаза (начало/конец движения)
    if phase == "start":
        base += 1
    elif phase == "end":
        base -= 1

    if base <= 0:
        return 0
    return int(max(1, min(30, base)))


# ==================== TELEGRAM ====================

def send_telegram(pair_name: str,
                  pair_code: str,
                  signal: str,
                  conf: int,
                  expiry: int,
                  mtype: str,
                  info: dict) -> None:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return

    arrow = "🟢" if signal == "BUY" else ("🔴" if signal == "SELL" else "⚪️")

    multi_str = (
        f"M1={info.get('M1','?')} | "
        f"M5={info.get('M5','?')} | "
        f"M15={info.get('M15','?')} | "
        f"M30={info.get('M30','?')}"
    )

    text = (
        f"🤖 AI FX Signal Bot v4.1 PRO\n"
        f"📌 Пара: {pair_name}\n"
        f"📊 Код для Pocket: {pair_code}\n"
        f"🏷 Тип: {mtype}\n"
        f"{arrow} Сигнал: {signal}\n\n"
        f"📉 Мульти-TF: {multi_str}\n"
        f"📈 Уверенность: {conf}%\n"
        f"⏱ Экспирация: {expiry} мин\n"
        f"🌍 Режим: {info.get('Regime','-')} | Фаза: {info.get('Phase','-')}\n"
        f"ADX30: {round(info.get('ADX30',0),2)}\n"
        f"❗ Бот для обучения. Не финсовет."
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass


# ==================== UI ====================

st.set_page_config(
    page_title="AI FX Bot v4.1 — M1+M5+M15+M30 + Telegram",
    layout="wide",
)

st.title("AI FX Bot v4.1 PRO — M1+M5+M15+M30 + Telegram")

st.markdown(
    "Режимы **Safe / Normal / Hard / Ultra** — это стиль фильтра, а не гарантия. "
    "Бот — инструмент для обучения, не финансовый совет."
)

col1, col2 = st.columns(2)

with col1:
    mode_name = st.selectbox("Режим отбора сигналов", list(MODES.keys()), index=0)

with col2:
    min_conf_slider = st.slider(
        "Минимальная уверенность (%) для сигнала",
        min_value=50,
        max_value=99,
        value=85,
    )

gap_input = st.number_input(
    "Пауза между сигналами по паре (сек)",
    min_value=10,
    max_value=3600,
    value=MIN_SEND_GAP_S,
    step=10,
)
MIN_SEND_GAP_S = int(gap_input)

work_threshold = max(MODES[mode_name], min_conf_slider)

st.markdown(
    f"**Текущий рабочий порог для отправки сигналов в Telegram: {work_threshold}%**"
)

# ==================== CЕССИЯ ====================

if "last_sent" not in st.session_state:
    st.session_state.last_sent = {}  # pair_name → timestamp

rows = []

# ==================== MAIN LOOP (один прогон) ====================

for name, symbol in PAIRS.items():
    # Загружаем 4 Tf
    df_m1  = get_or_fake(symbol, TF_M1)
    df_m5  = get_or_fake(symbol, TF_M5)
    df_m15 = get_or_fake(symbol, TF_M15)
    df_m30 = get_or_fake(symbol, TF_M30)

    # Анализ по каждому TF
    info_m1  = analyze_tf(df_m1)
    info_m5  = analyze_tf(df_m5)
    info_m15 = analyze_tf(df_m15)
    info_m30 = analyze_tf(df_m30)

    signal, conf, trade_class, mtf_info = combine_multi_tf(
        info_m1, info_m5, info_m15, info_m30
    )

    # Тип рынка (OTC/Биржа) — пока грубо: крипта и нефть → OTC/24/7
    if "BTC" in name or "ETH" in name or "OIL" in name:
        mtype = "OTC/24/7"
    else:
        mtype = "Биржевая"

    regime = mtf_info.get("Regime")
    phase = mtf_info.get("Phase")
    expiry = choose_expiry(conf, regime, phase)

    # Строка мульти-TF для таблицы
    multi_str = (
        f"M1={mtf_info['M1']} | "
        f"M5={mtf_info['M5']} | "
        f"M15={mtf_info['M15']} | "
        f"M30={mtf_info['M30']}"
    )

    # Для таблицы в интерфейсе
    rows.append(
        [
            name,
            mtype,
            signal,
            conf,
            trade_class,
            expiry,
            multi_str,
            round(mtf_info["ADX30"], 2),
        ]
    )

    # ====== Отправка в Telegram ======
    now_ts = time.time()
    last_ts = st.session_state.last_sent.get(name, 0)
    can_send = (
        signal in ("BUY", "SELL")
        and conf >= work_threshold
        and expiry > 0
        and (now_ts - last_ts) >= MIN_SEND_GAP_S
    )

    if can_send:
        send_telegram(
            pair_name=name,
            pair_code=name.replace(" ", "").split("(")[0],
            signal=signal,
            conf=conf,
            expiry=expiry,
            mtype=mtype,
            info=mtf_info,
        )
        st.session_state.last_sent[name] = now_ts

# ==================== ТАБЛИЦА СИГНАЛОВ ====================

st.markdown("## 📋 Таблица сигналов")

df_signals = pd.DataFrame(
    rows,
    columns=[
        "Пара",
        "Тип",
        "Сигнал",
        "Уверенность",
        "Класс",
        "Экспирация (мин)",
        "Multi-TF",
        "ADX30",
    ],
)

st.dataframe(df_signals, use_container_width=True)

st.caption(
    "При уверенности < 80% и классе C лучше пропускать сигнал. "
    "Уровень A — самые сильные точки входа по логике бота."
)
