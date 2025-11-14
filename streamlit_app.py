# ===================== AI FX Bot v4.1 PRO =====================
# M1 + M5 + M15 + M30 + Telegram
# Многотаймфреймовый анализ, мощный фильтр, умная экспирация.
# Бот — инструмент ДЛЯ ОБУЧЕНИЯ, не финсовет.

import time
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st

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

REFRESH_SEC = 1              # автообновление, сек (используем на стороне браузера)
ONLY_NEW = True              # не спамим одно и то же
MIN_SEND_GAP_S = 60          # пауза между сигналами по одной паре
BASE_CONF_THRESHOLD = 70     # базовый порог уверенности

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
                "low": [1.0],
                "close": [1.0],
                "volume": [0],
            },
            index=[now],
        )
    return df


# ==================== ИНДИКАТОРЫ ====================

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Безопасный RSI.
    Не ломается, если данных мало или попадаются NaN.
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)

    if len(series) < period + 1:
        # мало свечей → ровный RSI 50
        return pd.Series([50.0] * len(series), index=series.index)

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.fillna(50)

    return rsi


def calc_macd(series: pd.Series):
    """Безопасный MACD, всегда возвращает 3 серии."""
    if series is None or len(series) == 0:
        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )

    if len(series) < 35:
        s = pd.Series([0.0] * len(series), index=series.index)
        return s, s, s

    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Железобетонный ADX — никогда не вызывает ошибки 1-dimension."""
    try:
        if df is None or len(df) < period + 2:
            return pd.Series([20.0], index=[df.index[-1] if df is not None and len(df)>0 else datetime.now()])

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        up = high.diff()
        down = -low.diff()

        plus_dm_arr = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm_arr = np.where((down > up) & (down > 0), down, 0.0)

        plus_dm = pd.Series(plus_dm_arr, index=df.index)
        minus_dm = pd.Series(minus_dm_arr, index=df.index)

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(period).mean().replace(0, np.nan)

        plus_di = 100 * (plus_dm.rolling(period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(period).sum() / atr)

        dx = ((plus_di - minus_di).abs() /
             (plus_di + minus_di).replace(0, np.nan)) * 100

        adx = dx.rolling(period).mean().fillna(20.0)

        # всегда возвращаем однострочный Series → невозможно словить ошибку
        return pd.Series([float(adx.iloc[-1])], index=[df.index[-1]])
    except:
        return pd.Series([20.0], index=[datetime.now()])


# ==================== ЛОГИКА СИГНАЛОВ ====================

def analyze_tf(df: pd.DataFrame) -> dict:
    """Сигнал по одному tf."""
    close = df["close"]

    rsi = calc_rsi(close)
    macd, sig, hist = calc_macd(close)
    adx = calc_adx(df)

    last = df.index[-1]

    rsi_v = float(rsi.iloc[-1]) if len(rsi) else 50.0
    macd_v = float(macd.iloc[-1]) if len(macd) else 0.0
    sig_v = float(sig.iloc[-1]) if len(sig) else 0.0
    hist_v = float(hist.iloc[-1]) if len(hist) else 0.0
    adx_v = float(adx.iloc[-1]) if len(adx) else 20.0

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
        "RSI": rsi_v,
        "MACD": macd_v,
        "MACD_sig": sig_v,
        "MACD_hist": hist_v,
        "ADX": adx_v,
        "Regime": regime,
    }


def combine_multi_tf(m1_info, m5_info, m15_info, m30_info):
    """
    Ultra-PRO v2 — мощно, но не сверхжестко.
    Дает реальные сигналы A/B, фильтрует мусор, не режет рынок полностью.
    """

    signals = [
        m1_info["signal"],
        m5_info["signal"],
        m15_info["signal"],
        m30_info["signal"],
    ]

    buy_votes = signals.count("BUY")
    sell_votes = signals.count("SELL")

    # RSI + ADX старших TF
    adx30 = float(m30_info["ADX"])
    avg_rsi = (m5_info["RSI"] + m15_info["RSI"] + m30_info["RSI"]) / 3.0

    regimes = [m5_info["Regime"], m15_info["Regime"], m30_info["Regime"]]
    trend_votes = regimes.count("trend")

    # Базовый сигнал
    if buy_votes >= 2 and buy_votes > sell_votes:
        base_signal = "BUY"
    elif sell_votes >= 2 and sell_votes > buy_votes:
        base_signal = "SELL"
    else:
        base_signal = "FLAT"

    final_signal = base_signal

    # Фильтрация слабых ситуаций — но мягкая
    if base_signal == "BUY":
        if avg_rsi < 52 or adx30 < 18:
            final_signal = "FLAT"

    elif base_signal == "SELL":
        if avg_rsi > 48 or adx30 < 18:
            final_signal = "FLAT"

    # -------- Уверенность (0-100) --------
    score = 0

    # Голоса TF
    score += max(buy_votes, sell_votes) * 12  # максимум 48

    # Тренд старших TF
    score += trend_votes * 10  # максимум 30

    # ADX → сила движения
    score += min(int(adx30 * 1.2), 20)  # максимум 20

    # RSI (далеко ли от 50)
    score += min(int(abs(avg_rsi - 50) * 1.2), 15)

    conf = min(99, max(0, score))

    # Если FLAT — ограничиваем
    if final_signal == "FLAT":
        conf = min(conf, 60)

    # Классы
    if conf >= 90:
        trade_class = "A"
    elif conf >= 80:
        trade_class = "B"
    else:
        trade_class = "C"

    regime = "trend" if trend_votes >= 2 else "flat"

    if avg_rsi <= 40 or avg_rsi >= 60:
        phase = "start"
    elif 45 < avg_rsi < 55:
        phase = "mid"
    else:
        phase = "end"

    return final_signal, conf, trade_class, {
        "M1": m1_info["signal"],
        "M5": m5_info["signal"],
        "M15": m15_info["signal"],
        "M30": m30_info["signal"],
        "Regime": regime,
        "Phase": phase,
        "ADX30": round(adx30, 2),
                   }

    # --------- Ultra-PRO фильтр качества ---------
    strong = False
    if base_signal == "BUY":
        strong = (
            buy_votes >= 3
            and trend_votes >= 2          # тренд по старшим
            and avg_rsi >= 58             # не середина, а нормальный перекос
            and adx30 >= 22               # рынок не мёртвый
        )
    elif base_signal == "SELL":
        strong = (
            sell_votes >= 3
            and trend_votes >= 2
            and avg_rsi <= 42
            and adx30 >= 22
        )

    if not strong:
        # если условия не выполнены — сигнал считаем FLAT,
        # чтобы бот молчал в мусоре
        final_signal = "FLAT"

    # --------- Подсчёт уверенности (0–99) ---------
    score = 0.0

    # 1) Согласие TF
    score += max(buy_votes, sell_votes) * 8.0          # до ~32

    # 2) Тренд по старшим TF
    score += min(trend_votes, 3) * 8.0                 # до ~24

    # 3) ADX силы тренда
    adx_score = max(0.0, min(adx30, 50.0)) / 50.0 * 25.0
    score += adx_score                                  # до ~25

    # 4) Насколько RSI далеко от 50 (чем дальше — тем лучше)
    rsi_edge = abs(avg_rsi - 50.0)
    rsi_score = max(0.0, min(rsi_edge, 20.0)) / 20.0 * 18.0
    score += rsi_score                                  # до ~18

    conf = int(round(max(0.0, min(score, 99.0))))

    # Для FLAT не даём конфу выглядеть как "суперсигнал"
    if final_signal == "FLAT":
        conf = min(conf, 75)

    # --------- Класс сигнала ---------
    if conf >= 92:
        trade_class = "A+"
    elif conf >= 85:
        trade_class = "A"
    elif conf >= 80:
        trade_class = "B"
    else:
        trade_class = "C"

    # --------- Режим и фаза для инфо ---------
    regime = "trend" if trend_votes >= 2 else "flat"

    if avg_rsi <= 35 or avg_rsi >= 65:
        phase = "end"      # возможное окончание импульса
    elif avg_rsi < 45 or avg_rsi > 55:
        phase = "start"    # активная зона движения
    else:
        phase = "mid"      # середина, шум

    # --------- Информация для таблицы/телеги ---------
    info = {
        "M1": m1_info["signal"],
        "M5": m5_info["signal"],
        "M15": m15_info["signal"],
        "M30": m30_info["signal"],

        # можно потом показывать по ТФ, если захочешь
        "Conf_M1": conf if m1_info["signal"] == final_signal else 0,
        "Conf_M5": conf if m5_info["signal"] == final_signal else 0,
        "Conf_M15": conf if m15_info["signal"] == final_signal else 0,
        "Conf_M30": conf if m30_info["signal"] == final_signal else 0,

        "Regime": regime,
        "Phase": phase,
        "BW": round(abs(avg_rsi - 50.0), 2),  # ширина отклонения от середины
        "ADX30": round(adx30, 2),
    }

    return final_signal, conf, trade_class, info


# ==================== ЭКСПИРАЦИЯ ====================

def choose_expiry(conf: int, regime: str = None, phase: str = None) -> int:
    """
    Ultra-PRO экспирация:
    - Очень сильные сигналы → 2–3 минуты
    - Средние → 4–6 минут
    - Всё, что слабее 80% → не торговать (0)
    """
    # слишком слабые — пропуск
    if conf < 80:
        return 0

    # базовое время по уверенности
    if conf >= 95:
        base = 2
    elif conf >= 90:
        base = 3
    elif conf >= 85:
        base = 4
    else:  # 80–84
        base = 5

    # тренд → можно держать дольше
    if regime == "trend":
        base += 1
    elif regime == "flat":
        base -= 1

    # фаза движения
    if phase == "start":
        base += 1        # старт импульса → даём ещё минуту
    elif phase == "end":
        base -= 1        # конец — аккуратно

    if base <= 0:
        return 0

    return int(max(1, min(15, base)))  # ограничим 1–15 минут


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

    if signal == "BUY":
        arrow = "🟢"
    elif signal == "SELL":
        arrow = "🔴"
    else:
        arrow = "⚪️"

    multi_str = (
        f"M1={info.get('M1', '?')} | "
        f"M5={info.get('M5', '?')} | "
        f"M15={info.get('M15', '?')} | "
        f"M30={info.get('M30', '?')}"
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
        f"🌍 Режим: {info.get('Regime', '-')} | Фаза: {info.get('Phase', '-')}\n"
        f"ADX30: {round(info.get('ADX30', 0), 2)}\n"
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
    df_m1 = get_or_fake(symbol, TF_M1)
    df_m5 = get_or_fake(symbol, TF_M5)
    df_m15 = get_or_fake(symbol, TF_M15)
    df_m30 = get_or_fake(symbol, TF_M30)

    info_m1 = analyze_tf(df_m1)
    info_m5 = analyze_tf(df_m5)
    info_m15 = analyze_tf(df_m15)
    info_m30 = analyze_tf(df_m30)

    signal, conf, trade_class, mtf_info = combine_multi_tf(
        info_m1, info_m5, info_m15, info_m30
    )

    if "BTC" in name or "ETH" in name or "OIL" in name:
        mtype = "OTC/24/7"
    else:
        mtype = "Биржевая"

    regime = mtf_info.get("Regime")
    phase = mtf_info.get("Phase")
    expiry = choose_expiry(conf, regime, phase)

    multi_str = (
        f"M1={mtf_info['M1']} | "
        f"M5={mtf_info['M5']} | "
        f"M15={mtf_info['M15']} | "
        f"M30={mtf_info['M30']}"
    )

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
