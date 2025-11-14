# ===================== AI FX Bot v5.0 PRO =====================
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

REFRESH_SEC = 5              # автообновление, сек
ONLY_NEW = True              # не спамим одно и то же направление
MIN_SEND_GAP_S = 60          # пауза между сигналами по одной паре
BASE_CONF_THRESHOLD = 70     # базовый порог уверенности (запас на будущее)

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
            return pd.Series(
                [20.0],
                index=[df.index[-1] if df is not None and len(df) > 0 else datetime.now()]
            )

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

        # всегда возвращаем однострочный Series
        return pd.Series([float(adx.iloc[-1])], index=[df.index[-1]])
    except Exception:
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


def combine_multi_tf(
    m1_info: dict,
    m5_info: dict,
    m15_info: dict,
    m30_info: dict,
):
    """
    Ultra-PRO v3.
    Усиленный многотаймфреймовый сигнал.
    """

    # ---------- 1. Сбор направлений ----------
    infos = [m1_info, m5_info, m15_info, m30_info]
    signals = [i["signal"] for i in infos]

    buy_votes = signals.count("BUY")
    sell_votes = signals.count("SELL")

    # Итоговый сигнал
    if buy_votes == 0 and sell_votes == 0:
        final_signal = "FLAT"
    elif buy_votes > sell_votes:
        final_signal = "BUY"
    elif sell_votes > buy_votes:
        final_signal = "SELL"
    else:
        final_signal = "FLAT"

    # ---------- 2. Базовая уверенность по голосам ----------
    conf = 40  # старт

    max_votes = max(buy_votes, sell_votes)
    if max_votes == 4:
        conf += 35
    elif max_votes == 3:
        conf += 25
    elif max_votes == 2:
        conf += 15
    elif max_votes == 1:
        conf += 5

    # ---------- 3. Вес старших TF (M15 + M30) ----------
    high_tf_signals = [m15_info["signal"], m30_info["signal"]]
    if final_signal in ("BUY", "SELL"):
        high_tf_agree = high_tf_signals.count(final_signal)
        if high_tf_agree == 2:
            conf += 15
        elif high_tf_agree == 1:
            conf += 5
        else:
            conf -= 10  # старшие против — ослабляем

    # ---------- 4. ADX: сила тренда ----------
    avg_adx = (m5_info["ADX"] + m15_info["ADX"] + m30_info["ADX"]) / 3.0
    if avg_adx >= 35:
        conf += 10
    elif avg_adx >= 25:
        conf += 5
    elif avg_adx <= 15:
        conf -= 10

    # ---------- 5. RSI: зона перекупленности/перепроданности ----------
    avg_rsi = (m5_info["RSI"] + m15_info["RSI"] + m30_info["RSI"]) / 3.0

    if final_signal == "BUY":
        if avg_rsi >= 60:
            conf += 10
        elif avg_rsi <= 45:
            conf -= 10
    elif final_signal == "SELL":
        if avg_rsi <= 40:
            conf += 10
        elif avg_rsi >= 55:
            conf -= 10

    # Ограничиваем 0–100
    conf = int(max(0, min(100, conf)))

    # ---------- 6. Класс сигнала ----------
    if conf >= 92:
        trade_class = "A"
    elif conf >= 84:
        trade_class = "B"
    else:
        trade_class = "C"

    # ---------- 7. Режим и фаза рынка ----------
    regime_votes = [i["Regime"] for i in infos]
    trend_votes = regime_votes.count("trend")

    if trend_votes >= 3:
        regime = "trend"
    elif trend_votes == 2:
        regime = "mixed"
    else:
        regime = "flat"

    # Фазу возьмём по RSI M30
    rsi30 = m30_info["RSI"]
    if rsi30 < 40 or rsi30 > 60:
        phase = "start"   # начало импульса
    elif 40 <= rsi30 <= 45 or 55 <= rsi30 <= 60:
        phase = "mid"     # середина движения
    else:
        phase = "end"     # выдох движения / возможный разворот

    # ---------- 8. Детальная инфа для интерфейса/телеги ----------
    info = {
        "M1": m1_info["signal"],
        "M5": m5_info["signal"],
        "M15": m15_info["signal"],
        "M30": m30_info["signal"],

        # условные "локальные" уверенности для отображения
        "Conf_M1":  60 if m1_info["signal"] == final_signal else 40,
        "Conf_M5":  70 if m5_info["signal"] == final_signal else 40,
        "Conf_M15": 80 if m15_info["signal"] == final_signal else 40,
        "Conf_M30": 85 if m30_info["signal"] == final_signal else 40,

        "Regime": regime,
        "Phase": phase,
        "BW": abs(m30_info["RSI"] - 50),   # условная ширина тренда
        "ADX30": float(m30_info["ADX"]),
    }

    return final_signal, conf, trade_class, info


# ==================== ЭКСПИРАЦИЯ ====================

def choose_expiry(conf: int, regime: str = None, phase: str = None) -> int:
    """
    Умная экспирация под Pocket Option.
    - Очень сильные сигналы → 4–6 минут
    - Средние → 2–4 минуты
    - Всё, что слабее 80% → не торговать (0)
    """
    if conf < 80:
        return 0

    # базовое время по уверенности
    if conf >= 95:
        base = 6
    elif conf >= 90:
        base = 5
    elif conf >= 85:
        base = 4
    else:  # 80–84
        base = 3

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

def send_telegram(
    pair_name: str,
    pair_code: str,
    signal: str,
    conf: int,
    expiry: int,
    mtype: str,
    info: dict,
) -> None:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return

    # -------------------------
    # 1) Копируемый код валюты
    # -------------------------
    # BTCUSD → BTCUSD
    # EURUSD → EUR/USD
    if len(pair_code) == 6:
        pocket_code = pair_code[:3] + "/" + pair_code[3:]
    else:
        pocket_code = pair_code  # BTCUSD оставляем как есть

    # -------------------------
    # 2) Стрелка направления
    # -------------------------
    if signal == "BUY":
        arrow = "🟢 BUY"
    elif signal == "SELL":
        arrow = "🔴 SELL"
    else:
        arrow = "⚪ FLAT"

    # -------------------------
    # 3) Усиленная логика — отправляем только сильные точки входа
    # -------------------------
    m1 = info.get("M1", "?")
    m5 = info.get("M5", "?")
    m15 = info.get("M15", "?")
    m30 = info.get("M30", "?")
    adx = info.get("ADX30", 0.0)
    regime = info.get("Regime", "?")
    phase = info.get("Phase", "?")

    strong_trend = (m5 == signal and m15 == signal) or (m15 == signal and m30 == signal)
    multi_agree = sum([m1 == signal, m5 == signal, m15 == signal, m30 == signal])

    # Требования для "супер-сигналов"
    if conf < 80:
        return  # слабый

    if adx < 10:
        return  # тренда нет

    if multi_agree < 2:
        return  # слабая MTF структура

    # -------------------------
    # 4) Готовый текст (100% копируется)
    # -------------------------
    text = (
        f"🤖 AI FX Signal Bot v5.0 PRO\n"
        f"📌 Пара: {pair_name}\n"
        f"📋 Код для Pocket: {pocket_code}\n"
        f"🏷 Тип: {mtype}\n"
        f"{arrow}\n"
        f"\n"
        f"📊 Мульти-TF:\n"
        f"• M1: {m1}\n"
        f"• M5: {m5}\n"
        f"• M15: {m15}\n"
        f"• M30: {m30}\n"
        f"\n"
        f"💪 Уверенность: {conf}%\n"
        f"⏱ Экспирация: {expiry} мин\n"
        f"📈 ADX30: {adx}\n"
        f"🌍 Режим: {regime} | Фаза: {phase}\n"
        f"\n"
        f"❗ Бот для обучения. Не финсовет."
    )

    # -------------------------
    # 5) Отправка
    # -------------------------
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}

    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass


# ==================== UI ====================

st.set_page_config(
    page_title="AI FX Bot v5.0 — M1+M5+M15+M30 + Telegram",
    layout="wide",
)

st.title("AI FX Bot v5.0 PRO — M1+M5+M15+M30 + Telegram")

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

# ==================== CЕССИЯ (память) ====================

if "last_sent" not in st.session_state:
    st.session_state.last_sent = {}       # pair_name → timestamp

if "last_dir" not in st.session_state:
    st.session_state.last_dir = {}        # pair_name → 'BUY'/'SELL'/'FLAT'

if "last_conf" not in st.session_state:
    st.session_state.last_conf = {}       # pair_name → последняя уверенность

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

    # Тип рынка для текста
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

    # ------------ Anti-Spam v5.0 ------------
    now_ts = time.time()
    last_ts = st.session_state.last_sent.get(name, 0)

    prev_dir = st.session_state.last_dir.get(name, "NONE")
    prev_conf = st.session_state.last_conf.get(name, 0)

    dir_changed = (signal in ("BUY", "SELL")) and (signal != prev_dir)
    conf_jump = conf >= prev_conf + 7 or conf >= 95 > prev_conf
    time_ok = (now_ts - last_ts) >= MIN_SEND_GAP_S

    can_send = (
        signal in ("BUY", "SELL")
        and conf >= work_threshold
        and expiry > 0
        and time_ok
        and (not ONLY_NEW or dir_changed or conf_jump)
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
        st.session_state.last_dir[name] = signal
        st.session_state.last_conf[name] = conf

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

# ==================== АВТООБНОВЛЕНИЕ ====================

time.sleep(REFRESH_SEC)
st.experimental_rerun()
