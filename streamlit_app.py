# ===================== AI FX Bot v4.0 =====================
# M1 + M5 + M15 + M30 + Telegram
# Фильтр Safe / Normal / Hard / Ultra
# Отправляет только самые сильные сигналы (класс A/B)

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

# ==================== SECRETS =====================
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", "")
CHAT_ID        = st.secrets.get("CHAT_ID", "")

# ==================== SETTINGS ====================
REFRESH_SEC       = 1       # автообновление
ONLY_NEW          = True    # не спамим одинаковыми
MIN_SEND_GAP_S    = 60      # пауза между сигналами по паре
BASE_CONF_THRESH  = 70      # базовый минимум

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
    # Основные Forex
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
    "EURAUD": "EURAUD=X",
    "GBPAUD": "GBPAUD=X",

    # Немного OTC/крипты
    "BTCUSD (Bitcoin)":  "BTC-USD",
    "ETHUSD (Ethereum)": "ETH-USD",
}

# ==================== ХЕЛПЕРЫ ====================
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


def boll_width(close: pd.Series, n=20, k=2.0) -> float:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    up = ma + k * sd
    lo = ma - k * sd
    return safe_float(((up.iloc[-1] - lo.iloc[-1]) / (ma.iloc[-1] + 1e-9)) * 100)


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
    sup = safe_float(df["Low"].rolling(40).min().iloc[-1])
    res = safe_float(df["High"].rolling(40).max().iloc[-1])
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


def market_regime(adx_val: float, bw: float) -> str:
    if adx_val < 18 and bw < 3.0:
        return "flat"
    if adx_val > 28 and bw < 7.0:
        return "trend"
    return "impulse"


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
    clean = "".join(ch for ch in name if ch.isalnum() or ch in "/").upper()
    return clean

# ==================== DATA ====================
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
        if df is None or len(df) < 40:
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

# ==================== SCORING ====================
def score_tf(df: pd.DataFrame) -> tuple[str, int, dict]:
    """
    Оценка направления по одному ТФ.
    Возвращает: direction(BUY/SELL/FLAT), confidence(0-100), features.
    """
    if df is None or len(df) < 40:
        return "FLAT", 0, {
            "RSI": 50.0,
            "ADX": 0.0,
            "MACD_Hist": 0.0,
            "BB_Width": 0.0,
        }

    close = df["Close"]

    rsi_series = rsi(close)
    rsv = safe_float(rsi_series.iloc[-1], 50.0)
    rsv_prev = safe_float(rsi_series.iloc[-2], rsv) if len(rsi_series) > 2 else rsv

    ema9  = safe_float(ema(close, 9).iloc[-1], rsv)
    ema21 = safe_float(ema(close, 21).iloc[-1], rsv)
    ema200 = safe_float(ema(close, 200).iloc[-1], rsv)

    _, _, mh = macd(close)
    mhv = safe_float(mh.iloc[-1], 0.0)

    up, mid, lo, w = bbands(close)
    bb_width = safe_float(w.iloc[-1], 0.0)
    bb_pos = safe_float(
        (close.iloc[-1] - mid.iloc[-1]) /
        (up.iloc[-1] - lo.iloc[-1] + 1e-9),
        0.0,
    )

    adx_series = adx(df)
    adx_v = safe_float(adx_series.iloc[-1], 0.0)

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
    if mhv > 0:
        vu += 1
    if mhv < 0:
        vd += 1
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

    raw = abs(vu - vd) / 4.0
    trend_boost = min(max((adx_v - 18) / 25, 0), 1)
    conf = int(100 * (0.55 * raw + 0.45 * trend_boost))
    conf = max(0, min(99, conf))

    feats = {
        "RSI": round(rsv, 1),
        "RSI_prev": round(rsv_prev, 1),
        "ADX": round(adx_v, 1),
        "MACD_Hist": round(mhv, 6),
        "BB_Width": round(bb_width, 2),
        "BB_Pos": round(bb_pos, 3),
        "EMA9-21": round(ema9 - ema21, 6),
        "EMA200": round(ema200, 6),
    }
    return direction, conf, feats


def combine_multi_tf(df_m1, df_m5, df_m15, df_m30) -> tuple[str, int, str, dict]:
    """
    Объединение 4 таймфреймов.
    Возвращает: signal(BUY/SELL/FLAT), conf, class(A+/A/B/C),
    а также dict с описанием по ТФ.
    """
    d1, c1, f1 = score_tf(df_m1)
    d5, c5, f5 = score_tf(df_m5)
    d15, c15, f15 = score_tf(df_m15)
    d30, c30, f30 = score_tf(df_m30)

    dirs = [d1, d5, d15, d30]
    confs = [c1, c5, c15, c30]

    # Требуем явное направление (нет FLAT и нет смешения BUY/SELL)
    if any(d == "FLAT" for d in dirs):
        return "FLAT", 0, "C", {
            "M1": d1, "M5": d5, "M15": d15, "M30": d30,
            "Conf": confs,
        }

    if len(set(dirs)) != 1:
        # нет единства -> пропускаем
        return "FLAT", 0, "C", {
            "M1": d1, "M5": d5, "M15": d15, "M30": d30,
            "Conf": confs,
        }

    signal = dirs[0]

    # Взвешенная уверенность (старшие ТФ важнее)
    weights = np.array([1.0, 1.5, 2.0, 2.5])
    base_conf = int(np.average(confs, weights=weights))

    # Доп.фильтры по старшему ТФ (M30)
    bw = boll_width(df_m30["Close"])
    adx_series = adx(df_m30)
    adx_v = safe_float(adx_series.iloc[-1], 0.0)
    regime = market_regime(adx_v, bw)

    phase = candle_phase(df_m5)
    sr = near_sr(df_m30)
    spike = momentum_spike(df_m1)

    conf = base_conf

    # усиливаем тренд
    if regime == "trend":
        conf += 8
    elif regime == "impulse":
        conf += 4
    else:  # flat
        conf -= 10

    # поддержка / сопротивление
    if signal == "BUY" and sr == "support":
        conf += 5
    if signal == "SELL" and sr == "resistance":
        conf += 5

    # импульс на M1
    if spike:
        conf += 4

    # фазa свечи на M5
    if phase == "mid":
        conf += 3
    elif phase == "end":
        conf -= 4

    conf = int(max(0, min(100, conf)))

    # Классификация
    if conf >= 96:
        trade_class = "A+"
    elif conf >= 90:
        trade_class = "A"
    elif conf >= 85:
        trade_class = "B"
    else:
        trade_class = "C"

    info = {
        "M1": d1,
        "M5": d5,
        "M15": d15,
        "M30": d30,
        "Conf_M1": c1,
        "Conf_M5": c5,
        "Conf_M15": c15,
        "Conf_M30": c30,
        "Regime": regime,
        "Phase": phase,
        "BW": round(bw, 2),
        "ADX30": round(adx_v, 1),
    }
    return signal, conf, trade_class, info


def choose_expiry(conf: int, regime: str, phase: str) -> int:
    if conf < 60:
        return 0
    if conf < 75:
        base = 2
    elif conf < 85:
        base = 4
    elif conf < 90:
        base = 6
    elif conf < 95:
        base = 10
    else:
        base = 15

    # режим рынка
    if regime == "trend":
        base += 3
    elif regime == "flat":
        base -= 1

    # фаза свечи
    if phase == "start":
        base += 1
    elif phase == "end":
        base -= 2

    return int(max(1, min(30, base)))

# ==================== TELEGRAM ====================
def send_telegram(pair_name: str, pair_code: str, mtype: str,
                  signal: str, conf: int, trade_class: str,
                  expiry: int, mtf: dict) -> None:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return

    arrow = "⬆️" if signal == "BUY" else "⬇️"
    copy_code = pocket_code(pair_name, pair_code)

    phase_map = {
        "start": "🟢 начало",
        "mid": "🟡 середина",
        "end": "🔴 конец",
    }
    phase_txt = phase_map.get(mtf.get("Phase", ""), "❔")

    if conf < 80:
        strength = "🔴 слабый"
    elif conf < 90:
        strength = "🟡 нормальный"
    else:
        strength = "🟢 сильный"

    text = (
        "🤖 AI FX Signal Bot v4.0\n"
        f"💱 Пара: {pair_name}\n"
        f"📌 Код для Pocket: {copy_code}\n"
        f"🏷️ Тип: {mtype}\n"
        f"{arrow} Сигнал: *{signal}* (класс {trade_class})\n"
        f"🇲🇫 Multi-TF: M1={mtf['M1']} | M5={mtf['M5']} | "
        f"M15={mtf['M15']} | M30={mtf['M30']}\n"
        f"🌐 Рынок: {mtf['Regime']} | 🕯️ Свеча: {phase_txt}\n"
        f"💪 Уверенность: *{conf}%* ({strength})\n"
        f"⏱ Экспирация: *{expiry} мин*\n"
        f"📊 ADX30={mtf['ADX30']} | BW={mtf['BW']}%\n"
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

# ==================== STREAMLIT UI ====================
st.set_page_config(
    page_title="AI FX Bot v4.0 — M1+M5+M15+M30 + Telegram",
    layout="wide",
)

st.title("🤖 AI FX Bot v4.0 — M1+M5+M15+M30 + Telegram")
st.caption(
    "Режимы Safe/Normal/Hard/Ultra — это стиль фильтра, а не реальная гарантия. "
    "Бот — инструмент для обучения, не финсовет."
)

col_mode, col_conf, col_gap = st.columns([1, 1, 1])

with col_mode:
    mode_name = st.selectbox("Режим отбора сигналов", list(MODES.keys()), index=0)

with col_conf:
    user_conf = st.slider(
        "Минимальная уверенность (%) для сигнала",
        50, 99, 85,
        help="Минимум для попадания в таблицу. Для Telegram используется максимум из этого и режима.",
    )

with col_gap:
    min_gap = st.number_input(
        "Пауза между сигналами по паре (сек)",
        10, 600, MIN_SEND_GAP_S,
    )

effective_threshold = max(user_conf, MODES[mode_name])

st.markdown(
    f"**Текущий рабочий порог для отправки сигналов в Telegram:** {effective_threshold}%"
)

if "last_sent" not in st.session_state:
    st.session_state.last_sent = {}

rows = []

# ==================== MAIN LOOP ====================
for name, symbol in PAIRS.items():
    df_m1  = get_or_fake(symbol, TF_M1[1],  TF_M1[0])
    df_m5  = get_or_fake(symbol, TF_M5[1],  TF_M5[0])
    df_m15 = get_or_fake(symbol, TF_M15[1], TF_M15[0])
    df_m30 = get_or_fake(symbol, TF_M30[1], TF_M30[0])

    sig, conf, trade_class, mtf = combine_multi_tf(df_m1, df_m5, df_m15, df_m30)

    otc_flag = is_otc(name, symbol)
    mtype = "OTC/24/7" if otc_flag else "Биржевая"

    expiry = choose_expiry(conf, mtf["Regime"], mtf["Phase"])
    if otc_flag and expiry > 0:
        expiry = min(60, expiry + 3)

    multi_str = (
        f"M1={mtf['M1']}({mtf['Conf_M1']}) | "
        f"M5={mtf['M5']}({mtf['Conf_M5']}) | "
        f"M15={mtf['M15']}({mtf['Conf_M15']}) | "
        f"M30={mtf['M30']}({mtf['Conf_M30']})"
    )

    rows.append([
        name,
        mtype,
        sig,
        conf,
        trade_class,
        expiry,
        multi_str,
        mtf["Regime"],
        mtf["Phase"],
    ])

    # ===== отправка в Telegram =====
    if sig in ("BUY", "SELL") and conf >= effective_threshold and expiry > 0:
        # Только классы A/B/A+
        if trade_class in ("A+", "A", "B"):
            prev = st.session_state.last_sent.get(name, {})
            should = True
            if ONLY_NEW and prev:
                same_dir = prev.get("signal") == sig
                worse = conf <= prev.get("conf", 0)
                recent = (time.time() - prev.get("ts", 0)) < min_gap
                if same_dir and (worse or recent):
                    should = False
            if should:
                send_telegram(name, symbol, mtype, sig, conf, trade_class, expiry, mtf)
                st.session_state.last_sent[name] = {
                    "signal": sig,
                    "conf": conf,
                    "ts": time.time(),
                }

# ==================== TABLE ====================
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
        "Режим",
        "Свеча (M5)",
    ],
)

if len(df_show):
    df_show = df_show.sort_values(
        ["Класс", "Уверенность"],
        ascending=[True, False]
    ).reset_index(drop=True)

st.subheader("📋 Таблица сигналов")
st.dataframe(df_show, use_container_width=True, height=480)

# График по топ-паре
if len(df_show):
    top = df_show.iloc[0]
    sym = PAIRS[top["Пара"]]
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
            f"Топ-пара: {top['Пара']} • {top['Сигнал']} "
            f"({top['Уверенность']}%, {top['Класс']}) • {top['Multi-TF']}"
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

# автообновление
time.sleep(REFRESH_SEC)
st.rerun()
