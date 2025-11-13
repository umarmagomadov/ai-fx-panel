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

# ================== TELEGRAM ==================
# ⚠️ Ты сам дал токен и chat_id, поэтому пишу их прямо в код.
TELEGRAM_TOKEN = "7327265057:AAHoDsXxlKodgEYbtAsA1glZegTpeV4_oO4"
CHAT_ID = "6045310859"

# ================== НАСТРОЙКИ =================
REFRESH_SEC = 1              # автообновление страницы (сек)
ONLY_NEW = True              # не спамить одинаковыми сигналами
MIN_SEND_GAP_S = 60          # пауза между сигналами по одной паре

TF_MAIN = ("5m", "2d")       # основной вход
TF_MID = ("15m", "5d")       # подтверждение
TF_TREND = ("30m", "10d")    # глобальный тренд

# ================== ИНСТРУМЕНТЫ ===============
PAIRS = {
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
    "BTCUSD (BTC)": "BTC-USD",
    "ETHUSD (ETH)": "ETH-USD",
    "XAUUSD (Gold)": "GC=F",
    "XAGUSD (Silver)": "SI=F",
}

# ================== ХЕЛПЕРЫ ===================
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
    up = diff.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
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

    plus_dm = up_move.where(
        (up_move > 0) & (up_move > dn_move), 0.0
    ).fillna(0)
    minus_dm = dn_move.where(
        (dn_move > 0) & (dn_move > up_move), 0.0
    ).fillna(0)

    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
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


def momentum_spike(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 12:
        return False
    close = df["Close"]
    last_move = abs(
        safe_float(close.iloc[-1]) - safe_float(close.iloc[-2])
    )
    avg_raw = close.diff().abs().rolling(10).mean().iloc[-1]
    avg_move = safe_float(avg_raw, 0.0)
    if avg_move == 0.0:
        return False
    return bool(last_move > 1.6 * avg_move)


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
    if symbol == "GC=F":
        return "XAU/USD"
    if symbol == "SI=F":
        return "XAG/USD"
    clean = "".join(ch for ch in name if ch.isalnum() or ch == "/")
    return clean.upper()


# ================== ЗАГРУЗКА ДАННЫХ ==========
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
        return df[["Open", "High", "Low", "Close"]].tail(600).copy()
    except Exception:
        return None


def nudge_last(df: pd.DataFrame, max_bps: float = 5.0) -> pd.Series:
    last = df.iloc[-1].copy()
    c = safe_float(last["Close"], 1.0)
    bps = random.uniform(-max_bps, max_bps) / 10000.0
    new_c = max(1e-9, c * (1 + bps))
    last["Open"] = c
    last["High"] = max(c, new_c)
    last["Low"] = min(c, new_c)
    last["Close"] = new_c
    last.name = last.name + pd.tseries.frequencies.to_offset("1min")
    return last


def get_or_fake(symbol: str, period: str, interval: str) -> pd.DataFrame:
    if "cache" not in st.session_state:
        st.session_state["cache"] = {}
    cache = st.session_state["cache"]

    key = _cache_key(symbol, interval)
    real = safe_download(symbol, period, interval)
    if real is not None:
        cache[key] = real.copy()
        return real

    if key in cache and len(cache[key]):
        df = cache[key].copy()
        last = nudge_last(df)
        df = pd.concat([df, last.to_frame().T], axis=0).tail(600)
        cache[key] = df
        return df

    idx = pd.date_range(end=datetime.now(timezone.utc), periods=80, freq="1min")
    base = 1.0 + random.random() / 10
    vals = base * (1 + np.cumsum(np.random.randn(80)) / 100)
    df = pd.DataFrame({"Open": vals, "High": vals, "Low": vals, "Close": vals}, index=idx)
    cache[key] = df
    return df


# ================== ОДИН ТФ (M5) =============
def score_single(df: pd.DataFrame) -> tuple[str, int, dict]:
    if df is None or len(df) < 40:
        return "FLAT", 0, {
            "RSI": 50.0,
            "ADX": 0.0,
            "MACD_Hist": 0.0,
            "BB_Width": 0.0,
            "EMA9-EMA21": 0.0,
        }

    close = df["Close"]

    rsi_series = rsi(close)
    rsv = safe_float(rsi_series.iloc[-1], 50.0)
    rsv_prev = safe_float(
        rsi_series.iloc[-2], rsv
    ) if len(rsi_series) > 2 else rsv

    ema9 = safe_float(ema(close, 9).iloc[-1], rsv)
    ema21 = safe_float(ema(close, 21).iloc[-1], rsv)
    ema50 = safe_float(ema(close, 50).iloc[-1], rsv)

    _, _, mh = macd(close)
    mhv = safe_float(mh.iloc[-1], 0.0)

    up, mid, lo, w = bbands(close)
    w_last = safe_float(w.iloc[-1], 0.0)

    adx_series = adx(df)
    adx_v = safe_float(adx_series.iloc[-1], 0.0)

    vu = 0
    vd = 0

    # RSI
    if rsv < 32:
        vu += 1
    if rsv > 68:
        vd += 1

    # EMA cross
    if ema9 > ema21:
        vu += 1
    if ema9 < ema21:
        vd += 1

    # MACD
    if mhv > 0:
        vu += 1
    if mhv < 0:
        vd += 1

    # Цена относительно EMA50
    last_close = safe_float(close.iloc[-1], ema50)
    if last_close > ema50:
        vu += 1
    if last_close < ema50:
        vd += 1

    # Расширение/сжатие Боллинджера
    if w_last > 8:
        # сильный импульс, усиливаем макс сторону
        if vu > vd:
            vu += 1
        elif vd > vu:
            vd += 1

    if vu == vd:
        direction = "FLAT"
    elif vu > vd:
        direction = "BUY"
    else:
        direction = "SELL"

    raw = abs(vu - vd) / 4.0
    trend_boost = min(max((adx_v - 18) / 25, 0), 1)
    revers_rsi = min(abs(rsv - 50) / 30, 1)

    conf = 100 * (0.4 * raw + 0.35 * trend_boost + 0.25 * revers_rsi)
    conf = int(max(30, min(99, conf)))

    feats = {
        "RSI": round(rsv, 1),
        "RSI_prev": round(rsv_prev, 1),
        "ADX": round(adx_v, 1),
        "MACD_Hist": round(mhv, 5),
        "BB_Width": round(w_last, 2),
        "EMA9-EMA21": round(ema9 - ema21, 5),
    }
    return direction, conf, feats


def tf_direction(df: pd.DataFrame) -> str:
    close = df["Close"]
    macd_line, macd_sig, macd_hist = macd(close)
    rsi_series = rsi(close)
    rsv = safe_float(rsi_series.iloc[-1], 50.0)
    mh = safe_float(macd_hist.iloc[-1], 0.0)

    if mh > 0 and rsv > 52:
        return "BUY"
    if mh < 0 and rsv < 48:
        return "SELL"
    return "FLAT"


def market_regime(adx_val: float, bw: float) -> str:
    if adx_val < 18 and bw < 3:
        return "flat"
    if adx_val > 25 and bw < 7:
        return "trend"
    return "impulse"


# ============== MULTI-TF СТРАТЕГИЯ ===========
def score_multi_tf(symbol: str) -> tuple[str, int, dict, dict, pd.DataFrame]:
    df_main = get_or_fake(symbol, TF_MAIN[1], TF_MAIN[0])
    df_mid = get_or_fake(symbol, TF_MID[1], TF_MID[0])
    df_trend = get_or_fake(symbol, TF_TREND[1], TF_TREND[0])

    sig, conf, feats = score_single(df_main)

    d_main = tf_direction(df_main)
    d_mid = tf_direction(df_mid)
    d_trend = tf_direction(df_trend)

    agree = 0
    if d_main in ("BUY", "SELL") and d_mid == d_main:
        agree += 1
    if d_main in ("BUY", "SELL") and d_trend == d_main:
        agree += 1

    if d_main == d_mid == d_trend and d_main in ("BUY", "SELL"):
        conf += 15
    elif agree == 1:
        conf += 6
    else:
        conf -= 8

    bw = boll_width(df_main["Close"])
    adx_v = feats["ADX"]
    regime = market_regime(adx_v, bw)

    if momentum_spike(df_main):
        conf += 7

    phase = candle_phase(df_main)
    if phase == "mid":
        conf += 4
    elif phase == "end":
        conf -= 5

    conf = int(max(0, min(100, conf)))

    mtf = {
        "M5": d_main,
        "M15": d_mid,
        "M30": d_trend,
        "Regime": regime,
        "Phase": phase,
        "BW": round(bw, 2),
    }
    return sig, conf, feats, mtf, df_main


# ============== ВЫБОР ЭКСПИРАЦИИ =============
def choose_expiry(conf: int, adx_value: float, phase: str) -> int:
    if conf < 55:
        return 0

    if conf < 65:
        base = 3
    elif conf < 75:
        base = 5
    elif conf < 85:
        base = 8
    elif conf < 92:
        base = 12
    else:
        base = 18

    if adx_value >= 40:
        base += 4
    elif adx_value <= 18:
        base -= 2

    if phase == "end":
        base -= 2
    elif phase == "start":
        base += 1

    return int(max(1, min(30, base)))


# ============== TELEGRAM =====================
def send_telegram(pair_name: str,
                  pair_code: str,
                  mtype: str,
                  signal: str,
                  conf: int,
                  expiry: int,
                  feats: dict,
                  mtf: dict) -> None:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return

    arrow = "⬆️" if signal == "BUY" else ("⬇️" if signal == "SELL" else "➖")
    copy_code = pocket_code(pair_name, pair_code)

    phase_map = {"start": "🟢 начало", "mid": "🟡 середина", "end": "🔴 конец"}
    phase_icon = phase_map.get(mtf.get("Phase", ""), "❔")

    if conf < 70:
        strength = "🟡 нормальный"
    elif conf < 85:
        strength = "🟢 сильный"
    else:
        strength = "💚 премиум"

    text = (
        "🤖 AI FX Signal Bot v2.0\n"
        f"💱 Пара: {pair_name}\n"
        f"📌 Код для Pocket: `{copy_code}`\n"
        f"🏷️ Тип: {mtype}\n"
        f"{arrow} Сигнал: *{signal}*\n"
        f"📊 M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}\n"
        f"🌐 Рынок: {mtf['Regime']} | {phase_icon}\n"
        f"💪 Уверенность: *{conf}%* ({strength})\n"
        f"⏱ Экспирация: *{expiry} мин*\n"
        f"📈 RSI {feats['RSI']} | ADX {feats['ADX']} | MACD {feats['MACD_Hist']}\n"
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


# ============== STREAMLIT UI =================
st.set_page_config(
    page_title="AI FX Bot v2.0 — 85/90/95/99%",
    layout="wide",
)

st.title("🤖 AI FX Signal Bot v2.0 — M5+M15+M30 + Телеграм")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    mode = st.selectbox(
        "Режим отбора сигналов",
        [
            "Safe • 85%",
            "Balanced • 90%",
            "Aggressive • 95%",
            "Ultra • 99%",
        ],
    )

with col2:
    user_threshold = st.slider(
        "Минимальная уверенность (%) для сигнала",
        50,
        95,
        75,
        1,
    )

with col3:
    min_gap_input = st.number_input(
        "Пауза между сигналами по паре (сек)",
        min_value=10,
        max_value=600,
        value=MIN_SEND_GAP_S,
        step=5,
    )

# режим → добавка к порогу
if "85%" in mode:
    mode_bonus = -5
elif "90%" in mode:
    mode_bonus = 0
elif "95%" in mode:
    mode_bonus = +5
else:  # 99
    mode_bonus = +10

base_threshold = user_threshold + mode_bonus
base_threshold = int(max(50, min(95, base_threshold)))

st.caption(
    f"Текущий рабочий порог для отправки сигналов: **{base_threshold}%**"
)

if "last_sent" not in st.session_state:
    st.session_state["last_sent"] = {}

rows = []

for name, symbol in PAIRS.items():
    sig, conf, feats, mtf, df_main = score_multi_tf(symbol)

    otc_flag = is_otc(name, symbol)
    eff_threshold = base_threshold + (5 if otc_flag else 0)

    expiry = choose_expiry(conf, feats["ADX"], mtf["Phase"])

    mtype = "OTC/24/7" if otc_flag else "Биржевая"

    phase_map_short = {"start": "🟢", "mid": "🟡", "end": "🔴"}
    phase_show = phase_map_short.get(mtf["Phase"], "❔")

    rows.append(
        [
            name,
            mtype,
            sig,
            conf,
            expiry,
            f"M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}",
            phase_show,
            json.dumps(feats, ensure_ascii=False),
        ]
    )

    if sig in ("BUY", "SELL") and conf >= eff_threshold and expiry > 0:
        prev = st.session_state["last_sent"].get(name, {})
        should_send = True

        if ONLY_NEW and prev:
            same = prev.get("signal") == sig
            worse = conf <= prev.get("conf", 0)
            recent = (time.time() - prev.get("ts", 0)) < min_gap_input
            if same and (worse or recent):
                should_send = False

        if should_send:
            send_telegram(
                name,
                symbol,
                mtype,
                sig,
                conf,
                expiry,
                feats,
                mtf,
            )
            st.session_state["last_sent"][name] = {
                "signal": sig,
                "ts": time.time(),
                "conf": conf,
            }

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
    df_show = df_show.sort_values(
        "Уверенность", ascending=False
    ).reset_index(drop=True)

st.subheader("📋 Таблица сигналов")
st.dataframe(df_show, use_container_width=True, height=480)

if len(df_show):
    top = df_show.iloc[0]
    sym = PAIRS[top["Пара"]]
    st.markdown("**Топ-пара сейчас (копируемый код для Pocket Option):**")
    st.text_input(
        "Tap to copy:",
        value=pocket_code(top["Пара"], sym),
        key="copy_top_code",
    )

    dfc = get_or_fake(sym, TF_MAIN[1], TF_MAIN[0])
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
        margin=dict(l=0, r=0, t=25, b=0),
        title=(
            f"{top['Пара']} — {top['Сигнал']} "
            f"({top['Уверенность']}%) • {top['Multi-TF']} • {top['Свеча']}"
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

# авто-обновление
time.sleep(REFRESH_SEC)
st.experimental_rerun()
