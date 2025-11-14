# AI FX Signal Bot v102.2 — Triple-TF Safe Engine (M5+M15+M30)
# Автор: для Umar 🙂  | Язык: ru

import os
import time
import json
import random
from datetime import datetime, timezone
import streamlit as st

st.title("Тест Streamlit работает ✔️")

st.write("Если ты видишь этот текст — приложение загружается нормально.")
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
import plotly.graph_objects as go

# ================== КОНСТАНТЫ ==================

VERSION = "v102.2"
REFRESH_SECONDS = 1           # авто-обновление
DEFAULT_THRESHOLD = 70        # дефолт уверенности для Telegram
DEFAULT_MIN_GAP = 60          # сек, пауза между сигналами
ONLY_NEW = True               # не спамить одинаковым сигналом

# читаем токен/чат
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", os.getenv("TELEGRAM_TOKEN", ""))
CHAT_ID        = st.secrets.get("CHAT_ID",        os.getenv("CHAT_ID", ""))

# таймфреймы: (interval, period)
TF_CONFIG = {
    "M5":  ("5m",  "2d"),
    "M15": ("15m", "5d"),
    "M30": ("30m", "10d"),
}
MAIN_TF = "M5"

# ------- список инструментов (название → тикер Yahoo) -------
PAIRS = {
    # Forex majors & crosses
    "EURUSD":       "EURUSD=X",
    "GBPUSD":       "GBPUSD=X",
    "USDJPY":       "USDJPY=X",
    "USDCHF":       "USDCHF=X",
    "USDCAD":       "USDCAD=X",
    "AUDUSD":       "AUDUSD=X",
    "NZDUSD":       "NZDUSD=X",
    "EURJPY":       "EURJPY=X",
    "GBPJPY":       "GBPJPY=X",
    "AUDJPY":       "AUDJPY=X",
    "CADJPY":       "CADJPY=X",
    "CHFJPY":       "CHFJPY=X",
    "EURAUD":       "EURAUD=X",
    "EURCAD":       "EURCAD=X",
    "EURGBP":       "EURGBP=X",
    "EURCHF":       "EURCHF=X",
    "GBPCAD":       "GBPCAD=X",
    "GBPAUD":       "GBPAUD=X",
    "AUDCAD":       "AUDCAD=X",
    "NZDJPY":       "NZDJPY=X",

    # Commodities (фьючи – считаем OTC/24/7)
    "XAUUSD (Gold)":   "GC=F",
    "XAGUSD (Silver)": "SI=F",
    "WTI (Oil)":       "CL=F",
    "BRENT (Oil)":     "BZ=F",

    # Crypto (тоже OTC/24/7)
    "BTCUSD (Bitcoin)":   "BTC-USD",
    "ETHUSD (Ethereum)":  "ETH-USD",
    "SOLUSD (Solana)":    "SOL-USD",
    "BNBUSD (BNB)":       "BNB-USD",
    "XRPUSD (XRP)":       "XRP-USD",
    "DOGEUSD (Dogecoin)":"DOGE-USD",
}

# ================== ИНДИКАТОРЫ ==================


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-diff.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (dn + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, signal)
    h = m - s
    return m, s, h


def calc_bbands(close: pd.Series, n=20, k=2.0):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    up = ma + k * sd
    lo = ma - k * sd
    width = (up - lo) / (ma + 1e-9) * 100
    return up, ma, lo, width


def calc_adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    # классический ADX, но максимально безопасный
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    up_move = h.diff()
    dn_move = -l.diff()

    plus_dm = up_move.where((up_move > 0) & (up_move > dn_move), 0.0).fillna(0.0)
    minus_dm = dn_move.where((dn_move > 0) & (dn_move > up_move), 0.0).fillna(0.0)

    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()

    plus_di = 100 * (plus_dm.rolling(n).sum() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(n).sum() / (atr + 1e-9))

    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
    adx = dx.rolling(n).mean()
    return adx


# ================== ДАННЫЕ ==================


def download_ohlc(symbol: str, interval: str, period: str) -> pd.DataFrame | None:
    """Безопасная загрузка и очистка OHLC."""
    try:
        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or len(df) == 0:
            return None

        df = df.copy()

        # иногда индексы кривые
        df = df.reset_index(drop=True)

        # приводим все цены к числам
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Open", "High", "Low", "Close"])

        if len(df) < 50:
            return None

        return df.tail(400)
    except Exception:
        return None


def get_ohlc(symbol: str, tf: str) -> pd.DataFrame | None:
    interval, period = TF_CONFIG[tf]
    return download_ohlc(symbol, interval, period)


# ================== ВСПОМОГАТЕЛЬНЫЕ ==================


def is_otc(name: str, symbol: str) -> bool:
    n = name.lower()
    s = symbol.lower()
    if "otc" in n:
        return True
    if "=f" in s:        # фьючерсы
        return True
    if "-" in s:        # крипта BTC-USD
        return True
    return False


def pocket_code(name: str, symbol: str) -> str:
    # EURUSD=X -> EUR/USD
    if symbol.endswith("=X") and len(symbol) >= 7:
        base = symbol.replace("=X", "")
        if len(base) == 6:
            return f"{base[:3]}/{base[3:]}"
    # BTC-USD -> BTC/USD
    if symbol.endswith("-USD"):
        return symbol.replace("-USD", "/USD")
    # фьючи → маппинг
    if symbol in {"GC=F", "SI=F", "CL=F", "BZ=F"}:
        mapping = {
            "GC=F": "XAU/USD",
            "SI=F": "XAG/USD",
            "CL=F": "WTI/USD",
            "BZ=F": "BRENT/USD",
        }
        return mapping[symbol]
    # запасной вариант: чистим имя
    return "".join(ch for ch in name if ch.isalnum() or ch in "/").upper()


def candle_phase(df: pd.DataFrame) -> str:
    """start / mid / end по положению Close в диапазоне свечи."""
    last = df.iloc[-1]
    o, h, l, c = float(last["Open"]), float(last["High"]), float(last["Low"]), float(last["Close"])
    rng = max(1e-9, h - l)
    pos = (c - l) / rng
    if pos < 0.33:
        return "start"
    if pos < 0.66:
        return "mid"
    return "end"


def near_sr(df: pd.DataFrame) -> str | None:
    close = float(df["Close"].iloc[-1])
    sup = float(df["Low"].rolling(20).min().iloc[-1])
    res = float(df["High"].rolling(20).max().iloc[-1])
    if abs(close - sup) / max(1e-9, close) < 0.002:
        return "support"
    if abs(close - res) / max(1e-9, close) < 0.002:
        return "resistance"
    return None


def momentum_spike(df: pd.DataFrame) -> bool:
    """Резкое движение относительно среднего."""
    if len(df) < 12:
        return False
    close = df["Close"]
    last_move = abs(close.iloc[-1] - close.iloc[-2])
    avg_move = close.diff().abs().rolling(10, min_periods=5).mean().iloc[-1]
    if pd.isna(avg_move) or avg_move == 0:
        return False
    return bool(last_move > 1.5 * avg_move)


def boll_width_val(close: pd.Series, n: int = 20, k: float = 2.0) -> float:
    up, ma, lo, _ = calc_bbands(close, n=n, k=k)
    if pd.isna(ma.iloc[-1]) or pd.isna(up.iloc[-1]) or pd.isna(lo.iloc[-1]):
        return 0.0
    return float((up.iloc[-1] - lo.iloc[-1]) / (ma.iloc[-1] + 1e-9) * 100)


def tf_direction(df: pd.DataFrame) -> str:
    close = df["Close"]
    rsi = calc_rsi(close)
    _, _, h = calc_macd(close)
    rsv = float(rsi.iloc[-1])
    mh = float(h.iloc[-1])
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


# ================== СКОРИНГ ОДНОГО ТФ ==================


def score_single(df: pd.DataFrame) -> tuple[str, int, dict]:
    """Сигнал по MAIN_TF (M5)."""
    if df is None or len(df) < 50:
        return "FLAT", 0, {}

    close = df["Close"]

    rsi_series = calc_rsi(close)
    rsv = float(rsi_series.iloc[-1])
    rsv_prev = float(rsi_series.iloc[-2]) if len(rsi_series) > 2 else rsv

    ema9 = float(ema(close, 9).iloc[-1])
    ema21 = float(ema(close, 21).iloc[-1])
    ema200 = float(ema(close, 200).iloc[-1]) if len(close) >= 200 else ema21

    _, _, mh = calc_macd(close)
    mhv = float(mh.iloc[-1])

    up, mid, lo, bw_series = calc_bbands(close)
    bb_width_now = float(bw_series.iloc[-1])
    if pd.isna(bb_width_now):
        bb_width_now = 0.0
    if pd.isna(up.iloc[-1]) or pd.isna(lo.iloc[-1]) or pd.isna(mid.iloc[-1]):
        bb_pos = 0.0
    else:
        bb_pos = float((close.iloc[-1] - mid.iloc[-1]) /
                       (up.iloc[-1] - lo.iloc[-1] + 1e-9))

    adx_series = calc_adx(df)
    adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0.0

    # голоса BUY / SELL
    vu = vd = 0

    if rsv < 30:
        vu += 1
    if rsv > 70:
        vd += 1

    if ema9 > ema21:
        vu += 1
    if ema9 < ema21:
        vd += 1

    if mhv > 0:
        vu += 1
    if mhv < 0:
        vd += 1

    if bb_pos < -0.3:
        vu += 1
    if bb_pos > 0.3:
        vd += 1

    if vu == vd:
        direction = "FLAT"
    elif vu > vd:
        direction = "BUY"
    else:
        direction = "SELL"

    # базовая уверенность
    raw = abs(vu - vd) / 4.0
    trend_boost = min(max((adx_val - 18) / 25.0, 0), 1)
    conf = int(100 * (0.55 * raw + 0.45 * trend_boost))
    conf = max(0, min(99, conf))

    feats = {
        "RSI": round(rsv, 1),
        "RSI_prev": round(rsv_prev, 1),
        "ADX": round(adx_val, 1),
        "MACD_Hist": round(mhv, 6),
        "BB_Pos": round(bb_pos, 3),
        "BB_Width": round(bb_width_now, 2),
        "EMA9_minus_EMA21": round(ema9 - ema21, 6),
        "EMA200": round(ema200, 6),
    }

    return direction, conf, feats


# ================== MULTI-TF СКОРОСТЬ ==================


def score_multi(symbol: str) -> tuple[str, int, dict, dict, pd.DataFrame | None]:
    """Возвращает сигнальный direction, conf, feats, mtf_info, df_main."""
    df_main = get_ohlc(symbol, "M5")
    df_mid = get_ohlc(symbol, "M15")
    df_trend = get_ohlc(symbol, "M30")

    if df_main is None or df_mid is None or df_trend is None:
        return "FLAT", 0, {}, {}, df_main

    sig, conf, feats = score_single(df_main)

    # направления по всем таймфреймам
    d_main = tf_direction(df_main)
    d_mid = tf_direction(df_mid)
    d_trend = tf_direction(df_trend)

    agree = 0
    if sig in ("BUY", "SELL") and d_mid == sig:
        agree += 1
    if sig in ("BUY", "SELL") and d_trend == sig:
        agree += 1

    # усиливаем/ослабляем по согласованию
    if sig in ("BUY", "SELL") and d_main == d_mid == d_trend == sig:
        conf += 20
    elif agree == 1:
        conf += 8
    else:
        conf -= 12

    # режим
    bw = boll_width_val(df_main["Close"])
    adx_val = float(feats.get("ADX", 0.0))
    regime = market_regime(adx_val, bw)

    # импульс
    if momentum_spike(df_main):
        conf += 8

    # уровни
    sr = near_sr(df_main)
    if sig == "BUY" and sr == "support":
        conf += 7
    if sig == "SELL" and sr == "resistance":
        conf += 7

    # фаза свечи
    phase = candle_phase(df_main)
    if phase == "mid":
        conf += 4
    elif phase == "end":
        conf -= 6

    # резкое изменение RSI = осторожность
    if abs(feats["RSI"] - feats["RSI_prev"]) > 12:
        conf -= 6

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


# ================== ЭКСПИРАЦИЯ ==================


def choose_expiry(conf: int, adx_val: float, rsi_val: float, phase: str, bw: float) -> int:
    """Возвращает экспирацию в минутах (1–30)."""
    if conf < 50:
        return 0

    # базовое значение по уверенности
    if conf < 60:
        base = 2
    elif conf < 70:
        base = 3
    elif conf < 80:
        base = 5
    elif conf < 90:
        base = 8
    elif conf < 95:
        base = 12
    else:
        base = 18

    # тренд
    if adx_val >= 40:
        base += 6
    elif adx_val >= 30:
        base += 3
    elif adx_val < 18:
        base -= 2

    # волатильность
    if bw >= 8:
        base -= 3
    elif bw <= 2:
        base += 2

    # фаза свечи
    if phase == "end":
        base -= 2
    elif phase == "start":
        base += 1

    # экстремальные RSI → слегка меньше
    if rsi_val >= 75 or rsi_val <= 25:
        base -= 1

    return int(max(1, min(30, base)))


# ================== TELEGRAM ==================


def send_telegram(pair_name: str, symbol: str, mtype: str,
                  signal: str, conf: int, expiry: int,
                  feats: dict, mtf: dict):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return

    arrow = "⬆️" if signal == "BUY" else "⬇️"
    copy_code = pocket_code(pair_name, symbol)
    phase = mtf.get("Phase", "mid")
    phase_icon = "🟢 Начало" if phase == "start" else ("🟡 Середина" if phase == "mid" else "🔴 Конец")
    if conf < 60:
        strength = "🔴 слабый"
    elif conf < 80:
        strength = "🟡 средний"
    else:
        strength = "🟢 сильный"

    text = (
        f"🤖 AI FX СИГНАЛ {VERSION}\n"
        f"💱 Пара: {pair_name}\n"
        f"📌 Код (Pocket): `{copy_code}`\n"
        f"🏷️ Тип: {mtype}\n"
        f"{arrow} Сигнал: *{signal}*\n"
        f"📊 Multi-TF: M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}\n"
        f"🌐 Режим: {mtf['Regime']} | 🕯️ Свеча: {phase_icon}\n"
        f"💪 Уверенность: *{conf}%* ({strength})\n"
        f"⏱ Экспирация: *{expiry} мин*\n"
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
        st.toast(f"Ошибка Telegram: {e}", icon="⚠️")


# ================== UI ==================


st.set_page_config(page_title=f"AI FX {VERSION}", layout="wide")
st.title(f"🤖 AI FX Signal Bot {VERSION} — M5+M15+M30 + Pocket Copy")

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    threshold = st.slider("Порог уверенности для Telegram (%)", 50, 95, DEFAULT_THRESHOLD, 1)
with c2:
    min_gap = st.number_input("Мин. пауза между сигналами (сек)", 10, 300, DEFAULT_MIN_GAP, 5)
with c3:
    st.write("Авто-обновление:", f"каждые {REFRESH_SECONDS} сек")

if "last_sent" not in st.session_state:
    st.session_state.last_sent = {}

rows = []

# ================== СКАНИРОВАНИЕ ВСЕХ ПАР ==================

for pair_name, symbol in PAIRS.items():
    sig, conf, feats, mtf, df_main = score_multi(symbol)

    if df_main is None or not feats:
        # данных нет
        rows.append([pair_name, "нет данных", "FLAT", 0, 0, "-", "-", "-"])
        continue

    # тип инструмента
    otc_flag = is_otc(pair_name, symbol)
    mtype = "OTC/24/7" if otc_flag else "Биржевая"

    # эффективный порог для OTC (делаем строже)
    eff_threshold = threshold + 5 if otc_flag else threshold

    # рассчёт экспирации
    adx_val = float(feats["ADX"])
    rsi_val = float(feats["RSI"])
    bw_val = float(mtf["BW"])
    phase_val = mtf["Phase"]
    expiry = choose_expiry(conf, adx_val, rsi_val, phase_val, bw_val)

    # отображение фазы
    phase_show = "🟢 Начало" if phase_val == "start" else ("🟡 Середина" if phase_val == "mid" else "🔴 Конец")

    rows.append([
        pair_name,
        mtype,
        sig,
        conf,
        expiry,
        f"M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}",
        phase_show,
        json.dumps(feats, ensure_ascii=False),
    ])

    # ------- отправка в Telegram -------
    if sig in ("BUY", "SELL") and conf >= eff_threshold and expiry > 0:
        prev = st.session_state.last_sent.get(pair_name, {})
        should_send = True

        if ONLY_NEW and prev:
            same_dir = prev.get("signal") == sig
            worse = conf <= prev.get("conf", 0)
            recent = (time.time() - prev.get("ts", 0)) < min_gap
            if same_dir and (worse or recent):
                should_send = False

        if should_send:
            send_telegram(pair_name, symbol, mtype, sig, conf, expiry, feats, mtf)
            st.session_state.last_sent[pair_name] = {
                "signal": sig,
                "conf": conf,
                "ts": time.time(),
            }

# ================== ТАБЛИЦА ==================

df_show = pd.DataFrame(rows, columns=[
    "Пара", "Тип", "Сигнал", "Уверенность", "Экспирация (мин)",
    "Multi-TF", "Свеча", "Индикаторы"
])

if len(df_show):
    df_show = df_show.sort_values("Уверенность", ascending=False).reset_index(drop=True)

st.subheader("📋 Рейтинг сигналов")
st.dataframe(df_show, use_container_width=True, height=480)

# ===== КОД ДЛЯ Pocket (топ-пара) =====
if len(df_show):
    top_row = df_show.iloc[0]
    top_pair = top_row["Пара"]
    top_symbol = PAIRS[top_pair]
    st.markdown("**Код для Pocket Option (топ-пара):**")
    st.text_input("Нажми и удерживай для копирования", value=pocket_code(top_pair, top_symbol), key="copy_top")

    # небольшой график по топ-паре
    df_chart = get_ohlc(top_symbol, MAIN_TF)
    if df_chart is not None and len(df_chart):
        fig = go.Figure(data=[go.Candlestick(
            x=df_chart.index,
            open=df_chart["Open"],
            high=df_chart["High"],
            low=df_chart["Low"],
            close=df_chart["Close"],
        )])
        fig.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=20, b=0),
            title=f"{top_pair} — {top_row['Сигнал']} ({top_row['Уверенность']}%) • {top_row['Multi-TF']} • {top_row['Свеча']}",
        )
        st.plotly_chart(fig, use_container_width=True)

# ========== АВТО-ОБНОВЛЕНИЕ ==========
time.sleep(REFRESH_SECONDS)
st.experimental_rerun()
