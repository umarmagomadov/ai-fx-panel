import time, json, random, os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
import plotly.graph_objects as go

# ================== SECRETS ==================
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", os.getenv("TELEGRAM_TOKEN", ""))
CHAT_ID        = st.secrets.get("CHAT_ID",        os.getenv("CHAT_ID", ""))

# ================== SETTINGS =================
REFRESH_SEC         = 1     # автообновление, сек
ONLY_NEW            = True  # не спамим одно и то же
MIN_SEND_GAP_S      = 60    # пауза между сигналами по одной паре
BASE_CONF_THRESHOLD = 70    # базовый минимальный порог для отправки

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
TF_M15 = ("15m", "10d")
TF_M30 = ("30m", "30d")

# ================== ИНСТРУМЕНТЫ ==============
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
    "EURCAD": "EURCAD=X",
    "GBPCAD": "GBPCAD=X",
    "GBPAUD": "GBPAUD=X",
    "AUDCAD": "AUDCAD=X",
    "NZDJPY": "NZDJPY=X",

    # Commodities / фьючерсы (OTC/24)
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

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    up_move  = h.diff()
    dn_move  = -l.diff()
    plus_dm  = up_move.where((up_move > 0) & (up_move > dn_move), 0.0).fillna(0)
    minus_dm = dn_move.where((dn_move > 0) & (dn_move > up_move), 0.0).fillna(0)
    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di  = 100 * (plus_dm.rolling(n).sum()  / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(n).sum() / (atr + 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9))
    return dx.rolling(n).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()

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

# ================== DATA =====================
def download_data(symbol: str, interval: str, period: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            auto_adjust=True,
            progress=False,
        )
        if df is None or len(df) < 30:
            return None
        df = df[["Open", "High", "Low", "Close"]].copy()
        return df
    except Exception:
        return None

def get_df_cached(key: str, symbol: str, interval: str, period: str) -> pd.DataFrame | None:
    if "cache" not in st.session_state:
        st.session_state.cache = {}
    cache = st.session_state.cache

    df = cache.get(key)
    if df is not None and len(df) >= 30:
        return df

    df = download_data(symbol, interval, period)
    if df is not None:
        cache[key] = df
    return df

# ================== SCORING ==================
def score_single_tf(df: pd.DataFrame) -> tuple[str, float, dict]:
    """
    Анализ одного таймфрейма.
    Возвращает (direction, quality, feats)
    direction: BUY / SELL / FLAT
    quality: 0-1
    """
    if df is None or len(df) < 50:
        return "FLAT", 0.0, {}

    close = df["Close"]
    high = df["High"]
    low  = df["Low"]

    rsi_series = rsi(close)
    rsi_val  = safe_float(rsi_series.iloc[-1], 50.0)
    rsi_prev = safe_float(rsi_series.iloc[-2], 50.0) if len(rsi_series) > 2 else rsi_val

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)

    ema20_val  = safe_float(ema20.iloc[-1], close.iloc[-1])
    ema50_val  = safe_float(ema50.iloc[-1], close.iloc[-1])
    ema200_val = safe_float(ema200.iloc[-1], close.iloc[-1])

    macd_line, macd_sig, macd_hist = macd(close)
    macd_val = safe_float(macd_hist.iloc[-1], 0.0)

    up, mid, lo, w = bbands(close)
    bb_width = safe_float(w.iloc[-1], 0.0)
    bb_pos   = safe_float(
        (close.iloc[-1] - mid.iloc[-1]) /
        (up.iloc[-1] - lo.iloc[-1] + 1e-9),
        0.0,
    )

    adx_series = adx(df)
    adx_val = safe_float(adx_series.iloc[-1], 0.0)

    atr_series = atr(df)
    atr_val = safe_float(atr_series.iloc[-1], 0.0)
    atr_norm = atr_val / max(1e-9, safe_float(close.iloc[-1], 1.0)) * 10000

    vu = 0
    vd = 0

    if rsi_val < 35:
        vu += 1
    if rsi_val > 65:
        vd += 1
    if ema20_val > ema50_val:
        vu += 1
    if ema20_val < ema50_val:
        vd += 1
    if macd_val > 0:
        vu += 1
    if macd_val < 0:
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

    votes_delta = abs(vu - vd) / 4.0
    trend_boost = min(max((adx_val - 18) / 20, 0), 1)
    vola_boost  = min(max((atr_norm - 3) / 10, 0), 1)

    quality = 0.4 * votes_delta + 0.35 * trend_boost + 0.25 * vola_boost
    quality = max(0.0, min(1.0, quality))

    feats = {
        "RSI": round(rsi_val, 1),
        "RSI_prev": round(rsi_prev, 1),
        "ADX": round(adx_val, 1),
        "MACD_Hist": round(macd_val, 6),
        "BB_Width": round(bb_width, 2),
        "BB_Pos": round(bb_pos, 3),
        "ATR_norm": round(atr_norm, 2),
        "EMA20-EMA50": round(ema20_val - ema50_val, 6),
        "EMA200": round(ema200_val, 6),
    }
    return direction, float(quality), feats

def fuse_multi_tf(df_m1, df_m5, df_m15, df_m30):
    sig_m1, q1, f1 = score_single_tf(df_m1)
    sig_m5, q5, f5 = score_single_tf(df_m5)
    sig_m15, q15, f15 = score_single_tf(df_m15)
    sig_m30, q30, f30 = score_single_tf(df_m30)

    main_sig = sig_m5
    if main_sig == "FLAT":
        main_sig = sig_m15

    conf = 0.0
    weights = 0.0
    for sig, q, w in [
        (sig_m1, q1, 0.8),
        (sig_m5, q5, 1.2),
        (sig_m15, q15, 1.0),
        (sig_m30, q30, 0.8),
    ]:
        if sig == main_sig and sig in ("BUY", "SELL"):
            conf += q * w
        elif sig != "FLAT" and main_sig in ("BUY", "SELL") and sig != main_sig:
            conf -= q * w * 0.6
        weights += w

    if weights > 0:
        conf = conf / weights
    conf = max(0.0, min(1.0, conf))

    if main_sig not in ("BUY", "SELL"):
        main_sig = "FLAT"

    adx_val = f5.get("ADX", 0.0)
    if adx_val < 15:
        conf *= 0.6
    elif adx_val > 30:
        conf *= 1.1

    bw = f5.get("BB_Width", 0.0)
    if bw < 2:
        conf *= 0.7
    elif bw > 7:
        conf *= 1.05

    rsi_val = f5.get("RSI", 50.0)
    if main_sig == "BUY" and rsi_val > 75:
        conf *= 0.8
    if main_sig == "SELL" and rsi_val < 25:
        conf *= 0.8

    conf_pct = int(round(conf * 100))
    if conf_pct >= 90:
        klass = "A"
    elif conf_pct >= 80:
        klass = "B"
    else:
        klass = "C"

    mtf = {
        "M1": sig_m1,
        "M5": sig_m5,
        "M15": sig_m15,
        "M30": sig_m30,
    }

    feats = f5.copy()
    feats["RSI_M1"] = f1.get("RSI", 0)
    feats["RSI_M15"] = f15.get("RSI", 0)
    feats["RSI_M30"] = f30.get("RSI", 0)

    return main_sig, conf_pct, klass, mtf, feats, adx_val, bw, rsi_val

def choose_expiry(signal: str, conf: int, adx_val: float, bw: float) -> int:
    if signal not in ("BUY", "SELL"):
        return 0
    if conf < 60:
        return 0

    if conf < 70:
        base = 1
    elif conf < 80:
        base = 2
    elif conf < 90:
        base = 3
    elif conf < 95:
        base = 5
    else:
        base = 8

    if adx_val > 35:
        base += 2
    elif adx_val < 18:
        base -= 1

    if bw < 2:
        base = max(1, base - 1)
    elif bw > 7:
        base += 1

    return int(max(1, min(15, base)))

# ================== TELEGRAM =================
def send_telegram_signal(
    pair_name: str,
    pair_code: str,
    market_type: str,
    signal: str,
    conf: int,
    klass: str,
    expiry: int,
    mtf: dict,
    feats: dict,
) -> None:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    if signal not in ("BUY", "SELL"):
        return

    arrow = "⬆️" if signal == "BUY" else "⬇️"
    copy_code = pocket_code(pair_name, pair_code)

    if conf >= 90:
        strength = "🟢 сильный"
    elif conf >= 80:
        strength = "🟡 нормальный"
    else:
        strength = "🔴 слабый"

    text = (
        "🤖 AI FX Signal Bot v3.2\n"
        f"Пара: {pair_name}\n"
        f"Код для Pocket: {copy_code}\n"
        f"Тип: {market_type}\n"
        f"Сигнал: {arrow} {signal}\n"
        f"M1={mtf.get('M1')} | M5={mtf.get('M5')} | M15={mtf.get('M15')} | M30={mtf.get('M30')}\n"
        f"Класс: {klass}\n"
        f"Уверенность: {conf}% ({strength})\n"
        f"Экспирация: {expiry} мин\n"
        f"RSI: {feats.get('RSI')} | ADX: {feats.get('ADX')} | MACD: {feats.get('MACD_Hist')}\n"
        f"Время: {datetime.utcnow().strftime('%H:%M:%S')} UTC"
    )

    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text},
            timeout=10,
        )
    except Exception as e:
        st.toast(f"Ошибка Telegram: {e}", icon="⚠️")

# ================== STREAMLIT UI =============
st.set_page_config(
    page_title="AI FX Bot v3.2 — M1+M5+M15+M30 + Telegram",
    layout="wide",
)

st.title("🤖 AI FX Bot v3.2 — M1+M5+M15+M30 + Telegram")
st.markdown(
    "Режимы **Safe/Normal/Hard/Ultra** — это стиль фильтра, "
    "а **не** реальная гарантия. Бот — инструмент для обучения, не финсовет."
)

col_mode, col_min_conf, col_gap = st.columns([1, 2, 1])

with col_mode:
    mode = st.selectbox(
        "Режим отбора сигналов",
        list(MODES.keys()),
        index=0,
    )

with col_min_conf:
    user_min_conf = st.slider(
        "Минимальная уверенность (%) для сигнала",
        50,
        99,
        85,
    )

with col_gap:
    min_gap = st.number_input(
        "Пауза между сигналами по паре (сек)",
        min_value=10,
        max_value=600,
        value=60,
        step=5,
    )

mode_base_threshold = MODES.get(mode, 85)
effective_threshold = max(user_min_conf, mode_base_threshold)

st.markdown(
    f"**Текущий рабочий порог для отправки сигналов: {effective_threshold}%**"
)

if "last_sent" not in st.session_state:
    st.session_state.last_sent = {}

rows = []

for pair_name, symbol in PAIRS.items():
    df_m1  = get_df_cached(f"{symbol}_1m",  symbol, TF_M1[0],  TF_M1[1])
    df_m5  = get_df_cached(f"{symbol}_5m",  symbol, TF_M5[0],  TF_M5[1])
    df_m15 = get_df_cached(f"{symbol}_15m", symbol, TF_M15[0], TF_M15[1])
    df_m30 = get_df_cached(f"{symbol}_30m", symbol, TF_M30[0], TF_M30[1])

    main_sig, conf, klass, mtf, feats, adx_val, bw, rsi_val = fuse_multi_tf(
        df_m1, df_m5, df_m15, df_m30
    )

    market_type = "OTC/24/7" if is_otc(pair_name, symbol) else "Биржевая"
    expiry = choose_expiry(main_sig, conf, adx_val, bw)

    rows.append(
        [
            pair_name,
            market_type,
            main_sig,
            conf,
            klass,
            expiry,
            f"M1={mtf['M1']} | M5={mtf['M5']} | M15={mtf['M15']} | M30={mtf['M30']}",
            json.dumps(feats, ensure_ascii=False),
        ]
    )

    if (
        main_sig in ("BUY", "SELL")
        and conf >= effective_threshold
        and expiry > 0
    ):
        prev = st.session_state.last_sent.get(pair_name)
        now_ts = time.time()
        should_send = True
        if prev is not None:
            same_sig = prev.get("signal") == main_sig
            worse = conf <= prev.get("conf", 0)
            recent = (now_ts - prev.get("ts", 0)) < min_gap
            if ONLY_NEW and same_sig and (worse or recent):
                should_send = False

        if should_send:
            send_telegram_signal(
                pair_name,
                symbol,
                market_type,
                main_sig,
                conf,
                klass,
                expiry,
                mtf,
                feats,
            )
            st.session_state.last_sent[pair_name] = {
                "signal": main_sig,
                "conf": conf,
                "ts": now_ts,
            }

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
        "Индикаторы",
    ],
)

if len(df_show):
    df_show = df_show.sort_values("Уверенность", ascending=False).reset_index(drop=True)

st.subheader("📋 Таблица сигналов")
st.dataframe(df_show, use_container_width=True, height=480)

if len(df_show):
    top = df_show.iloc[0]
    top_pair = top["Пара"]
    sym = PAIRS[top_pair]
    st.markdown(f"**Топ-пара сейчас: {top_pair} ({top['Сигнал']} {top['Уверенность']}%)**")
    st.text_input(
        "Код для Pocket Option (копируй):",
        value=pocket_code(top_pair, sym),
        key="pocket_copy",
    )

    df_chart = get_df_cached(f"{sym}_chart", sym, TF_M5[0], TF_M5[1])
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
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            title=f"{top_pair} — M5 график",
        )
        st.plotly_chart(fig, use_container_width=True)

time.sleep(REFRESH_SEC)
st.experimental_rerun()
