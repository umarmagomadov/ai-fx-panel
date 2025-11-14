import time, random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# --- НАСТРОЙКИ ---
REFRESH_SEC = 5  # автообновление каждые 5 секунд

# --- СПИСОК ИНСТРУМЕНТОВ ---
PAIRS = {
    # Forex
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CHF": "USDCHF=X",
    "USD/CAD": "USDCAD=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "NZD/JPY": "NZDJPY=X",
    # Crypto (как OTC / 24/7)
    "BTC/USD (crypto)": "BTC-USD",
    "ETH/USD (crypto)": "ETH-USD",
    "LTC/USD (crypto)": "LTC-USD",
}

# Мульти-таймфреймы
TF_CONFIG = {
    "M5":  ("5m",  "5d"),
    "M15": ("15m", "10d"),
    "M30": ("30m", "30d"),
}

# ---------- ИНДИКАТОРЫ ----------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (dn + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, signal)
    return m, s, m - s

def adx(df: pd.DataFrame, n=14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    up_move   = h.diff()
    dn_move   = -l.diff()
    plus_dm   = up_move.where((up_move > 0) & (up_move > dn_move), 0.0).fillna(0)
    minus_dm  = dn_move.where((dn_move > 0) & (dn_move > up_move), 0.0).fillna(0)
    tr = pd.concat(
        [(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()],
        axis=1
    ).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di  = 100 * (plus_dm.rolling(n).sum() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(n).sum() / (atr + 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9))
    return dx.rolling(n).mean()

# ---------- ВСПОМОГАТЕЛЬНЫЕ ----------

def is_otc(name: str, symbol: str) -> bool:
    n = name.lower()
    if "crypto" in n:
        return True
    if "-usd" in symbol.lower():
        return True
    return False

def pocket_code(name: str, symbol: str) -> str:
    # EURUSD=X -> EUR/USD ; BTC-USD -> BTC/USD
    if symbol.endswith("=X") and len(symbol) >= 7:
        base = symbol.replace("=X", "")
        if len(base) == 6:
            return f"{base[:3]}/{base[3:]}"
    if symbol.endswith("-USD"):
        return symbol.replace("-USD", "/USD")
    # запасной вариант
    return "".join(ch for ch in name if ch.isalnum() or ch in "/").upper()

def safe_download(symbol: str, period: str, interval: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        if df is None or df.empty or len(df) < 50:
            return None
        df = df.tail(600)
        df = df.rename(columns=str.capitalize)
        return df
    except Exception:
        return None

def score_tf(df: pd.DataFrame) -> dict | None:
    """Сигнал по одному таймфрейму."""
    if df is None or len(df) < 60:
        return None

    close = df["Close"]

    rsv = float(rsi(close).iloc[-1])
    ema_fast = float(ema(close, 12).iloc[-1])
    ema_slow = float(ema(close, 26).iloc[-1])
    ema_trend = float(ema(close, 50).iloc[-1])
    macd_line, sig, hist = macd(close)
    macdv = float(hist.iloc[-1])
    adx_v = float(adx(df).iloc[-1])
    c = float(close.iloc[-1])

    # направление тренда
    trend = "flat"
    if c > ema_trend * 1.0005:
        trend = "up"
    elif c < ema_trend * 0.9995:
        trend = "down"

    # базовый сигнал
    side = "FLAT"
    if trend == "up" and macdv > 0 and rsv >= 55:
        side = "BUY"
    elif trend == "down" and macdv < 0 and rsv <= 45:
        side = "SELL"

    regime = "trend" if adx_v >= 20 else "flat"

    conf = 50
    if side in ("BUY", "SELL"):
        conf += 10
    if regime == "trend":
        conf += 10
    if abs(rsv - 50) > 15:
        conf += 10

    conf = int(max(0, min(99, conf)))

    return {
        "side": side,
        "trend": trend,
        "regime": regime,
        "rsi": rsv,
        "adx": adx_v,
        "macd": macdv,
        "price": c,
        "conf": conf,
    }

def analyze_pair(symbol: str) -> dict | None:
    """Мульти-TF анализ пары."""
    tf_res = {}
    for tf_name, (interval, period) in TF_CONFIG.items():
        df = safe_download(symbol, period, interval)
        res = score_tf(df)
        if res is not None:
            tf_res[tf_name] = res

    if not tf_res:
        return None

    buy_votes  = sum(1 for r in tf_res.values() if r["side"] == "BUY")
    sell_votes = sum(1 for r in tf_res.values() if r["side"] == "SELL")

    if buy_votes > sell_votes and buy_votes >= 2:
        final_side = "BUY"
    elif sell_votes > buy_votes and sell_votes >= 2:
        final_side = "SELL"
    else:
        final_side = "FLAT"

    avg_conf = int(np.mean([r["conf"] for r in tf_res.values()]))

    # выбор экспирации
    if final_side == "FLAT":
        expiry = 0
    elif avg_conf < 60:
        expiry = 2
    elif avg_conf < 75:
        expiry = 5
    else:
        expiry = 10

    mtf_line = " | ".join(f"{tf}={r['side']}" for tf, r in tf_res.items())
    main_tf = "M5" if "M5" in tf_res else list(tf_res.keys())[0]
    main = tf_res[main_tf]

    return {
        "side": final_side,
        "avg_conf": avg_conf,
        "expiry": expiry,
        "mtf_line": mtf_line,
        "main_rsi": main["rsi"],
        "main_adx": main["adx"],
        "main_macd": main["macd"],
        "main_price": main["price"],
    }

# ---------- STREAMLIT UI ----------

st.set_page_config(page_title="AI FX Panel Simple", layout="wide")
st.title("🤖 AI FX Panel — сигналы (без Telegram)")

c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(
        "Панель показывает сигналы **BUY/SELL** по мульти-таймфрейму "
        "(M5 / M15 / M30) для валют и крипты.\n\n"
        "Каждые несколько секунд всё обновляется автоматически."
    )
with c2:
    threshold = st.slider(
        "Минимальная уверенность для показа сигнала",
        50, 95, 70, 1
    )

rows = []

for name, symbol in PAIRS.items():
    res = analyze_pair(symbol)
    if not res:
        continue
    if res["side"] == "FLAT":
        continue
    if res["avg_conf"] < threshold:
        continue

    mtype = "OTC / 24/7" if is_otc(name, symbol) else "Биржевая"
    rows.append([
        name,
        pocket_code(name, symbol),
        mtype,
        res["side"],
        res["avg_conf"],
        res["expiry"],
        res["mtf_line"],
        f"RSI {res['main_rsi']:.1f} | ADX {res['main_adx']:.1f} | MACD {res['main_macd']:.5f}",
    ])

if rows:
    df_show = pd.DataFrame(
        rows,
        columns=[
            "Пара",
            "Код для Pocket",
            "Тип",
            "Сигнал",
            "Уверенность",
            "Экспирация (мин)",
            "Multi-TF",
            "Индикаторы",
        ],
    ).sort_values("Уверенность", ascending=False).reset_index(drop=True)

    st.subheader("📋 Текущие сильные сигналы")
    st.dataframe(df_show, use_container_width=True, height=450)

    # График по топ-сигналу
    top = df_show.iloc[0]
    sym = PAIRS[top["Пара"]]
    dfc = safe_download(sym, TF_CONFIG["M5"][1], TF_CONFIG["M5"][0])
    if dfc is not None:
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
            margin=dict(l=0, r=0, t=30, b=0),
            title=(
                f"Топ: {top['Пара']} — {top['Сигнал']} "
                f"({top['Уверенность']}%)"
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Пока нет сильных сигналов по заданному порогу уверенности.")

st.caption("⏱ Автообновление каждые 5 секунд.")
time.sleep(REFRESH_SEC)
st.experimental_rerun()
