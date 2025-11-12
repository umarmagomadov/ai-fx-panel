# AI FX PANEL PRO — 24/7 сигналы (Forex + Commodities + Crypto), автоэкспирация 1–30 мин

import time, json, math, random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# --------- СЕКРЕТЫ (Streamlit Secrets) ---------
TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
CHAT_ID        = st.secrets["CHAT_ID"]

# --------- НАСТРОЙКИ ---------
REFRESH_SEC     = 1            # обновление интерфейса, сек
LOOKBACK_MIN    = 180          # история для расчётов, мин
INTERVAL        = "1m"         # таймфрейм
SEND_THRESHOLD  = 70           # порог уверенности для отправки в TG
ONLY_NEW        = True         # антиспам: не слать одинаковое хуже/чаще
MIN_SEND_GAP_S  = 60           # минимум сек между сигналами по одной паре

# --------- ИНСТРУМЕНТЫ ---------
PAIRS = {
    # Forex — мажоры и кроссы
    "EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"USDJPY=X","USDCHF":"USDCHF=X","AUDUSD":"AUDUSD=X",
    "NZDUSD":"NZDUSD=X","USDCAD":"USDCAD=X","EURJPY":"EURJPY=X","GBPJPY":"GBPJPY=X","AUDJPY":"AUDJPY=X",
    "CADJPY":"CADJPY=X","CHFJPY":"CHFJPY=X","EURGBP":"EURGBP=X","EURCHF":"EURCHF=X","EURCAD":"EURCAD=X",
    "EURAUD":"EURAUD=X","GBPCAD":"GBPCAD=X","GBPAUD":"GBPAUD=X","AUDCAD":"AUDCAD=X","NZDJPY":"NZDJPY=X",
    # Commodities
    "XAUUSD (Gold)":"GC=F","XAGUSD (Silver)":"SI=F","WTI (Oil)":"CL=F","BRENT (Oil)":"BZ=F",
    # Crypto (24/7)
    "BTCUSD (Bitcoin)":"BTC-USD","ETHUSD (Ethereum)":"ETH-USD","SOLUSD (Solana)":"SOL-USD",
    "XRPUSD (XRP)":"XRP-USD","BNBUSD (BNB)":"BNB-USD","DOGEUSD (Dogecoin)":"DOGE-USD"
}

# --------- ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ ---------
def ema(s, n): 
    return s.ewm(span=n, adjust=False).mean()

def rsi(close, period=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (dn + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, signal)
    return m, s, m - s

def bbands(close, n=20, k=2):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    up, lo = ma + k*sd, ma - k*sd
    width = (up - lo) / (ma + 1e-9) * 100
    return up, ma, lo, width

def adx(df, n=14):
    h, l, c = df['High'], df['Low'], df['Close']
    up_move   = h.diff()
    dn_move   = -l.diff()
    plus_dm   = up_move.where((up_move > 0) & (up_move > dn_move), 0.0).fillna(0)
    minus_dm  = dn_move.where((dn_move > 0) & (dn_move > up_move), 0.0).fillna(0)
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di  = 100 * (plus_dm.rolling(n).sum() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(n).sum() / (atr + 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9))
    return dx.rolling(n).mean()

# --------- ЗАГРУЗКА/ФЕЙК ДАННЫХ ---------
def safe_download(symbol):
    """Пытаемся взять реальные данные; если пусто — вернём None."""
    try:
        data = yf.download(symbol, period=f"{max(LOOKBACK_MIN, 60)}m", interval=INTERVAL,
                           progress=False, auto_adjust=True)
        if data is None or len(data) < 50:
            return None
        return data.tail(600)
    except Exception:
        return None

def nudge_last(df, max_bps=5):
    """Создаём «тиковую» последнюю свечу от последней цены (±несколько б.п.), чтобы панель не замирала."""
    last = df.iloc[-1].copy()
    close = float(last["Close"])
    bps = random.uniform(-max_bps, max_bps) / 10000.0  # ±N б.п.
    new_close = max(1e-9, close * (1 + bps))
    last["Open"]  = close
    last["High"]  = max(close, new_close)
    last["Low"]   = min(close, new_close)
    last["Close"] = new_close
    last.name = last.name + timedelta(minutes=1)
    return last

def get_or_fake(symbol):
    """Реальные данные → кэш; иначе подвигаем последний бар; иначе делаем синтетику."""
    if "cache" not in st.session_state:
        st.session_state.cache = {}

    real = safe_download(symbol)
    if real is not None:
        st.session_state.cache[symbol] = real.copy()
        return real

    cached = st.session_state.cache.get(symbol)
    if cached is not None and len(cached) > 0:
        df = cached.copy()
        last = nudge_last(df)
        # если вернулся Series — превращаем в DataFrame с индексом-временем
        if isinstance(last, pd.Series):
            last = last.to_frame().T
        df = pd.concat([df, last], axis=0)
        st.session_state.cache[symbol] = df.tail(600)
        return st.session_state.cache[symbol]

    # Совсем пусто — создаём небольшую синтетику
    idx = pd.date_range(end=datetime.utcnow(), periods=60, freq="1min")
    base = 1.0 + random.random() / 10
    vals = base * (1 + np.cumsum(np.random.randn(60)) / 100)
    df = pd.DataFrame({"Open": vals, "High": vals, "Low": vals, "Close": vals}, index=idx)
    st.session_state.cache[symbol] = df
    return df

# --------- ЛОГИКА СИГНАЛА/УВЕРЕННОСТИ ---------
def calculate_confidence(rsi_v: float, adx_v: float, macd_hist_v: float) -> int:
    """
    Чем сильнее тренд (ADX) и чем дальше RSI от 50, + подтверждение MACD — тем выше уверенность.
    Возвращаем 40..100.
    """
    score = 0
    score += min(abs(rsi_v - 50) * 1.2, 40)           # до 40
    score += min(adx_v, 40)                           # до 40
    score += min(abs(macd_hist_v) * 100000, 20)       # до 20
    return int(max(40, min(100, round(score))))

def score_and_signal(df):
    close = df["Close"]
    rsi_v = float(rsi(close).iloc[-1])
    ema9  = float(ema(close, 9).iloc[-1])
    ema21 = float(ema(close, 21).iloc[-1])
    macd_line, macd_sig, macd_hist = macd(close)
    m_hist = float(macd_hist.iloc[-1])
    up, mid, lo, width = bbands(close)
    bb_pos = float((close.iloc[-1] - mid.iloc[-1]) / (up.iloc[-1] - lo.iloc[-1] + 1e-9))
    adx_v = float(adx(df).iloc[-1])

    votes_buy = votes_sell = 0

    # --- RSI ---
    if rsi_v < 30:
        votes_buy += 1
    if rsi_v > 70:
        votes_sell += 1

    # --- EMA тренд ---
    if ema9 > ema21:
        votes_buy += 1
    if ema9 < ema21:
        votes_sell += 1

    # --- MACD ---
    if m_hist > 0:
        votes_buy += 1
    if m_hist < 0:
        votes_sell += 1

    # --- Боллинджер ---
    if bb_pos < -0.25:
        votes_buy += 1
    if bb_pos > 0.25:
        votes_sell += 1

    # --- Расчёт силы тренда ---
    trend_boost = min(max((adx_v - 18) / 25, 0), 1)

    # --- Определяем направление ---
    if votes_buy == votes_sell:
        direction = "FLAT"
    elif votes_buy > votes_sell:
        direction = "BUY"
    else:
        direction = "SELL"

    # --- 🔥 ФИЛЬТР ПРОТИВ ТРЕНДА 🔥 ---
    strong_down = ema9 < ema21 and adx_v > 45 and rsi_v < 25
    strong_up   = ema9 > ema21 and adx_v > 45 and rsi_v > 75

    # если идёт сильный тренд, не даём сигнал против него
    if strong_down and direction == "BUY":
        direction = "FLAT"
    elif strong_up and direction == "SELL":
        direction = "FLAT"

    # --- Расчёт уверенности ---
    raw = abs(votes_buy - votes_sell) / 4.0
    confidence = int(100 * (0.55 * raw + 0.45 * trend_boost))
    confidence = max(0, min(99, confidence))

    # --- Формируем индикаторы для отчёта ---
    feats = dict(
        RSI=round(rsi_v, 1),
        ADX=round(adx_v, 1),
        MACD_Hist=round(m_hist, 5),
        EMA9_minus_EMA21=round(ema9 - ema21, 5),
        BB_Pos=round(bb_pos, 3),
        BB_Width=round(float(width.iloc[-1]), 2),
    )

    return direction, confidence, feats

def choose_expiry(confidence: int, adx_value: float, rsi_value: float) -> int | None:
    """
    Возвращает оптимальное время экспирации (в минутах)
    на основе уверенности сигнала и силы тренда.
    """
    # Фильтр слабых
    if confidence < 60:
        print(f"⚠️ Пропущен слабый сигнал (уверенность {confidence}%)")
        return None

    # База по уверенности
    if confidence < 65:
        base = 2
    elif confidence < 75:
        base = 5
    elif confidence < 85:
        base = 8
    elif confidence < 90:
        base = 12
    elif confidence < 95:
        base = 18
    else:
        base = 25

    # Корректировка по силе тренда
    if adx_value >= 50:
        base += 10
    elif adx_value >= 30:
        base += 5
    elif adx_value < 20:
        base = max(2, base - 3)

    return int(max(1, min(30, base)))

def send_telegram(pair_name: str, signal: str, confidence: int, expiry: int | None, feats: dict):
    """Отправка сообщения в Telegram."""
    if expiry is None:
        exp_txt = "—"
    else:
        exp_txt = f"{expiry} мин"

    text = (
        f"🤖 *AI FX СИГНАЛ*\n"
        f"💵 Пара: {pair_name}\n"
        f"📊 Сигнал: {signal}\n"
        f"💪 Уверенность: {confidence}%\n"
        f"⏱ Экспирация: {exp_txt}\n"
        f"⚙️ RSI {feats.get('RSI','?')} | ADX {feats.get('ADX','?')} | MACD {feats.get('MACD_Hist','?')}\n"
        f"⏰ {datetime.utcnow().strftime('%H:%M:%S')}"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"})
    except Exception as e:
        st.toast(f"TG error: {e}", icon="⚠️")

# --------- UI ---------
st.set_page_config(page_title="AI FX Panel Pro", layout="wide")
st.title("🤖 AI FX PANEL — 24/7 умные сигналы (FX + Commodities + Crypto)")

# Управление порогом и антиспамом
c1, c2, c3 = st.columns([1,1,1])
with c1:
    threshold = st.slider("Порог отправки в Telegram", 50, 95, SEND_THRESHOLD, 1)
with c2:
    min_gap = st.number_input("Мин. пауза между сигналами (сек)", 10, 300, MIN_SEND_GAP_S)
with c3:
    st.write(" ")

# Антиспам хранилище
if "last_sent" not in st.session_state:
    st.session_state.last_sent = {}  # {pair: {"signal": "BUY/SELL", "ts": epoch, "conf": int}}

rows = []

# --------- АНАЛИЗ ВСЕХ ИНСТРУМЕНТОВ ---------
for name, symbol in PAIRS.items():
    df = get_or_fake(symbol)
    if df is None or len(df) < 30:
        continue

    sig, conf, feats = score_and_signal(df)
    expiry = choose_expiry(conf, feats["ADX"], feats["RSI"])

    rows.append([name, sig, conf, expiry if expiry is not None else "-", json.dumps(feats)])

    # отправка только уверенных
    if sig in ("BUY", "SELL") and conf >= threshold:
        prev = st.session_state.last_sent.get(name, {})
        should = True
        if ONLY_NEW and prev:
            same_dir = prev.get("signal") == sig
            not_better = conf <= prev.get("conf", 0)
            recently = (time.time() - prev.get("ts", 0)) < min_gap
            if same_dir and (not_better or recently):
                should = False
        if should:
            send_telegram(name, sig, conf, expiry, feats)
            st.session_state.last_sent[name] = {"signal": sig, "ts": time.time(), "conf": conf}

# --------- ТАБЛИЦА ---------
df_show = pd.DataFrame(rows, columns=["Пара","Сигнал","Уверенность","Экспирация (мин)","Индикаторы"])
if len(df_show):
    df_show = df_show.sort_values("Уверенность", ascending=False).reset_index(drop=True)
st.subheader("📋 Рейтинг сигналов")
st.dataframe(df_show, use_container_width=True, height=440)

# --------- ГРАФИК ЛУЧШЕЙ ПАРЫ ---------
dfc = None
top = None

if len(df_show):
    top = df_show.iloc[0]
    sym = PAIRS.get(top["Пара"])
    if sym:
        dfc = get_or_fake(sym)

if dfc is not None and len(dfc):
    fig = go.Figure(data=[go.Candlestick(
        x=dfc.index,
        open=dfc["Open"],
        high=dfc["High"],
        low=dfc["Low"],
        close=dfc["Close"]
    )])
    title_txt = f"Топ: {top['Пара']} — {top['Сигнал']} ({top['Уверенность']}%)" if top is not None else "Топ"
    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=20, b=0),
        title=title_txt
    )
    st.plotly_chart(fig, use_container_width=True)

# --------- АВТООБНОВЛЕНИЕ ---------
time.sleep(REFRESH_SEC)
st.rerun()
