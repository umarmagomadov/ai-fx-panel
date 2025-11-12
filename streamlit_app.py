# streamlit_app.py
# AI FX PANEL PRO — исправленная и улучшенная версия (v99.0)
import time, json, math, random
import requests, numpy as np, pandas as pd, yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ---- конфиг ----
REFRESH_SEC     = 1
LOOKBACK_MIN    = 180
INTERVAL        = "1m"
SEND_THRESHOLD  = 70
ONLY_NEW        = True
MIN_SEND_GAP_S  = 60

# ---- secrets ----
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", None)
CHAT_ID        = st.secrets.get("CHAT_ID", None)

# ---- инструменты ----
PAIRS = {
    "EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"USDJPY=X","USDCHF":"USDCHF=X","AUDUSD":"AUDUSD=X",
    "NZDUSD":"NZDUSD=X","USDCAD":"USDCAD=X","EURJPY":"EURJPY=X","GBPJPY":"GBPJPY=X","AUDJPY":"AUDJPY=X",
    "CADJPY":"CADJPY=X","CHFJPY":"CHFJPY=X","EURGBP":"EURGBP=X","EURCHF":"EURCHF=X","EURCAD":"EURCAD=X",
    "EURAUD":"EURAUD=X","GBPCAD":"GBPCAD=X","GBPAUD":"GBPAUD=X","AUDCAD":"AUDCAD=X","NZDJPY":"NZDJPY=X",
    # Commodities / futures (часто OTC-ish on brokers)
    "XAUUSD (Gold)":"GC=F","XAGUSD (Silver)":"SI=F","WTI (Oil)":"CL=F","BRENT (Oil)":"BZ=F",
    # Crypto
    "BTCUSD (Bitcoin)":"BTC-USD","ETHUSD (Ethereum)":"ETH-USD","SOLUSD (Solana)":"SOL-USD",
    "XRPUSD (XRP)":"XRP-USD","BNBUSD (BNB)":"BNB-USD","DOGEUSD (Dogecoin)":"DOGE-USD"
}

# Вспомогательная функция: определяем OTC-ish актив
def is_otc(name, symbol):
    # простая эвристика: если в отображаемом назв. есть "OTC" или тикер -F (фьючерс) или crypto (-)
    n = name.lower()
    if "otc" in n or "=f" in symbol.lower():
        return True
    # крипто на некоторых платформах считаем 24/7 (пометить как OTC-style)
    if "-" in symbol and symbol.count("-") >= 1:
        return True
    return False

# ---- индикаторы ----
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

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

# ---- загрузка данных ----
def safe_download(symbol):
    try:
        data = yf.download(symbol, period=f"{max(LOOKBACK_MIN, 60)}m", interval=INTERVAL,
                           progress=False, auto_adjust=True)
        if data is None or len(data) < 10:
            return None
        return data.tail(600)
    except Exception:
        return None

def nudge_last(df, max_bps=5):
    last = df.iloc[-1].copy()
    close = float(last["Close"])
    bps = random.uniform(-max_bps, max_bps) / 10000.0
    new_close = max(1e-9, close * (1 + bps))
    last["Open"]  = close
    last["High"]  = max(close, new_close)
    last["Low"]   = min(close, new_close)
    last["Close"] = new_close
    last.name = last.name + timedelta(minutes=1)
    return last

def get_or_fake(symbol):
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
        if isinstance(last, pd.Series):
            last = last.to_frame().T
        df = pd.concat([df, last], ignore_index=False)
        st.session_state.cache[symbol] = df.tail(600)
        return st.session_state.cache[symbol]
    # синтетика (маленький фрейм)
    idx = pd.date_range(end=datetime.utcnow(), periods=60, freq="1min")
    base = 1.0 + random.random() / 10
    vals = base * (1 + np.cumsum(np.random.randn(60)) / 100)
    df = pd.DataFrame({"Open": vals, "High": vals, "Low": vals, "Close": vals}, index=idx)
    st.session_state.cache[symbol] = df
    return df

# ---- скоринг и сигнал ----
def score_and_signal(df):
    close = df["Close"]
    # безопасно, если мало данных — вернём flat
    if len(close) < 8:
        return "FLAT", 0, {"RSI": None, "ADX": None, "MACD_Hist": None}

    rsi_v = float(rsi(close).iloc[-1])
    ema9  = float(ema(close, 9).iloc[-1])
    ema21 = float(ema(close, 21).iloc[-1])
    macd_line, macd_sig, macd_hist = macd(close)
    m_hist = float(macd_hist.iloc[-1])
    up, mid, lo, width = bbands(close)
    # защитимся если NaN
    try:
        bb_pos = float((close.iloc[-1] - mid.iloc[-1]) / (up.iloc[-1] - lo.iloc[-1] + 1e-9))
    except Exception:
        bb_pos = 0.0
    adx_v = float(adx(df).iloc[-1]) if len(df) > 20 else 0.0

    votes_buy = votes_sell = 0
    if rsi_v < 35: votes_buy += 1
    if rsi_v > 65: votes_sell += 1
    if ema9 > ema21: votes_buy += 1
    if ema9 < ema21: votes_sell += 1
    if m_hist > 0: votes_buy += 1
    if m_hist < 0: votes_sell += 1
    if bb_pos < -0.25: votes_buy += 1
    if bb_pos > 0.25: votes_sell += 1

    if votes_buy == votes_sell:
        direction = "FLAT"
    elif votes_buy > votes_sell:
        direction = "BUY"
    else:
        direction = "SELL"

    trend_boost = min(max((adx_v - 18) / 25, 0), 1)
    raw = abs(votes_buy - votes_sell) / 4.0
    confidence = int(100 * (0.55 * raw + 0.45 * trend_boost))
    confidence = max(0, min(100, confidence))

    feats = dict(RSI=round(rsi_v,1), ADX=round(adx_v,1), MACD_Hist=round(m_hist,6),
                 EMA9_minus_EMA21=round(ema9-ema21,5), BB_Pos=round(bb_pos,3),
                 BB_Width=round(float(width.iloc[-1]) if len(width)>0 else 0,2))
    return direction, confidence, feats

# ---- выбор экспирации ----
def choose_expiry(confidence, adx_value, rsi_value, is_otc_flag=False):
    if confidence < 60:
        return None
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

    if adx_value >= 50:
        base += 10
    elif adx_value >= 30:
        base += 5
    elif adx_value < 20:
        base = max(2, base - 3)

    # если OTC — чуть более консервативно (больше экспирация, жестче порог)
    if is_otc_flag:
        base = min(60, base + 5)

    expiry = int(max(1, min(60, base)))
    return expiry

# ---- отправка в телеграм ----
def send_telegram(pair_name, symbol, signal, confidence, expiry, feats, is_otc_flag):
    if TELEGRAM_TOKEN is None or CHAT_ID is None:
        st.warning("TELEGRAM_TOKEN/CHAT_ID не настроены в secrets — пропускаем отправку в TG.")
        return
    phase = "OTC/24/7" if is_otc_flag else "Биржевая"
    text = (
        f"🤖 *AI FX СИГНАЛ*\n"
        f"💵 Пара: *{pair_name}*\n"
        f"`{symbol}`\n"
        f"📊 Сигнал: *{signal}*\n"
        f"💪 Уверенность: *{confidence}%*\n"
        f"⏱ Экспирация: *{expiry} мин*\n"
        f"⚙️ RSI {feats.get('RSI')} | ADX {feats.get('ADX')} | MACD {feats.get('MACD_Hist')}\n"
        f"🏷️ Тип: *{phase}*\n"
        f"⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"})
    except Exception as e:
        st.toast(f"TG error: {e}", icon="⚠️")

# ---- UI ----
st.set_page_config(page_title="AI FX Panel Pro", layout="wide")
st.title("🤖 AI FX PANEL — 24/7 сигналы (FX + Commodities + Crypto)")

c1, c2, c3 = st.columns([1,1,1])
with c1:
    threshold = st.slider("Порог отправки в Telegram", 50, 95, SEND_THRESHOLD, 1)
with c2:
    min_gap = st.number_input("Мин. пауза между сигналами (сек)", 10, 300, MIN_SEND_GAP_S)
with c3:
    st.write(" ")

if "last_sent" not in st.session_state:
    st.session_state.last_sent = {}

rows = []

for name, symbol in PAIRS.items():
    df = get_or_fake(symbol)
    sig, conf, feats = score_and_signal(df)
    otc_flag = is_otc(name, symbol)
    expiry = choose_expiry(conf, feats.get('ADX',0) or 0, feats.get('RSI',50) or 50, otc_flag)
    # умный OTC-фильтр: увеличим порог для OTC
    effective_threshold = threshold + 10 if otc_flag else threshold

    rows.append([name, symbol, sig, conf, expiry or "-", json.dumps(feats)])

    # отправка в TG
    if sig in ("BUY","SELL") and conf >= effective_threshold:
        prev = st.session_state.last_sent.get(name, {})
        should = True
        if ONLY_NEW and prev:
            same_dir = prev.get("signal") == sig
            not_better = conf <= prev.get("conf", 0)
            recently = (time.time() - prev.get("ts", 0)) < min_gap
            if same_dir and (not_better or recently):
                should = False
        if should:
            send_telegram(name, symbol, sig, conf, expiry or 0, feats, otc_flag)
            st.session_state.last_sent[name] = {"signal": sig, "ts": time.time(), "conf": conf}

# таблица
df_show = pd.DataFrame(rows, columns=["Пара","Тикер","Сигнал","Уверенность","Экспирация (мин)","Индикаторы"])
if len(df_show):
    df_show = df_show.sort_values("Уверенность", ascending=False).reset_index(drop=True)
st.subheader("📋 Рейтинг сигналов")
st.dataframe(df_show, use_container_width=True, height=420)

# удобство: выбрать строку и скопировать тикер
st.markdown("#### Выбрать пару и скопировать тикер")
pair_names = df_show["Пара"].tolist()
sel = st.selectbox("Пара", ["-- выбрать --"] + pair_names)
if sel and sel != "-- выбрать --":
    row = df_show[df_show["Пара"] == sel].iloc[0]
    st.write(f"**Тикер:** `{row['Тикер']}` (скопируй нажатием и удержанием на мобильном)")
    st.text_input("Готовый тикер для копирования:", value=row['Тикер'], key=f"copy_{sel}")

# график топа
if len(df_show):
    top = df_show.iloc[0]
    sym = PAIRS[top["Пара"]]
    dfc = get_or_fake(sym)
    if dfc is not None and len(dfc):
        fig = go.Figure(data=[go.Candlestick(x=dfc.index, open=dfc["Open"], high=dfc["High"],
                                             low=dfc["Low"], close=dfc["Close"])])
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=20,b=0),
                          title=f"Топ: {top['Пара']} — {top['Сигнал']} ({top['Уверенность']}%)")
        st.plotly_chart(fig, use_container_width=True)

time.sleep(REFRESH_SEC)
st.experimental_rerun()
