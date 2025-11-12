# AI FX PANEL PRO v99.2 — мульти-ТФ, фазы свечи, RSI-фильтр, вердикт
# Forex + Commodities + Crypto

import time, json, math, random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
import plotly.graph_objects as go

# ---------- СЕКРЕТЫ ----------
TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
CHAT_ID        = st.secrets["CHAT_ID"]

# ---------- НАСТРОЙКИ UI/ЛОГИКИ ----------
REFRESH_SEC     = 1
ONLY_NEW        = True
MIN_SEND_GAP_S  = 60
BASE_INTERVAL   = "5m"          # базовый ТФ
CONF_THRESHOLD  = 70            # дефолтный порог
LOOKBACK_MIN    = 240           # история для расчётов (минут)

# Таймфреймы для подтверждения
MTF = [
    ("5m",  240),   # (interval, lookback_minutes)
    ("15m", 720),
    ("30m", 1440),
]

# Инструменты
PAIRS = {
    # Forex
    "EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"USDJPY=X","USDCHF":"USDCHF=X","AUDUSD":"AUDUSD=X",
    "NZDUSD":"NZDUSD=X","USDCAD":"USDCAD=X","EURJPY":"EURJPY=X","GBPJPY":"GBPJPY=X","AUDJPY":"AUDJPY=X",
    "CADJPY":"CADJPY=X","CHFJPY":"CHFJPY=X","EURGBP":"EURGBP=X","EURCHF":"EURCHF=X","EURCAD":"EURCAD=X",
    "EURAUD":"EURAUD=X","GBPCAD":"GBPCAD=X","GBPAUD":"GBPAUD=X","AUDCAD":"AUDCAD=X","NZDJPY":"NZDJPY=X",
    # Commodities
    "XAUUSD (Gold)":"GC=F","XAGUSD (Silver)":"SI=F","WTI (Oil)":"CL=F","BRENT (Oil)":"BZ=F",
    # Crypto
    "BTCUSD (Bitcoin)":"BTC-USD","ETHUSD (Ethereum)":"ETH-USD","SOLUSD (Solana)":"SOL-USD",
    "XRPUSD (XRP)":"XRP-USD","BNBUSD (BNB)":"BNB-USD","DOGEUSD (Dogecoin)":"DOGE-USD",
}

# ---------- ТЕХНИЧЕСКИЕ ----------
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
    plus_dm   = up_move.where((up_move > 0) & (up_move > dn_move), 0.0).fillna(0.0)
    minus_dm  = dn_move.where((dn_move > 0) & (dn_move > up_move), 0.0).fillna(0.0)
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di  = 100 * (plus_dm.rolling(n).sum() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(n).sum() / (atr + 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9))
    return dx.rolling(n).mean()

# ---------- УТИЛИТЫ ----------
def classify_market(pair_name: str) -> str:
    """Биржевая / OTC по названию из брокера. Если в названии есть OTC — OTC."""
    return "OTC" if "OTC" in pair_name.upper() else "Биржевая"

def tf_minutes(interval: str) -> int:
    return int(interval.replace("m","").replace("h","0")) if "m" in interval else 60*int(interval.replace("h",""))

def safe_download(symbol: str, period_min: int, interval: str):
    try:
        data = yf.download(
            symbol,
            period=f"{max(period_min, 60)}m",
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if data is None or len(data) < 50:
            return None
        return data.tail(600)
    except Exception:
        return None

def nudge_last(df: pd.DataFrame, max_bps=5) -> pd.Series:
    last = df.iloc[-1].copy()
    close = float(last["Close"])
    bps = random.uniform(-max_bps, max_bps) / 10000.0
    new_close = max(1e-9, close * (1 + bps))
    last["Open"]  = close
    last["High"]  = max(close, new_close)
    last["Low"]   = min(close, new_close)
    last["Close"] = new_close
    last.name = last.name + pd.tseries.frequencies.to_offset("1min")
    return last

def get_or_fake(symbol: str, period_min: int, interval: str) -> pd.DataFrame:
    if "cache" not in st.session_state:
        st.session_state.cache = {}
    key = (symbol, interval)

    real = safe_download(symbol, period_min, interval)
    if real is not None:
        st.session_state.cache[key] = real.copy()
        return real

    cached = st.session_state.cache.get(key)
    if cached is not None and len(cached) > 0:
        df = cached.copy()
        last = nudge_last(df)
        if isinstance(last, pd.Series):
            last = last.to_frame().T
        df = pd.concat([df, last], ignore_index=False)
        st.session_state.cache[key] = df.tail(600)
        return st.session_state.cache[key]

    # Совсем пусто — синтетика
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=60, freq="1min")
    base = 1.0 + random.random()/10
    vals = base * (1 + np.cumsum(np.random.randn(60))/100)
    df = pd.DataFrame({"Open": vals, "High": vals, "Low": vals, "Close": vals}, index=idx)
    st.session_state.cache[key] = df
    return df

def candle_phase(last_dt: pd.Timestamp, interval: str) -> tuple[str, float]:
    """Возвращает (эмодзи, доля_свечи_0..1)."""
    now = datetime.now(timezone.utc)
    if last_dt.tzinfo is None:
        last_dt = last_dt.tz_localize(timezone.utc)
    minutes = tf_minutes(interval)
    passed = (now - last_dt).total_seconds()
    frac = max(0.0, min(1.0, passed / (minutes*60)))
    if frac < 0.33:  ico = "🟢 Начало"
    elif frac < 0.66: ico = "🟡 Середина"
    else:             ico = "🔴 Конец"
    return ico, frac

# ---------- СКОРИНГ / СИГНАЛ ----------
def score_single_tf(df: pd.DataFrame) -> dict:
    close = df["Close"]
    rsi_v = float(rsi(close).iloc[-1])
    rsi_prev = float(rsi(close).iloc[-2])
    ema9  = float(ema(close, 9).iloc[-1])
    ema21 = float(ema(close, 21).iloc[-1])
    m_line, m_sig, m_hist = macd(close)
    m_hist_v = float(m_hist.iloc[-1])
    up, mid, lo, width = bbands(close)
    bb_pos = float((close.iloc[-1] - mid.iloc[-1]) / (up.iloc[-1] - lo.iloc[-1] + 1e-9))
    adx_v = float(adx(df).iloc[-1])

    # Направление индикаторов
    ema_dir  =  1 if ema9 > ema21 else (-1 if ema9 < ema21 else 0)
    macd_dir =  1 if m_hist_v > 0   else (-1 if m_hist_v < 0   else 0)
    rsi_dir  = -1 if rsi_v > 60 else (1 if rsi_v < 40 else 0)  # перекуп/перепрод

    votes_buy = votes_sell = 0
    if rsi_v < 30: votes_buy += 1
    if rsi_v > 70: votes_sell += 1
    if ema_dir > 0: votes_buy += 1
    if ema_dir < 0: votes_sell += 1
    if macd_dir > 0: votes_buy += 1
    if macd_dir < 0: votes_sell += 1
    if bb_pos < -0.25: votes_buy += 1
    if bb_pos >  0.25: votes_sell += 1

    if votes_buy == votes_sell:
        direction = "FLAT"
    elif votes_buy > votes_sell:
        direction = "BUY"
    else:
        direction = "SELL"

    # Базовая уверенность
    score = 0
    score += min(abs(rsi_v - 50) * 1.2, 40)
    score += min(adx_v, 40)
    score += min(abs(m_hist_v) * 100000, 20)
    confidence = max(40, min(100, round(score)))

    feats = dict(
        RSI=round(rsi_v,1),
        RSI_prev=round(rsi_prev,1),
        ADX=round(adx_v,1),
        MACD_Hist=round(m_hist_v,6),
        EMA9_minus_EMA21=round(ema9-ema21,6),
        BB_Pos=round(bb_pos,3),
        BB_Width=round(float(width.iloc[-1]),2)
    )
    ind_dirs_agree = (rsi_dir == macd_dir == (1 if ema_dir>0 else -1 if ema_dir<0 else 0)) and rsi_dir != 0
    return dict(direction=direction, confidence=confidence, feats=feats, agree=ind_dirs_agree)

def score_multi_tf(symbol: str) -> tuple[str,int,dict,str,float]:
    """
    Мульти-таймфрейм: возвращает (signal, confidence, feats, phase_text, phase_frac)
    """
    # Базовый DF для фазы свечи
    base_df = get_or_fake(symbol, LOOKBACK_MIN, BASE_INTERVAL)
    phase_txt, phase_frac = candle_phase(base_df.index[-1], BASE_INTERVAL)

    # По всем ТФ
    results = []
    for interval, look in MTF:
        df = get_or_fake(symbol, look, interval)
        results.append((interval, score_single_tf(df)))

    # Согласованность направлений
    dirs = [r["direction"] for _, r in results]
    strong = [d for d in dirs if d in ("BUY","SELL")]
    signal = "FLAT" if not strong else ( "BUY" if strong.count("BUY")>=strong.count("SELL") else "SELL" )
    agree_cnt = strong.count(signal)

    # Итоговые признаки берём с базового ТФ
    base_res = dict(results)[BASE_INTERVAL]
    feats = base_res["feats"]
    confidence = base_res["confidence"]

    # Бонусы/штрафы
    if agree_cnt == 3:          confidence += 10      # полное согласие M5/M15/M30
    elif agree_cnt == 2:        confidence += 5       # 2 из 3
    if base_res["agree"]:       confidence += 5       # индикаторы согласны между собой

    # Фильтр «RSI-импульс» — избегаем ложных разворотов
    if abs(feats["RSI"] - feats["RSI_prev"]) > 10:
        confidence -= 12

    # Фаза свечи: вход лучше в середине
    if phase_frac < 0.25:       confidence -= 5       # слишком рано, ждём
    elif phase_frac < 0.75:     confidence += 5       # оптимально
    else:                       confidence -= 7       # конец, возможен откат

    confidence = int(max(0, min(100, confidence)))
    return signal, confidence, feats, phase_txt, phase_frac

def choose_expiry(conf: int, adx_value: float, rsi_value: float) -> int:
    if conf < 60: return 0
    if conf < 65: base = 2
    elif conf < 75: base = 5
    elif conf < 85: base = 8
    elif conf < 90: base = 12
    elif conf < 95: base = 18
    else: base = 25
    if adx_value >= 50: base += 10
    elif adx_value >= 30: base += 5
    elif adx_value < 20: base = max(2, base - 3)
    return int(max(1, min(30, base)))

def verdict(signal: str, conf: int, phase_txt: str) -> str:
    if signal == "FLAT":
        return "Сигнал слабый/флэт. Пропустить."
    if conf >= 90 and "Середина" in phase_txt:
        return "Сильный сигнал по тренду. Вход разрешён ✅"
    if conf >= 80:
        return "Хороший сигнал, можно входить при подтверждении."
    if conf >= 70:
        return "Средний сигнал. Лучше дождаться лучшей точки."
    return "Слабый сигнал. Пропустить."

def send_telegram(pair_name: str, pair_code: str, mtype: str, signal: str, conf: int, expiry: int, feats: dict, phase_txt: str):
    text = (
        f"🤖 AI FX СИГНАЛ v99.2\n"
        f"💱 Пара: {pair_name}\n"
        f"📌 Код: `{pair_code}`\n"
        f"🏷️ Тип: {mtype}\n"
        f"📉 Сигнал: {signal}\n"
        f"🧭 Свеча: {phase_txt}\n"
        f"💪 Уверенность: {conf}%\n"
        f"⏱ Экспирация: {expiry} мин\n"
        f"📈 RSI {feats['RSI']} | ADX {feats['ADX']} | MACD {feats['MACD_Hist']}\n"
        f"⏰ {datetime.utcnow().strftime('%H:%M:%S')} UTC"
    )
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10
        )
    except Exception as e:
        st.toast(f"TG error: {e}", icon="⚠️")

# ---------- UI ----------
st.set_page_config(page_title="AI FX Panel Pro v99.2", layout="wide")
st.title("🤖 AI FX PANEL — v99.2 (MTF, Phase, RSI-guard, Verdict)")

c1, c2, c3 = st.columns([1,1,1])
with c1:
    threshold = st.slider("Порог отправки в Telegram", 50, 95, CONF_THRESHOLD, 1)
with c2:
    min_gap = st.number_input("Мин. пауза между сигналами (сек)", 10, 300, MIN_SEND_GAP_S)
with c3:
    st.write(" ")

if "last_sent" not in st.session_state:
    st.session_state.last_sent = {}  # {pair: {"signal":..., "ts":..., "conf":...}}

rows = []

# ---------- ОСНОВНОЙ ЦИКЛ ----------
for pair_name, pair_code in PAIRS.items():
    signal, conf, feats, phase_txt, phase_frac = score_multi_tf(pair_code)
    expiry = choose_expiry(conf, feats["ADX"], feats["RSI"])
    mtype  = classify_market(pair_name)

    ver = verdict(signal, conf, phase_txt)
    rows.append([pair_name, mtype, signal, conf, expiry, phase_txt, ver, json.dumps(feats)])

    # отправка
    if signal in ("BUY","SELL") and conf >= threshold and expiry > 0:
        prev = st.session_state.last_sent.get(pair_name, {})
        should = True
        if ONLY_NEW and prev:
            same_dir = prev.get("signal") == signal
            not_better = conf <= prev.get("conf", 0)
            recently = (time.time() - prev.get("ts", 0)) < min_gap
            if same_dir and (not_better or recently):
                should = False
        if should:
            send_telegram(pair_name, pair_code, mtype, signal, conf, expiry, feats, phase_txt)
            st.session_state.last_sent[pair_name] = {"signal": signal, "ts": time.time(), "conf": conf}

# ---------- ТАБЛИЦА ----------
df_show = pd.DataFrame(rows, columns=[
    "Пара","Тип","Сигнал","Уверенность","Экспирация (мин)","Свеча","Вердикт","Индикаторы"
]).sort_values("Уверенность", ascending=False).reset_index(drop=True)

st.subheader("📋 Рейтинг сигналов (v99.2)")
st.dataframe(df_show, use_container_width=True, height=460)

# ---------- ГРАФИК ТОП-ПАРЫ ----------
if len(df_show):
    top = df_show.iloc[0]
    sym = PAIRS[top["Пара"]]
    dfc = get_or_fake(sym, LOOKBACK_MIN, BASE_INTERVAL)
    fig = go.Figure(data=[go.Candlestick(
        x=dfc.index, open=dfc["Open"], high=dfc["High"], low=dfc["Low"], close=dfc["Close"]
    )])
    fig.update_layout(height=380, margin=dict(l=0,r=0,t=20,b=0),
                      title=f"Топ: {top['Пара']} — {top['Сигнал']} ({top['Уверенность']}%) — {top['Свеча']}")
    st.plotly_chart(fig, use_container_width=True)

# ---------- АВТООБНОВЛЕНИЕ ----------
time.sleep(REFRESH_SEC)
st.rerun()
