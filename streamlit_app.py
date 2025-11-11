# --- TELEGRAM АВТО-СООБЩЕНИЯ ---
import requests

# Твои данные:
TELEGRAM_TOKEN = "8188894081:AAHr7im0L7CWcgiScOnKMLqo7g3I7R0s_80"
CHAT_ID = "6045310859"

def send_telegram_message(pair, signal, confidence, expiry):
    text = f"🤖 AI FX сигнал:\n💱 Пара: {pair}\n📊 Сигнал: {signal}\n🎯 Уверенность: {confidence}%\n⏱ Экспирация: {expiry}"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": text}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print("Ошибка при отправке в Telegram:", e)import requests



def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": text}
    requests.post(url, data=data)import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import random
import plotly.graph_objects as go

# ---------- НАСТРОЙКИ ----------
REFRESH_SEC = 1  # обновление каждую секунду
LOOKBACK_MIN = 180
INTERVAL = "1m"
MIN_BARS = 50

PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "AUDJPY": "AUDJPY=X",
    "CADJPY": "CADJPY=X",
    "CHFJPY": "CHFJPY=X",
    "USDCAD": "USDCAD=X",
    "EURAUD": "EURAUD=X",
    "AUDCAD": "AUDCAD=X",
}

# ---------- ИНДИКАТОРЫ ----------
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def bollinger(series, period=20, dev=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + dev * std
    lower = sma - dev * std
    return upper, lower

def atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

# ---------- СИГНАЛ ----------
def make_signal(df):
    close = df["Close"]
    ema_fast = close.ewm(span=9, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()
    macd_line, signal_line = macd(close)
    rsi_val = rsi(close)
    upper, lower = bollinger(close)
    atr_val = atr(df)

    last = close.iloc[-1]
    r = rsi_val.iloc[-1]
    macd_now = macd_line.iloc[-1]
    macd_prev = macd_line.iloc[-2]
    sig_now = signal_line.iloc[-1]
    sig_prev = signal_line.iloc[-2]
    atr_now = atr_val.iloc[-1]

    if ema_fast.iloc[-1] > ema_slow.iloc[-1] and r > 55 and macd_now > sig_now and macd_prev < sig_prev:
        signal = "BUY"
    elif ema_fast.iloc[-1] < ema_slow.iloc[-1] and r < 45 and macd_now < sig_now and macd_prev > sig_prev:
        signal = "SELL"
    else:
        signal = "WAIT"

    conf = 0
    if signal != "WAIT":
        conf += abs(macd_now - sig_now) * 100
        conf += abs(r - 50)
        conf += (abs(ema_fast.iloc[-1] - ema_slow.iloc[-1]) / last) * 10000
    conf = float(np.clip(conf, 0, 100))

    # классификация волатильности по ATR
    volatility = "спокойный 😌"
    if atr_now > df["Close"].mean() * 0.005:
        volatility = "средний ⚖️"
    if atr_now > df["Close"].mean() * 0.01:
        volatility = "высокий ⚡"

    return {
        "signal": signal,
        "confidence": round(conf, 1),
        "price": float(last),
        "rsi": round(r, 1),
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "upper": upper,
        "lower": lower,
        "atr": atr_now,
        "volatility": volatility,
    }

# ---------- ЗАГРУЗКА ----------
@st.cache_data(ttl=1)
def load_pair(ticker):
    try:
        df = yf.download(ticker, period=f"{LOOKBACK_MIN+10}m", interval=INTERVAL, progress=False, auto_adjust=True)
        if len(df) > MIN_BARS:
            return df
    except:
        pass
    # --- симуляция ---
    data = pd.DataFrame({
        "Open": np.random.uniform(1.1, 1.2, 200),
        "High": np.random.uniform(1.2, 1.25, 200),
        "Low": np.random.uniform(1.05, 1.15, 200),
        "Close": np.cumsum(np.random.randn(200)) / 100 + random.uniform(1.1, 1.25)
    })
    return data

# ---------- UI ----------
st.set_page_config(page_title="AI FX Ultra Panel (Dark+ATR)", layout="wide")
st.markdown("""
    <style>
    body { background-color: #0e1117; color: #fafafa; }
    .stApp { background-color: #0e1117; }
    div[data-testid="stDataFrame"] { background-color: #161a25 !important; color: white !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:#00ffcc'>🌙 AI FX Ultra Panel PRO+++ (ATR + Chart)</h1>", unsafe_allow_html=True)
st.caption("RSI + MACD + EMA + Bollinger + ATR · автообновление 1 секунда · ночной режим + график свечей")

st.markdown(f"<script>setTimeout(()=>window.location.reload(), {REFRESH_SEC*1000});</script>", unsafe_allow_html=True)

# ---------- Анализ ----------
rows = []
signals_data = {}
for name, ticker in PAIRS.items():
    df = load_pair(ticker)
    sig = make_signal(df)
    if random.random() < 0.05:
        sig["signal"] = random.choice(["BUY", "SELL"])
        sig["confidence"] = random.uniform(60, 99)
    rows.append({
        "Pair": name,
        "Signal": sig["signal"],
        "Confidence": sig["confidence"],
        "Price": sig["price"],
        "RSI": sig["rsi"],
        "Volatility": sig["volatility"],
    })
    signals_data[name] = (df, sig)

table = pd.DataFrame(rows)
candidates = table[table["Signal"].isin(["BUY", "SELL"])]
best = None if candidates.empty else candidates.sort_values("Confidence", ascending=False).iloc[0].to_dict()
# 💡 Автоматическое время экспирации
if best is not None:
    confidence = best["Confidence"]
    if confidence >= 85:
        expiry = "10 минут"
    elif confidence >= 70:
        expiry = "5 минут"
    elif confidence >= 50:
        expiry = "3 минуты"
    else:
        expiry = "1–2 минуты (осторожно, слабый сигнал)"
    st.write(f"⏱ Рекомендуемое время экспирации: **{expiry}**")
# ---------- Уведомление ----------
if "last_signal" not in st.session_state:
    st.session_state["last_signal"] = ""

def notify_with_popup(pair, signal, conf):
    color = "#2ecc71" if signal == "BUY" else "#e74c3c"
    sound_url = "https://actions.google.com/sounds/v1/alarms/beep_short.ogg"
    js = f"""
    <script>
        var audio = new Audio("{sound_url}");
        audio.play();
        if (navigator.vibrate) navigator.vibrate(400);
        let popup = document.createElement("div");
        popup.style.position = "fixed";
        popup.style.bottom = "30px";
        popup.style.left = "50%";
        popup.style.transform = "translateX(-50%)";
        popup.style.background = "{color}";
        popup.style.color = "white";
        popup.style.padding = "14px 20px";
        popup.style.borderRadius = "10px";
        popup.style.fontSize = "18px";
        popup.style.boxShadow = "0 4px 10px rgba(0,0,0,0.3)";
        popup.innerHTML = "🔥 {signal} сигнал — {pair} ({conf}%)";
        document.body.appendChild(popup);
        setTimeout(()=>popup.remove(), 4000);
    </script>
    """
    st.markdown(js, unsafe_allow_html=True)

# ---------- Вывод ----------
if best:
    color = "#2ecc71" if best["Signal"] == "BUY" else "#e74c3c"
    emoji = "🟢" if best["Signal"] == "BUY" else "🔴"
    st.markdown(
        f"""
        <div style='border:2px solid {color};padding:15px;border-radius:10px;background:#161a25'>
        <h3 style='color:{color}'>{emoji} Лучший сигнал: {best['Pair']} — {best['Signal']} ({best['Confidence']}%)</h3>
        <p>Цена: {best['Price']} | RSI: {best['RSI']} | Волатильность: {best['Volatility']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    key = f"{best['Pair']}_{best['Signal']}"
    if best["Confidence"] >= 70 and st.session_state["last_signal"] != key:
        notify_with_popup(best["Pair"], best["Signal"], best["Confidence"])
        st.session_state["last_signal"] = key

    df, sig = signals_data[best["Pair"]]
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color='#00ffcc', decreasing_line_color='#ff4b4b'
    )])
    fig.add_trace(go.Scatter(x=df.index, y=sig["ema_fast"], line=dict(color='yellow', width=1), name='EMA 9'))
    fig.add_trace(go.Scatter(x=df.index, y=sig["ema_slow"], line=dict(color='orange', width=1), name='EMA 21'))
    fig.add_trace(go.Scatter(x=df.index, y=sig["upper"], line=dict(color='gray', width=0.5, dash='dot'), name='Bollinger Top'))
    fig.add_trace(go.Scatter(x=df.index, y=sig["lower"], line=dict(color='gray', width=0.5, dash='dot'), name='Bollinger Bottom'))
    fig.update_layout(template='plotly_dark', height=400, margin=dict(l=10, r=10, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Нет активных сигналов (данные симулируются).")

st.divider()
st.subheader("📊 Все пары по уверенности:")
st.dataframe(table.sort_values("Confidence", ascending=False).reset_index(drop=True), use_container_width=True)
st.caption(f"Последнее обновление: {pd.Timestamp.utcnow().strftime('%H:%M:%S')} UTC")
# 🔔 Уведомление при сильном сигнале
import streamlit.components.v1 as components

# Если есть активный сигнал с уверенностью выше 70%
if not best_signal.empty and best_signal['Confidence'].iloc[0] > 70:
    signal_type = best_signal['Signal'].iloc[0]
    pair = best_signal['Pair'].iloc[0]
    confidence = best_signal['Confidence'].iloc[0]
    st.success(f"🚀 Сильный сигнал: {pair} — {signal_type} ({confidence:.1f}%)")

    # Встроенное уведомление браузера + звук
    components.html("""
        <script>
        const audio = new Audio('https://actions.google.com/sounds/v1/alarms/beep_short.ogg');
        audio.play();
        if (Notification.permission !== "granted") {
            Notification.requestPermission();
        }
        new Notification("AI FX Panel", {
            body: "Сильный сигнал на рынке! Проверь панель 👇",
            icon: "https://em-content.zobj.net/source/microsoft-teams/337/chart-increasing_1f4c8.png"
        });
        if (navigator.vibrate) {
            navigator.vibrate([200, 100, 200]);
        }
        </script>
    """, height=0)
# ✅ Безошибочная версия с рекомендацией по экспирации

if "best_signal" in locals() and not best_signal.empty:
    pair = best_signal["Pair"].iloc[0]
    signal_type = best_signal["Signal"].iloc[0]
    confidence = best_signal["Confidence"].iloc[0]
    price = best_signal["Price"].iloc[0]
    rsi = best_signal["RSI"].iloc[0]

    # Определение рекомендуемой экспирации
    if confidence >= 85:
        expiry = "10 минут"
    elif confidence >= 70:
        expiry = "5 минут"
    else:
        expiry = "2–3 минуты"

    st.markdown(f"""
    <div style='background-color:#330000;padding:15px;border-radius:12px;border:2px solid #ff4444;'>
    <h3 style='color:#ff4444;'>🔴 Лучший сигнал: {pair} — {signal_type}</h3>
    <p style='color:white;'>📊 Уверенность: <b>{confidence:.1f}%</b><br>
    💰 Цена: {price:.5f} | RSI: {rsi:.1f}<br>
    ⏱ Рекомендуемая экспирация: <b>{expiry}</b></p>
    </div>
    """, unsafe_allow_html=True)
