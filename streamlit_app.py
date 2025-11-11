import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# ---------- НАСТРОЙКИ ----------
REFRESH_SEC = 1  # 🔁 Обновление каждую секунду
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

# ---------- СИГНАЛ ----------
def make_signal(df):
    close = df["Close"]
    ema_fast = close.ewm(span=9, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()
    macd_line, signal_line = macd(close)
    rsi_val = rsi(close)
    upper, lower = bollinger(close)

    last = close.iloc[-1]
    r = rsi_val.iloc[-1]
    macd_now = macd_line.iloc[-1]
    macd_prev = macd_line.iloc[-2]
    sig_now = signal_line.iloc[-1]
    sig_prev = signal_line.iloc[-2]

    if ema_fast.iloc[-1] > ema_slow.iloc[-1] and r > 55 and macd_now > sig_now and macd_prev < sig_prev and last < upper.iloc[-1]:
        signal = "BUY"
    elif ema_fast.iloc[-1] < ema_slow.iloc[-1] and r < 45 and macd_now < sig_now and macd_prev > sig_prev and last > lower.iloc[-1]:
        signal = "SELL"
    else:
        signal = "WAIT"

    conf = 0
    if signal != "WAIT":
        conf += abs(macd_now - sig_now) * 100
        conf += abs(r - 50)
        conf += (abs(ema_fast.iloc[-1] - ema_slow.iloc[-1]) / last) * 10000
    conf = float(np.clip(conf, 0, 100))

    return {"signal": signal, "confidence": round(conf, 1), "price": float(last), "rsi": round(r, 1)}

# ---------- ЗАГРУЗКА ----------
@st.cache_data(ttl=1)
def load_pair(ticker):
    try:
        df = yf.download(ticker, period=f"{LOOKBACK_MIN+10}m", interval=INTERVAL, progress=False, auto_adjust=True)
        if len(df) > MIN_BARS:
            return df
    except:
        return None
    return None

# ---------- UI ----------
st.set_page_config(page_title="AI FX Ultra Panel 1s", layout="wide")
st.title("⚡ AI FX Ultra Panel (обновление 1 сек) 🔔📳💬")
st.caption("RSI + MACD + EMA + Bollinger · Обновление каждую секунду · Вибрация + звук + pop-up уведомление")

# Автообновление каждую секунду
st.markdown(f"<script>setTimeout(()=>window.location.reload(), {REFRESH_SEC*1000});</script>", unsafe_allow_html=True)

# ---------- Анализ всех валют ----------
rows = []
for name, ticker in PAIRS.items():
    df = load_pair(ticker)
    if df is None:
        rows.append({"Pair": name, "Signal": "—", "Confidence": 0.0, "Price": None, "RSI": None})
        continue
    sig = make_signal(df)
    rows.append({
        "Pair": name,
        "Signal": sig["signal"],
        "Confidence": sig["confidence"],
        "Price": sig["price"],
        "RSI": sig["rsi"],
    })

table = pd.DataFrame(rows)
candidates = table[table["Signal"].isin(["BUY", "SELL"])]
best = None if candidates.empty else candidates.sort_values("Confidence", ascending=False).iloc[0].to_dict()

# ---------- УВЕДОМЛЕНИЯ ----------
if "last_signal" not in st.session_state:
    st.session_state["last_signal"] = ""

def notify_with_popup(pair, signal, conf):
    color = "green" if signal == "BUY" else "red"
    sound_url = "https://actions.google.com/sounds/v1/alarms/beep_short.ogg"
    js = f"""
    <script>
        var audio = new Audio("{sound_url}");
        audio.play();
        if (navigator.vibrate) navigator.vibrate(500);
        let popup = document.createElement("div");
        popup.style.position = "fixed";
        popup.style.bottom = "30px";
        popup.style.left = "50%";
        popup.style.transform = "translateX(-50%)";
        popup.style.background = "{color}";
        popup.style.color = "white";
        popup.style.padding = "16px 24px";
        popup.style.borderRadius = "12px";
        popup.style.fontSize = "18px";
        popup.style.boxShadow = "0 4px 10px rgba(0,0,0,0.3)";
        popup.innerHTML = "🔥 {signal} сигнал — {pair} ({conf}%)";
        document.body.appendChild(popup);
        setTimeout(()=>popup.remove(), 3000);
    </script>
    """
    st.markdown(js, unsafe_allow_html=True)

# ---------- ВЫВОД ----------
if best:
    color = "green" if best["Signal"] == "BUY" else "red"
    emoji = "🟢" if best["Signal"] == "BUY" else "🔴"
    st.markdown(
        f"""
        <div style='border:2px solid {color};padding:15px;border-radius:10px'>
        <h3>{emoji} Лучший сигнал: {best['Pair']} — {best['Signal']} ({best['Confidence']}%)</h3>
        Цена: {best['Price']} | RSI: {best['RSI']}
        </div>
        """,
        unsafe_allow_html=True
    )

    key = f"{best['Pair']}_{best['Signal']}"
    if best["Confidence"] >= 70 and st.session_state["last_signal"] != key:
        notify_with_popup(best["Pair"], best["Signal"], best["Confidence"])
        st.session_state["last_signal"] = key
else:
    st.warning("Нет активных сигналов.")

st.divider()
st.subheader("📊 Все пары по уверенности:")
st.dataframe(table.sort_values("Confidence", ascending=False).reset_index(drop=True), use_container_width=True)
st.caption(f"Последнее обновление: {pd.Timestamp.utcnow().strftime('%H:%M:%S')} UTC")
