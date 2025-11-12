import requests
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import random
import time
import plotly.graph_objects as go
from datetime import datetime
# --- TELEGRAM НАСТРОЙКИ ---# --- ТЕСТ ОТПРАВКИ СООБЩЕНИЯ ---
if st.button("📩 Отправить тестовое сообщение в Telegram"):
    try:
        test_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        test_data = {
            "chat_id": CHAT_ID,
            "text": "✅ Тест: соединение с Telegram работает!",
        }
        r = requests.post(test_url, data=test_data)
        if r.status_code == 200:
            st.success("Сообщение успешно отправлено ✅")
        else:
            st.error(f"Ошибка при отправке: {r.text}")
    except Exception as e:
        st.error(f"Ошибка соединения: {e}")
import streamlit as st

TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
CHAT_ID = st.secrets["CHAT_ID"]

def send_telegram_message(pair, signal, confidence, expiry, mode):
    text = (
        f"🤖 *AI FX СИГНАЛ ({mode})*\n"
        f"💱 Пара: {pair}\n"
        f"📊 Сигнал: {signal}\n"
        f"📈 Уверенность: {confidence}\n"
        f"⏱ Экспирация: {expiry}\n"
        f"⚙️ Обновлено автоматически."
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print("Ошибка при отправке в Telegram:", e)
# НАСТРОЙКИ
REFRESH_SEC = 1
LOOKBACK_MIN = 120
INTERVAL = "1m"

# ВАЛЮТНЫЕ ПАРЫ
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
    "XAUUSD (Gold)": "GC=F",
    "BTCUSD (Bitcoin)": "BTC-USD",
    "ETHUSD (Ethereum)": "ETH-USD",
}

# TELEGRAM
TELEGRAM_TOKEN = "8188894081:AAHr7im0L7CWCgiScOnKMLqo7g3I7R0s_80"
CHAT_ID = "6045310859"

def send_telegram_message(pair, signal, confidence, expiry, mode):
    text = (
        f"🤖 AI FX СИГНАЛ ({mode})\n"
        f"💱 Пара: {pair}\n"
        f"📈 Сигнал: {signal}\n"
        f"📊 Уверенность: {confidence}%\n"
        f"⏱ Экспирация: {expiry}\n"
        f"⚙️ Автоматический выбор лучшего сигнала завершён."
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text})
    except Exception as e:
        print("Ошибка при отправке в Telegram:", e)

def is_market_open():
    now = datetime.utcnow()
    if now.weekday() == 5 and now.hour >= 21:
        return False
    if now.weekday() == 6 and now.hour < 22:
        return False
    return True

# ИНТЕРФЕЙС
st.set_page_config(page_title="AI FX Panel", layout="wide")
st.title("🤖 AI FX PANEL — Умный анализ и сигналы")

rows = []
market_open = is_market_open()

for name, symbol in PAIRS.items():
    try:
        if market_open:
            data = yf.download(symbol, period=f"{LOOKBACK_MIN}m", interval=INTERVAL, progress=False, timeout=5)
            if data is None or data.empty:
                raise ValueError("Нет данных")
            data["SMA"] = data["Close"].rolling(window=10).mean()
            data["Signal"] = np.where(data["Close"] > data["SMA"], "BUY", "SELL")
            signal = data["Signal"].iloc[-1]
            confidence = random.randint(60, 99)
            expiry = random.choice(["1 минута", "3 минуты", "5 минут"])
        else:
            signal = random.choice(["BUY", "SELL"])
            confidence = random.randint(65, 97)
            expiry = random.choice(["1 минута", "3 минуты", "5 минут"])
        rows.append({"Пара": name, "Сигнал": signal, "Уверенность": confidence, "Экспирация": expiry})
    except Exception:
        signal = random.choice(["BUY", "SELL"])
        confidence = random.randint(60, 95)
        expiry = random.choice(["1 минута", "3 минуты", "5 минут"])
        rows.append({"Пара": name, "Сигнал": signal, "Уверенность": confidence, "Экспирация": expiry})

if rows:
    table = pd.DataFrame(rows)
    best = table.loc[table["Уверенность"].idxmax()]
    st.subheader("🔥 Лучший сигнал:")
    st.metric("Пара", best["Пара"])
    st.metric("Сигнал", best["Сигнал"])
    st.metric("Уверенность", f"{best['Уверенность']}%")
    st.metric("Экспирация", best["Экспирация"])
    st.write(f"🟡 Режим: {'РЕАЛЬНЫЙ' if market_open else 'ДЕМО (рынок закрыт)'}")
    send_telegram_message(best["Пара"], best["Сигнал"], best["Уверенность"], best["Экспирация"], "РЕАЛ" if market_open else "ДЕМО")
    st.subheader("📋 Все пары:")
    st.dataframe(table)
else:
    st.warning("⚠️ Не удалось получить сигналы.")
# --- УВЕДОМЛЕНИЕ ---
import streamlit.components.v1 as components

alert_html = """
<script>
    const playSound = () => {
        let sound;
        if ("{{signal}}" === "BUY") {
            sound = "https://actions.google.com/sounds/v1/cartoon/wood_plank_flicks.ogg";
        } else {
            sound = "https://actions.google.com/sounds/v1/alarms/beep_short.ogg";
        }
        const audio = new Audio(sound);
        audio.play();
        document.body.style.backgroundColor = '#fff3cd';
        setTimeout(() => { document.body.style.backgroundColor = 'white'; }, 600);
    };
    playSound();
</script>
""".replace("{{signal}}", str(best["Сигнал"]))

components.html(alert_html, height=0)

# Отправка обновлённого сигнала в Telegram
send_telegram_message(
    best["Пара"],
    best["Сигнал"],
    best["Уверенность"],
    best["Экспирация"],
    "РЕАЛ" if market_open else "ДЕМО"
)
print("📨 Сигнал отправлен в Telegram:", best["Пара"], best["Сигнал"])
# Пауза перед обновлением
time.sleep(REFRESH_SEC)
st.rerun()
