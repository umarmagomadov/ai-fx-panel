import streamlit as st
from collections import deque
import statistics

# ============================
#   LuckyJet Analyzer (AI)
# ============================

class LuckyJetAnalyzer:
    def __init__(self, max_history=200):
        self.history = deque(maxlen=max_history)

    def add_multiplier(self, value):
        try:
            x = float(value)
        except:
            return
        self.history.append(x)

    def clean_history(self):
        cleaned = deque(maxlen=self.history.maxlen)
        for i in self.history:
            try:
                cleaned.append(float(i))
            except:
                pass
        self.history = cleaned

    def get_signal_advanced(self):
        """AI-стиль сигналов как у интернет-ботов"""
        if len(self.history) < 6:
            return "⚪ Недостаточно данных"

        self.clean_history()

        last5 = list(self.history)[-5:]
        last = last5[-1]

        low_count = sum(1 for x in last5 if x < 1.5)
        high_count = sum(1 for x in last5 if x > 3)

        # 1 — серия низких → шанс высокого ↑
        if low_count >= 4:
            return "🟩 СТАВИТЬ — серия низких, шанс высокого выше среднего"

        # 2 — был высокий → обычно затем низкий
        if last > 5:
            return "🟥 НЕ СТАВИТЬ — только что был высокий"

        # 3 — серия хаотичная → осторожно
        if high_count >= 2:
            return "🟥 НЕ СТАВИТЬ — хаотичная серия"

        # 4 — нормальная стабильная зона
        if 1.4 <= last <= 3:
            return "🟧 ОСТОРОЖНО — зона средней волатильности"

        # 5 — очень низкий множитель → возможен средний
        if last < 1.2:
            return "🟩 СТАВИТЬ — возможен средний множитель"

        return "🟧 ОСТОРОЖНО — нет чёткого паттерна"


# ======================================
#   STREAMLIT UI (интерфейс приложения)
# ======================================

st.title("🟣 LuckyJet — AI Signal Panel")

# создаём объект анализатора в сессии
if "lj" not in st.session_state:
    st.session_state.lj = LuckyJetAnalyzer()

lj = st.session_state.lj

st.subheader("Введите множитель")
value = st.text_input("Пример: 1.42, 17.15")

if st.button("Добавить"):
    lj.add_multiplier(value)
    st.success("Добавлено!")

# показываем историю
st.subheader("📜 История")
st.write(list(lj.history))

# показываем сигнал
st.subheader("📡 AI Сигнал")
signal = lj.get_signal_advanced()
st.write(signal)
