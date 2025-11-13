import streamlit as st
import statistics
from collections import deque

# ============================
#   LUCKYJET ANALYZER MODULE
# ============================

class LuckyJetAnalyzer:
    def __init__(self, max_history=200):
        self.history = deque(maxlen=max_history)

    def add_multiplier(self, value):
        """Добавляет множитель, гарантируя что это float"""
        try:
            x = float(value)
        except:
            return  # игнорируем некорректный ввод
        self.history.append(x)

    def clean_history(self):
        """Удаляет все некорректные значения"""
        cleaned = deque(maxlen=self.history.maxlen)
        for i in self.history:
            try:
                cleaned.append(float(i))
            except:
                pass
        self.history = cleaned

    def get_stats(self):
        if len(self.history) == 0:
            return None

        self.clean_history()

        avg = statistics.mean(self.history)
        low = len([i for i in self.history if i < 1.5])
        mid = len([i for i in self.history if 1.5 <= i < 3])
        high = len([i for i in self.history if i >= 3])

        return {
            "count": len(self.history),
            "average": round(avg, 2),
            "low_runs": low,
            "mid_runs": mid,
            "high_runs": high,
            "last": self.history[-1]
        }

    def get_signal(self):
        if len(self.history) < 5:
            return "Мало данных для анализа."

        self.clean_history()

        last_values = list(self.history)[-5:]

        last_values = [i for i in last_values if isinstance(i, (int, float))]

        if len(last_values) < 5:
            return "Недостаточно корректных данных."

        low_series = sum(1 for i in last_values if i < 1.5)
        if low_series >= 4:
            return "⚠ Серия низких коэффициентов — шанс высокого × выше среднего."

        last = last_values[-1]

        if last > 5:
            return "⚠ Последний × был высоким — следующий может быть низким."

        if 1.5 <= last <= 3:
            return "🟢 Стабильная зона — риск средний."

        if last < 1.2:
            return "🟠 Очень низкий множитель — шанс среднего увеличен."

        return "🟣 Нет явного сигнала."


# ============================
#   STREAMLIT INTERFACE
# ============================

st.title("🟣 LuckyJet Analyzer — AI Panel")

# создаём объект анализатора
if "lj" not in st.session_state:
    st.session_state.lj = LuckyJetAnalyzer()

lj = st.session_state.lj

st.subheader("Добавить множитель")
new_value = st.text_input("Введите коэффициент (например: 1.42, 17.15):")

if st.button("Добавить"):
    lj.add_multiplier(new_value)
    st.success("Добавлено!")

st.subheader("История множителей")
st.write(list(lj.history))

stats = lj.get_stats()
if stats:
    st.subheader("📊 Статистика")
    st.write(stats)

st.subheader("📡 Сигнал")
st.write(lj.get_signal())
