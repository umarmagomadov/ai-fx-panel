# ============================
#   LUCKYJET ANALYZER MODULE
# ============================

import statistics
from collections import deque

class LuckyJetAnalyzer:
    def __init__(self, max_history=100):
        self.history = deque(maxlen=max_history)

    def add_multiplier(self, x):
        """Добавить новый множитель после окончания раунда"""
        try:
            x = float(x)
        except:
            return None
        self.history.append(x)

    def get_stats(self):
        """Возвращает статистические данные"""
        if len(self.history) == 0:
            return None

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
        """Выдаёт сигналы на основе статистики (НЕ прогноз)"""

        if len(self.history) < 5:
            return "Мало данных…"

        last = self.history[-1]

        # Серия низких множителей
        low_series = sum(1 for i in self.history[-5:] if i < 1.5)
        if low_series >= 4:
            return "⚠ Возможен высокий множитель (по серии низких)."

        # После высокого обычно идёт низкий
        if last > 5:
            return "⚠ Последний выстрел был высокий: сейчас повышенный риск низкого."

        # Стабильная зона
        if 1.5 <= last <= 3:
            return "🟢 Стабильная зона. Риск умеренный."

        # Очень низкий множитель
        if last < 1.2:
            return "🟠 Очень низкий множитель: возможен средний."

        return "🟣 Нет чёткого сигнала."


# ============================
#     ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================

if __name__ == "__main__":
    lj = LuckyJetAnalyzer()

    # добавляем историю
    for x in [1.24, 1.12, 1.45, 2.1, 3.5, 1.03, 1.11]:
        lj.add_multiplier(x)

    print(lj.get_stats())
    print(lj.get_signal())
