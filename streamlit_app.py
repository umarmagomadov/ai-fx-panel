import os
from datetime import datetime, date

import requests
import streamlit as st
import pandas as pd

# ========= КЛЮЧИ =========
API_FOOTBALL_KEY = st.secrets.get("API_FOOTBALL_KEY", os.getenv("API_FOOTBALL_KEY", ""))
NEWS_API_KEY     = st.secrets.get("NEWS_API_KEY",     os.getenv("NEWS_API_KEY", ""))

API_URL   = "https://v3.football.api-sports.io"
NEWS_URL  = "https://newsapi.org/v2/everything"

# ========= ЛИГИ =========
LEAGUES = {
    "Premier League": 39,
    "La Liga":        140,
    "Serie A":        135,
    "Bundesliga":     78,
    "Ligue 1":        61,
}

# ========= ХЕЛПЕРЫ =========
def has_api() -> bool:
    return bool(API_FOOTBALL_KEY)

def get_session():
    s = requests.Session()
    s.headers.update({"x-apisports-key": API_FOOTBALL_KEY})
    return s

def get_fixtures(session, league_id: int):
    today = date.today().isoformat()
    try:
        r = session.get(
            f"{API_URL}/fixtures",
            params={"league": league_id,
                    "season": datetime.now().year,
                    "date": today},
            timeout=10,
        )
        return r.json().get("response", [])
    except Exception:
        return []

def get_standings(session, league_id: int):
    try:
        r = session.get(
            f"{API_URL}/standings",
            params={"league": league_id,
                    "season": datetime.now().year},
            timeout=10,
        )
        data = r.json()
        return data["response"][0]["league"]["standings"][0]
    except Exception:
        return []

def get_news():
    """Простые новости про футбол (если есть NEWS_API_KEY)."""
    if not NEWS_API_KEY:
        return []
    try:
        params = {
            "q": "football OR soccer",
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 10,
            "apiKey": NEWS_API_KEY,
        }
        r = requests.get(NEWS_URL, params=params, timeout=10)
        return r.json().get("articles", [])
    except Exception:
        return []


# ========= UI =========
st.set_page_config(page_title="Football Center Live", layout="wide")
st.title("⚽ Football Center Live")

st.markdown(
    "Приложение показывает **матчи дня, таблицы лиг и новости футбола**. "
    "Никаких ставок, только информация и статистика. "
    "Хочешь — смотри, анализируй, делись с друзьями."
)

# ---------- Матчи сегодня ----------
st.header("📅 Матчи сегодня")

if not has_api():
    st.error("Не найден API_FOOTBALL_KEY. Добавь ключ от API-Football в Secrets.")
    st.stop()

session = get_session()
all_matches = []

for league_name, league_id in LEAGUES.items():
    fixtures = get_fixtures(session, league_id)
    if not fixtures:
        continue

    for f in fixtures:
        fx = f["fixture"]
        status = fx["status"]["short"]
        # уже сыгранные пропускаем
        if status in {"FT", "AET", "PEN", "CANC", "ABD", "PST"}:
            continue

        home = f["teams"]["home"]["name"]
        away = f["teams"]["away"]["name"]

        kickoff = datetime.fromisoformat(
            fx["date"].replace("Z", "+00:00")
        ).strftime("%H:%M")

        all_matches.append({
            "Лига": league_name,
            "Хозяева": home,
            "Гости": away,
            "Время (UTC)": kickoff,
        })

if all_matches:
    df_matches = pd.DataFrame(all_matches)
    df_matches = df_matches.sort_values(["Лига", "Время (UTC)"]).reset_index(drop=True)
    st.dataframe(df_matches, use_container_width=True, height=360)
else:
    st.info("Сегодня нет матчей по выбранным лигам или нет данных.")

# ---------- Простые “интересные матчи” ----------
st.header("🔥 Популярные матчи дня (по названиям)")

popular = []
hot_keywords = ["real", "barca", "barselona", "barcelona",
                "chelsea", "arsenal", "milan", "inter",
                "psg", "city", "liverpool", "bayern"]

for m in all_matches:
    name = f"{m['Хозяева']} {m['Гости']}".lower()
    if any(k in name for k in hot_keywords):
        popular.append(m)

if popular:
    st.success("Матчи с топ-клубами (по простому фильтру названий):")
    st.dataframe(pd.DataFrame(popular), use_container_width=True, height=220)
else:
    st.info("Сегодня нет ярко выраженных топ-матчей по названиям команд.")

# ---------- Таблицы лиг ----------
st.header("🏆 Таблицы топ-лиг")

league_choice = st.selectbox("Выбери лигу", list(LEAGUES.keys()))
table = get_standings(session, LEAGUES[league_choice])

rows = []
for pos in table:
    rows.append({
        "Поз.": pos["rank"],
        "Команда": pos["team"]["name"],
        "И": pos["all"]["played"],
        "Голы": f"{pos['all']['goals']['for']} : {pos['all']['goals']['against']}",
        "Очки": pos["points"],
    })

df_table = pd.DataFrame(rows)
st.dataframe(df_table, use_container_width=True, height=360)

# ---------- Новости ----------
st.header("📰 Новости футбола")

articles = get_news()
if articles:
    for art in articles:
        st.subheader(art.get("title", "Без заголовка"))
        if art.get("description"):
            st.write(art["description"])
        if art.get("source", {}).get("name"):
            st.caption(f"Источник: {art['source']['name']}")
        if art.get("url"):
            st.write(f"[Читать подробнее]({art['url']})")
        st.write("---")
else:
    st.info("Для новостей нужен NEWS_API_KEY (NewsAPI.org). Можно оставить пустым, тогда блок новостей скрывает реальные статьи.")
