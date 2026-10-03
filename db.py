import sqlite3
import os
from typing import Optional

DB_PATH = "app.db"

def get_conn(path: str = DB_PATH) -> sqlite3.Connection:
    """Подключение к БД с включёнными внешними ключами."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn



def create_db(db_path: str = DB_PATH) -> None:
    # Если файл уже есть — не перезаписываем (убери проверку, если хочешь всегда заново)
    if os.path.exists(db_path):
        print(f"⚠️  Файл {db_path} уже существует. Удаляю и создаю заново...")
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Включаем поддержку внешних ключей
    cur.execute("PRAGMA foreign_keys = ON;")

    # Таблица пользователей
    cur.execute("""
        CREATE TABLE users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            id_tg         INTEGER,
            first_name    TEXT,
            last_name     TEXT,
            username      TEXT,
            phone_number  TEXT,
            telegram_link TEXT
        );
    """)

    # Таблица фотографий
    cur.execute("""
        CREATE TABLE photos (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB NOT NULL,
            path     TEXT NOT NULL,
            user_id  INTEGER NOT NULL,
            FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE
        );
    """)

    # Индекс
    cur.execute("CREATE INDEX idx_photos_user_id ON photos(user_id);")

    conn.commit()
    conn.close()

    print(f"✅ База данных создана: {db_path}")


# ---------- Пользователи ----------

def add_user(
    id_tg,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    username: Optional[str] = None,
    phone_number: Optional[str] = None,
    telegram_link: Optional[str] = None,
) -> int:
    """Добавляет пользователя и возвращает его id."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO users (id_tg, first_name, last_name, username, phone_number, telegram_link)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (id_tg, first_name, last_name, username, phone_number, telegram_link),
        )
        return cur.lastrowid

# ---------- Фото ----------

def add_photo(path: str, user_id: int, embedding) -> int:
    """Добавляет фото и возвращает его id."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO photos (path, user_id, embedding) VALUES (?, ?, ?)",
            (path, user_id, embedding),
        )
        return cur.lastrowid