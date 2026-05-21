import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = "data/chatbot.db"


def init_db():
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            user_message TEXT,
            bot_response TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_message(user_id, user_message, bot_response):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO conversations (user_id, user_message, bot_response, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, user_message, bot_response, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_recent_history(user_id, limit=5):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT user_message, bot_response
        FROM conversations
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    rows.reverse()
    return rows
