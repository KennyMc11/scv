import sqlite3
import os
from typing import Optional
from db import add_user, add_photo, get_all_photo, delete_user, get_user, get_photo
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from fa import find_same_emb

# Инициализация модели (один раз при старте программы)
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

DB_PATH = "app.db"

img = cv2.imread("photos/i.jpg")
faces = app.get(img)
for face in faces:
    u = face.normed_embedding
    c = find_same_emb(u)

def get_conn(path: str = DB_PATH) -> sqlite3.Connection:
    """Подключение к БД с включёнными внешними ключами."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

"""
add_user(222, "Юлия", "Мальцева", "Julia", "899999999", "t.me",)
for face in faces:
    add_photo("photos/y.jpg", 3, face.normed_embedding)
"""
v = get_photo(c)

print(v)
