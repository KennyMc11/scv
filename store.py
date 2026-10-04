import time
import threading

# ---------- Хранилище распознанных лиц за сессию ----------

class KnownFacesStore:
    """Хранилище распознанных лиц за сессию.
    """

    def __init__(self, max_items: int = 10, ttl_sec: float = 300.0):
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}
        self.max_items = max_items
        self.ttl = ttl_sec

    @staticmethod
    def _key(f: dict) -> str:
        # Основной ключ — id_tg
        if f.get("id") is not None:
            return f"tg_{f['id']}"
        # Fallback — username (стабилен между кадрами)
        if f.get("id_tg") is not None:
            return f"tg_{f['id_tg']}"
        if f.get("username"):
            return f"un_{f['username']}"
        # Крайний случай — ФИО
        name = f"{f.get('last_name','')}|{f.get('first_name','')}"
        return f"nm_{name}"

    def update(self, known_faces: list[dict]) -> None:
        now_ms = time.time() * 1000.0
        with self._lock:
            for f in known_faces:
                key = self._key(f)
                prev = self._items.get(key)
                if prev is None:
                    self._items[key] = {
                        **f,
                        "firstSeen": now_ms,
                        "lastSeen": now_ms,
                    }
                else:
                    # обновляем данные (bbox, score, age могут меняться),
                    # но firstSeen сохраняем
                    prev.update(f)
                    prev["lastSeen"] = now_ms

            self._evict_locked(now_ms)

    def _evict_locked(self, now_ms: float) -> None:
        # 1) удаляем устаревшие по TTL
        stale = [k for k, v in self._items.items()
                 if now_ms - v.get("lastSeen", 0) > self.ttl * 1000.0]
        for k in stale:
            self._items.pop(k, None)

        # 2) если всё ещё больше лимита — режем самые старые
        if len(self._items) > self.max_items:
            ordered = sorted(
                self._items.items(),
                key=lambda kv: kv[1].get("lastSeen", 0),
                reverse=True,
            )
            keep = dict(ordered[: self.max_items])
            self._items = keep

    def snapshot(self) -> list[dict]:
        now_ms = time.time() * 1000.0
        with self._lock:
            self._evict_locked(now_ms)
            return sorted(
                self._items.values(),
                key=lambda v: v.get("lastSeen", 0),
                reverse=True,
            )
