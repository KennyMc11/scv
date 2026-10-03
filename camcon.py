from dvrip import DVRIPCam

class CamStream:
    def __init__(self, ip="178.72.91.183", user="tavr9k4",
                 password="9k4", port=34568):
        self.cam = DVRIPCam(ip, user=user, password=password, port=port)
        self._connected = False

    def open(self):
        if not self.cam.login():
            raise ConnectionError("Не удалось подключиться к камере")
        self._connected = True
        return self

    def get_frame(self):
        if not self._connected:
            raise RuntimeError("Камера не подключена")
        return self.cam.snapshot()   # bytes

    def close(self):
        if self._connected:
            self.cam.close()
            self._connected = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


# --- Использование ---
"""
with CamStream() as s:  
    data = s.get_frame()
"""