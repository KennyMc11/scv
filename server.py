# server.py
import os
import sys
import ctypes
from pathlib import Path

# --- Загрузка NVRTC ДО onnxruntime ---
site_packages = Path(sys.prefix) / "Lib" / "site-packages"
nvrtc_bin = site_packages / "nvidia" / "cuda_nvrtc" / "bin"

if nvrtc_bin.is_dir():
    os.environ["PATH"] = str(nvrtc_bin) + os.pathsep + os.environ["PATH"]
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(nvrtc_bin))

    for dll in sorted(nvrtc_bin.glob("nvrtc*.dll")):
        try:
            ctypes.CDLL(str(dll))
            print(f"✓ {dll.name}")
        except OSError as e:
            print(f"✗ {dll.name}: {e}")
else:
    print(f"⚠ NVRTC не найден: {nvrtc_bin}")

# --- Только теперь импортируем onnxruntime ---
import onnxruntime as ort

import cv2
import asyncio
import websockets
import base64
import os
import site
from insightface.app import FaceAnalysis

ort.set_default_logger_severity(0)

# GPU или CPU
#app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# Только GPU
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


def visual(frame):
    faces = app.get(frame)
    # Рамка с точками от insightface
    #app.draw_on(frame, faces)

    # Кастомная рамка
    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # подпись
        if face.gender == 1:
            face_gender = "M"
        else:
            face_gender = "Ж"
        label = f"{face.det_score:.2f}, Age {face.age}, {face_gender}"
        cv2.putText(frame, label, (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def read_frames(cap: cv2.VideoCapture, width=640, height=480, quality=70):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр. Поток мог завершиться.")
            break

        frame1 = visual(frame)

        #frame = cv2.resize(frame, (width, height))
        ok, buffer = cv2.imencode('.jpg', frame1, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            continue

        data = base64.b64encode(buffer).decode('utf-8')
        yield f"data:image/jpeg;base64,{data}"


async def stream(websocket):
    """WebSocket-обработчик: гоняет кадры из генератора клиенту."""
    #cap = cv2.VideoCapture("rtsp://localhost:8554/my_camera")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await websocket.send("error: не удалось открыть камеру")
        return

    try:
        for payload in read_frames(cap):
            await websocket.send(payload)
            await asyncio.sleep(0.03)  # ~30 FPS
    except websockets.ConnectionClosed:
        pass
    finally:
        cap.release()


async def main():
    async with websockets.serve(stream, "0.0.0.0", 8765):
        print("WS сервер на ws://0.0.0.0:8765")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())