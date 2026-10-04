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

import onnxruntime as ort
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
)
import cv2
import asyncio
import websockets
import base64
import json
from insightface.app import FaceAnalysis
from fa import find_same_emb
from db import get_user, get_photo

ort.set_default_logger_severity(0)

# GPU или CPU
#app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# Только GPU
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


def draw_faces(frame, faces):
    """Рисует рамки и подписи поверх кадра."""
    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        gender = "M" if face.gender == 1 else "Ж"
        label = f"{face.det_score:.2f}, Age {face.age}, {gender}"
        cv2.putText(
            frame, label, (box[0], box[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
    return frame


def face_to_dict(face):
    """Базовые поля лица (общие для known/unknown)."""
    return {
        "bbox": face.bbox.astype(int).tolist(),
        "gender": "M" if face.gender == 1 else "Ж",
        "age": int(face.age),
        "score": float(face.det_score),
    }


def process_frame(frame):
    """Синхронная обработка кадра: детекция + распознавание.
    Возвращает (jpeg_base64, faces_known, faces_unknown)."""
    faces = app.get(frame)
    draw_faces(frame, faces)

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return None, [], []

    image_b64 = base64.b64encode(buffer).decode("utf-8")

    known, unknown = [], []
    for face in faces:
        base = face_to_dict(face)
        match_user_id = find_same_emb(face.normed_embedding)
        if match_user_id is None:
            unknown.append(base)
            continue

        # Известное лицо: обогащаем данными из БД
        user_rows = get_user(match_user_id) or []
        photo_path = get_photo(match_user_id)
        for u in user_rows:
            known.append({
                **base,
                "id_tg": u["id_tg"],
                "first_name": u["first_name"],
                "last_name": u["last_name"],
                "username": u["username"],
                "phone_number": u["phone_number"],
                "telegram_link": u["telegram_link"],
                "photo": photo_path,
            })

    return image_b64, known, unknown


async def stream(websocket):
    """WebSocket-обработчик."""
    cap = cv2.VideoCapture("rtsp://localhost:8554/my_camera", cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    #cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        await websocket.send(json.dumps({
            "type": "error",
            "msg": "не удалось открыть камеру",
        }))
        return

    try:
        while True:
            # Блокирующее чтение + обработку уводим в отдельный поток,
            # чтобы не блокировать event loop.
            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                await websocket.send(json.dumps({
                    "type": "error",
                    "msg": "поток завершился",
                }))
                break

            image_b64, known, unknown = await asyncio.to_thread(process_frame, frame)

            if image_b64 is not None:
                await websocket.send(json.dumps({
                    "type": "frame",
                    "image": f"data:image/jpeg;base64,{image_b64}",
                    "faces_known": known,
                    "faces_unknown": unknown,
                }))

            # Небольшая пауза, чтобы не забивать сеть
            # await asyncio.sleep(0.01)
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