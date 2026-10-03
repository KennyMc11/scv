import cv2
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l')  # Легковесная модель для real-time
app.prepare(ctx_id=0, det_size=(640, 480))  # ctx_id=-1 для CPU

def process_frame(frame):
    """Обработка кадра и отправка координат на фронтенд"""
    # InsightFace принимает BGR (формат OpenCV)
    faces = app.get(frame)

    detections = []
    for face in faces:
        # Координаты рамки [x1, y1, x2, y2]
        bbox = face.bbox.astype(int).tolist()
        # Опционально: ключевые точки для более точной отрисовки
        kps = face.kps.astype(int).tolist()
        gender = int(face.gender)
        age = int(face.age)

        detections.append({
            'bbox': bbox,          # [x1, y1, x2, y2]
            'kps': kps,            # 5 точек (глаза, нос, углы рта)
            'gender': gender,
            'age': age,
            'score': float(face.det_score)  # Уверенность детекции
        })
        
    return detections