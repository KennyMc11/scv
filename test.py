import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from camcon import CamStream

# Инициализация модели (один раз при старте программы)
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

# 2. Загрузка изображения
img = cv2.imread('photos/a.jpg')


# 3. Анализ
faces = app.get(img)
ref_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
ref_emb = ref_face.normed_embedding
x = [int(face.age) for face in faces]
print(x)

# 5. Визуализация
result_img = app.draw_on(img, faces)
cv2.imwrite('result.jpg', result_img)