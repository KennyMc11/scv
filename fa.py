import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Инициализация модели (один раз при старте программы)
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))


def cosine_similarity(emb1, emb2):
    """Косинусная близость между двумя эмбеддингами."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def find_same_person(reference_path, folder_path, threshold=0.5, verbose=True):
    """
    Сравнивает эталонное фото с фото из папки и возвращает те, где тот же человек.

    :param reference_path: путь к эталонному фото
    :param folder_path: путь к папке с фото для сравнения
    :param threshold: порог косинусной близости (обычно 0.4–0.6)
    :param verbose: печатать ли прогресс
    :return: список кортежей (путь_к_файлу, схожесть)
    """
    # 1. Загружаем эталонное фото
    ref_img = cv2.imread(reference_path)
    if ref_img is None:
        raise FileNotFoundError(f"Не удалось открыть эталонное фото: {reference_path}")

    ref_faces = app.get(ref_img)
    if not ref_faces:
        raise ValueError(f"На эталонном фото не найдено лиц: {reference_path}")

    # Берём самое крупное лицо (по площади bbox) как эталон
    ref_face = max(ref_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    ref_emb = ref_face.normed_embedding

    if verbose:
        print(f"Эталон: {reference_path} — найдено лиц: {len(ref_faces)}")

    # 2. Проходим по всем фото в папке
    matches = []
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(extensions):
            continue

        filepath = os.path.join(folder_path, filename)

        img = cv2.imread(filepath)
        if img is None:
            if verbose:
                print(f"  [skip] не удалось открыть: {filename}")
            continue

        faces = app.get(img)
        if not faces:
            if verbose:
                print(f"  [skip] лиц не найдено: {filename}")
            continue

        # Сравниваем с каждым лицом на фото, берём максимальную схожесть
        best_sim = -1.0
        for face in faces:
            sim = cosine_similarity(ref_emb, face.normed_embedding)
            if sim > best_sim:
                best_sim = sim

        is_match = best_sim >= threshold
        if verbose:
            status = "MATCH" if is_match else "     "
            print(f"  [{status}] {filename}: схожесть = {best_sim:.4f}")

        if is_match:
            matches.append((filepath, float(best_sim)))

    # 3. Сортируем по убыванию схожести
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

if __name__ == "__main__":
    results = find_same_person(
        reference_path="photos/i.jpg",
        folder_path="photos/",
        threshold=0.5,
        verbose=True
    )

    print("\n=== Найденные совпадения ===")
    for path, sim in results:
        print(f"{path}  →  схожесть: {sim:.4f}")