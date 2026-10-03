import cv2

# Укажите ваш RTSP-адрес
rtsp_url = "rtsp://localhost:8554/my_camera"
# Создаем объект захвата видео
cap = cv2.VideoCapture(rtsp_url)

# Проверяем, удалось ли подключиться
if not cap.isOpened():
    print("Ошибка: Не удалось подключиться к потоку")
    exit()

while True:
    # Читаем кадр из потока
    ret, frame = cap.read()
    
    if not ret:
        print("Не удалось получить кадр. Поток мог завершиться.")
        break

    # Показываем кадр в окне
    cv2.imshow('RTSP Stream', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()