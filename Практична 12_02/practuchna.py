import cv2
import numpy as np
import yt_dlp
from ultralytics import YOLO
import time
import pandas as pd

# 1. Налаштування
YOUTUBE_URL = "https://www.youtube.com/watch?v=Lxqcg1qt0XU"
MODEL_PATH = '../yolov8n.pt'
DISTANCE_METERS = 20

# Координати двох ліній (як на вашому малюнку жовтим)
# Формат: [x1, y1, x2, y2]
LINE_1 = [700, 280, 1075, 325]  # Верхня лінія (біля переходу)
LINE_2 = [100, 450, 860, 670]  # Нижня лінія


def is_crossing_line(point, line):
    """Перевірка, чи знаходиться точка нижче або вище лінії (спрощено для горизонтальних ліній)"""
    # Для похилих ліній використовуємо перевірку положення точки відносно прямої
    x, y = point
    x1, y1, x2, y2 = line
    # Обчислюємо положення точки через рівняння прямої
    v = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    return v > 0


# Отримання стріму
ydl_opts = {'format': 'best', 'quiet': True}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(YOUTUBE_URL, download=False)
    stream_url = info['url']

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(stream_url)

track_data = {}
total_cars_crossed = 0
speeds_list = []
log_entries = []
csv_filename = f"traffic_report_{time.strftime('%Y%m%d_%H%M%S')}.csv"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (1280, 720))
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    current_time = time.time()

    # Малюємо лінії
    cv2.line(frame, (LINE_1[0], LINE_1[1]), (LINE_1[2], LINE_1[3]), (0, 255, 255), 3)  # Жовта лінія 1
    cv2.line(frame, (LINE_2[0], LINE_2[1]), (LINE_2[2], LINE_2[3]), (0, 255, 255), 3)  # Жовта лінія 2

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, clss):
            x1, y1, x2, y2 = box
            center_p = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            class_name = model.names[cls]

            if class_name in ['car', 'bus', 'truck']:
                # Логіка перетину ліній
                crossed_l1 = is_crossing_line(center_p, LINE_1)
                crossed_l2 = is_crossing_line(center_p, LINE_2)

                if track_id not in track_data:
                    track_data[track_id] = {'l1': crossed_l1, 'l2': crossed_l2, 'start_time': None, 'counted': False}

                # Якщо машина перетнула першу лінію (зверху вниз)
                if crossed_l1 and not track_data[track_id]['l1']:
                    track_data[track_id]['l1'] = True
                    track_data[track_id]['start_time'] = current_time

                # Якщо машина вже перетнула Л1 і тепер перетинає Л2
                if track_data[track_id]['l1'] and crossed_l2 and not track_data[track_id]['counted']:
                    if track_data[track_id]['start_time'] is not None:
                        duration = current_time - track_data[track_id]['start_time']
                        if duration > 0.5:
                            speed_kmh = (DISTANCE_METERS / duration) * 0.36
                            track_data[track_id]['speed'] = speed_kmh
                            track_data[track_id]['counted'] = True  # Цей прапорець не дає рахувати машину двічі

                            total_cars_crossed += 1
                            speeds_list.append(speed_kmh)

                            log_entries.append({
                                'ID': track_id, 'Type': class_name,
                                'Speed_KMH': round(speed_kmh, 2),
                                'Timestamp': time.strftime('%H:%M:%S')
                            })
                            pd.DataFrame(log_entries).to_csv(csv_filename, index=False)

                # Візуалізація швидкості
                display_speed = ""
                if track_id in track_data and 'speed' in track_data[track_id]:
                    display_speed = f"{int(track_data[track_id]['speed'])} km/h"

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} ID:{track_id} {display_speed}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Статистика
    avg_speed = np.mean(speeds_list) if speeds_list else 0
    cv2.putText(frame, f"Total Cars: {total_cars_crossed}", (30, 50), 0, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Avg Speed: {avg_speed:.1f} km/h", (30, 90), 0, 1, (0, 0, 255), 2)

    cv2.imshow("YouTube Traffic Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()