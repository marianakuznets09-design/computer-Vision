import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)

VIDEO_DIR = os.path.join(PROJECT_DIR, '')

OUT_DIR = os.path.join(PROJECT_DIR, 'out')

os.makedirs(OUT_DIR, exist_ok=True)

USE_WEBCAM = False

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)

else:
    VIDEO_PATH = os.path.join(VIDEO_DIR, 'zalikova/video.cars.mp4')
    print(VIDEO_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

model = YOLO('yolov8n.pt')

CONFIDENCE_THRESHOLD = 0.4

RESIZE_WIDTH = 960

prev_time = time.time()
fps = 0.0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]

        scale = RESIZE_WIDTH / w

        new_w = int(scale * w)
        new_h = int(scale * h)

        frame = cv2.resize(frame, (new_w, new_h))



    result = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    car_count = 0
    bus_count = 0
    motorcycle_count = 0
    truck_count = 0
    psevdo_id = 0


    CAR_CLASS_ID = 2
    BUS_CLASS_ID = 5
    TRUCK_CLASS_ID = 7
    MOTORCYCLE_CLASS_ID = 3
    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            classes = [CAR_CLASS_ID, BUS_CLASS_ID, MOTORCYCLE_CLASS_ID, TRUCK_CLASS_ID]

            if cls in classes:
                psevdo_id += 1

                name ="name"

                if cls == CAR_CLASS_ID:
                    car_count += 1
                    name = "car"
                if cls == BUS_CLASS_ID:
                    bus_count += 1
                    name = "bus"
                if cls == MOTORCYCLE_CLASS_ID:
                    motorcycle_count += 1
                    name = "motorcycle"
                if cls == TRUCK_CLASS_ID:
                    truck_count += 1
                    name = "truck"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # label = f'ID: {psevdo_id} conf {conf:.2f}'
                # cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

                now = time.time()
                dt = now - prev_time
                prev_time = now

                if dt > 0:
                    fps = 1.0 / dt


                cv2.putText(frame, f'{name}', (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 255, 0),2)
                cv2.putText(frame, f'Car count: {car_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 0, 0), 1)
                cv2.putText(frame, f'Bus count: {bus_count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
                cv2.putText(frame, f'Truck count: {truck_count}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

                cv2.putText(frame, f'Motorcycle count: {motorcycle_count}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
    final_image_path = os.path.join(OUT_DIR, "final_result.jpg")
    cv2.imwrite(final_image_path, frame)

    # cap.release()
    cv2.imshow('YOLO', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()