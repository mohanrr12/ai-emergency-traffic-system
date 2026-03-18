from ultralytics import YOLO
import cv2

# load model
model = YOLO("yolov8n.pt")

# open video
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # run detection
    results = model(frame)

    annotated_frame = results[0].plot()

    ambulance_detected = False

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            # simulate ambulance detection
            if label == "car":
                ambulance_detected = True

    # traffic signal logic
    if ambulance_detected:
        print("🚑 Emergency detected!")
        print("North Signal: GREEN")
        print("Others: RED")

    # show video
    cv2.imshow("Traffic Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
