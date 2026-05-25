import cv2
import mediapipe as mp
from ultralytics import YOLO

# Khởi tạo models
yolo_model = YOLO('model_optimizefps.pt')
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Lưu ý: static_image_mode=False để tối ưu cho luồng video
pose_estimator = mp_pose.Pose(
    static_image_mode=False, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Mở webcam (số 0 là camera mặc định)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc từ camera.")
        break

    height, width, _ = frame.shape

    # Chạy YOLO. verbose=False để terminal không bị spam log liên tục
    results = yolo_model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue

            cropped_person = frame[y1:y2, x1:x2]
            cropped_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
            pose_results = pose_estimator.process(cropped_rgb)

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    cropped_person, 
                    pose_results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
                
                frame[y1:y2, x1:x2] = cropped_person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Hiển thị FPS hoặc kết quả lên màn hình
    cv2.imshow('Real-time Pose Estimation', frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp tài nguyên
cap.release()
cv2.destroyAllWindows()
pose_estimator.close()