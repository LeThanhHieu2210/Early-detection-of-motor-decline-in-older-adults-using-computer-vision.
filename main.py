import cv2
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
import numpy as np

# 1. Khởi tạo Models
yolo_model = YOLO('model_optimizefps.pt')
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose_estimator = mp_pose.Pose(
    static_image_mode=False, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# 2. Biến theo dõi lịch sử để tính vận tốc
# Lưu tọa độ Y của Hông trong 10 frame gần nhất. 
hip_y_history = deque(maxlen=10)
fall_alert_timer = 0 # Dùng để giữ cảnh báo ngã trên màn hình một lúc

# 3. Mở Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w, _ = frame.shape
    results = yolo_model(frame, verbose=False)
    
    # Tìm bounding box lớn nhất (người chính trong khung hình) để phân tích
    largest_box = None
    max_area = 0
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_box = (x1, y1, x2, y2)

    # Nếu tìm thấy người
    if largest_box:
        x1, y1, x2, y2 = largest_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)
        
        box_w, box_h = x2 - x1, y2 - y1
        
        if box_w > 20 and box_h > 20:
            # Crop và chạy MediaPipe
            cropped_person = frame[y1:y2, x1:x2]
            cropped_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
            pose_results = pose_estimator.process(cropped_rgb)

            state = "Khong xac dinh"
            box_color = (255, 255, 0) # Mặc định Vàng

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                
                # --- CHUYỂN ĐỔI TỌA ĐỘ TỪ CROP RA ẢNH GỐC ---
                # Lấy Y của Mũi
                nose_y = y1 + int(landmarks[mp_pose.PoseLandmark.NOSE.value].y * box_h)
                
                # Lấy Y của Hông (Trung bình Hông trái và phải)
                hip_left_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                hip_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                hip_y = y1 + int(((hip_left_y + hip_right_y) / 2) * box_h)
                
                # Lấy Y của Mắt cá chân
                ankle_left_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                ankle_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                ankle_y = y1 + int(((ankle_left_y + ankle_right_y) / 2) * box_h)

                # --- LƯU LỊCH SỬ VÀ TÍNH VẬN TỐC ---
                hip_y_history.append(hip_y)
                
                # Tốc độ rơi: Y hiện tại trừ đi Y ở frame cũ nhất trong lịch sử
                # (Trục Y hướng xuống, nên Y tăng nghĩa là đang rơi)
                fall_speed = hip_y - hip_y_history[0] if len(hip_y_history) == 10 else 0

                # --- LOGIC PHÂN LOẠI TRẠNG THÁI ---
                aspect_ratio = box_w / box_h
                
                # 1. Kiểm tra Lying (Nằm)
                # Nằm khi chiều rộng lớn hơn chiều cao, hoặc Mũi/Hông/Chân ngang nhau
                if aspect_ratio > 1.2 or abs(nose_y - ankle_y) < (frame_h * 0.2):
                    state = "Nam (Lying)"
                    box_color = (255, 165, 0) # Cam
                    
                    # 2. Kích hoạt Té Ngã (Fall) nếu Tốc độ rơi nhanh + Đang ở tư thế nằm
                    # Bạn có thể tinh chỉnh số 50 (pixels) này tùy vào độ phân giải camera
                    if fall_speed > 50: 
                        fall_alert_timer = 30 # Giữ cảnh báo trong 30 frames
                        
                # 3. Kiểm tra Ngồi (Sitting)
                # Ngồi khi khoảng cách từ Hông đến Chân ngắn lại so với lúc đứng
                elif (ankle_y - hip_y) < (box_h * 0.45):
                    state = "Ngoi (Sitting)"
                    box_color = (0, 255, 255) # Vàng
                    
                # 4. Đứng (Standing)
                else:
                    state = "Dung (Standing)"
                    box_color = (0, 255, 0) # Xanh lá

                # Vẽ Khung xương lên ảnh crop và ghép lại
                mp_drawing.draw_landmarks(
                    cropped_person, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
                frame[y1:y2, x1:x2] = cropped_person

            # Vẽ Bounding Box và Trạng thái
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, state, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    # --- XỬ LÝ GIAO DIỆN CẢNH BÁO TÉ NGÃ ---
    if fall_alert_timer > 0:
        cv2.putText(frame, "CANH BAO: TE NGA!", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        cv2.rectangle(frame, (0, 0), (frame_w, frame_h), (0, 0, 255), 15)
        fall_alert_timer -= 1 # Trừ dần thời gian cảnh báo

    # Hiển thị
    cv2.imshow('Fall Detection System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose_estimator.close()