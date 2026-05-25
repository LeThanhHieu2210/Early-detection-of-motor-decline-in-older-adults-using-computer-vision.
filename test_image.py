import cv2
import mediapipe as mp
from ultralytics import YOLO

# Khởi tạo các model
yolo_model = YOLO('model_optimizefps.pt')
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Cấu hình MediaPipe Pose
pose_estimator = mp_pose.Pose(
    static_image_mode=True, 
    min_detection_confidence=0.5,
    model_complexity=1 # Có thể tăng lên 2 nếu cần độ chính xác cao hơn
)

# Đọc ảnh đầu vào
image_path = "C:/Users/LE THANH HIEU/Downloads/model/Person VOC.v1-roboflow-instant-1--eval-.yolo26/valid/images/standing_8_11_23-19-_jpg.rf.65f5542520dce94c49b65f74a9619029.jpg"
image = cv2.imread(image_path)
height, width, _ = image.shape

# YOLOv8 yêu cầu ảnh RGB hoặc BGR đều được, nhưng thư viện chuẩn xử lý tốt hơn với RGB
# Tiến hành dự đoán với YOLO
results = yolo_model(image)

for result in results:
    boxes = result.boxes
    for box in boxes:
        # 1. Lấy tọa độ Bounding Box và ép kiểu về số nguyên
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Đảm bảo tọa độ không vượt quá kích thước ảnh gốc
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        # Bỏ qua nếu khung cắt quá nhỏ (tránh lỗi do nhiễu)
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue

        # 2. Crop Person (Cắt người ra khỏi ảnh)
        cropped_person = image[y1:y2, x1:x2]
        
        # MediaPipe yêu cầu ảnh đầu vào hệ màu RGB
        cropped_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)

        # 3. Chạy MediaPipe Pose trên vùng ảnh đã crop
        pose_results = pose_estimator.process(cropped_rgb)

        # 4. Draw Skeleton (Vẽ khung xương)
        if pose_results.pose_landmarks:
            # Vẽ các điểm mốc và đường nối trực tiếp lên ảnh đã crop
            mp_drawing.draw_landmarks(
                cropped_person, 
                pose_results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # Điểm
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)  # Đường nối
            )
            
            # 5. Cập nhật lại vùng ảnh đã vẽ vào ảnh gốc
            image[y1:y2, x1:x2] = cropped_person
            
            # (Tùy chọn) Vẽ thêm khung YOLO để dễ quan sát
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)



# Hiển thị kết quả cuối cùng
cv2.imshow('YOLO + MediaPipe Pose Estimation', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Giải phóng bộ nhớ của MediaPipe
pose_estimator.close()