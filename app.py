import atexit
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, send_from_directory
from ultralytics import YOLO


class KalmanFilter:
    """
    Bộ lọc Kalman với mô hình vận tốc không đổi (Constant Velocity) cho trục dọc (y).
    Trạng thái: X = [y, v_y]^T (vị trí y và vận tốc theo y, đơn vị: pixel / bước khung hình).
    """

    def __init__(self, dt: float = 1.0):
        self.dt = float(dt)
        # A: ma trận chuyển trạng thái (mô hình CV rời rạc)
        self.A = np.array([[1.0, self.dt], [0.0, 1.0]], dtype=np.float64)
        # H: ma trận quan sát — chỉ đo được vị trí y (không đo trực tiếp v_y)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        # Q: hiệp phương sai nhiễu quá trình
        self.Q = np.array([[4.0, 0.0], [0.0, 9.0]], dtype=np.float64)
        # R: hiệp phương sai nhiễu đo (pixel^2)
        self.R = np.array([[36.0]], dtype=np.float64)
        # P: ma trận hiệp phương sai lỗi ước lượng trạng thái
        self.P = np.eye(2, dtype=np.float64) * 500.0
        self.x = np.zeros((2, 1), dtype=np.float64)
        self._initialized = False

    def predict(self) -> None:
        """Bước dự đoán (time update): x^- = A x, P^- = A P A^T + Q."""
        if not self._initialized:
            return
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, measurement: float) -> None:
        """Bước cập nhật (measurement update) với đo y từ cảm biến (MediaPipe)."""
        z = np.array([[float(measurement)]], dtype=np.float64)
        if not self._initialized:
            self.x = np.array([[float(measurement)], [0.0]], dtype=np.float64)
            self.P = np.eye(2, dtype=np.float64) * 1000.0
            self._initialized = True
            return
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        innovation = z - self.H @ self.x
        self.x = self.x + K @ innovation
        I = np.eye(2, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

    def reset(self) -> None:
        """Đặt lại bộ lọc khi mất đối tượng theo dõi (tránh kéo trạng thái cũ)."""
        self._initialized = False
        self.x.fill(0.0)
        self.P = np.eye(2, dtype=np.float64) * 500.0

    @property
    def y(self) -> float:
        """Vị trí y ước lượng sau lọc."""
        return float(self.x[0, 0])

    @property
    def v_y(self) -> float:
        """Vận tốc v_y ước lượng (pixel / bước khung)."""
        return float(self.x[1, 0])


@dataclass(frozen=True)
class DetectorStatus:
    label: str
    level: str  # "normal" | "warning" | "alert"
    fall_detected: bool
    person_detected: bool
    posture: str
    updated_at: float


class FrameGrabber:
    """
    Owns cv2.VideoCapture and continuously grabs frames on a background thread.
    This avoids blocking Flask request threads and prevents multiple consumers
    from creating duplicate camera handles.
    """

    def __init__(self, src: str | int = 0, width: int | None = None, height: int | None = None):
        self.src = int(src) if isinstance(src, str) and src.isdigit() else src
        self.width = width
        self.height = height

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._last_ok_ts: float = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._open_capture()
        self._thread = threading.Thread(target=self._loop, name="frame-grabber", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._release_capture()

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def _open_capture(self) -> None:
        self._release_capture()

        # On Windows, CAP_DSHOW often reduces camera open latency.
        if isinstance(self.src, int):
            cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.src)

        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self._cap = cap

    def _release_capture(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def _loop(self) -> None:
        backoff_s = 0.2
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                time.sleep(backoff_s)
                self._open_capture()
                backoff_s = min(2.0, backoff_s * 1.4)
                continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            with self._lock:
                self._frame = frame
                self._last_ok_ts = time.time()

            backoff_s = 0.2


class FallDetector:
    """
    Refactored from main.py:
    - YOLO finds the primary person (largest box)
    - MediaPipe Pose draws skeletal landmarks on that crop
    - A simple temporal rule uses hip Y history + posture to trigger a fall alert
    """

    def __init__(self, yolo_weights_path: str):
        self.yolo_model = YOLO(yolo_weights_path)

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_estimator = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

        self.hip_y_history: deque[int] = deque(maxlen=10)
        self.fall_alert_timer: int = 0

        # Khởi tạo một lần: bộ lọc Kalman cho tọa độ y của hông (không tạo lại mỗi khung hình).
        self.hip_y_kalman = KalmanFilter(dt=1.0)
        # v_y sau Kalman — gán mỗi khung khi có pose; logic phát hiện ngã có thể dùng thêm sau.
        self.hip_v_y_smoothed: float = 0.0

        self._status_lock = threading.Lock()
        self._status = DetectorStatus(
            label="Trạng thái: Đang khởi tạo",
            level="warning",
            fall_detected=False,
            person_detected=False,
            posture="Chưa phát hiện",
            updated_at=time.time(),
        )

    def close(self) -> None:
        try:
            self.pose_estimator.close()
        except Exception:
            pass

    def get_status(self) -> DetectorStatus:
        with self._status_lock:
            return self._status

    def _set_status(self, status: DetectorStatus) -> None:
        with self._status_lock:
            self._status = status

    def process(self, frame: np.ndarray) -> np.ndarray:
        frame_h, frame_w, _ = frame.shape
        person_detected = False
        posture_ui = "Chưa phát hiện"

        # Run YOLO and select the largest detection (primary person).
        results = self.yolo_model(frame, verbose=False)
        largest_box = None
        max_area = 0
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_box = (x1, y1, x2, y2)

        state_label = "Khong phat hien nguoi"
        box_color = (120, 120, 120)

        if largest_box:
            x1, y1, x2, y2 = largest_box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)

            box_w, box_h = x2 - x1, y2 - y1
            if box_w > 20 and box_h > 20:
                person_detected = True
                cropped_person = frame[y1:y2, x1:x2]
                cropped_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
                pose_results = self.pose_estimator.process(cropped_rgb)

                state_label = "Khong ro"
                posture_ui = "Chưa rõ"
                box_color = (255, 255, 0)

                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark

                    nose_y = y1 + int(
                        landmarks[self.mp_pose.PoseLandmark.NOSE.value].y * box_h
                    )

                    hip_left_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
                    hip_right_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y
                    # Đo y thô của điểm hông (pixel, toàn khung) — làm đo đầu vào cho Kalman
                    hip_y_measured = y1 + ((hip_left_y + hip_right_y) / 2.0) * box_h
                    # Bước 1: dự đoán (predict) trạng thái [y, v_y] theo mô hình vận tốc không đổi
                    self.hip_y_kalman.predict()
                    # Bước 2: cập nhật (update) với đo y từ MediaPipe
                    self.hip_y_kalman.update(float(hip_y_measured))
                    # Bước 3: lấy y đã làm mịn và v_y — gán vào biến dùng cho lịch sử hông / tốc độ / tư thế ngồi
                    hip_y = int(round(self.hip_y_kalman.y))
                    self.hip_v_y_smoothed = self.hip_y_kalman.v_y

                    ankle_left_y = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                    ankle_right_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                    ankle_y = y1 + int(((ankle_left_y + ankle_right_y) / 2) * box_h)

                    self.hip_y_history.append(hip_y)
                    fall_speed = (
                        hip_y - self.hip_y_history[0] if len(self.hip_y_history) == 10 else 0
                    )

                    aspect_ratio = box_w / max(1, box_h)

                    # Posture logic (kept equivalent to main.py).
                    is_lying = aspect_ratio > 1.2 or abs(nose_y - ankle_y) < (frame_h * 0.2)
                    is_sitting = (ankle_y - hip_y) < (box_h * 0.45)

                    if is_lying:
                        state_label = "Nam"
                        posture_ui = "Nằm"
                        box_color = (255, 165, 0)
                        if fall_speed > 50:
                            self.fall_alert_timer = 30
                    elif is_sitting:
                        state_label = "Ngoi"
                        posture_ui = "Ngồi"
                        box_color = (0, 255, 255)
                    else:
                        state_label = "Dung"
                        posture_ui = "Đứng"
                        box_color = (0, 255, 0)

                    # Draw skeleton on crop, then place it back.
                    self.mp_drawing.draw_landmarks(
                        cropped_person,
                        pose_results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    )
                    frame[y1:y2, x1:x2] = cropped_person

            # Draw bounding box + state label.
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame,
                state_label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                box_color,
                2,
                cv2.LINE_AA,
            )
        else:
            self.hip_y_history.clear()
            self.fall_alert_timer = 0
            # Đặt lại Kalman khi không còn người — tránh “kéo” trạng thái sang lần xuất hiện sau
            self.hip_y_kalman.reset()

        # Fall alert overlay (timer persists across frames).
        if self.fall_alert_timer > 0:
            cv2.putText(
                frame,
                "CANH BAO: PHAT HIEN TE NGA!",
                (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                4,
                cv2.LINE_AA,
            )
            cv2.rectangle(frame, (0, 0), (frame_w, frame_h), (0, 0, 255), 12)
            self.fall_alert_timer -= 1

        # Update status for the UI.
        now = time.time()
        if not person_detected:
            self._set_status(
                DetectorStatus(
                    label="Trạng thái: Chưa phát hiện người",
                    level="warning",
                    fall_detected=False,
                    person_detected=False,
                    posture=posture_ui,
                    updated_at=now,
                )
            )
        elif self.fall_alert_timer > 0:
            self._set_status(
                DetectorStatus(
                    label="Trạng thái: Đã phát hiện",
                    level="alert",
                    fall_detected=True,
                    person_detected=True,
                    posture=posture_ui,
                    updated_at=now,
                )
            )
        else:
            self._set_status(
                DetectorStatus(
                    label="Trạng thái: Đã phát hiện",
                    level="normal",
                    fall_detected=False,
                    person_detected=True,
                    posture=posture_ui,
                    updated_at=now,
                )
            )

        return frame


def _encode_jpeg(frame_bgr: np.ndarray) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return buf.tobytes()


def create_app() -> Flask:
    app = Flask(__name__)

    weights_path = os.environ.get("YOLO_WEIGHTS", "model_optimizefps.pt")
    video_source = os.environ.get("VIDEO_SOURCE", "0")
    width = int(os.environ.get("VIDEO_WIDTH", "0")) or None
    height = int(os.environ.get("VIDEO_HEIGHT", "0")) or None
    demo_mode = os.environ.get("DEMO_MODE", "0") == "1"
    app_started_at = time.time()

    grabber = FrameGrabber(video_source, width=width, height=height)
    detector = FallDetector(weights_path)

    grabber.start()

    @atexit.register
    def _cleanup() -> None:
        grabber.stop()
        detector.close()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/sound.mp3")
    def sound_file():
        return send_from_directory(app.root_path, "sound.mp3")

    @app.get("/status")
    def status():
        if demo_mode:
            # Demo flow for UI testing without camera input:
            # wait 5s, then run through a short sequence of states once.
            t = time.time() - app_started_at
            if t < 5:
                s = DetectorStatus(
                    label="Trạng thái: Chưa phát hiện người",
                    level="warning",
                    fall_detected=False,
                    person_detected=False,
                    posture="Chưa phát hiện",
                    updated_at=time.time(),
                )
            else:
                timeline = [
                    # duration, status
                    (
                        2.0,
                        DetectorStatus(
                            label="Trạng thái: Chưa phát hiện người",
                            level="warning",
                            fall_detected=False,
                            person_detected=False,
                            posture="Chưa phát hiện",
                            updated_at=time.time(),
                        ),
                    ),
                    (
                        2.0,
                        DetectorStatus(
                            label="Trạng thái: Đã phát hiện",
                            level="normal",
                            fall_detected=False,
                            person_detected=True,
                            posture="Đứng",
                            updated_at=time.time(),
                        ),
                    ),
                    (
                        2.0,
                        DetectorStatus(
                            label="Trạng thái: Đã phát hiện",
                            level="normal",
                            fall_detected=False,
                            person_detected=True,
                            posture="Ngồi",
                            updated_at=time.time(),
                        ),
                    ),
                    (
                        2.0,
                        DetectorStatus(
                            label="Trạng thái: Đã phát hiện",
                            level="normal",
                            fall_detected=False,
                            person_detected=True,
                            posture="Nằm",
                            updated_at=time.time(),
                        ),
                    ),
                    (
                        2.0,
                        DetectorStatus(
                            label="Trạng thái: Đã phát hiện",
                            level="alert",
                            fall_detected=True,
                            person_detected=True,
                            posture="Nằm",
                            updated_at=time.time(),
                        ),
                    ),
                    (
                        2.0,
                        DetectorStatus(
                            label="Trạng thái: Đã phát hiện",
                            level="normal",
                            fall_detected=False,
                            person_detected=True,
                            posture="Đứng",
                            updated_at=time.time(),
                        ),
                    ),
                ]

                phase_t = t - 5
                total_duration = sum(duration_s for duration_s, _ in timeline)
                if total_duration > 0:
                    phase_t = phase_t % total_duration
                selected = None
                for duration_s, status_obj in timeline:
                    if phase_t < duration_s:
                        selected = status_obj
                        break
                    phase_t -= duration_s

                # Fallback safety (should rarely happen).
                if selected is None:
                    selected = DetectorStatus(
                        label="Trạng thái: Chưa phát hiện người",
                        level="warning",
                        fall_detected=False,
                        person_detected=False,
                        posture="Chưa phát hiện",
                        updated_at=time.time(),
                    )

                s = DetectorStatus(
                    label=selected.label,
                    level=selected.level,
                    fall_detected=selected.fall_detected,
                    person_detected=selected.person_detected,
                    posture=selected.posture,
                    updated_at=time.time(),
                )
        else:
            s = detector.get_status()
        return jsonify(
            {
                "label": s.label,
                "level": s.level,
                "fall_detected": s.fall_detected,
                "person_detected": s.person_detected,
                "posture": s.posture,
                "updated_at": s.updated_at,
            }
        )

    @app.get("/video_feed")
    def video_feed():
        def gen():
            while True:
                frame = grabber.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                processed = detector.process(frame)
                jpg = _encode_jpeg(processed)
                if jpg is None:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


app = create_app()


if __name__ == "__main__":
    # For LAN testing, you can set: FLASK_RUN_HOST=0.0.0.0 or just edit below.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), threaded=True, debug=False)
