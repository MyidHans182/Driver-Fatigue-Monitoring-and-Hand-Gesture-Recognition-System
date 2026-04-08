"""
Detect_Sleep.py  –  Hệ thống giám sát buồn ngủ tài xế + Mô phỏng lái xe.

Tính năng:
- EAR (Eye Aspect Ratio) cả 2 mắt – Soukupova & Cech 2016
- MAR (Mouth Aspect Ratio) detect ngáp
- Head Pose Estimation (pitch/yaw/roll) bằng solvePnP
- Angle-based hand gesture + temporal smoothing
- Driver Attention Score (composite 0-100)
- Tích hợp driving simulation qua pygame
- Data logging CSV
- Âm thanh qua pygame.mixer (multi-channel)
"""
import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector
import cvzone
import os
import math
import time as time_module
import collections
import csv
import datetime
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto

from cvzone.PlotModule import LivePlot
from driving_simulation import DrivingSimulation

# ── Cấu hình ──────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Camera
    CAMERA_INDEX: int = 0
    DISPLAY_W: int = 640
    DISPLAY_H: int = 360

    # EAR (Eye Aspect Ratio)
    EAR_THRESHOLD: float = 35.0        # dưới ngưỡng này = nhắm mắt
    EAR_SMOOTH_WINDOW: int = 5
    EYES_CLOSED_FRAMES: int = 3        # nhắm liên tục bao nhiêu frame → drowsy
    EYES_DANGER_FRAMES: int = 15       # nhắm rất lâu → danger

    # MAR (Mouth Aspect Ratio)
    MAR_THRESHOLD: float = 94.0
    MAR_SMOOTH_WINDOW: int = 5
    YAWN_ALERT_COUNT: int = 60         # tổng yawn frames → cảnh báo
    YAWN_DECAY_RATE: int = 30          # mỗi N frame không ngáp, giảm yawn 1

    # Hand gesture
    GESTURE_HOLD_FRAMES: int = 15
    GESTURE_CONFIDENCE: float = 0.7
    GESTURE_SMOOTH_WINDOW: int = 7

    # Blink tracking
    BLINK_ALERT_COUNT: int = 30        # số blink trong chu kỳ → cảnh báo
    BLINK_RESET_CYCLE: int = 900       # reset blink counter mỗi N frame

    # Head pose
    HEAD_PITCH_THRESHOLD: float = -15.0   # gật đầu
    HEAD_YAW_THRESHOLD: float = 30.0      # quay đầu
    HEAD_ROLL_THRESHOLD: float = 20.0     # nghiêng đầu

    # Attention score weights
    ATT_W_EYE: float = 0.35
    ATT_W_MOUTH: float = 0.20
    ATT_W_HEAD: float = 0.30
    ATT_W_BLINK: float = 0.15

    # Logging
    LOG_ENABLED: bool = True

CFG = Config()


# ── Enum trạng thái ───────────────────────────────────────────────────────────
class SystemState(Enum):
    IDLE = auto()
    MONITORING = auto()
    MUSIC = auto()

class EyeState(Enum):
    OPEN = auto()
    CLOSED = auto()


# ── Sound Manager (pygame.mixer – multi-channel) ─────────────────────────────
class SoundManager:
    """Quản lý âm thanh qua pygame.mixer – channel riêng cho alert vs music."""

    def __init__(self):
        try:
            import pygame.mixer
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self._alert_channel = pygame.mixer.Channel(0)
            self._music_channel = pygame.mixer.Channel(1)
            self._available = True
        except Exception as e:
            print(f"[SOUND] pygame.mixer init failed: {e}, falling back to silent")
            self._available = False

        self._sound_cache: dict = {}
        self.music_playing = False

    def _load(self, filename):
        """Load và cache sound file."""
        if not self._available or not filename:
            return None
        if filename in self._sound_cache:
            return self._sound_cache[filename]
        if os.path.exists(filename):
            import pygame.mixer
            snd = pygame.mixer.Sound(filename)
            self._sound_cache[filename] = snd
            return snd
        else:
            print(f"[WARNING] Sound not found: {filename}")
            return None

    def play_alert(self, filename):
        """Phát alert sound (channel 0) – không ảnh hưởng music."""
        if not self._available:
            return
        snd = self._load(filename)
        if snd:
            self._alert_channel.play(snd)

    def stop_alert(self):
        """Dừng alert sound – KHÔNG dừng music."""
        if self._available:
            self._alert_channel.stop()

    def play_music(self, filename):
        """Phát nhạc nền (channel 1) – loop."""
        if not self._available:
            return
        snd = self._load(filename)
        if snd:
            self._music_channel.play(snd, loops=-1)
            self.music_playing = True

    def stop_music(self):
        """Dừng nhạc nền."""
        if self._available:
            self._music_channel.stop()
        self.music_playing = False

    def stop_all(self):
        """Dừng tất cả."""
        self.stop_alert()
        self.stop_music()


# ── Head Pose Estimation ─────────────────────────────────────────────────────
class HeadPoseEstimator:
    """Ước lượng hướng đầu (pitch/yaw/roll) từ face mesh landmarks."""

    # 3D model points (generic face model, mm)
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (1)
        (0.0, -330.0, -65.0),        # Chin (199)
        (-225.0, 170.0, -135.0),     # Left eye left corner (33)
        (225.0, 170.0, -135.0),      # Right eye right corner (263)
        (-150.0, -150.0, -125.0),    # Left mouth corner (61)
        (150.0, -150.0, -125.0),     # Right mouth corner (291)
    ], dtype=np.float64)

    LANDMARK_IDS = [1, 199, 33, 263, 61, 291]

    def __init__(self, img_w=640, img_h=480):
        self.img_w = img_w
        self.img_h = img_h
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1))

        # Smoothing
        self._pitch_buf = collections.deque(maxlen=5)
        self._yaw_buf = collections.deque(maxlen=5)
        self._roll_buf = collections.deque(maxlen=5)

    def estimate(self, face_landmarks):
        """
        Tính pitch, yaw, roll từ face mesh landmarks.
        face_landmarks: list of (x, y) tuples từ FaceMeshDetector.
        Returns: (pitch, yaw, roll) in degrees, hoặc (0, 0, 0) nếu fail.
        """
        try:
            image_points = np.array([
                face_landmarks[lid] for lid in self.LANDMARK_IDS
            ], dtype=np.float64)

            # Chỉ lấy x, y (face mesh cho tuple (x, y))
            if len(image_points[0]) > 2:
                image_points = image_points[:, :2]

            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.MODEL_POINTS, image_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return 0.0, 0.0, 0.0

            # Rotation matrix → Euler angles
            rmat, _ = cv2.Rodrigues(rotation_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            pitch = angles[0]
            yaw = angles[1]
            roll = angles[2]

            # Smooth
            self._pitch_buf.append(pitch)
            self._yaw_buf.append(yaw)
            self._roll_buf.append(roll)

            smooth_pitch = sum(self._pitch_buf) / len(self._pitch_buf)
            smooth_yaw = sum(self._yaw_buf) / len(self._yaw_buf)
            smooth_roll = sum(self._roll_buf) / len(self._roll_buf)

            return smooth_pitch, smooth_yaw, smooth_roll

        except Exception:
            return 0.0, 0.0, 0.0


# ── Angle-based finger counting ──────────────────────────────────────────────
def _angle(a, b, c):
    """Tính góc tại điểm b (độ)."""
    ba = np.array(a[:2]) - np.array(b[:2])
    bc = np.array(c[:2]) - np.array(b[:2])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def count_fingers_angle(hand_data, angle_threshold=160):
    """Đếm ngón tay bằng góc giữa các đốt (ổn định hơn fingersUp)."""
    lm = hand_data["lmList"]
    fingers = []

    # Ngón cái
    thumb_angle = _angle(lm[1], lm[2], lm[3])
    thumb_tip_angle = _angle(lm[2], lm[3], lm[4])
    thumb_open = thumb_angle > 140 and thumb_tip_angle > 140
    thumb_tip_dist = np.linalg.norm(np.array(lm[4][:2]) - np.array(lm[2][:2]))
    thumb_base_dist = np.linalg.norm(np.array(lm[3][:2]) - np.array(lm[2][:2]))
    if thumb_base_dist > 0:
        thumb_extended = thumb_open and (thumb_tip_dist / thumb_base_dist > 1.2)
    else:
        thumb_extended = thumb_open
    fingers.append(1 if thumb_extended else 0)

    # 4 ngón còn lại
    finger_joints = [
        (5, 6, 7, 8), (9, 10, 11, 12),
        (13, 14, 15, 16), (17, 18, 19, 20),
    ]
    for mcp, pip_, dip, tip in finger_joints:
        angle_pip = _angle(lm[mcp], lm[pip_], lm[dip])
        angle_dip = _angle(lm[pip_], lm[dip], lm[tip])
        is_open = angle_pip > angle_threshold and angle_dip > angle_threshold
        fingers.append(1 if is_open else 0)

    return fingers


# ── Gesture Smoother ──────────────────────────────────────────────────────────
class GestureSmoother:
    """Majority vote + hold-to-activate."""

    def __init__(self, window_size=7, hold_frames=15):
        self.history = collections.deque(maxlen=window_size)
        self.hold_frames = hold_frames
        self.current_gesture = 0
        self.hold_counter = 0
        self.activated = False

    def update(self, raw_finger_count):
        """Trả về (stable_gesture, just_activated)."""
        self.history.append(raw_finger_count)
        if len(self.history) < 3:
            return 0, False

        counts = {}
        for g in self.history:
            counts[g] = counts.get(g, 0) + 1
        voted = max(counts, key=counts.get)

        if voted != self.current_gesture:
            self.current_gesture = voted
            self.hold_counter = 0
            self.activated = False
            return voted, False

        self.hold_counter += 1
        if self.hold_counter >= self.hold_frames and not self.activated:
            self.activated = True
            return voted, True

        return voted, False

    def reset(self):
        self.history.clear()
        self.current_gesture = 0
        self.hold_counter = 0
        self.activated = False


# ── Driver Attention Score ────────────────────────────────────────────────────
class AttentionScorer:
    """Tính attention score tổng hợp 0-100."""

    def __init__(self):
        self._score = 100.0
        self._blink_times = collections.deque(maxlen=60)

    def record_blink(self):
        """Ghi nhận 1 lần chớp mắt."""
        self._blink_times.append(time_module.time())

    def _blink_freq_score(self):
        """Tần suất chớp mắt bình thường = 15-20/phút → score 100.
        Quá nhiều (>25) hoặc quá ít (<10) → giảm."""
        now = time_module.time()
        recent = [t for t in self._blink_times if now - t < 60]
        bpm = len(recent)

        if 10 <= bpm <= 25:
            return 100.0
        elif bpm < 10:
            return max(30, 100 - (10 - bpm) * 7)
        else:
            return max(30, 100 - (bpm - 25) * 5)

    def compute(self, ear_avg, mar_avg, head_pitch, head_yaw, head_roll):
        """Tính attention score dựa trên EAR, MAR, head pose, blink freq."""
        eye_score = np.clip((ear_avg - 25) / 10 * 100, 0, 100)
        mouth_score = np.clip((94 - mar_avg) / 34 * 100, 0, 100)

        pitch_penalty = max(0, (-head_pitch - 10)) * 5
        yaw_penalty = max(0, abs(head_yaw) - 15) * 4
        roll_penalty = max(0, abs(head_roll) - 10) * 3
        head_score = np.clip(100 - pitch_penalty - yaw_penalty - roll_penalty, 0, 100)

        blink_score = self._blink_freq_score()

        raw = (CFG.ATT_W_EYE * eye_score +
               CFG.ATT_W_MOUTH * mouth_score +
               CFG.ATT_W_HEAD * head_score +
               CFG.ATT_W_BLINK * blink_score)

        alpha = 0.15
        self._score = alpha * raw + (1 - alpha) * self._score
        return self._score


# ── Data Logger ───────────────────────────────────────────────────────────────
class DataLogger:
    """Log metrics mỗi frame vào CSV."""

    HEADERS = [
        "timestamp", "ear_left", "ear_right", "ear_avg", "mar",
        "head_pitch", "head_yaw", "head_roll",
        "attention_score", "drowsy_level", "blink_count", "yawn_count",
        "vehicle_speed", "lane_offset", "system_state",
    ]

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._file = None
        self._writer = None
        if enabled:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{ts}.csv"
            self._file = open(filename, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(self.HEADERS)
            print(f"[LOG] Logging to {filename}")

    def log(self, **kwargs):
        if not self.enabled or not self._writer:
            return
        row = [kwargs.get(h, "") for h in self.HEADERS]
        row[0] = datetime.datetime.now().isoformat()
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()


# ── EAR helper (top-level, không define trong loop) ───────────────────────────
def calc_ear(face, eye_ids, face_detector):
    """Tính EAR cho 1 mắt. Trả về int (EAR * 100)."""
    top1 = face[eye_ids["top1"]]
    bot1 = face[eye_ids["bot1"]]
    top2 = face[eye_ids["top2"]]
    bot2 = face[eye_ids["bot2"]]
    left = face[eye_ids["left"]]
    right = face[eye_ids["right"]]
    v1, _ = face_detector.findDistance(top1, bot1)
    v2, _ = face_detector.findDistance(top2, bot2)
    h, _ = face_detector.findDistance(left, right)
    if h == 0:
        return 50
    return int(((v1 + v2) / (2.0 * h)) * 100)


# ── Hệ thống chính ───────────────────────────────────────────────────────────
def Start():
    # Khởi tạo detectors
    faceDetector = FaceMeshDetector(maxFaces=1)
    handDetector = HandDetector(detectionCon=CFG.GESTURE_CONFIDENCE, maxHands=1)

    cap = cv2.VideoCapture(CFG.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Không mở được camera!")
        return

    # Driving simulation
    sim = DrivingSimulation()

    # Sound manager (dùng pygame.mixer – multi-channel, check init trước)
    sound = SoundManager()

    # Head pose estimator
    head_pose = HeadPoseEstimator(640, 480)

    # Attention scorer
    attention = AttentionScorer()

    # Data logger
    logger = DataLogger(enabled=CFG.LOG_ENABLED)

    # Landmark IDs – EAR (Soukupova & Cech 2016)
    LEFT_EYE = {
        "top1": 159, "bot1": 145, "top2": 158, "bot2": 153,
        "left": 133, "right": 33
    }
    RIGHT_EYE = {
        "top1": 386, "bot1": 374, "top2": 385, "bot2": 380,
        "left": 362, "right": 263
    }
    idListEye = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243,
                 253, 254, 256, 339, 384, 385, 386, 387, 388, 359, 463]
    idListMouth = [185, 39, 37, 0, 267, 269, 325, 321, 404, 315, 16, 85, 180, 90]

    # Plots
    ploty_Eye = LivePlot(640, 360, [20, 50], invert=True)
    ploty_Mouth = LivePlot(640, 360, [30, 140], invert=True)
    ploty_Attention = LivePlot(640, 360, [0, 100], invert=True)

    # Trạng thái
    state = SystemState.IDLE
    eye_state = EyeState.OPEN
    prev_drowsy_level = 0          # theo dõi transition để không spam sound
    gesture_smoother = GestureSmoother(
        window_size=CFG.GESTURE_SMOOTH_WINDOW,
        hold_frames=CFG.GESTURE_HOLD_FRAMES
    )

    ratioList_Eye = collections.deque(maxlen=CFG.EAR_SMOOTH_WINDOW)
    ratioList_Mouth = collections.deque(maxlen=CFG.MAR_SMOOTH_WINDOW)

    # Counters
    blink_count = 0
    eyes_closed_frames = 0
    yawn_frames = 0
    yawn_no_yawn_counter = 0
    frame_count = 0

    # Drowsy level
    drowsy_level = 0

    # Current metrics (cho logging)
    cur_ear_left = 0.0
    cur_ear_right = 0.0
    cur_ear_avg = 0.0
    cur_mar = 0.0
    cur_pitch = 0.0
    cur_yaw = 0.0
    cur_roll = 0.0
    cur_attention = 100.0

    # FPS tracking
    prev_time = time_module.time()

    # Phát âm thanh khởi động
    sound.play_alert('start.wav')

    while True:
        success, img = cap.read()
        if not success or img is None:
            print("Không đọc được camera!")
            break

        # FPS
        curr_time = time_module.time()
        fps = 1.0 / max(curr_time - prev_time, 0.001)
        prev_time = curr_time

        # ── HAND DETECTION ──
        hands, img = handDetector.findHands(img, draw=True)

        if hands:
            hand = hands[0]
            fingers = count_fingers_angle(hand)
            raw_count = fingers.count(1)
            stable_gesture, just_activated = gesture_smoother.update(raw_count)

            gesture_color = (0, 255, 0) if just_activated else (255, 255, 0)
            cvzone.putTextRect(img, f'Gesture: {stable_gesture}', (0, 50),
                               scale=2, thickness=2, colorR=gesture_color)

            if just_activated:
                if stable_gesture == 1:
                    sound.play_alert('eyes-start.wav')
                    state = SystemState.MONITORING
                    sim.update_fatigue_state(alert_msg="MONITORING STARTED")
                elif stable_gesture == 2:
                    sound.play_alert('goodbye.wav')
                    state = SystemState.IDLE
                    drowsy_level = 0
                    prev_drowsy_level = 0
                    eyes_closed_frames = 0
                    sim.update_fatigue_state(0, 0, 0, False, "MONITORING STOPPED")
                elif stable_gesture == 3 and not sound.music_playing:
                    sound.play_alert('music_escape.wav')
                    sound.play_music('music.wav')
                    state = SystemState.MUSIC
                elif stable_gesture == 5 and sound.music_playing:
                    sound.stop_music()
                    if state == SystemState.MUSIC:
                        state = SystemState.MONITORING
        else:
            gesture_smoother.update(0)

        # ── FACE DETECTION ──
        _, faces = faceDetector.findFaceMesh(img, draw=False)

        if state in (SystemState.MONITORING, SystemState.MUSIC):
            if faces:
                face = faces[0]

                # Vẽ landmarks mắt
                for fid in idListEye:
                    if fid < len(face):
                        cv2.circle(img, face[fid], 3, (255, 0, 255), cv2.FILLED)

                # ── EAR cả 2 mắt (top-level function) ──
                cur_ear_left = calc_ear(face, LEFT_EYE, faceDetector)
                cur_ear_right = calc_ear(face, RIGHT_EYE, faceDetector)
                cur_ear_avg = (cur_ear_left + cur_ear_right) / 2.0

                ratioList_Eye.append(cur_ear_avg)
                ear_smooth = sum(ratioList_Eye) / len(ratioList_Eye)

                # Vẽ đường EAR mắt trái
                leftUp = face[159]; leftDown = face[23]
                leftLeft = face[130]; leftRight = face[243]
                cv2.line(img, leftUp, leftDown, (100, 200, 100), 2)
                cv2.line(img, leftLeft, leftRight, (100, 200, 100), 2)

                # ── Eye State Machine ──
                if ear_smooth < CFG.EAR_THRESHOLD:
                    # Mắt nhắm
                    if eye_state == EyeState.OPEN:
                        eye_state = EyeState.CLOSED
                        blink_count += 1
                        attention.record_blink()

                    eyes_closed_frames += 1

                    # Cảnh báo theo mức
                    if eyes_closed_frames >= CFG.EYES_DANGER_FRAMES:
                        drowsy_level = 2
                    elif eyes_closed_frames >= CFG.EYES_CLOSED_FRAMES:
                        drowsy_level = 1
                else:
                    # Mắt mở
                    if eye_state == EyeState.CLOSED:
                        eye_state = EyeState.OPEN
                        sound.stop_alert()

                    eyes_closed_frames = 0
                    drowsy_level = 0

                # Phát alert CHỈ khi vừa chuyển level (không spam mỗi frame)
                if drowsy_level != prev_drowsy_level:
                    if drowsy_level == 2:
                        sound.play_alert('warning.wav')
                    elif drowsy_level == 1:
                        sound.play_alert('sleepy.wav')
                    prev_drowsy_level = drowsy_level

                # Blink count: cảnh báo khi quá nhiều + reset mỗi chu kỳ
                if blink_count >= CFG.BLINK_ALERT_COUNT:
                    drowsy_level = max(drowsy_level, 1)

                if frame_count > 0 and frame_count % CFG.BLINK_RESET_CYCLE == 0:
                    blink_count = 0

                # ── MAR (Miệng) ──
                for fid in idListMouth:
                    if fid < len(face):
                        cv2.circle(img, face[fid], 3, (255, 0, 255), cv2.FILLED)

                MouthUp = face[0]; MouthDown = face[16]
                MouthLeft = face[185]; MouthRight = face[325]
                lenghtVer, _ = faceDetector.findDistance(MouthUp, MouthDown)
                lenghtHor, _ = faceDetector.findDistance(MouthLeft, MouthRight)

                cur_mar = int((lenghtVer / max(lenghtHor, 1)) * 100)
                ratioList_Mouth.append(cur_mar)
                mar_smooth = sum(ratioList_Mouth) / len(ratioList_Mouth)

                if mar_smooth > CFG.MAR_THRESHOLD:
                    yawn_frames += 1
                    yawn_no_yawn_counter = 0
                else:
                    yawn_no_yawn_counter += 1
                    if yawn_no_yawn_counter >= CFG.YAWN_DECAY_RATE and yawn_frames > 0:
                        yawn_frames -= 1
                        yawn_no_yawn_counter = 0

                if yawn_frames > CFG.YAWN_ALERT_COUNT:
                    sound.play_alert('sleepy_stop_car.wav')
                    drowsy_level = max(drowsy_level, 2)
                    yawn_frames = 0

                # ── Head Pose Estimation ──
                cur_pitch, cur_yaw, cur_roll = head_pose.estimate(face)

                head_nodding = cur_pitch < CFG.HEAD_PITCH_THRESHOLD
                head_distracted = abs(cur_yaw) > CFG.HEAD_YAW_THRESHOLD
                head_leaning = abs(cur_roll) > CFG.HEAD_ROLL_THRESHOLD

                if head_nodding:
                    drowsy_level = max(drowsy_level, 1)
                if head_leaning:
                    drowsy_level = max(drowsy_level, 1)
                if head_distracted:
                    drowsy_level = max(drowsy_level, 1)

                # ── Attention Score ──
                cur_attention = attention.compute(
                    ear_smooth, mar_smooth,
                    cur_pitch, cur_yaw, cur_roll
                )

                # ── HUD text ──
                status_color = (0, 255, 0) if drowsy_level == 0 else (
                    (255, 200, 0) if drowsy_level == 1 else (0, 0, 255))

                cvzone.putTextRect(img, f'EAR: {int(ear_smooth)}', (0, 100),
                                   scale=2, thickness=2, colorR=status_color)
                cvzone.putTextRect(img, f'Blink: {blink_count}', (0, 150),
                                   scale=2, thickness=2, colorR=status_color)
                cvzone.putTextRect(img, f'Yawn: {yawn_frames // 20}', (0, 200),
                                   scale=2, thickness=2, colorR=status_color)
                cvzone.putTextRect(img, f'ATT: {int(cur_attention)}%', (0, 250),
                                   scale=2, thickness=2, colorR=status_color)

                pose_txt = f'P:{int(cur_pitch)} Y:{int(cur_yaw)} R:{int(cur_roll)}'
                cvzone.putTextRect(img, pose_txt, (350, 50),
                                   scale=1.5, thickness=2, colorR=(200, 200, 200))

                if head_nodding:
                    cvzone.putTextRect(img, 'NODDING!', (350, 100),
                                       scale=2, thickness=2, colorR=(0, 0, 255))
                if head_distracted:
                    cvzone.putTextRect(img, 'DISTRACTED!', (350, 150),
                                       scale=2, thickness=2, colorR=(0, 0, 255))

                # ── Update simulation ──
                alert_msg = ""
                if drowsy_level == 2:
                    alert_msg = "DANGER! WAKE UP!"
                elif drowsy_level == 1:
                    alert_msg = "WARNING: DROWSY"
                elif head_distracted:
                    alert_msg = "LOOK AT THE ROAD!"

                sim.update_fatigue_state(
                    drowsy_level=drowsy_level,
                    yawn_count=yawn_frames // 20,
                    blink_count=blink_count,
                    is_monitoring=True,
                    alert_msg=alert_msg,
                    attention_score=cur_attention,
                    head_pitch=cur_pitch,
                    head_yaw=cur_yaw,
                    head_roll=cur_roll,
                )

                # Plots
                imgPlotEye = ploty_Eye.update(ear_smooth, (255, 0, 0))
                imgPlotMouth = ploty_Mouth.update(mar_smooth, (0, 255, 0))
                imgPlotAtt = ploty_Attention.update(cur_attention, (0, 200, 255))
                img = cv2.resize(img, (CFG.DISPLAY_W, CFG.DISPLAY_H))
                imgStack = cvzone.stackImages(
                    [img, imgPlotEye, imgPlotMouth, imgPlotAtt], 2, 1)

                # ── Log ──
                vehicle_state = sim.get_vehicle_state()
                logger.log(
                    ear_left=cur_ear_left, ear_right=cur_ear_right,
                    ear_avg=cur_ear_avg, mar=cur_mar,
                    head_pitch=round(cur_pitch, 1),
                    head_yaw=round(cur_yaw, 1),
                    head_roll=round(cur_roll, 1),
                    attention_score=round(cur_attention, 1),
                    drowsy_level=drowsy_level,
                    blink_count=blink_count,
                    yawn_count=yawn_frames // 20,
                    vehicle_speed=vehicle_state["speed_kmh"],
                    lane_offset=vehicle_state["lane_offset_px"],
                    system_state=state.name,
                )

            else:
                # Không thấy mặt
                img = cv2.resize(img, (CFG.DISPLAY_W, CFG.DISPLAY_H))
                imgStack = cvzone.stackImages([img], 1, 1)
                sim.update_fatigue_state(
                    0, yawn_frames // 20, blink_count, True,
                    "NO FACE DETECTED", cur_attention,
                )
        else:
            # IDLE
            img = cv2.resize(img, (CFG.DISPLAY_W, CFG.DISPLAY_H))
            imgStack = cvzone.stackImages([img], 1, 1)
            sim.update_fatigue_state(0, 0, 0, False, "", 100.0)

        # FPS + State overlay
        cv2.putText(imgStack, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(imgStack, f'State: {state.name}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        frame_count += 1
        cv2.imshow("Driver Fatigue Monitor", imgStack)

        # Update driving simulation
        if not sim.tick():
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    logger.close()
    cap.release()
    cv2.destroyAllWindows()
    sim.quit()
    sound.stop_all()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    Start()
