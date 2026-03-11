"""
Detect_Sleep.py  –  Hệ thống giám sát buồn ngủ tài xế + Mô phỏng lái xe.
- Detect mắt (EAR cả 2 mắt) + ngáp (MAR)
- Hand gesture điều khiển (angle-based + temporal smoothing cho ổn định)
- Tích hợp driving simulation qua pygame
"""
import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector
import cvzone
import os
import math
import time as time_module
import threading
import collections
import numpy as np
from dataclasses import dataclass, field
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
    EAR_SMOOTH_WINDOW: int = 5         # số frame trung bình hóa
    BLINK_ALERT_COUNT: int = 30        # số blink liên tục → cảnh báo nhẹ
    SLEEP_DANGER_COUNT: int = 6        # sleep counter → cảnh báo nặng

    # MAR (Mouth Aspect Ratio)
    MAR_THRESHOLD: float = 94.0        # trên ngưỡng này = đang ngáp
    MAR_SMOOTH_WINDOW: int = 5
    YAWN_ALERT_COUNT: int = 60         # yawn counter → cảnh báo dừng xe

    # Hand gesture
    GESTURE_HOLD_FRAMES: int = 15      # giữ gesture bao nhiêu frame mới kích hoạt
    GESTURE_CONFIDENCE: float = 0.7    # ngưỡng confidence tay
    GESTURE_SMOOTH_WINDOW: int = 7     # bao nhiêu frame gần nhất để vote gesture

    # Thời gian reset
    FRAME_RESET_CYCLE: int = 900       # reset blink counter mỗi N frame

CFG = Config()


# ── Enum trạng thái ───────────────────────────────────────────────────────────
class SystemState(Enum):
    IDLE = auto()           # chờ gesture bật monitoring
    MONITORING = auto()     # đang giám sát
    MUSIC = auto()          # đang phát nhạc nền


# ── Helper: phát âm thanh không blocking ──────────────────────────────────────
def play_sound_async(filename, flags=None):
    """Phát .wav trong thread riêng, không block video loop."""
    def _play():
        try:
            import winsound
            if filename is None:
                winsound.PlaySound(None, 0)
                return
            if os.path.exists(filename):
                f = flags if flags is not None else winsound.SND_FILENAME
                winsound.PlaySound(filename, f)
            else:
                print(f"[WARNING] Sound not found: {filename}")
        except Exception as e:
            print(f"[SOUND ERROR] {e}")

    threading.Thread(target=_play, daemon=True).start()


def stop_sound():
    """Dừng mọi âm thanh."""
    play_sound_async(None)


# ── Angle-based finger counting (thay thế fingersUp của cvzone) ──────────────
def _angle(a, b, c):
    """Tính góc tại điểm b (độ), với a-b-c là 3 landmarks [x,y,z]."""
    ba = np.array(a[:2]) - np.array(b[:2])
    bc = np.array(c[:2]) - np.array(b[:2])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def count_fingers_angle(hand_data, angle_threshold=160):
    """
    Đếm số ngón tay đang giơ dựa trên góc giữa các đốt ngón.
    Ổn định hơn fingersUp vì không phụ thuộc hướng tay (x/y).

    Ngón duỗi thẳng → góc lớn (gần 180°).
    Ngón gập        → góc nhỏ (< threshold).

    MediaPipe hand landmarks:
      Ngón cái:  1-2-3-4    (CMC-MCP-IP-TIP)
      Ngón trỏ:  5-6-7-8    (MCP-PIP-DIP-TIP)
      Ngón giữa: 9-10-11-12
      Ngón áp út: 13-14-15-16
      Ngón út:    17-18-19-20
    """
    lm = hand_data["lmList"]
    fingers = []

    # Ngón cái: dùng góc tại MCP (landmark 2) và IP (landmark 3)
    # Ngón cái cần ngưỡng thấp hơn vì phạm vi xoay khác
    thumb_angle = _angle(lm[1], lm[2], lm[3])
    thumb_tip_angle = _angle(lm[2], lm[3], lm[4])
    # Cái duỗi khi cả 2 góc đều lớn
    thumb_open = thumb_angle > 140 and thumb_tip_angle > 140
    # Thêm kiểm tra khoảng cách: tip phải xa wrist
    thumb_tip_dist = np.linalg.norm(np.array(lm[4][:2]) - np.array(lm[2][:2]))
    thumb_base_dist = np.linalg.norm(np.array(lm[3][:2]) - np.array(lm[2][:2]))
    if thumb_base_dist > 0:
        thumb_extended = thumb_open and (thumb_tip_dist / thumb_base_dist > 1.2)
    else:
        thumb_extended = thumb_open
    fingers.append(1 if thumb_extended else 0)

    # 4 ngón còn lại: góc tại PIP và DIP
    finger_joints = [
        (5, 6, 7, 8),      # trỏ
        (9, 10, 11, 12),    # giữa
        (13, 14, 15, 16),   # áp út
        (17, 18, 19, 20),   # út
    ]
    for mcp, pip, dip, tip in finger_joints:
        angle_pip = _angle(lm[mcp], lm[pip], lm[dip])
        angle_dip = _angle(lm[pip], lm[dip], lm[tip])
        # Ngón duỗi khi cả 2 góc đều lớn
        is_open = angle_pip > angle_threshold and angle_dip > angle_threshold
        fingers.append(1 if is_open else 0)

    return fingers


# ── Gesture Smoother ──────────────────────────────────────────────────────────
class GestureSmoother:
    """Lọc gesture bằng majority vote trên N frame gần nhất."""

    def __init__(self, window_size=7, hold_frames=15):
        self.history = collections.deque(maxlen=window_size)
        self.hold_frames = hold_frames
        self.current_gesture = 0
        self.hold_counter = 0
        self.activated = False   # đã kích hoạt gesture chưa

    def update(self, raw_finger_count):
        """Cập nhật gesture mới. Trả về (stable_gesture, just_activated)."""
        self.history.append(raw_finger_count)

        # Majority vote
        if len(self.history) < 3:
            return 0, False

        counts = {}
        for g in self.history:
            counts[g] = counts.get(g, 0) + 1
        voted = max(counts, key=counts.get)

        # Nếu gesture thay đổi → reset hold
        if voted != self.current_gesture:
            self.current_gesture = voted
            self.hold_counter = 0
            self.activated = False
            return voted, False

        # Cùng gesture → tăng hold counter
        self.hold_counter += 1

        if self.hold_counter >= self.hold_frames and not self.activated:
            self.activated = True
            return voted, True  # just activated!

        return voted, False

    def reset(self):
        self.history.clear()
        self.current_gesture = 0
        self.hold_counter = 0
        self.activated = False


# ── Hệ thống chính ───────────────────────────────────────────────────────────
def Start():
    # Khởi tạo detector
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

    # Landmark IDs
    # Mắt trái: 6 điểm theo công thức EAR (Soukupova & Cech 2016)
    LEFT_EYE  = {"top1": 159, "bot1": 145, "top2": 158, "bot2": 153, "left": 133, "right": 33}
    RIGHT_EYE = {"top1": 386, "bot1": 374, "top2": 385, "bot2": 380, "left": 362, "right": 263}
    # Vẽ mắt
    idListEye = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243,
                 253, 254, 256, 339, 384, 385, 386, 387, 388, 359, 463]
    idListMouth = [185, 39, 37, 0, 267, 269, 325, 321, 404, 315, 16, 85, 180, 90]

    # Plot
    ploty_Eye   = LivePlot(640, 360, [20, 50],  invert=True)
    ploty_Mouth = LivePlot(640, 360, [30, 140], invert=True)
    ploty_Time  = LivePlot(640, 360, [-2, 8],   invert=True)

    # Trạng thái
    state = SystemState.IDLE
    gesture_smoother = GestureSmoother(
        window_size=CFG.GESTURE_SMOOTH_WINDOW,
        hold_frames=CFG.GESTURE_HOLD_FRAMES
    )

    ratioList_Eye = collections.deque(maxlen=CFG.EAR_SMOOTH_WINDOW)
    ratioList_Mouth = collections.deque(maxlen=CFG.MAR_SMOOTH_WINDOW)
    blinkCounter = 0
    time_sleep_Counter = 0
    counter = 0
    color = (0, 0, 255)
    yawn = 0
    frame_count = 0
    music_playing = False

    # FPS tracking
    prev_time = time_module.time()

    # Phát âm thanh khởi động
    play_sound_async('start.wav')

    while True:
        success, img = cap.read()
        if not success or img is None:
            print("Không đọc được camera!")
            break

        # FPS
        curr_time = time_module.time()
        fps = 1.0 / max(curr_time - prev_time, 0.001)
        prev_time = curr_time

        # ── HAND DETECTION (mỗi frame) ──
        hands, img = handDetector.findHands(img, draw=True)

        if hands:
            hand = hands[0]
            # Dùng angle-based counting thay vì cvzone fingersUp
            fingers = count_fingers_angle(hand)
            raw_count = fingers.count(1)

            stable_gesture, just_activated = gesture_smoother.update(raw_count)

            # Hiển thị gesture (ổn định)
            gesture_color = (0, 255, 0) if just_activated else (255, 255, 0)
            cvzone.putTextRect(img, f'Gesture: {stable_gesture}', (0, 50),
                               scale=2, thickness=2, colorR=gesture_color)

            # Xử lý gesture chỉ khi mới kích hoạt (hold đủ lâu)
            if just_activated:
                if stable_gesture == 1:
                    play_sound_async('eyes-start.wav')
                    state = SystemState.MONITORING
                    sim.update_fatigue_state(alert_msg="MONITORING STARTED")
                elif stable_gesture == 2:
                    play_sound_async('goodbye.wav')
                    state = SystemState.IDLE
                    sim.update_fatigue_state(0, 0, 0, False, "MONITORING STOPPED")
                elif stable_gesture == 3 and not music_playing:
                    play_sound_async('music_escape.wav')
                    try:
                        import winsound
                        play_sound_async('music.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
                    except ImportError:
                        pass
                    music_playing = True
                    if state == SystemState.IDLE:
                        state = SystemState.MONITORING
                elif stable_gesture == 5 and music_playing:
                    stop_sound()
                    music_playing = False
        else:
            gesture_smoother.update(0)  # không thấy tay → vote 0

        # ── FACE DETECTION ──
        _, faces = faceDetector.findFaceMesh(img, draw=False)

        drowsy_level = 0  # cho simulation

        if state in (SystemState.MONITORING, SystemState.MUSIC):
            if faces:
                face = faces[0]

                # Vẽ landmarks mắt
                for fid in idListEye:
                    if fid < len(face):
                        cv2.circle(img, face[fid], 3, color, cv2.FILLED)

                # ── EAR cả 2 mắt (Soukupova & Cech 2016) ──
                def calc_ear(eye_ids):
                    top1 = face[eye_ids["top1"]]
                    bot1 = face[eye_ids["bot1"]]
                    top2 = face[eye_ids["top2"]]
                    bot2 = face[eye_ids["bot2"]]
                    left = face[eye_ids["left"]]
                    right = face[eye_ids["right"]]
                    v1, _ = faceDetector.findDistance(top1, bot1)
                    v2, _ = faceDetector.findDistance(top2, bot2)
                    h,  _ = faceDetector.findDistance(left, right)
                    if h == 0:
                        return 50  # tránh chia 0
                    return int(((v1 + v2) / (2.0 * h)) * 100)

                ear_left = calc_ear(LEFT_EYE)
                ear_right = calc_ear(RIGHT_EYE)
                ear_avg = (ear_left + ear_right) / 2.0

                # Vẽ đường EAR mắt trái (cho trực quan)
                leftUp = face[159]; leftDown = face[23]
                leftLeft = face[130]; leftRight = face[243]
                cv2.line(img, leftUp, leftDown, (100, 200, 100), 2)
                cv2.line(img, leftLeft, leftRight, (100, 200, 100), 2)

                ratioList_Eye.append(ear_avg)
                ratioAvg_Eye = sum(ratioList_Eye) / len(ratioList_Eye)

                # Phát hiện nhắm mắt
                if ratioAvg_Eye < CFG.EAR_THRESHOLD and counter == 0:
                    time_sleep_Counter += 1
                    blinkCounter += 1
                    if blinkCounter > CFG.BLINK_ALERT_COUNT:
                        play_sound_async('sleepy.wav')
                        blinkCounter = 0
                    if time_sleep_Counter >= CFG.SLEEP_DANGER_COUNT:
                        play_sound_async('warning.wav')
                        drowsy_level = 2
                        sim.update_fatigue_state(2, int(yawn / 20), blinkCounter, True,
                                                 "WAKE UP! DANGER!")
                    elif time_sleep_Counter >= 3:
                        drowsy_level = 1
                        sim.update_fatigue_state(1, int(yawn / 20), blinkCounter, True,
                                                 "DROWSY WARNING!")
                    else:
                        stop_sound()
                    color = (100, 200, 100)
                    counter = 1
                elif ratioAvg_Eye >= CFG.EAR_THRESHOLD and counter == 0:
                    time_sleep_Counter = 0
                    stop_sound()
                    drowsy_level = 0
                    sim.update_fatigue_state(0, int(yawn / 20), blinkCounter, True)

                if counter != 0:
                    counter += 1
                    if counter > 10:
                        counter = 0
                        color = (255, 0, 255)

                if frame_count > CFG.FRAME_RESET_CYCLE:
                    blinkCounter = 0
                    frame_count = 0

                cvzone.putTextRect(img, f'Sleep: {time_sleep_Counter}', (0, 100),
                                   scale=2, thickness=2, colorR=color)
                cvzone.putTextRect(img, f'Blink: {blinkCounter}', (0, 150),
                                   scale=2, thickness=2, colorR=color)

                # ── Miệng (MAR) ──
                for fid in idListMouth:
                    cv2.circle(img, face[fid], 3, color, cv2.FILLED)

                MouthUp   = face[0];   MouthDown  = face[16]
                MouthLeft = face[185]; MouthRight = face[325]
                lenghtVerMouth, _ = faceDetector.findDistance(MouthUp, MouthDown)
                lenghtHorMouth, _ = faceDetector.findDistance(MouthLeft, MouthRight)

                if lenghtHorMouth > 0:
                    ratioMouth = int((lenghtVerMouth / lenghtHorMouth) * 100)
                else:
                    ratioMouth = 0
                ratioList_Mouth.append(ratioMouth)
                ratioAvg_Mouth = sum(ratioList_Mouth) / len(ratioList_Mouth)

                if ratioAvg_Mouth > CFG.MAR_THRESHOLD:
                    yawn += 1
                    time_sleep_Counter = 0

                if yawn > CFG.YAWN_ALERT_COUNT:
                    play_sound_async('sleepy_stop_car.wav')
                    sim.update_fatigue_state(2, int(yawn / 20), blinkCounter, True,
                                             "TOO MANY YAWNS! STOP!")
                    yawn = 0

                cvzone.putTextRect(img, f'Yawn: {int(yawn / 20)}', (0, 200),
                                   scale=2, thickness=2, colorR=color)

                # Plot
                imgPlotEye   = ploty_Eye.update(ratioAvg_Eye, (255, 0, 0))
                imgPlotMouth = ploty_Mouth.update(ratioAvg_Mouth, (0, 255, 0))
                imgPlotTime  = ploty_Time.update(time_sleep_Counter, (0, 0, 255))
                img = cv2.resize(img, (CFG.DISPLAY_W, CFG.DISPLAY_H))
                imgStack = cvzone.stackImages([img, imgPlotEye, imgPlotMouth, imgPlotTime], 2, 1)
            else:
                # Không thấy mặt
                img = cv2.resize(img, (CFG.DISPLAY_W, CFG.DISPLAY_H))
                imgStack = cvzone.stackImages([img], 1, 1)
                sim.update_fatigue_state(0, int(yawn / 20), blinkCounter, True, "NO FACE DETECTED")
        else:
            # IDLE – chỉ hiện camera
            img = cv2.resize(img, (CFG.DISPLAY_W, CFG.DISPLAY_H))
            imgStack = cvzone.stackImages([img], 1, 1)
            sim.update_fatigue_state(0, 0, 0, False)

        # FPS overlay
        cv2.putText(imgStack, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # State overlay
        state_text = f'State: {state.name}'
        cv2.putText(imgStack, state_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        frame_count += 1
        cv2.imshow("Driver Fatigue Monitor", imgStack)

        # Update driving simulation
        if not sim.tick():
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sim.quit()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    Start()
