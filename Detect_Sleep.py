import cv2                              #pip install opencv-python
import mediapipe as mp                  #pip install mediapipe
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector
import winsound          #phát wav
import cvzone                       #pip install cvzone
import os
from cvzone.PlotModule import LivePlot

# ── Helper: play sound safely ──
def play_sound(filename, flags=winsound.SND_FILENAME):
    """Play a .wav file only if it exists, otherwise skip silently."""
    if filename is None:
        winsound.PlaySound(None, 0)
        return
    if os.path.exists(filename):
        winsound.PlaySound(filename, flags)
    else:
        print(f"[WARNING] Sound file not found: {filename}")

#==============================================HÀM DETECT BÀN TAY==============================================================
def Start():
    voice_repeat = 0

    faceDetector = FaceMeshDetector(maxFaces=1)
    handDetector = HandDetector(detectionCon=0.5, maxHands=2)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không mở được camera!")
        return

    idListEye = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
    idListMouth = [185,39,37,0,267,269,325,321,404,315,16,85,180,90]
    ploty_Eye   = LivePlot(640, 360, [20, 50],  invert=True)
    ploty_Mouth = LivePlot(640, 360, [30, 140], invert=True)
    ploty_Time  = LivePlot(640, 360, [-2, 8],   invert=True)

    ratioList_Eye = []
    ratioList_Mouth = []
    blinkCounter = 0
    time_sleep_Counter = 0
    counter = 0
    color = (0, 0, 255)
    state = 0           # renamed from 'dir' to avoid shadowing built-in
    yawn = 0
    frame_count = 0     # renamed from 'time' to avoid shadowing built-in
    maxframes = 12
    num_frames1 = num_frames2 = num_frames3 = 0
    detect_time = False
    music_playing = False

    while True:
        if voice_repeat == 0:
            play_sound('start.wav')
            voice_repeat += 1

        success, img = cap.read()
        if not success or img is None:
            print("Không đọc được camera!")
            break

        # ── Detect bàn tay bằng cvzone (không cần mp.solutions) ──
        hands, img = handDetector.findHands(img)

        if hands:
            hand_count = hands[0]
            fingers = handDetector.fingersUp(hand_count)
            count = fingers.count(1)

            cvzone.putTextRect(img, f'Finger: {count}', (0, 50), scale=2, thickness=2, colorR=color)

            if count == 1:
                num_frames1 += 1; num_frames2 = 0; num_frames3 = 0
                if num_frames1 > maxframes: detect_time = True
            elif count == 2:
                num_frames2 += 1; num_frames1 = 0; num_frames3 = 0
                if num_frames2 > maxframes: detect_time = True
            elif count == 3:
                num_frames3 += 1; num_frames2 = 0; num_frames1 = 0
                if num_frames3 > maxframes: detect_time = True
            else:
                num_frames1 = num_frames2 = num_frames3 = 0
                detect_time = False

            if detect_time and count == 1:
                play_sound('eyes-start.wav')
                state = 1
                detect_time = False
            elif detect_time and count == 2:
                play_sound('goodbye.wav')
                state = 0
                detect_time = False
            elif detect_time and count == 3:
                play_sound('music_escape.wav')
                play_sound('music.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
                music_playing = True
                state = 1
                detect_time = False
            elif detect_time and count == 5 and music_playing:
                # 5 fingers to stop music (replaces blocking input())
                play_sound(None)
                music_playing = False

        # ── Detect khuôn mặt ──
        _, faces = faceDetector.findFaceMesh(img, draw=False)

        if state > 0:
            if faces:
                detect_time = False
                face = faces[0]
                for fid in idListEye:
                    cv2.circle(img, face[fid], 5, color, cv2.FILLED)

                # Mắt
                leftUp    = face[159]; leftDown  = face[23]
                leftLeft  = face[130]; leftRight = face[243]
                lenghtVer, _ = faceDetector.findDistance(leftUp, leftDown)
                lenghtHor, _ = faceDetector.findDistance(leftLeft, leftRight)
                cv2.line(img, leftUp, leftDown, (100, 200, 100), 3)
                cv2.line(img, leftLeft, leftRight, (100, 200, 100), 3)

                ratio = int((lenghtVer / lenghtHor) * 100)
                ratioList_Eye.append(ratio)
                if len(ratioList_Eye) > 3: ratioList_Eye.pop(0)
                ratioAvg_Eye = sum(ratioList_Eye) / len(ratioList_Eye)

                if ratioAvg_Eye < 35 and counter == 0:
                    time_sleep_Counter += 1
                    blinkCounter += 1
                    if blinkCounter > 30:
                        play_sound('sleepy.wav')
                        blinkCounter = 0
                    if time_sleep_Counter >= 6:
                        play_sound('warning.wav')
                    elif time_sleep_Counter < 3:
                        play_sound(None)
                    color = (100, 200, 100)
                    counter = 1
                elif ratioAvg_Eye >= 35 and counter == 0:
                    time_sleep_Counter = 0
                    play_sound(None)

                if counter != 0:
                    counter += 1
                    if counter > 10:
                        counter = 0
                        color = (255, 0, 255)

                if frame_count > 900:
                    blinkCounter = 0
                    frame_count = 0

                cvzone.putTextRect(img, f'Sleep: {time_sleep_Counter}', (0, 100), scale=2, thickness=2, colorR=color)
                cvzone.putTextRect(img, f'Blink: {blinkCounter}',       (0, 150), scale=2, thickness=2, colorR=color)

                # Miệng
                for fid in idListMouth:
                    cv2.circle(img, face[fid], 5, color, cv2.FILLED)

                MouthUp    = face[0];   MouthDown  = face[16]
                MouthLeft  = face[185]; MouthRight = face[325]
                lenghtVerMouth, _ = faceDetector.findDistance(MouthUp, MouthDown)
                lenghtHorMouth, _ = faceDetector.findDistance(MouthLeft, MouthRight)

                ratioMouth = int((lenghtVerMouth / lenghtHorMouth) * 100)
                ratioList_Mouth.append(ratioMouth)
                if len(ratioList_Mouth) > 3: ratioList_Mouth.pop(0)
                ratioAvg_Mouth = sum(ratioList_Mouth) / len(ratioList_Mouth)

                if ratioAvg_Mouth > 94:
                    yawn += 1
                    time_sleep_Counter = 0
                print(f"Mouth ratio: {ratioAvg_Mouth:.1f}")

                if yawn > 60:
                    play_sound('sleepy_stop_car.wav')
                    yawn = 0
                cvzone.putTextRect(img, f'Yawn: {int(yawn/20)}', (0, 200), scale=2, thickness=2, colorR=color)

                imgPlotEye   = ploty_Eye.update(ratioAvg_Eye,   (255, 0, 0))
                imgPlotMouth = ploty_Mouth.update(ratioAvg_Mouth, (0, 255, 0))
                imgPlotTime  = ploty_Time.update(time_sleep_Counter, (0, 0, 255))
                img = cv2.resize(img, (640, 360))
                imgStack = cvzone.stackImages([img, imgPlotEye, imgPlotMouth, imgPlotTime], 2, 1)
            else:
                # Face detection active but no face found — show camera only
                img = cv2.resize(img, (640, 360))
                imgStack = cvzone.stackImages([img], 1, 1)
        else:
            img = cv2.resize(img, (640, 360))
            imgStack = cvzone.stackImages([img], 1, 1)

        frame_count += 1
        cv2.imshow("Image", imgStack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =====================================================Chạy trực tiếp=======================================================
if __name__ == "__main__":
    Start()
