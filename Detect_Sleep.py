import cv2                              #pip install opencv-python
import mediapipe as mp                  #pip install mediapipe
from cvzone.HandTrackingModule import HandDetector  
from cvzone.FaceMeshModule import FaceMeshDetector
import winsound          #phát wav
import cvzone                       #pip install cvzone
# from tkinter import *
# from PIL import ImageTk,Image 
from cvzone.PlotModule import LivePlot

#==============================================HÀM DETECT BÀN TAY==============================================================
def Start():
    voice_repeat = 0

    detectorEye = detectorMouth = FaceMeshDetector(maxFaces=1)
    handDetector = HandDetector(detectionCon=0.5, maxHands=2)
    cap = cv2.VideoCapture(0)

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
    dir = 0
    yawn = 0
    time = 0
    maxframes = 12
    num_frames1 = num_frames2 = num_frames3 = 0
    detect_time = False
    
    while True:
        if voice_repeat == 0:
            winsound.PlaySound('start.wav', winsound.SND_FILENAME)
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
                winsound.PlaySound('eyes-start.wav', winsound.SND_FILENAME)
                dir += 1
            elif detect_time and count == 2:
                winsound.PlaySound('goodbye.wav', winsound.SND_FILENAME)
                dir = 0
            elif detect_time and count == 3:
                winsound.PlaySound('music_escape.wav', winsound.SND_FILENAME)
                winsound.PlaySound('music.wav', winsound.SND_FILENAME + winsound.SND_ASYNC)
                val = input("Enter any input to stop sound: ")
                if val:
                    winsound.PlaySound(None, 0)
                dir += 1
            else:
                dir *= 2

        # ── Detect khuôn mặt ──
        img_eye, faces = detectorEye.findFaceMesh(img, draw=False)

        if dir > 0:
            if faces:
                detect_time = False
                face = faces[0]
                for id in idListEye:
                    cv2.circle(img, face[id], 5, color, cv2.FILLED)

                # Mắt
                leftUp    = face[159]; leftDown  = face[23]
                leftLeft  = face[130]; leftRight = face[243]
                lenghtVer, _ = detectorEye.findDistance(leftUp, leftDown)
                lenghtHor, _ = detectorEye.findDistance(leftLeft, leftRight)
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
                        winsound.PlaySound('sleepy.wav', winsound.SND_FILENAME)
                        blinkCounter = 0
                    if time_sleep_Counter >= 6:
                        winsound.PlaySound('warning.wav', winsound.SND_FILENAME)
                    elif time_sleep_Counter < 3:
                        winsound.PlaySound(None, 0)
                    color = (100, 200, 100)
                    counter = 1
                elif ratioAvg_Eye > 35 and counter == 0:
                    time_sleep_Counter = 0
                    winsound.PlaySound(None, 0)

                if counter != 0:
                    counter += 1
                    if counter > 10:
                        counter = 0
                        color = (255, 0, 255)

                if time > 900:
                    blinkCounter = 0
                    time = 0

                cvzone.putTextRect(img, f'Sleep: {time_sleep_Counter}', (0, 100), scale=2, thickness=2, colorR=color)
                cvzone.putTextRect(img, f'Blink: {blinkCounter}',       (0, 150), scale=2, thickness=2, colorR=color)

                # Miệng
                for id in idListMouth:
                    cv2.circle(img, face[id], 5, color, cv2.FILLED)

                MouthUp    = face[0];   MouthDown  = face[16]
                MouthLeft  = face[185]; MouthRight = face[325]
                lenghtVerMouth, _ = detectorMouth.findDistance(MouthUp, MouthDown)
                lenghtHorMouth, _ = detectorMouth.findDistance(MouthLeft, MouthRight)

                ratioMouth = int((lenghtVerMouth / lenghtHorMouth) * 100)
                ratioList_Mouth.append(ratioMouth)
                if len(ratioList_Mouth) > 3: ratioList_Mouth.pop(0)
                ratioAvg_Mouth = sum(ratioList_Mouth) / len(ratioList_Mouth)

                if ratioAvg_Mouth > 94:
                    yawn += 1
                    time_sleep_Counter = 0
                print(f"Mouth ratio: {ratioAvg_Mouth:.1f}")

                if yawn > 60:
                    winsound.PlaySound('sleepy_stop_car.wav', winsound.SND_FILENAME)
                    yawn = 0
                cvzone.putTextRect(img, f'Yawn: {int(yawn/20)}', (0, 200), scale=2, thickness=2, colorR=color)

                imgPlotEye   = ploty_Eye.update(ratioAvg_Eye,   (255, 0, 0))
                imgPlotMouth = ploty_Mouth.update(ratioAvg_Mouth, (0, 255, 0))
                imgPlotTime  = ploty_Time.update(time_sleep_Counter, (0, 0, 255))
                img = cv2.resize(img, (640, 360))
                imgStack = cvzone.stackImages([img, imgPlotEye, imgPlotMouth, imgPlotTime], 2, 1)
        else:
            img = cv2.resize(img, (640, 360))
            imgStack = cvzone.stackImages([img], 1, 1)

        time += 1
        cv2.imshow("Image", imgStack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =====================================================Chạy trực tiếp=======================================================
if __name__ == "__main__":
    Start()