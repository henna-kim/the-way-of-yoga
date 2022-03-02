import cv2
import mediapipe as mp
import time
import math
import numpy as np

class poseDetector():

    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    def __int__(self, mode=False, upBody=False, smooth=True,
                detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

    def getAngle(a, b, c):
        ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        return ang + 360 if ang < 0 else ang

    def distance(self, x1, y1, x2, y2):
        dist = math.sqrt((math.fabs(x2 - x1)) ** 2 + ((math.fabs(y2 - y1))) ** 2)
        return dist

    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y*h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lmlist

    def runDetector(self, path, live=False, scale_percent=30, mark_part=False, pose_number=14):
        if live:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(path)

        pTime = 0
        while True:
            success, image = cap.read()
            try:
                width = int(image.shape[1] * scale_percent / 100)
            except AttributeError:
                print("Video just finished")
                break
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            img = self.findPose(img)
            lmlist = self.findPosition(img)

            if len(lmlist) != 0 and mark_part:
                number = pose_number
                cv2.circle(img, (lmlist[number][1], lmlist[number][2]), 10, (0, 0, 255), cv2.FILLED)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (70, 50),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            a = np.array([lmlist[27][1], lmlist[27][2]])
            b = np.array([lmlist[25][1], lmlist[25][2]])
            c = np.array([lmlist[23][1], lmlist[23][2]])

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.degrees(np.arccos(cosine_angle))

            # CHANGE FONT HERE
            angle_text = str(int(angle))
            cv2.putText(img, angle_text, (b[0] - 30, b[1]), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0),2)

            a = np.array([lmlist[11][1], lmlist[11][2]])
            b = np.array([lmlist[23][1], lmlist[23][2]])
            c = np.array([lmlist[25][1], lmlist[25][2]])

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle1 = np.degrees(np.arccos(cosine_angle))

            # CHANGE FONT HERE
            angle_text1 = str(int(angle1))
            cv2.putText(img, angle_text1, (b[0] - 30, b[1]), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

            a = np.array([lmlist[24][1], lmlist[24][2]])
            b = np.array([lmlist[26][1], lmlist[26][2]])
            c = np.array([lmlist[28][1], lmlist[28][2]])

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle2 = np.degrees(np.arccos(cosine_angle))

            # CHANGE FONT HERE
            angle_text2 = str(int(angle2))
            cv2.putText(img, angle_text2, (b[0] - 30, b[1]), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

            a = np.array([lmlist[11][1], lmlist[11][2]])
            b = np.array([lmlist[13][1], lmlist[13][2]])
            c = np.array([lmlist[15][1], lmlist[15][2]])

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle3 = np.degrees(np.arccos(cosine_angle))

            # CHANGE FONT HERE
            angle_text3 = str(int(angle3))
            cv2.putText(img, angle_text3, (b[0] - 30, b[1]), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Image', img)
            cv2.waitKey(1)

if __name__ == '__main__':
    detector = poseDetector()
    path = '../videos/yoga.mp4'
    detector.runDetector(path, live=True)