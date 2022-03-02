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

            x_1, y_1 = lmlist[25][1], lmlist[25][2]
            x_2, y_2 = lmlist[23][1], lmlist[23][2]

            # quantifies the hypotenuse of the triangle
            hypotenuse = self.distance(x_1, x_1, x_2, y_2)
            # quantifies the horizontal of the triangle
            horizontal = self.distance(x_1, y_1, x_2, y_1)
            # makes the third-line of the triangle
            thirdline = self.distance(x_2, y_2, x_2, y_1)
            # calculates the angle using trigonometry
            angle = np.arcsin((thirdline / hypotenuse)) * 180 / math.pi

            # draws all 3 lines
            cv2.line(img, (x_1, y_1), (x_2, y_2), (0, 0, 255), 2)
            cv2.line(img, (x_1, y_1), (x_2, y_1), (0, 0, 255), 2)
            cv2.line(img, (x_2, y_2), (x_2, y_1), (0, 0, 255), 2)

            # put angle text (allow for calculations upto 180 degrees)
            angle_text = ""
            if y_2 < y_1 and x_2 > x_1:
                angle_text = str(int(angle))
            elif y_2 < y_1 and x_2 < x_1:
                angle_text = str(int(180 - angle))
            elif y_2 > y_1 and x_2 < x_1:
                angle_text = str(int(180 + angle))
            elif y_2 > y_1 and x_2 > x_1:
                angle_text = str(int(360 - angle))

            # CHANGE FONT HERE
            cv2.putText(img, angle_text, (x_1 - 30, y_1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0),2)

            cv2.imshow('Image', img)
            cv2.waitKey(1)

if __name__ == '__main__':
    detector = poseDetector()
    path = '../videos/yoga1.mp4'
    while True:
        detector.runDetector(path)