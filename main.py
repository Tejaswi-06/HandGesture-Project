import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Variables
width, height = 1280, 720
folderPath = "presentation"

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Get the list of presentation images
pathImages = sorted(os.listdir("C:/Users/sushm/OneDrive/Desktop/gest/HandGestureControle_ppt/presentation"), key=len)

# Variables
imgNumber = 0
hs, ws = int(120 * 1), int(213 * 1)
gestureThreshold = 300
buttonPressed = False
buttonCounter = 0
buttonDelay = 10
annotation = [[]]
annotationNumber = -1
annotationStart = False

# Zoom variables
scale_factor = 1.0
zoom_increment = 0.1

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    # Import Images
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Resize image based on scale factor for zoom functionality
    imgCurrent = cv2.resize(imgCurrent, (int(imgCurrent.shape[1] * scale_factor), int(imgCurrent.shape[0] * scale_factor)))

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height - 200], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:  # If the hand is at the height of the face
            annotationStart = False

            # Gesture 1 - Move to previous slide (Thumb up)
            if fingers == [1, 0, 0, 0, 0]:  # Only thumb is up
                print("Previous Slide")
                if imgNumber > 0:
                    buttonPressed = True
                    annotation = [[]]
                    annotationNumber = 0
                    imgNumber -= 1

            # Gesture 2 - Move to next slide (Index finger up)
            elif fingers == [0, 1, 0, 0, 0]:  # Only index finger is up
                print("Next Slide")
                if imgNumber < len(pathImages) - 1:
                    buttonPressed = True
                    annotation = [[]]
                    annotationNumber = 0
                    imgNumber += 1
            
            # Gesture 3 - Zoom In (Index finger and thumb together)
            elif fingers == [0, 1, 1, 0, 0]:  # Thumb and Index finger up
                print("Zoom In")
                scale_factor += zoom_increment

            # Gesture 4 - Zoom Out (Thumb down and Index finger up)
            elif fingers == [1, 1, 0, 0, 0]:  # Only index finger is up
                print("Zoom Out")
                scale_factor -= zoom_increment
                scale_factor = max(1.0, scale_factor)  # Prevent zooming out too much

            # Gesture 5 - Exit Slide (Little finger up)
            elif fingers == [0, 0, 0, 0, 1]:  # Only little finger is up
                print("Exiting Slide Show")
                break  # This will break the while loop and exit the program

        else:
            annotationStart = False

    else:
        annotationStart = False

    # Button press iterations
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    # Adding webcam image on the slides
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:w] = imgSmall

    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurrent)
    

    key = cv2.waitKey(1)
    if key == ord('q'):  # Exit with 'q' if you want to manually close it
        break

cap.release()
cv2.destroyAllWindows()
