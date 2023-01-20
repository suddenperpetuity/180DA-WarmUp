# started with code from https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html

import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while True:
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # color processing code from https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 150, 20])
    upper1 = np.array([10, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160, 150, 20])
    upper2 = np.array([179, 255, 255])

    # using two different masks
    lower_mask = cv.inRange(hsv, lower1, upper1)
    upper_mask = cv.inRange(hsv, lower2, upper2)
    mask = cv.bitwise_or(lower_mask, upper_mask)

    _, thresh = cv.threshold(mask, 127, 255, 0)
    contours, _ = cv.findContours(thresh, 1, 2)

    # bounding rect code from https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    for c in contours:
        (x, y, w, h) = cv.boundingRect(c)
        c_area = cv.contourArea(c)
        # print(contour_area)
        if c_area < 10000:
            continue
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('frame', frame)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
