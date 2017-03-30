import cv2
import imutils

cam = cv2.VideoCapture(0)
firstFrame = None
# fgbg = cv2.createBackgroundSubtractorMOG2()
count = 0

while cam.isOpened():
    success, image = cam.read()
    image = imutils.resize(image, width=500)
    img2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # converts image to gray & returns another image
    img2 = cv2.GaussianBlur(img2, (21, 21), 0)

    if firstFrame is None:
        firstFrame = img2
        continue

    frameDelta = cv2.absdiff(firstFrame, img2)
    img2 = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    # img3 = fgbg.apply(img2)
    cv2.imwrite("frame%d.jpg" % count, img2)  # save frame as JPEG file+
    count += 1

    cv2.imshow('mog', img2)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
