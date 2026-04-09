from ultralytics import YOLO
import cv2
import pygame

#initialize alert sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert_sound.mp3")

# load model
model = YOLO("optimized150.pt")

# load image
img = cv2.imread("test.jpg")

# run detection
results = model(img)

# show results
for r in results:
    img = r.plot()
    if len(r.boxes) > 0:  # fire/smoke detected
     alert_sound.play()  # play alert sound

cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()