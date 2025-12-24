# jenifer NDU202400100
import cv2
import numpy as np

# Read input image
img = cv2.imread("sample.jpg")
if img is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)


# 1. Contour-based Detection

_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_img = img.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Contour Detection", contour_img)


# 2. Template Matching (Optional)

template = cv2.imread("template.jpg", 0)  # Cropped object template
if template is not None:
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where(res >= threshold)
    tm_img = img.copy()
    for pt in zip(*loc[::-1]):
        cv2.rectangle(tm_img, pt, (pt[0]+template.shape[1], pt[1]+template.shape[0]), (0,0,255), 2)
    cv2.imshow("Template Matching", tm_img)
else:
    print("Template not found. Skipping template matching.")


# 3. Haar Cascade Detection (Optional)

# Example: Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
haar_img = img.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(haar_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imshow("Haar Cascade Detection", haar_img)

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
