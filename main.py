import cv2
import numpy as np

img = cv2.imread('text.jpg')
img = cv2.pyrDown(img)
small = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5, 8), np.uint8)
grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel) #берем градиент

thresh = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_OTSU)[1] #отсекаем пиксели по пороговому значению

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) #сближаем контуры текста
# фильтр специально взят большим по x, маленьким по y, чтобы искать абзацы, а не отдельные слова

contours, _ = cv2.findContours(close.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

n = len(contours)
for idx in range(n):
    x, y, w, h = cv2.boundingRect(contours[idx]) #ограничивааем контуры прямоугольником

    if w > 10 and h > 10:
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 2) #рисуем прямоугольники

cv2.imwrite("./result.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
#В решении есть проблема: алгоритм выделяет абзацы раньше, чем они на самом деле начинаются.
# Это происходит из-за того, что из-за заглавной буквы абзацы "слипаются".
# Для того, чтобы решить эту проблему, надо как-то ориентироваться на пустое место в конце абзаца,
# но к сожалению я не смогла реализовать это в срок.