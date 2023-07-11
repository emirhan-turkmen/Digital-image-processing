import cv2
import numpy as np


image = cv2.imread('kutu5.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 20, 150)

cv2.imshow("kenar", edges)
cv2.imwrite("kutu5_kenar_goruntu.jpg", edges)

# Kenarları bulunan alanları kapatın
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

cv2.imshow("closed", closed)

contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv2.contourArea)

mask = np.zeros_like(gray)
cv2.drawContours(mask, [largest_contour], 0, (255), thickness=cv2.FILLED)

# adım 3
masked_image = cv2.bitwise_and(image, image, mask=mask)

cv2.imwrite("maskelenmis_goruntu.jpg", masked_image)



median_image = masked_image.copy()

ksize = 3

filtered_image = cv2.medianBlur(median_image, ksize)

cv2.imwrite("median_goruntu.jpg", filtered_image)

cv2.imshow("Evrak", image)
cv2.imshow("Maskelenmiş Görüntü", masked_image)
cv2.imshow("median filtre", filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
