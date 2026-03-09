import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("gambar.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

equalized = cv2.equalizeHist(gray)

avg_blur = cv2.blur(gray, (5,5))

median_blur = cv2.medianBlur(gray, 5)

kernel_sharpen = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
])

sharpened = cv2.filter2D(gray, -1, kernel_sharpen)

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

sobel = cv2.magnitude(sobelx, sobely)
sobel = cv2.convertScaleAbs(sobel)

plt.figure(figsize=(10,6))

plt.subplot(2,3,1)
plt.title("Original")
plt.imshow(gray, cmap='gray')
plt.axis("off")

plt.subplot(2,3,2)
plt.title("Equalized")
plt.imshow(equalized, cmap='gray')
plt.axis("off")

plt.subplot(2,3,3)
plt.title("Avg Blur")
plt.imshow(avg_blur, cmap='gray')
plt.axis("off")

plt.subplot(2,3,4)
plt.title("Median Blur")
plt.imshow(median_blur, cmap='gray')
plt.axis("off")

plt.subplot(2,3,5)
plt.title("Sharpened")
plt.imshow(sharpened, cmap='gray')
plt.axis("off")

plt.subplot(2,3,6)
plt.title("Sobel")
plt.imshow(sobel, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()