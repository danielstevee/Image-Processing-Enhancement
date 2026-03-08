import cv2
import numpy as np
import matplotlib.pyplot as plt

# membaca gambar
image = cv2.imread("gambar.jpg")

# ubah ke grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# =========================
# 1. HISTOGRAM EQUALIZATION
# =========================
hist_eq = cv2.equalizeHist(gray)

# =========================
# 2. SPATIAL SMOOTHING
# =========================

# Average filter (Lowpass)
smooth_avg = cv2.blur(gray, (5,5))

# Median filter
smooth_median = cv2.medianBlur(gray, 5)

# =========================
# 3. SPATIAL SHARPENING
# =========================

# Kernel sharpening
kernel_sharpen = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
])

sharpen = cv2.filter2D(gray, -1, kernel_sharpen)

# =========================
# MENAMPILKAN HASIL
# =========================

titles = [
    "Original",
    "Histogram Equalization",
    "Smoothing Average",
    "Median Filter",
    "Sharpening"
]

images = [
    gray,
    hist_eq,
    smooth_avg,
    smooth_median,
    sharpen
]

plt.figure(figsize=(12,6))

for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.show()