# ============================================================
# SEGMENTASI CITRA DIGITAL - TEKNIK THRESHOLDING
# Ujian Tengah Semester (UTS) Pengolahan Citra Digital
# Dibuat oleh: Kessya Shalsabilla Fahlevi
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Membaca dan Mengonversi Citra ke Grayscale
# ------------------------------------------------------------
image = cv2.imread('images/document.jpg')

if image is None:
    print("Gambar tidak ditemukan! Pastikan file ada di folder 'images/'.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ------------------------------------------------------------
# 2. Global Thresholding (Ambang Tetap)
# ------------------------------------------------------------
T = 127
_, thresh_global = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)

# ------------------------------------------------------------
# 3. Otsu’s Thresholding (Ambang Otomatis)
# ------------------------------------------------------------
_, thresh_otsu = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
print("Nilai threshold optimal (Otsu):", _)

# ------------------------------------------------------------
# 4. Adaptive Thresholding (Mean & Gaussian)
# ------------------------------------------------------------
# a. Adaptive Mean
thresh_mean = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

# b. Adaptive Gaussian
thresh_gaussian = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

# ------------------------------------------------------------
# 5. Workflow Segmentasi Dokumen Lengkap
# ------------------------------------------------------------
def segment_document(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cleaned


# ------------------------------------------------------------
# 6. Menampilkan Hasil Perbandingan Thresholding
# ------------------------------------------------------------
methods = {
    'Global (T=127)': thresh_global,
    'Otsu': thresh_otsu,
    'Adaptive Mean': thresh_mean,
    'Adaptive Gaussian': thresh_gaussian
}

plt.figure(figsize=(12, 8))
for i, (name, result) in enumerate(methods.items(), 1):
    plt.subplot(2, 2, i)
    plt.imshow(result, cmap='gray')
    plt.title(name)
    plt.axis('off')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. Menampilkan Hasil Segmentasi Dokumen Lengkap
# ------------------------------------------------------------
segmented_result = segment_document('images/document.jpg')

plt.figure(figsize=(6, 6))
plt.imshow(segmented_result, cmap='gray')
plt.title('Hasil Segmentasi Dokumen Lengkap')
plt.axis('off')
plt.show()
