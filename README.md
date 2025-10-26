
---

# SegmentasiCitra — Implementasi Teknik Thresholding Menggunakan VSCode

Proyek ini dibuat sebagai bagian dari **Ujian Tengah Semester (UTS)** mata kuliah **Pengolahan Citra Digital**.
Fokus pembahasan adalah **segmentasi citra menggunakan berbagai teknik thresholding** untuk memisahkan objek dari latar belakang.
Seluruh proses percobaan dilakukan menggunakan **Visual Studio Code (VSCode)** dengan bahasa **Python** serta pustaka **OpenCV**, **NumPy**, dan **Matplotlib**.

---

## Tentang Proyek Ini

Praktikum ini menitikberatkan pada penerapan metode thresholding untuk memisahkan area penting pada citra (foreground) dari latar belakang (background).
Hasil segmentasi ini menjadi dasar untuk berbagai aplikasi computer vision seperti:

* Pengenalan teks (OCR)
* Deteksi objek
* Analisis dokumen atau pola

Tiga metode utama thresholding yang digunakan adalah:

1. **Global Thresholding** – menggunakan satu nilai ambang tetap untuk seluruh piksel.
2. **Otsu’s Method** – menentukan nilai ambang otomatis berdasarkan distribusi histogram.
3. **Adaptive Thresholding** – menetapkan nilai ambang berbeda untuk setiap area lokal citra.

---

## Sasaran Pembelajaran

Setelah menyelesaikan proyek ini, mahasiswa diharapkan dapat:

1. Memahami prinsip dasar segmentasi citra.
2. Menjelaskan cara kerja berbagai metode thresholding.
3. Menganalisis kelebihan dan kekurangan setiap metode.
4. Menerapkan metode thresholding secara praktis menggunakan VSCode.

---

## Persiapan Awal

### Struktur Folder

Buat struktur direktori proyek seperti berikut:

```
SegmentasiCitra/
│
├── images/
│   └── document.jpg        ← contoh gambar uji
│
├── main.py
└── README.md
```

### Instalasi Library

Jalankan perintah berikut di **terminal VSCode**:

```bash
pip install opencv-python matplotlib numpy
```

---

## Pengantar Teori

**Segmentasi citra** adalah proses membagi citra menjadi beberapa region yang memiliki makna tertentu.
Dengan thresholding, piksel citra dibagi menjadi dua kelompok: **objek (foreground)** dan **latar belakang (background)**.

Rumus dasar thresholding:

```
g(x, y) = 1 jika f(x, y) > T
g(x, y) = 0 jika f(x, y) ≤ T
```

Keterangan:
`T` adalah nilai ambang (threshold), `f(x, y)` intensitas piksel asli, dan `g(x, y)` hasil biner (0 atau 1).

---

## Uraian Program Utama (`main.py`)

### 1. Import Library

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

Ketiga library ini digunakan untuk:

* `cv2`: operasi pemrosesan citra (baca, ubah warna, thresholding)
* `numpy`: operasi numerik dan pembuatan kernel
* `matplotlib`: menampilkan hasil citra secara visual

---

### 2. Membaca dan Mengonversi Citra

```python
image = cv2.imread('images/document.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

Citra dibaca dari folder `images` dan dikonversi menjadi grayscale agar thresholding bisa dilakukan berdasarkan nilai intensitas.

---

### 3. Global Thresholding (Ambang Tetap)

```python
T = 127
_, thresh_global = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
```

Metode paling sederhana dengan satu nilai ambang.
Jika intensitas piksel > 127 maka menjadi putih (255), jika ≤ 127 maka menjadi hitam (0).
Kelemahan: tidak bekerja baik pada pencahayaan tidak merata.

---

### 4. Otsu’s Thresholding (Ambang Otomatis)

```python
_, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Nilai threshold optimal (Otsu):", _)
```

Otsu’s method secara otomatis mencari nilai ambang yang memaksimalkan perbedaan antar kelas (foreground dan background).
Cocok untuk citra dengan histogram dua puncak (bimodal).

---

### 5. Adaptive Thresholding (Ambang Lokal)

```python
# Mean Adaptive
thresh_mean = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Gaussian Adaptive
thresh_gaussian = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
```

Berbeda dari metode sebelumnya, setiap area kecil memiliki nilai ambang tersendiri.
Parameter penting:

* `blockSize`: ukuran area lokal (harus ganjil, misal 11)
* `C`: nilai konstan untuk mengatur batas ambang

Kelebihan: efektif untuk citra dengan pencahayaan tidak rata.
Kekurangan: waktu proses lebih lama.

---

### 6. Workflow Segmentasi Lengkap

```python
def segment_document(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    binary = cv2.adaptiveThreshold(blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cleaned
```

Langkah-langkah:

1. **Preprocessing** — ubah ke grayscale dan haluskan dengan Gaussian Blur.
2. **Thresholding adaptif** — segmentasi berdasarkan area lokal.
3. **Morphological cleaning** — memperbaiki noise dan area kecil.

---

### 7. Visualisasi Hasil

```python
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
```

Menampilkan hasil keempat metode dalam satu tampilan agar mudah dibandingkan secara visual.

---

## Analisis Hasil

| Metode              | Keunggulan Utama       | Kondisi Ideal                          | Kelemahan                      |
| ------------------- | ---------------------- | -------------------------------------- | ------------------------------ |
| Global Thresholding | Cepat dan sederhana    | Pencahayaan merata                     | Gagal jika ada shadow          |
| Otsu’s Method       | Nilai ambang otomatis  | Histogram bimodal                      | Tidak stabil pada noise tinggi |
| Adaptive Mean       | Sesuai tiap area lokal | Pencahayaan tidak rata                 | Agak lambat                    |
| Adaptive Gaussian   | Lebih halus dan akurat | Dokumen dengan bayangan atau highlight | Perlu atur parameter           |

Dari hasil percobaan, **Otsu** bekerja baik pada citra dengan kontras tinggi, sedangkan **Adaptive Gaussian** unggul untuk citra dokumen dengan pencahayaan tidak seragam.

---

## Cara Menjalankan Program

1. Letakkan gambar uji di dalam folder `images/` (misal `document.jpg`).
2. Jalankan terminal di VSCode.
3. Ketik perintah:

   ```bash
   python main.py
   ```
4. Hasil akan muncul dalam jendela visual yang menampilkan perbandingan metode thresholding.

---

## Kesimpulan

1. Thresholding merupakan teknik dasar dalam segmentasi citra.
2. Otsu’s Method mampu menentukan nilai ambang otomatis.
3. Adaptive Thresholding efektif untuk kondisi pencahayaan tidak seragam.
4. Kombinasi thresholding dan operasi morfologi memberikan hasil segmentasi yang lebih bersih.
5. Pemilihan metode bergantung pada karakteristik citra yang diuji.

---

## Catatan Pribadi

Selama percobaan, saya menemukan bahwa **parameter blockSize dan C** sangat berpengaruh terhadap kualitas hasil.
Nilai kecil memberikan detail tinggi namun rawan noise, sedangkan nilai besar menghasilkan area halus namun kehilangan detail kecil.
Menentukan parameter ideal menjadi bagian penting dalam proses segmentasi yang efektif.

---


## Tentang Penulis

Proyek ini dikerjakan oleh **Kessya Shalsabilla Fahlevi dengan NIM 231011400427 kelas 05TPLP009** sebagai bagian proyek dari UTS mata kuliah **Pengolahan Citra Digital**.
Dosen pengampu: **Dr. Arya Adhyaksa Waskita**


