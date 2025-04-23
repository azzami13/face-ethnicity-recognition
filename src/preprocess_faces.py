import os
import cv2
import numpy as np

# Load pre-trained model for face detection (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fungsi untuk memproses gambar
def process_image(image_path, output_path_prefix):
    print(f"Memproses gambar: {image_path}")
    
    # Membaca gambar
    img = cv2.imread(image_path)

    if img is None:
        print(f"Gambar tidak dapat dibaca atau rusak: {image_path}")
        return
    else:
        print(f"Gambar berhasil dibaca: {image_path}")
    
    # Mengubah gambar ke grayscale untuk deteksi wajah
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Jumlah wajah yang terdeteksi: {len(faces)}")

    if len(faces) == 0:
        print(f"Tidak ada wajah ditemukan di {image_path}")
        return

    print(f"Wajah ditemukan, memotong dan mengubah ukuran...")
    # Menggunakan wajah pertama yang ditemukan
    for (x, y, w, h) in faces:
        # Memotong wajah dari gambar
        face_img = img[y:y+h, x:x+w]
        
        # Menambahkan filter dasar pada gambar (contoh: konversi warna, kontras, pencahayaan)
        face_img = apply_filters(face_img)
        
        # Menyimpan gambar yang sudah diproses
        save_processed_images(face_img, output_path_prefix)

# Fungsi untuk menerapkan filter dasar pada gambar (konversi warna, pencahayaan, dan kontras)
def apply_filters(img):
    # Mengubah gambar menjadi lebih terang (menambah pencahayaan)
    brightness = 30
    img_bright = cv2.convertScaleAbs(img, beta=brightness)
    
    # Mengubah gambar menjadi lebih kontras dengan mengatur alpha
    alpha = 2.0  # Kontras
    img_contrast = cv2.convertScaleAbs(img_bright, alpha=alpha)
    
    return img_contrast

# Fungsi untuk menyimpan hasil pemrosesan gambar (filter: Gaussian, Median, Histogram Equalization)
def save_processed_images(img, output_path_prefix):
    # Histogram Equalization
    img_hist = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cv2.imwrite(f"{output_path_prefix}_histogram.jpg", img_hist)
    print("Histogram equalization selesai")
    
    # Gaussian Blur (Filter Penghalusan)
    img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)  # Ukuran kernel 5x5
    cv2.imwrite(f"{output_path_prefix}_gaussian.jpg", img_gaussian)
    print("Gaussian blur selesai")
    
    # Median Filter (Filter Penghalusan)
    img_median = cv2.medianBlur(img, 5)  # Ukuran kernel 5
    cv2.imwrite(f"{output_path_prefix}_median.jpg", img_median)
    print("Median filter selesai")
    

dataset_folder = r'C:\Users\tehbo\Downloads\PCD ETS non final 1\face-ethnicity-detector\dataset_source'  
output_folder = r'C:\Users\tehbo\Downloads\PCD ETS non final 1\face-ethnicity-detector\processed_images' 

if not os.path.exists(dataset_folder):
    print(f"Folder dataset tidak ditemukan: {dataset_folder}")
else:
    print(f"Folder dataset ditemukan: {dataset_folder}")

os.makedirs(output_folder, exist_ok=True)

files = os.listdir(dataset_folder)
print(f"File yang ditemukan: {files}")

for filename in files:
    file_path = os.path.join(dataset_folder, filename)
    print(f"Membaca file: {filename}")

    if os.path.isfile(file_path):  
        valid_extensions = (".jpg", ".jpeg", ".png")
        if filename.lower().endswith(valid_extensions):
            print(f"Memproses gambar: {filename}")  
            output_path_prefix = os.path.join(output_folder, f"{filename.split('.')[0]}")
            process_image(file_path, output_path_prefix)
        else:
            print(f"File {filename} tidak diproses karena tidak memiliki ekstensi yang valid: {filename}")
