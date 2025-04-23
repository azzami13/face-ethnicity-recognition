# Face & Ethnicity Detection System

Environment ini digunakan untuk mencari hasil visualisasi dari dua implementasi, yaitu:
1. Face Similarity
2. Ethnicity Detection

Jadi kode disini hanya untuk menghasilkan visual, tidak include dengan API.

Langkah Penggunaan:
1. Install requirement dengan perintah pip instal -r requirement
2. Masukan Dataset pada folder dataset dengan klasifikasi suku/nama/image
3. setelah dataset diklasifikasi, jalan kode processes_face dengan perintah python processes_faces.py
   Output akan membuat folder processed_image dengan didalamnya sudah ada gambar processing dengan 3 augmentasi yaitu Histogram, Median, dan Gaussian.
4. Selanjutnya untuk membuat visualisasi, jalankan kode extract_embedding.py dengan perintah python
   extract_embedding.py , Hasil output akan menjadi file embedding.json
5. Jalankan kode check_similarity dengan perintah python check_similarity.py , output berupa file csv yang berisi
   similarity matriks
6. Jalankan kode visualize_tsne.py untuk mendapatkan visualisasi sebaran kemiripan.
7. Jalankan kode plot_roc_curve.py untuk mendapatkan visualisasi ROC yang berisi perbandingan True Positive dan
   False Positive
8. Jalankan kode clasify_ethnicity.py untuk mendapatkan hasil deteksi berupa akurasi yang berbentuk matriks.
