# 🏗️ Final Project Data Science Beginner 2025

Berisi source code untuk final project berjudul:
**"Deteksi Pelanggaran K3 di Lokasi Konstruksi Menggunakan Deep Learning"**

Disini saya melakukan klasifikasi gambar pada lokasi konstruksi berdasarkan penggunaan APD (helm dan rompi), ke dalam dua kondisi: "Safe" atau "Unsafe", untuk mendeteksi potensi pelanggaran K3 (Keselamatan dan Kesehatan Kerja).

📂 File:
- final_project_Citra_R.ipynb: Notebook utama berisi training & evaluasi model
- utils :
    - gradcam.py : Untuk membuat hasil heatmap yang menampilkan fokus model pada gambar yang mempengaruhi klasifikasi kelas
    - preprocessing.py : Untuk memproses gambar agar sesuai format yang diterima model
- app.py: Script aplikasi streamlit
- README.md: Panduan singkat

