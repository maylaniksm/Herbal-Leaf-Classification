<h1 align="center">Herbal Leaf Classification</h1>
---
<p align="center">
  <img src="assets/images/cover.webp" width="70%">
</p>

<p align="center">
  Sumber Image : <a href="https://www.freepik.com/free-photos-vectors/herbal-leaves">Access Here</a>
</p>

---

<h1 align="center">📑 Table of Contents 📑</h1>

- [Deskripsi Proyek](#deskripsi-proyek)
- [Latar Belakang](#latar-belakang)
- [Tujuan Pengembangan](#tujuan-pengembangan)
- [Sumber Dataset](#sumber-dataset)
- [Preprocessing dan Pemodelan](#preprocessing-dan-pemodelan)
  - [Preprocessing Data](#preprocessing-data)
  - [Pemodelan](#pemodelan)
- [Hasil & Evaluasi](#hasil--evaluasi)
- [Dashboard](#dashboard)

---

<h1 id="deskripsi-proyek" align="center">🌿 Klasifikasi Daun Herbal Indonesia 🌿</h1>

Proyek ini berfokus pada pengembangan sistem klasifikasi citra otomatis untuk mengenali berbagai jenis daun herbal khas Indonesia menggunakan pendekatan Deep Learning. Sistem ini memanfaatkan citra daun sebagai input dan bertujuan untuk membantu proses identifikasi tanaman herbal secara cepat, konsisten, dan akurat, khususnya dalam konteks edukasi, penelitian, dan pengembangan sistem pertanian cerdas.

Dataset yang digunakan dalam penelitian ini adalah <b>Indonesian Herb Leaf Dataset 3500</b> yang diperoleh dari Mendeley Data. Dataset ini berisi ribuan citra daun herbal yang diambil dalam berbagai kondisi pencahayaan dan sudut pandang, sehingga menantang model untuk belajar fitur visual yang representatif.

Untuk memenuhi kebutuhan jumlah data pada Ujian Akhir Praktikum (UAP) dan meningkatkan kemampuan generalisasi model, dilakukan proses <i>data augmentation</i> seperti rotasi, pergeseran, zoom, shear, dan flip horizontal. Setelah proses augmentasi, total dataset diperluas hingga lebih dari 5.500 citra dan kemudian dibagi ke dalam data latih, validasi, dan uji.

Model dilatih untuk mengklasifikasikan citra daun ke dalam 10 kelas tanaman herbal berikut:

<ul>
  <li><b>Kemangi</b></li>
  <li><b>Belimbing Wuluh</b></li>
  <li><b>Jeruk Nipis</b></li>
  <li><b>Nangka</b></li>
  <li><b>Sirih</b></li>
  <li><b>Lidah Buaya</b></li>
  <li><b>Seledri</b></li>
  <li><b>Jambu Biji</b></li>
  <li><b>Pandan</b></li>
  <li><b>Pepaya</b></li>
</ul>

Setiap citra diproses dengan ukuran input <b>224×224 piksel</b> dan dilakukan normalisasi nilai piksel agar sesuai dengan kebutuhan arsitektur Deep Learning modern.

Untuk memperoleh performa klasifikasi yang optimal, proyek ini membandingkan empat pendekatan arsitektur model Convolutional Neural Network (CNN) yang berbeda, yaitu:

<ul>
  <li>
    <b>1. Custom CNN (Baseline Model)</b><br>
    Model CNN yang dibangun secara manual sebagai baseline, terdiri dari beberapa lapisan Convolution, MaxPooling, dan Fully Connected. Model ini digunakan untuk memahami kemampuan dasar CNN dalam mengekstraksi fitur visual daun herbal.
  </li>
  <br>
  <li>
    <b>2. MobileNetV2 (Transfer Learning)</b><br>
    Arsitektur ringan yang dioptimalkan untuk efisiensi komputasi. MobileNetV2 digunakan dengan bobot pre-trained ImageNet dan dikombinasikan dengan lapisan klasifikasi baru untuk meningkatkan akurasi dengan waktu pelatihan yang lebih cepat.
  </li>
  <br>
  <li>
    <b>3. ResNet50 (Deep Residual Network)</b><br>
    Model deep learning dengan konsep residual connection yang memungkinkan pelatihan jaringan yang lebih dalam tanpa degradasi performa. ResNet50 digunakan untuk menangkap fitur kompleks pada pola daun.
  </li>
  <br>
  <li>
    <b>4. VGG16 (Advanced Transfer Learning)</b><br>
    Arsitektur CNN klasik dengan struktur yang dalam dan konsisten. VGG16 dimanfaatkan sebagai pembanding performa model transfer learning berbasis layer konvolusi berurutan.
  </li>
</ul>

Melalui perbandingan keempat model ini, proyek ini bertujuan untuk menganalisis pengaruh arsitektur jaringan terhadap performa klasifikasi daun herbal serta menentukan model terbaik berdasarkan akurasi dan stabilitas pada data validasi dan pengujian.


---

<h1 id="latar-belakang" align="center">🧠 Latar Belakang 🧠</h1>

Penyakit Alzheimer merupakan gangguan neurodegeneratif progresif yang menjadi penyebab paling umum dari demensia di seluruh dunia. Penyakit ini menyebabkan penyusutan sel otak secara bertahap (atrofi), yang berdampak pada penurunan daya ingat, kemampuan berpikir, hingga perubahan perilaku secara drastis.
Tantangan utama dalam penanganan Alzheimer adalah gejalanya yang seringkali dianggap sebagai proses penuaan normal pada tahap awal (Very Mild Demented). Padahal, intervensi medis akan jauh lebih efektif jika dilakukan sebelum kerusakan otak meluas ke stadium lanjut (Moderate Demented).
Penggunaan citra Magnetic Resonance Imaging (MRI) merupakan salah satu standar emas untuk melihat perubahan struktur otak. Namun, menganalisis ratusan scan MRI secara manual memerlukan ketelitian tinggi, waktu yang lama, dan keahlian spesialis radiologi yang terbatas jumlahnya.

Proyek ini hadir untuk menjawab tantangan tersebut dengan memanfaatkan teknologi Deep Learning. Dengan menggunakan model Custom CNN, MobileNetV2, dan VGG16, sistem ini dikembangkan untuk:

- Otomatisasi Skrining: Membantu tenaga medis dalam melakukan klasifikasi stadium Alzheimer secara cepat dan objektif.
- Deteksi Dini: Mengidentifikasi pola halus pada citra MRI yang mungkin terlewatkan dalam observasi visual manual, terutama pada fase Very Mild.
- Akurasi Diagnostik: Memberikan perbandingan performa antara arsitektur model konvensional dan Transfer Learning untuk mendapatkan hasil klasifikasi yang paling reliabel.

Melalui pendekatan ini, diharapkan proses pemantauan kesehatan saraf pasien dapat dilakukan secara lebih efisien, mendukung keputusan klinis yang lebih tepat, dan membantu perencanaan perawatan pasien dengan lebih baik.

---

<h1 id="tujuan-pengembangan" align="center">🎯 Tujuan Pengembangan 🎯</h1>

- **Mengembangkan sistem klasifikasi citra MRI otak untuk mendeteksi penyakit Alzheimer secara otomatis ke dalam 4 kategori: Non Demented, Very Mild Demented, Mild Demented, dan Moderate Demented.**
- **Mengevaluasi dan membandingkan performa tiga arsitektur Deep Learning, meliputi:**
  - **Custom CNN**: Menguji efektivitas model yang dibangun dari awal dengan lapisan konvolusi mandiri.
  - **MobileNetV2**: Menguji performa model pre-trained yang ringan dan efisien menggunakan teknik partial fine-tuning pada 20 layer terakhir.
  - **VGG16**: Menguji kekuatan arsitektur yang lebih dalam dengan melakukan fine-tuning spesifik pada Block 4 dan Block 5 untuk menangkap detail tekstur otak yang kompleks.
- **Mengoptimalkan akurasi model melalui teknik Transfer Learning dan Fine-Tuning, guna mengatasi keterbatasan data medis dan meningkatkan kemampuan generalisasi model terhadap data baru.**
- **Menerapkan strategi pelatihan yang cerdas dengan menggunakan Callbacks seperti Early Stopping dan Learning Rate Reduction untuk mencegah overfitting dan memastikan konvergensi model yang stabil.**
- **Menyediakan alat skrining awal (Decision Support System) yang dapat membantu tenaga medis dalam mendeteksi indikasi Alzheimer secara objektif, cepat, dan konsisten berdasarkan data citra digital.**
- **Menentukan model terbaik (Best Model) berdasarkan metrik evaluasi seperti Akurasi, Loss, dan Confusion Matrix untuk digunakan sebagai standar dalam deteksi dini penyakit neurodegeneratif.**

---

<h1 id="sumber-dataset" align="center">📊 Sumber Dataset 📊</h1>

Dataset yang digunakan dalam proyek ini adalah **Augmented Alzheimer MRI Dataset**, yang diperoleh dari platform Kaggle. Dataset ini terdiri dari citra medis MRI otak yang telah dikumpulkan dan diproses untuk membantu tugas klasifikasi penyakit neurodegeneratif.

Dataset ini mencakup 4 kelas tingkat keparahan Alzheimer:

- **Non Demented**: Citra otak normal tanpa tanda-tanda atrofi yang signifikan.
- **Very Mild Demented**: Tahap awal di mana gejala mulai muncul secara samar.
- **Mild Demented**: Tahap ringan dengan pola penyusutan jaringan otak yang mulai terlihat jelas.
- **Moderate Demented**: Tahap menengah dengan indikasi atrofi yang kuat pada area hipokampus dan korteks.

Detail Pemrosesan Dataset:

- **Keseimbangan Data: Dataset telah melalui proses augmentation (penambahan data buatan) untuk memastikan setiap kelas memiliki jumlah sampel yang seimbang (1.250 citra per kelas dalam folder training), guna menghindari bias pada model.**
- **Pre-processing: Citra diproses ke dalam format $224 \times 224$ piksel dan dilakukan normalisasi nilai pixel sesuai standar masing-masing arsitektur (VGG16 & MobileNetV2).**
- **Pembagian Data: Dataset dibagi secara sistematis ke dalam tiga bagian: Train, Validation, dan Test.**

Link Original Dataset: [Augmented Alzheimer MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)

---

<h1 id="preprocessing-dan-pemodelan" align="center">🧼 Preprocessing dan Pemodelan 🧼</h1>

<h2 id="preprocessing-data" align="center">✨ Preprocessing Data ✨</h2>

Tahap preprocessing dilakukan secara sistematis menggunakan pipeline tf.data untuk memastikan efisiensi memori dan kecepatan training. Seluruh citra MRI dimuat dengan ukuran 224×224 piksel dalam format warna RGB.

Proses utama dalam tahap ini meliputi:

- **Normalisasi Spesifik Model: Menggunakan fungsi preprocess_input yang berbeda untuk setiap arsitektur. MobileNetV2 memerlukan rentang piksel [-1, 1], sementara VGG16 menggunakan normalisasi berbasis mean ImageNet.**
- **Data Augmentation: Untuk meningkatkan variasi data dan mencegah overfitting, diterapkan teknik rotasi acak (0.05), random zoom (0.05), serta penyesuaian kontras secara real-time pada data training.**
- **Optimization: Dataset dioptimalkan menggunakan fungsi .cache() untuk mempercepat akses data dari memori dan .prefetch(tf.data.AUTOTUNE) agar proses penyiapan data tidak menghambat proses pelatihan model di GPU.**
- **Dataset Splitting: Data dibagi menjadi tiga bagian (Train, Val, Test) untuk memastikan model diuji pada data yang belum pernah dilihat sebelumnya guna menjamin objektivitas hasil.**

<h2 id="pemodelan" align="center">🤖 Pemodelan 🤖</h2>

Penelitian ini menggunakan tiga pendekatan Deep Learning yang berbeda untuk membandingkan efektivitas antara arsitektur sederhana dengan arsitektur state-of-the-art berbasis Transfer Learning.

### 🟦 A. Custom CNN (Baseline)
Model ini dibangun sebagai standar dasar untuk melihat seberapa baik jaringan saraf konvensional menangkap fitur MRI tanpa bantuan pre-trained weights.

- **Arsitektur: 4 Blok Konvolusi dengan jumlah filter yang meningkat (32, 64, 128, 256).**
- **Teknik Khusus: Penggunaan BatchNormalization setelah setiap layer konvolusi untuk menstabilkan proses training dan GlobalAveragePooling2D untuk efisiensi parameter.**

### 🟨 B. MobileNetV2 (Efficient Transfer Learning)
Dipilih karena keseimbangannya yang luar biasa antara kecepatan dan akurasi, sangat cocok untuk implementasi perangkat medis portabel.

- **Partial Fine-Tuning: Membuka 20 layer terakhir untuk melatih ulang bobot agar lebih sensitif terhadap tekstur MRI Alzheimer.**
- **Regularisasi: Menambahkan layer Dense (512 unit) dengan L2 Regularization dan Dropout(0.6) untuk menangani kompleksitas data.**

### 🟥 C. VGG16 (Advanced Fine-Tuning)
Model ini digunakan untuk mengekstraksi fitur yang lebih mendalam melalui arsitektur yang sangat terstruktur.

- **Deep Adaptation: Melakukan unfreeze pada Block 4 dan Block 5 sehingga filter konvolusi tingkat tinggi dapat beradaptasi dengan pola atrofi otak.**
- **Heavy Classifier: Menggunakan dua lapisan Dense besar (1024 dan 512 unit) untuk memastikan seluruh fitur visual yang diekstraksi dapat terklasifikasi dengan tepat ke dalam 4 stadium Alzheimer.**

### 🟩 D. Strategi Optimasi & Pelatihan
Seluruh model dilatih menggunakan:

- **Optimizer: Adam dengan learning rate yang disesuaikan (lebih rendah untuk transfer learning agar tidak merusak bobot asli).**
- **Callbacks: - EarlyStopping: Menghentikan latih jika tidak ada perbaikan pada akurasi validasi.**
- **ReduceLROnPlateau: Menurunkan learning rate saat model mulai jenuh untuk menemukan titik minimum loss yang lebih global.**
- **ModelCheckpoint: Menyimpan versi terbaik dari setiap model secara otomatis.**

---

<h1 id="hasil--evaluasi" align="center">📊 Hasil & Evaluasi 📊</h1>

**Evaluasi Model**

Model dievaluasi menggunakan beberapa metrik, termasuk **classification report** dan **confusion matrix**.

**Classification Report**

Berikut adalah penjelasan tentang metrik yang digunakan dalam classification report:

- **Precision**: Mengukur proporsi prediksi positif yang benar.
- **Recall**: Mengukur proporsi sampel aktual positif yang berhasil diidentifikasi dengan benar.
- **F1-Score**: Rata-rata harmonis dari precision dan recall.
- **Accuracy**: Mengukur keseluruhan performa model.

**Tabel Perbandingan Classification Report**

Berikut adalah perbandingan metrik evaluasi untuk setiap model:

| Model & Pendekatan                | Arsitektur      | Akurasi | Precision | Recall | F1-Score |
|-----------------------------------|-----------------|---------|-----------|--------|----------|
| Custom CNN Baseline               | CNN             | 0.78    | 0.78      | 0.78   | 0.77     |
| MobileNetV2 Fine-Tuning           | MobileNetV2     | 0.86    | 0.85      | 0.86   | 0.85     |
| VGG16 Fine-Tuning                 | VGG16           | 0.84    | 0.89      | 0.84   | 0.84     |

Analisis Singkat:

- MobileNetV2 memberikan performa keseluruhan terbaik dengan akurasi 0.86 (86%), menunjukkan bahwa arsitektur yang ringan dengan Inverted Residuals sangat efektif untuk mengenali pola citra MRI ini.
- VGG16 unggul dalam nilai Precision (0.89), yang berarti model ini sangat baik dalam meminimalkan kesalahan prediksi positif (sangat akurat dalam menentukan stadium tertentu tanpa banyak salah tebak).
- Custom CNN berfungsi sebagai baseline yang cukup solid dengan akurasi 0.78, namun masih di bawah performa model Transfer Learning yang memiliki pengetahuan awal dari ImageNet.

<h2><b>Confusion Matrix 🔴🟢</b></h2>
<p>Di bawah ini adalah confusion matrix untuk 3 model.</p>

<table align="center">
  <tr>
    <td align="center">
      <b>CNN Baseline</b><br>
      <img src="assets/images/Confusion_Matrix_Baseline.PNG" width="350px">
    </td>
    <td align="center">
      <b>MobileNetV2</b><br>
      <img src="assets/images/Confusion_Matrix_MobileNetV2.PNG" width="350px">
    </td>
    <td align="center">
      <b>VGG16</b><br>
      <img src="assets/images/Confusion_Matrix_VGG16.PNG" width="350px">
    </td>
  </tr>
</table>

<h2><b>Learning Curves 📈</b></h2>
<p>Berikut adalah learning curves untuk setiap model.</p>

<table align="center">
  <tr>
    <td align="center">
      <b>CNN Learning Curve</b><br>
      <img src="assets/images/Grafik_CNN.PNG" width="350px">
    </td>
    <td align="center">
      <b>MobileNetV2 Learning Curve</b><br>
      <img src="assets/images/Grafik_MobileNetV2.PNG" width="350px">
    </td>
    <td align="center" colspan="2">
      <b>VGG16 Learning Curve</b><br>
      <img src="assets/images/Grafik_VGG16.PNG" width="350px">
    </td>
  </tr>
</table>

<h1 id="dashboard" align="center">🧠 Alzheimer Disease Stage Diagnostics 🧠</h1>

<p align="center">
  <a href="https://alzheimer-brain-mri-classification.streamlit.app/" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
  </a>
</p>

<p align="center">
  <strong>Live Demo:</strong> 
  <a href="https://alzheimer-brain-mri-classification.streamlit.app/">alzheimer-brain-mri-classification.streamlit.app</a>
</p>

**GrowthVision AI** adalah sistem berbasis web yang dirancang untuk melakukan klasifikasi morfologi wajah pada anak guna mendukung analisis pertumbuhan pediatrik. Proyek ini memanfaatkan teknologi *Deep Learning* dengan arsitektur **CNN** dan **EfficientNet-B0** yang dioptimalkan menggunakan teknik **LoRA (Low-Rank Adaptation)**.

---

## 🚀 Fitur Utama
- **Batch Processing**: Mampu melakukan analisis hingga 20 subjek secara acak sekaligus.
- **Inference Models**: Pilihan arsitektur model antara EfficientNet + LoRA, CNN Fine-Tuning, atau SVM Klasik.
- **Visualisasi Real-time**: Hasil prediksi dilengkapi dengan *Confidence Score* menggunakan Gauge Chart interaktif.
- **Export Data**: Pengguna dapat mengunduh hasil analisis dalam format CSV untuk keperluan statistik lebih lanjut.

---

## 🛠️ Cara Menggunakan Dashboard

### 1. Memilih Sumber Data
Terdapat dua metode input pada panel kiri (Sidebar):
* **Sampel Acak GitHub**: Sistem akan mengambil 20 gambar secara acak dari dataset penelitian yang tersimpan di folder `samples`.
* **Upload Manual**: Pengguna dapat mengunggah foto subjek sendiri (format .jpg, .png, atau .jpeg).

### 2. Menjalankan Analisis
* Pilih arsitektur model yang diinginkan pada menu drop-down.
* Klik tombol **🚀 RUN INFERENCE**.
* Tunggu hingga progress bar mencapai 100%.

### 3. Membaca Hasil
* **Classification Summary**: Ringkasan total jumlah subjek yang terdeteksi sebagai **VP-0 (Proportional)** dan **VP-1 (Linear)**.
* **Individual Analysis**: Detail hasil per gambar lengkap dengan persentase keyakinan model.
* **Download Report**: Klik tombol unduh di bagian bawah untuk menyimpan tabel hasil.

---

## 📂 Struktur Repositori
- `app.py`: File utama aplikasi Streamlit.
- `samples/`: Folder berisi dataset gambar sampel untuk demo.
- `requirements.txt`: Daftar library Python yang dibutuhkan (PyTorch, Streamlit, Plotly, dll).
- `README.md`: Dokumentasi proyek.

---

## 🔬 Metodologi & Riset
Sistem ini dikembangkan sebagai bagian dari tugas besar mata kuliah **Machine Learning**. Fokus riset ini adalah mengimplementasikan teknik *transfer learning* dan efisiensi model melalui **LoRA** untuk mengenali fitur morfologi wajah yang berkaitan dengan pola pertumbuhan (Visual Proxy) pada anak-anak.

---

## ⚖️ Lisensi
Proyek ini didistribusikan di bawah **MIT License**. Data yang digunakan dalam demo ini bertujuan untuk kepentingan edukasi dan riset teknologi *screening* awal non-medis.

---
**© 2024 | Machine Learning**
