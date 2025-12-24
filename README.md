<h1 align="center">Herbal Leaf Classification</h1>
---
<p align="center">
  <img src="assets/images/cover.avif" width="70%">
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

<h1 id="latar-belakang" align="center">🌿 Latar Belakang 🌿</h1>

Indonesia merupakan negara dengan kekayaan hayati yang sangat melimpah, termasuk berbagai jenis tanaman herbal yang telah lama dimanfaatkan dalam pengobatan tradisional, industri pangan, dan kosmetik. Setiap tanaman herbal memiliki ciri khas daun yang berbeda, baik dari segi bentuk, tekstur, maupun pola warna, yang menjadi kunci utama dalam proses identifikasi.

Namun, proses identifikasi daun herbal secara manual masih memiliki berbagai tantangan. Beberapa jenis daun memiliki kemiripan visual yang tinggi, sehingga rawan terjadi kesalahan klasifikasi, terutama bagi masyarakat awam atau praktisi non-ahli. Selain itu, faktor pencahayaan, sudut pengambilan gambar, dan kualitas citra juga dapat mempengaruhi ketepatan identifikasi secara visual.

Perkembangan teknologi Computer Vision dan Deep Learning membuka peluang untuk mengatasi permasalahan tersebut. Dengan memanfaatkan citra daun sebagai data masukan, model Convolutional Neural Network (CNN) mampu mengekstraksi fitur visual penting secara otomatis, seperti pola tulang daun, tekstur permukaan, dan karakteristik warna, yang sulit diukur secara manual.

Dalam proyek ini, sistem klasifikasi daun herbal dikembangkan menggunakan pendekatan Deep Learning dengan memanfaatkan dataset <b>Indonesian Herb Leaf Dataset 3500</b>. Untuk meningkatkan jumlah data dan kemampuan generalisasi model, dilakukan proses data augmentation sehingga dataset memenuhi kebutuhan pelatihan model berskala besar.

Proyek ini mengimplementasikan dan membandingkan beberapa arsitektur model, yaitu Custom CNN sebagai baseline, serta model Transfer Learning seperti MobileNetV2, ResNet50, dan VGG16, dengan tujuan untuk:

<ul>
  <li><b>Otomatisasi Identifikasi</b>: Membantu proses pengenalan jenis daun herbal secara cepat dan konsisten.</li>
  <li><b>Mengurangi Kesalahan Manual</b>: Meminimalkan kesalahan identifikasi akibat kemiripan visual antar daun.</li>
  <li><b>Evaluasi Arsitektur Model</b>: Menganalisis pengaruh kompleksitas model terhadap performa klasifikasi citra daun herbal.</li>
</ul>

Melalui pendekatan ini, diharapkan sistem klasifikasi daun herbal dapat menjadi dasar pengembangan aplikasi cerdas di bidang pertanian, kesehatan tradisional, dan edukasi, serta mendukung pelestarian dan pemanfaatan tanaman herbal Indonesia secara lebih optimal.

---

<h1 id="tujuan-pengembangan" align="center">🎯 Tujuan Pengembangan 🎯</h1>

<ul>
  <li>
    <b>Mengembangkan sistem klasifikasi citra daun herbal Indonesia secara otomatis</b>
    untuk mengenali dan mengelompokkan citra daun ke dalam 10 kelas tanaman herbal, yaitu:
    Kemangi, Belimbing Wuluh, Jeruk Nipis, Nangka, Sirih, Lidah Buaya, Seledri, Jambu Biji,
    Pandan, dan Pepaya.
  </li>
  <br>

  <li>
    <b>Mengevaluasi dan membandingkan performa beberapa arsitektur Deep Learning</b>,
    meliputi:
    <ul>
      <li><b>Custom CNN</b>: Sebagai baseline model untuk mengukur kemampuan dasar CNN dalam mengekstraksi fitur visual daun.</li>
      <li><b>MobileNetV2</b>: Model transfer learning yang ringan dan efisien untuk meningkatkan akurasi dengan biaya komputasi lebih rendah.</li>
      <li><b>ResNet50</b>: Arsitektur deep residual network untuk mempelajari fitur kompleks pada pola daun.</li>
      <li><b>VGG16</b>: Model CNN klasik yang dalam untuk menangkap detail tekstur dan struktur daun secara lebih mendalam.</li>
    </ul>
  </li>
  <br>

  <li>
    <b>Meningkatkan performa klasifikasi melalui teknik Data Augmentation</b>
    guna memperkaya variasi data latih dan meningkatkan kemampuan generalisasi model
    terhadap citra daun dengan kondisi pencahayaan dan sudut pengambilan yang berbeda.
  </li>
  <br>

  <li>
    <b>Menerapkan teknik Transfer Learning</b> dengan memanfaatkan bobot pre-trained ImageNet
    untuk mengatasi keterbatasan jumlah data dan mempercepat proses konvergensi pelatihan model.
  </li>
  <br>

  <li>
    <b>Mengimplementasikan strategi pelatihan yang optimal</b>
    menggunakan callback seperti <i>Early Stopping</i> untuk mencegah overfitting
    serta memastikan proses training berjalan stabil dan efisien.
  </li>
  <br>

  <li>
    <b>Menentukan model terbaik (Best Model)</b>
    berdasarkan metrik evaluasi seperti Akurasi, Loss, Classification Report,
    dan Confusion Matrix untuk digunakan sebagai acuan dalam sistem identifikasi daun herbal.
  </li>
  <br>

  <li>
    <b>Menyediakan dasar pengembangan sistem pendukung keputusan (Decision Support System)</b>
    yang dapat dimanfaatkan dalam bidang pertanian, edukasi, dan pengenalan tanaman herbal
    berbasis citra digital.
  </li>
</ul>

---

<h1 id="sumber-dataset" align="center">📊 Sumber Dataset 📊</h1>

Dataset yang digunakan dalam proyek ini adalah <b>Indonesian Herb Leaf Dataset 3500</b> yang diperoleh dari platform Mendeley Data. Dataset ini berisi citra daun dari berbagai jenis tanaman herbal Indonesia yang dikumpulkan untuk mendukung penelitian di bidang pengolahan citra dan pengenalan tanaman berbasis kecerdasan buatan.

Dataset asli terdiri dari sekitar 3.500 citra daun yang terbagi ke dalam 10 kelas tanaman herbal, dengan variasi kondisi pencahayaan, sudut pengambilan gambar, serta karakteristik visual daun yang beragam.

Dataset ini mencakup 10 kelas daun herbal berikut:

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

<b>Detail Pemrosesan Dataset:</b>

<ul>
  <li>
    <b>Data Augmentation</b>: Untuk memenuhi kebutuhan jumlah data dan meningkatkan kemampuan generalisasi model, dilakukan proses augmentasi citra menggunakan teknik rotasi, pergeseran (width & height shift), zoom, shear, dan horizontal flip. Proses ini memperluas dataset hingga lebih dari <b>5.500 citra</b>.
  </li>
  <br>

  <li>
    <b>Pre-processing</b>: Seluruh citra diubah ukurannya menjadi <b>224 × 224 piksel</b> dan dilakukan normalisasi nilai piksel ke rentang [0,1] sebelum digunakan dalam proses pelatihan model.
  </li>
  <br>

  <li>
    <b>Pembagian Data</b>: Dataset hasil augmentasi dibagi ke dalam tiga subset, yaitu:
    <ul>
      <li><b>Training</b>: 80%</li>
      <li><b>Validation</b>: 10%</li>
      <li><b>Testing</b>: 10%</li>
    </ul>
    Pembagian dilakukan secara acak dan terpisah untuk setiap kelas guna menjaga proporsi data yang seimbang.
  </li>
</ul>

<b>Link Dataset Asli:</b><br>
<a href="https://data.mendeley.com/datasets/s82j8dh4rr/1" target="_blank">
Indonesian Herb Leaf Dataset 3500 (Mendeley Data)
</a>

---

<h1 id="preprocessing-dan-pemodelan" align="center">🧼 Preprocessing dan Pemodelan 🧼</h1>

<h2 id="preprocessing-data" align="center">✨ Preprocessing Data ✨</h2>

Tahap preprocessing dilakukan secara sistematis menggunakan pipeline <i>tf.data</i> untuk memastikan efisiensi pemrosesan data dan stabilitas proses pelatihan model. Seluruh citra daun herbal dimuat dalam format RGB dan diubah ukurannya menjadi <b>224 × 224 piksel</b>, sesuai dengan kebutuhan arsitektur CNN modern dan model transfer learning berbasis ImageNet.

Proses utama pada tahap preprocessing meliputi:

<ul>
  <li>
    <b>Data Augmentation</b>: Untuk meningkatkan jumlah data dan memperkaya variasi citra, diterapkan teknik augmentasi seperti rotasi, pergeseran horizontal dan vertikal, zoom, shear, dan horizontal flip. Proses ini dilakukan secara <i>offline</i> hingga jumlah dataset mencapai lebih dari 5.500 citra.
  </li>
  <br>

  <li>
    <b>Normalisasi Data</b>: Seluruh citra dinormalisasi ke rentang nilai [0,1] menggunakan layer <i>Rescaling(1./255)</i> agar mempercepat konvergensi dan menjaga kestabilan proses training.
  </li>
  <br>

  <li>
    <b>Dataset Splitting</b>: Dataset hasil augmentasi dibagi menjadi tiga subset, yaitu:
    <ul>
      <li><b>Training</b>: 80%</li>
      <li><b>Validation</b>: 10%</li>
      <li><b>Testing</b>: 10%</li>
    </ul>
    Pembagian dilakukan secara acak dan terpisah untuk setiap kelas agar distribusi data tetap seimbang.
  </li>
  <br>

  <li>
    <b>Pipeline Optimization</b>: Dataset dioptimalkan menggunakan fungsi <i>.cache()</i> dan <i>.prefetch(tf.data.AUTOTUNE)</i> untuk meningkatkan kecepatan akses data dan mencegah bottleneck saat pelatihan model.
  </li>
</ul>

<h2 id="pemodelan" align="center">🤖 Pemodelan 🤖</h2>

Pada tahap pemodelan, penelitian ini membandingkan empat pendekatan arsitektur Deep Learning untuk mengklasifikasikan citra daun herbal Indonesia ke dalam 10 kelas. Model yang digunakan terdiri dari satu model CNN yang dibangun dari awal dan tiga model berbasis <i>Transfer Learning</i> dengan bobot pre-trained ImageNet.

<h3>🟦 A. Custom CNN (Baseline Model)</h3>

Model Custom CNN digunakan sebagai baseline untuk mengevaluasi kemampuan dasar jaringan konvolusional dalam mengekstraksi fitur visual daun tanpa bantuan bobot pre-trained.

<ul>
  <li><b>Arsitektur</b>: Terdiri dari tiga blok Conv2D dan MaxPooling dengan jumlah filter bertingkat (32, 64, 128).</li>
  <li><b>Fully Connected Layer</b>: Dense 128 unit dengan aktivasi ReLU.</li>
  <li><b>Regularisasi</b>: Dropout sebesar 0.3 untuk mengurangi risiko overfitting.</li>
  <li><b>Output Layer</b>: Softmax dengan 10 neuron sesuai jumlah kelas daun herbal.</li>
</ul>

<h3>🟨 B. MobileNetV2 (Transfer Learning)</h3>

MobileNetV2 digunakan sebagai model transfer learning yang efisien dan ringan, cocok untuk klasifikasi citra dengan kebutuhan komputasi yang relatif rendah.

<ul>
  <li><b>Pre-trained Weights</b>: ImageNet.</li>
  <li><b>Feature Extractor</b>: Seluruh layer base model dibekukan (freeze).</li>
  <li><b>Classifier Head</b>: GlobalAveragePooling2D, Dense 256 unit (ReLU), Dropout 0.4.</li>
  <li><b>Output</b>: Dense Softmax 10 kelas.</li>
</ul>

<h3>🟥 C. ResNet50 (Deep Residual Network)</h3>

ResNet50 digunakan untuk mempelajari fitur visual daun yang lebih kompleks melalui arsitektur residual network yang dalam.

<ul>
  <li><b>Pre-trained Weights</b>: ImageNet.</li>
  <li><b>Freeze Base Model</b>: Bobot awal dikunci untuk menjaga fitur umum dari ImageNet.</li>
  <li><b>Classifier Head</b>: GlobalAveragePooling2D, Dense 256 unit (ReLU), Dropout 0.4.</li>
  <li><b>Output</b>: Softmax 10 kelas.</li>
</ul>

<h3>🟩 D. VGG16 (Transfer Learning)</h3>

VGG16 digunakan sebagai pembanding arsitektur CNN klasik yang memiliki struktur konvolusi berurutan dan dalam.

<ul>
  <li><b>Pre-trained Weights</b>: ImageNet.</li>
  <li><b>Freeze Layer</b>: Seluruh layer konvolusi VGG16 dibekukan.</li>
  <li><b>Classifier</b>: Flatten, Dense 256 unit (ReLU), Dropout 0.4, Dense 128 unit (ReLU).</li>
  <li><b>Output</b>: Softmax 10 kelas.</li>
</ul>

<h3>🟪 E. Strategi Optimasi dan Pelatihan</h3>

Seluruh model dilatih menggunakan strategi pelatihan yang konsisten untuk menjaga keadilan perbandingan performa, yaitu:

<ul>
  <li><b>Optimizer</b>: Adam dengan learning rate 0.001 (Custom CNN) dan 0.0001 (Transfer Learning).</li>
  <li><b>Loss Function</b>: Categorical Crossentropy.</li>
  <li><b>Callback</b>: EarlyStopping untuk menghentikan pelatihan ketika performa validasi tidak meningkat.</li>
  <li><b>Evaluation Metrics</b>: Accuracy, Classification Report, dan Confusion Matrix.</li>
</ul>

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

| Model & Pendekatan  | Arsitektur  | Akurasi (%) | Precision (%) | Recall (%) | F1-Score (%) | Hasil Analisis                                                                                                                                                                                          |
| ------------------- | ----------- | ----------- | ------------- | ---------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Custom CNN Baseline | CNN Manual  | 78.31       | 79.06         | 78.31      | 78.38        | Model baseline mampu mempelajari pola dasar daun herbal dengan cukup baik, namun masih terbatas dalam menangkap fitur kompleks sehingga performanya berada di bawah model transfer learning.            |
| MobileNetV2         | MobileNetV2 | 98.16       | 98.20         | 98.16      | 98.15        | Menunjukkan performa terbaik secara keseluruhan. Arsitektur yang efisien dan bobot pre-trained ImageNet sangat efektif dalam mengekstraksi fitur daun herbal dengan tingkat akurasi yang sangat tinggi. |
| ResNet50            | ResNet50    | 68.93       | 70.15         | 68.93      | 69.16        | Performa relatif rendah dibandingkan model lain. Hal ini kemungkinan disebabkan oleh kompleksitas model yang tinggi sehingga kurang optimal pada dataset berukuran terbatas tanpa fine-tuning lanjutan. |
| VGG16               | VGG16       | 95.77       | 95.79         | 95.77      | 95.75        | Memberikan performa yang sangat baik dengan kemampuan menangkap detail tekstur daun. Namun, kebutuhan parameter yang besar membuatnya sedikit kurang efisien dibandingkan MobileNetV2.                  |

<h2><b>Confusion Matrix 🔴🟢</b></h2>
<p>Di bawah ini adalah confusion matrix untuk setiap model.</p>

<table align="center">
  <tr>
    <td align="center">
      <b>CNN Baseline</b><br>
      <img src="assets/images/Confusion_Matrix_CNN.png" width="350px">
    </td>
    <td align="center">
      <b>MobileNetV2</b><br>
      <img src="assets/images/Confusion_Matrix_MobileNetV2.png" width="350px">
    </td>
    <td align="center">
      <b>VGG16</b><br>
      <img src="assets/images/Confusion_Matrix_ResNet50.png" width="350px">
    </td>
    <td align="center">
      <b>VGG16</b><br>
      <img src="assets/images/Confusion_Matrix_VGG16.png" width="350px">
    </td>
  </tr>
</table>

<h2><b>Learning Curves 📈</b></h2>
<p>Berikut adalah learning curves untuk setiap model.</p>

<table align="center">
  <tr>
    <td align="center">
      <b>CNN Learning Curve</b><br>
      <img src="assets/images/Grafik_CNN.png" width="350px">
    </td>
    <td align="center">
      <b>MobileNetV2 Learning Curve</b><br>
      <img src="assets/images/Grafik_MobileNetV2.png" width="350px">
    </td>
    <td align="center" colspan="2">
      <b>ResNet50 Learning Curve</b><br>
      <img src="assets/images/Grafik_ResNet50.png" width="350px">
    </td>
    <td align="center" colspan="2">
      <b>VGG16 Learning Curve</b><br>
      <img src="assets/images/Grafik_VGG16.png" width="350px">
    </td>
  </tr>
</table>

<h2><b>Visualisasi Perbandingan 📊</b></h2>
<p>Berikut adalah visualisasi perbandingan untuk setiap model.</p>

<table align="center">
  <tr>
    <td align="center">
      <b>Bar Chart Perbandingan Akurasi Model</b><br>
      <img src="assets/images/Perbandingan_Akurasi.png" width="350px">
    </td>
    <td align="center">
      <b>Bar Chart Perbandingan Presisi Recall F1-Score</b><br>
      <img src="assets/images/Perbandingan_PRF1.png" width="350px">
    </td>
  </tr>
</table>

<h1 id="dashboard" align="center">🌿 Herbal Leaf Classification Dashboard 🌿</h1>

<p align="center">
  <a href="https://herbal-leaf-classification.streamlit.app/" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
  </a>
</p>

<p align="center">
  <strong>Live Demo:</strong> 
  <a href="https://herbal-leaf-classification.streamlit.app/">alzheimer-brain-mri-classification.streamlit.app</a>
</p>

<p align="center">
  <strong>Web Dashboard:</strong><br>
  Aplikasi interaktif berbasis <i>Streamlit</i> untuk menguji dan membandingkan performa
  model Deep Learning dalam mengklasifikasikan jenis daun herbal Indonesia.
</p>

## 🧠 Deskripsi Dashboard

<b>Herbal Leaf Classification Dashboard</b> adalah aplikasi berbasis web yang dirancang untuk
melakukan identifikasi otomatis jenis daun herbal Indonesia menggunakan citra digital.
Aplikasi ini memanfaatkan model Deep Learning yang telah dilatih sebelumnya dan disimpan
dalam format <code>.keras</code>.

Dashboard ini memungkinkan pengguna untuk:
- Menguji satu model tertentu secara individual
- Membandingkan hasil prediksi dari seluruh model secara bersamaan
- Melakukan klasifikasi pada satu atau banyak gambar (batch upload)

Model yang tersedia dalam sistem ini meliputi:
- <b>CNN Manual</b>
- <b>MobileNetV2</b>
- <b>ResNet50</b>
- <b>VGG16</b>

---

## 🚀 Fitur Utama
- **Single Model Testing**  
  Menguji satu arsitektur model pilihan (CNN, MobileNetV2, ResNet50, atau VGG16) terhadap gambar daun herbal yang diunggah.

- **Multi-Model Evaluation**  
  Menampilkan hasil prediksi dari seluruh model secara paralel untuk satu gambar yang sama, sehingga memudahkan analisis perbandingan performa.

- **Batch Image Upload**  
  Mendukung unggah banyak gambar sekaligus dan menampilkan hasil klasifikasi untuk setiap gambar secara terpisah.

- **Confidence Score Visualization**  
  Menampilkan tingkat kepercayaan prediksi dalam bentuk persentase dan progress bar.

- **User-Friendly Interface**  
  Antarmuka sederhana dengan sidebar navigasi untuk memudahkan pengguna dalam melakukan eksperimen model.

---

## 🛠️ Cara Menggunakan Dashboard

### 1️⃣ Upload Gambar
- Unggah satu atau beberapa gambar daun herbal melalui menu <b>Upload Center</b> di sidebar.
- Format file yang didukung: <code>.jpg</code>, <code>.jpeg</code>, dan <code>.png</code>.

### 2️⃣ Pilih Mode Analisis
- <b>Uji Single Model</b>  
  Pilih satu arsitektur model untuk melakukan klasifikasi.
- <b>Evaluasi Seluruh Model</b>  
  Menampilkan hasil prediksi dari seluruh model sekaligus untuk perbandingan langsung.

### 3️⃣ Jalankan Analisis
- Sistem akan memproses gambar (resize 224×224 dan normalisasi).
- Model akan melakukan inferensi dan menghasilkan label daun herbal beserta confidence score.

### 4️⃣ Interpretasi Hasil
- <b>Jenis Daun Teridentifikasi</b>: Kelas daun dengan probabilitas tertinggi.
- <b>Confidence Score</b>: Tingkat keyakinan model terhadap prediksi yang dihasilkan.
- Pada mode evaluasi, setiap model ditampilkan dalam panel terpisah untuk memudahkan perbandingan.

---

## 📂 Struktur Repositori
- `app.py`: File utama aplikasi Streamlit.
- `models/`: Folder berisi model-model yang sudah dilatih.
- `requirements.txt`: Daftar library Python yang dibutuhkan.
- `README.md`: Dokumentasi proyek.

---

## 🔬 Metodologi & Implementasi
Aplikasi ini memuat model Deep Learning yang telah dilatih sebelumnya menggunakan dataset
<i>Indonesian Herb Leaf Dataset 3500</i> yang telah melalui proses data augmentation.
Setiap gambar input diproses dengan:
- Konversi ke RGB
- Resize ke 224×224 piksel
- Normalisasi nilai piksel ke rentang [0,1]

Inferensi dilakukan menggunakan TensorFlow/Keras tanpa proses training ulang di sisi aplikasi.

---

## ⚖️ Lisensi
Proyek ini dikembangkan untuk keperluan edukasi dan penelitian akademik.
Seluruh dataset dan model digunakan khusus untuk eksperimen klasifikasi citra daun herbal
dan tidak ditujukan sebagai sistem identifikasi botani medis resmi.

---
**© 2025 | Machine Learning**
