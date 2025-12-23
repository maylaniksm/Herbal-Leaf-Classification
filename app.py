import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Leaf Disease Classification",
    page_icon="🍃",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <h1 style="text-align:center;">🍃 Leaf Disease Classification</h1>
    <p style="text-align:center; font-size:16px;">
        Sistem Klasifikasi Penyakit Daun Menggunakan Deep Learning
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# =====================================================
# LOAD CLASS NAMES
# =====================================================
@st.cache_data
def load_class_names():
    with open("model/class_names.json", "r") as f:
        return json.load(f)

class_names = load_class_names()

# =====================================================
# LOAD MODELS (LAZY LOADING - EFISIEN RAM)
# =====================================================
@st.cache_resource
def load_specific_model(model_key):
    model_paths = {
        "CNN Manual": "model/cnn_leaf_model.keras",
        "MobileNetV2": "model/mobilenetv2_leaf_model.keras",
        "ResNet50": "model/resnet50_leaf_model.keras",
        "VGG16": "model/vgg16_leaf_model.keras"
    }
    path = model_paths[model_key]
    
    if not os.path.exists(path):
        st.error(f"File {path} tidak ditemukan!")
        return None

    tf.keras.backend.clear_session()
    
    # Gunakan try-except yang lebih agresif
    try:
        # Tambahkan custom_objects={} jika kosong
        return tf.keras.models.load_model(path, compile=False)
    except ValueError:
        # Jika ValueError (Input Compatibility), kita paksa Keras memuatnya
        # lewat API fungsional jika Sequential gagal
        import keras
        return keras.saving.load_model(path, compile=False)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.markdown("## ⚙️ Konfigurasi Model")
st.sidebar.info(
    "Pilih arsitektur model deep learning yang akan digunakan."
)

# User memilih nama model
model_name = st.sidebar.selectbox(
    "Model Deep Learning",
    ["CNN Manual", "MobileNetV2", "ResNet50", "VGG16"]
)

# Proses Loading Model hanya yang dipilih
with st.sidebar:
    with st.spinner(f"Loading {model_name}..."):
        model = load_specific_model(model_name)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Catatan Akademik** - Dataset sama untuk semua model  
    - Model menggunakan pre-trained ImageNet  
    - Output berupa confidence score
    """
)

# =====================================================
# PREPROCESS FUNCTION
# =====================================================
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img).astype('float32') # Pastikan float32 di sini
    
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    # NORMALISASI (Sesuai Colab: 1/255)
    img_array = img_array / 255.0

    # Pastikan dimensinya (1, 224, 224, 3)
    img_array = np.reshape(img_array, (1, 224, 224, 3))
    
    return img_array

# =====================================================
# MAIN CONTENT
# =====================================================
st.markdown("### 📤 Upload Gambar Daun")
st.write(
    "Unggah gambar daun tanaman dalam format **JPG / PNG**."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# =====================================================
# PREDICTION LOGIC
# =====================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("### 🖼️ Preview Gambar")
    st.image(image, use_column_width=True)

    st.markdown("---")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        predict_btn = st.button("🔍 Lakukan Prediksi", use_container_width=True)

    if predict_btn:
        if model is not None:
            with st.spinner(f"Menggunakan {model_name} untuk inferensi..."):
                # Preprocessing
                processed_img = preprocess_image(image)
                
                # Predict
                preds = model.predict(processed_img)
                confidence = float(np.max(preds))
                predicted_class = class_names[int(np.argmax(preds))]

                st.success("Prediksi berhasil dilakukan")

                st.markdown("### 🧠 Hasil Klasifikasi")
                st.markdown(
                    f"""
                    <div style="
                        background-color:#f9f9f9;
                        padding:20px;
                        border-radius:10px;
                        border-left:5px solid #2ecc71;
                    ">
                        <p><b>Model</b> : {model_name}</p>
                        <p><b>Predicted Class</b> : {predicted_class}</p>
                        <p><b>Confidence Score</b> : {confidence*100:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.error("Model gagal dimuat. Periksa log atau file model Anda.")

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:13px; color:gray;">
        © 2025 | Leaf Disease Classification System  
        <br>
        Deep Learning – CNN, MobileNetV2, ResNet50, VGG16
    </p>
    """,
    unsafe_allow_html=True
)