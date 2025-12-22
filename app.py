import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

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
    # Mapping path model
    model_paths = {
        "CNN Manual": "model/cnn_leaf_model.keras",
        "MobileNetV2": "model/mobilenetv2_leaf_model.keras",
        "ResNet50": "model/resnet50_leaf_model.keras",
        "VGG16": "model/vgg16_leaf_model.keras"
    }
    
    path = model_paths[model_key]
    
    if not os.path.exists(path):
        st.error(f"File model {path} tidak ditemukan!")
        return None

    # Bersihkan session lama untuk menghemat RAM sebelum load model baru
    tf.keras.backend.clear_session()
    
    # Load model
    return tf.keras.models.load_model(path, compile=False)

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
def preprocess_image(img, model_key):
    img = img.resize((224, 224))
    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)

    if model_key == "MobileNetV2":
        img_array = mobilenet_preprocess(img_array)
    elif model_key == "ResNet50":
        img_array = resnet_preprocess(img_array)
    elif model_key == "VGG16":
        img_array = vgg_preprocess(img_array)
    else:  # CNN Manual
        img_array = img_array / 255.0

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
                processed_img = preprocess_image(image, model_name)
                
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