import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import pandas as pd
from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Leaf Disease Multi-Model Comparison",
    page_icon="🍃",
    layout="wide", # Diubah ke wide agar perbandingan kolom lebih enak dilihat
    initial_sidebar_state="expanded"
)

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <h1 style="text-align:center;">🍃 Leaf Disease Classification System</h1>
    <p style="text-align:center; font-size:16px;">
        Bandingkan hasil prediksi dari berbagai arsitektur Deep Learning secara real-time.
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
# LOAD MODELS (OPTIMIZED FOR COMPARISON)
# =====================================================
def load_specific_model(model_key):
    model_paths = {
        "CNN Manual": "model/cnn_leaf_model.keras",
        "MobileNetV2": "model/mobilenetv2_leaf_model.keras",
        "ResNet50": "model/resnet50_leaf_model.keras",
        "VGG16": "model/vgg16_leaf_model.keras"
    }
    path = model_paths[model_key]
    
    if not os.path.exists(path):
        return None

    # Membersihkan session sebelumnya agar RAM tidak penuh
    tf.keras.backend.clear_session()
    
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        import keras
        return keras.saving.load_model(path, compile=False)

# =====================================================
# PREPROCESS FUNCTION
# =====================================================
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =====================================================
# SIDEBAR & UPLOAD
# =====================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/892/892926.png", width=100)
st.sidebar.markdown("## 📤 Upload Center")

# Fitur Batch Upload
uploaded_files = st.sidebar.file_uploader(
    "Pilih satu atau beberapa gambar",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

st.sidebar.info(f"Terdeteksi: {len(uploaded_files)} gambar")

# =====================================================
# MAIN CONTENT
# =====================================================
if not uploaded_files:
    st.info("Silakan unggah gambar daun di sidebar untuk memulai analisis.")
else:
    # List model yang akan dijalankan
    model_list = ["CNN Manual", "MobileNetV2", "ResNet50", "VGG16"]

    # Loop setiap gambar yang di-upload
    for i, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        
        with st.expander(f"🖼️ Analisis Gambar: {uploaded_file.name}", expanded=True):
            col_img, col_info = st.columns([1, 2])
            
            with col_img:
                st.image(image, caption="Original Image", use_container_width=True)
            
            with col_info:
                st.write("### Perbandingan Model")
                if st.button(f"🔍 Jalankan Semua Model ({uploaded_file.name})", key=f"btn_{i}"):
                    processed_img = preprocess_image(image)
                    
                    # Layout kolom untuk hasil
                    res_cols = st.columns(len(model_list))
                    
                    for idx, m_name in enumerate(model_list):
                        with st.spinner(f"Running {m_name}..."):
                            model = load_specific_model(m_name)
                            
                            if model is not None:
                                preds = model.predict(processed_img, verbose=0)
                                confidence = float(np.max(preds))
                                predicted_class = class_names[int(np.argmax(preds))]
                                
                                # Tampilan Card Hasil
                                color = "#2ecc71" if confidence > 0.8 else "#f1c40f"
                                with res_cols[idx]:
                                    st.markdown(f"""
                                        <div style="border:1px solid #ddd; padding:10px; border-radius:10px; text-align:center; border-top: 5px solid {color};">
                                            <small>{m_name}</small><br>
                                            <b style="font-size:14px;">{predicted_class}</b><br>
                                            <h3 style="color:{color}; margin:0;">{confidence*100:.1f}%</h3>
                                        </div>
                                    """, unsafe_allow_html=True)
                            else:
                                with res_cols[idx]:
                                    st.error(f"{m_name} Fail")

# =====================================================
# FOOTER
# =====================================================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("© 2025 Leaf Disease System | Academic Comparison Mode | Built with Keras 3")