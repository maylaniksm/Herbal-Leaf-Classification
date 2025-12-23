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
    page_title="Leaf Disease Analysis",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD CLASS NAMES & MODELS
# =====================================================
@st.cache_data
def load_class_names():
    with open("model/class_names.json", "r") as f:
        return json.load(f)

class_names = load_class_names()

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

    tf.keras.backend.clear_session()
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        import keras
        return keras.saving.load_model(path, compile=False)

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.markdown("# 🍃 Menu Utama")
app_mode = st.sidebar.radio(
    "Pilih Mode Analisis:",
    ["Uji Single Model", "Evaluasi Seluruh Model"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📤 Upload Center")
uploaded_files = st.sidebar.file_uploader(
    "Upload Gambar (Single/Batch)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

selected_model_name = None
if app_mode == "Uji Single Model":
    st.sidebar.markdown("### ⚙️ Konfigurasi")
    selected_model_name = st.sidebar.selectbox(
        "Pilih Model:",
        ["CNN Manual", "MobileNetV2", "ResNet50", "VGG16"]
    )

# =====================================================
# MAIN CONTENT
# =====================================================
st.markdown(f"# 🔍 Mode: {app_mode}")
st.write("Sistem Klasifikasi Penyakit Daun Berbasis Deep Learning")

if not uploaded_files:
    st.info("Silakan unggah gambar melalui sidebar untuk memulai.")
else:
    for i, file in enumerate(uploaded_files):
        image = Image.open(file)
        processed_img = preprocess_image(image)
        
        with st.expander(f"🖼️ Hasil Analisis: {file.name}", expanded=True):
            col_preview, col_res = st.columns([1, 3])
            
            with col_preview:
                st.image(image, use_container_width=True, caption="Gambar Input")
            
            with col_res:
                if app_mode == "Uji Single Model":
                    # LOGIK SINGLE MODEL
                    st.subheader(f"Prediksi {selected_model_name}")
                    with st.spinner(f"Memproses dengan {selected_model_name}..."):
                        model = load_specific_model(selected_model_name)
                        if model:
                            preds = model.predict(processed_img, verbose=0)
                            conf = float(np.max(preds))
                            label = class_names[int(np.argmax(preds))]
                            
                            st.metric("Label Penyakit", label)
                            st.progress(conf)
                            st.write(f"Confidence Score: **{conf*100:.2f}%**")
                        else:
                            st.error("Gagal memuat model.")

                else:
                    # LOGIK EVALUASI SELURUH MODEL (Perbandingan)
                    st.subheader("Perbandingan Akurasi Antar Model")
                    model_list = ["CNN Manual", "MobileNetV2", "ResNet50", "VGG16"]
                    res_cols = st.columns(4)
                    
                    for idx, m_name in enumerate(model_list):
                        with res_cols[idx]:
                            with st.spinner(f"{m_name}..."):
                                model = load_specific_model(m_name)
                                if model:
                                    preds = model.predict(processed_img, verbose=0)
                                    conf = float(np.max(preds))
                                    label = class_names[int(np.argmax(preds))]
                                    
                                    color = "#2ecc71" if conf > 0.8 else "#f1c40f"
                                    st.markdown(f"""
                                        <div style="border:1px solid #ddd; padding:10px; border-radius:10px; border-top: 5px solid {color};">
                                            <small>{m_name}</small><br>
                                            <p style="margin:5px 0;"><b>{label}</b></p>
                                            <h4 style="color:{color}; margin:0;">{conf*100:.1f}%</h4>
                                        </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.error("Error")

# =====================================================
# FOOTER
# =====================================================
st.markdown("<br><hr><p style='text-align:center;'>© 2025 Leaf Disease Classifier</p>", unsafe_allow_html=True)