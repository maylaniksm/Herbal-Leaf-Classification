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
    page_title="Herbal Leaf Classification",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD CLASS NAMES & MODELS
# =====================================================
@st.cache_data
def load_class_names():
    # Pastikan file ini berisi list nama daun herbal Anda
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
st.sidebar.markdown("# 🌿 Menu Navigasi")
app_mode = st.sidebar.radio(
    "Pilih Mode Analisis:",
    ["Uji Single Model", "Evaluasi Seluruh Model"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📤 Upload Center")
uploaded_files = st.sidebar.file_uploader(
    "Upload Gambar Daun Herbal (Single/Batch)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

selected_model_name = None
if app_mode == "Uji Single Model":
    st.sidebar.markdown("### ⚙️ Konfigurasi Model")
    selected_model_name = st.sidebar.selectbox(
        "Pilih Arsitektur Model:",
        ["CNN Manual", "MobileNetV2", "ResNet50", "VGG16"]
    )

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <h1 style="text-align:center;">🌿 Herbal Leaf Classification</h1>
    <p style="text-align:center; font-size:16px;">
        Sistem Identifikasi Jenis Daun Herbal Menggunakan Arsitektur Deep Learning
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# =====================================================
# MAIN CONTENT
# =====================================================
if not uploaded_files:
    st.info("Silakan unggah satu atau beberapa gambar daun herbal melalui sidebar untuk memulai klasifikasi.")
else:
    st.write(f"### Hasil Analisis Mode: {app_mode}")
    
    for i, file in enumerate(uploaded_files):
        image = Image.open(file)
        processed_img = preprocess_image(image)
        
        # Gunakan expander agar hasil batch upload tidak menumpuk terlalu panjang
        with st.expander(f"🖼️ Hasil Klasifikasi: {file.name}", expanded=True):
            col_preview, col_res = st.columns([1, 3])
            
            with col_preview:
                st.image(image, use_container_width=True, caption="Gambar Input")
            
            with col_res:
                if app_mode == "Uji Single Model":
                    # --- LOGIKA SINGLE MODEL ---
                    st.subheader(f"Prediksi Menggunakan {selected_model_name}")
                    with st.spinner(f"Sedang mengidentifikasi dengan {selected_model_name}..."):
                        model = load_specific_model(selected_model_name)
                        if model:
                            preds = model.predict(processed_img, verbose=0)
                            conf = float(np.max(preds))
                            label = class_names[int(np.argmax(preds))]
                            
                            st.metric("Jenis Daun Teridentifikasi", label)
                            st.progress(conf)
                            st.write(f"Tingkat Kepercayaan (Confidence): **{conf*100:.2f}%**")
                        else:
                            st.error(f"File model {selected_model_name} tidak ditemukan di folder 'model/'.")

                else:
                    # --- LOGIKA EVALUASI SELURUH MODEL ---
                    st.subheader("Perbandingan Hasil Antar Arsitektur Model")
                    model_list = ["CNN Manual", "MobileNetV2", "ResNet50", "VGG16"]
                    res_cols = st.columns(4)
                    
                    for idx, m_name in enumerate(model_list):
                        with res_cols[idx]:
                            with st.spinner(f"Loading {m_name}..."):
                                model = load_specific_model(m_name)
                                if model:
                                    preds = model.predict(processed_img, verbose=0)
                                    conf = float(np.max(preds))
                                    label = class_names[int(np.argmax(preds))]
                                    
                                    # Warna indikator (Hijau jika conf > 80%, Kuning jika kurang)
                                    color = "#2ecc71" if conf > 0.8 else "#f1c40f"
                                    
                                    st.markdown(f"""
                                        <div style="border:1px solid #ddd; padding:15px; border-radius:10px; border-top: 5px solid {color}; text-align:center;">
                                            <p style="color:gray; font-size:12px; margin-bottom:5px;">{m_name}</p>
                                            <p style="margin:5px 0; font-size:16px;"><b>{label}</b></p>
                                            <h3 style="color:{color}; margin:0;">{conf*100:.1f}%</h3>
                                        </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.error(f"{m_name} Not Found")

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    """
    <br><hr>
    <p style="text-align:center; font-size:13px; color:gray;">
        © 2025 | <b>Herbal Leaf Classification System</b><br>
        Deep Learning Comparison: CNN Manual, MobileNetV2, ResNet50, VGG16
    </p>
    """,
    unsafe_allow_html=True
)