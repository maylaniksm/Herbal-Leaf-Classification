import streamlit as st
import tensorflow as tf
import numpy as np
import json
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
with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

# =====================================================
# LOAD MODELS (CACHE)
# =====================================================
@st.cache_resource
def load_models():
    return {
        "CNN Manual": tf.keras.models.load_model(
            "model/cnn_leaf_model.keras", compile=False
        ),
        "MobileNetV2": tf.keras.models.load_model(
            "model/mobilenetv2_leaf_model.keras", compile=False
        ),
        "ResNet50": tf.keras.models.load_model(
            "model/resnet50_leaf_model.keras", compile=False
        ),
        "VGG16": tf.keras.models.load_model(
            "model/vgg16_leaf_model.keras", compile=False
        ),
    }

models = load_models()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.markdown("## ⚙️ Konfigurasi Model")
st.sidebar.info(
    "Pilih arsitektur model deep learning yang akan digunakan "
    "untuk melakukan klasifikasi penyakit daun."
)

model_name = st.sidebar.selectbox(
    "Model Deep Learning",
    list(models.keys())
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Catatan Akademik**  
    - Dataset sama untuk semua model  
    - Model menggunakan pre-trained ImageNet  
    - Output berupa confidence score
    """
)

# =====================================================
# MAIN CONTENT
# =====================================================
st.markdown("### 📤 Upload Gambar Daun")
st.write(
    "Unggah gambar daun tanaman dalam format **JPG / PNG** untuk dilakukan "
    "klasifikasi penyakit menggunakan model yang dipilih."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# =====================================================
# PREPROCESS FUNCTION
# =====================================================
def preprocess_image(img, model_name):
    img = img.resize((224, 224))
    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)

    if model_name == "MobileNetV2":
        img_array = mobilenet_preprocess(img_array)
    elif model_name == "ResNet50":
        img_array = resnet_preprocess(img_array)
    elif model_name == "VGG16":
        img_array = vgg_preprocess(img_array)
    else:  # CNN Manual
        img_array = img_array / 255.0

    return img_array

# =====================================================
# PREDICTION
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
        with st.spinner("Model sedang melakukan inferensi..."):
            processed_img = preprocess_image(image, model_name)
            model = models[model_name]

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

        st.markdown("---")

        st.caption(
            "Confidence score menunjukkan tingkat keyakinan model terhadap hasil prediksi."
        )

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
