import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="Herbal Leaf Classification",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
    }
    .stApp {
        background: transparent;
    }
    .upload-box {
        border: 2px dashed #2ecc71;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background: rgba(255, 255, 255, 0.9);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache untuk loading model
@st.cache_resource
def load_model(model_name):
    """Load model berdasarkan nama file dari folder models"""
    try:
        # Path ke folder models
        model_path = os.path.join("models", model_name)
        
        # Cek apakah file ada
        if not os.path.exists(model_path):
            st.error(f"❌ Model tidak ditemukan: {model_path}")
            st.info("💡 Pastikan file model ada di folder 'models/' di repository GitHub Anda")
            return None
        
        # Load model
        if model_name.endswith('.h5'):
            model = keras.models.load_model(model_path)
        else:
            model = keras.models.load_model(model_path)
        
        st.success(f"✅ Model {model_name} berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model {model_name}: {str(e)}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess gambar untuk prediksi"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    img = image.resize(target_size)
    
    # Convert to array
    img_array = np.array(img)
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def create_confidence_chart(predictions, class_names):
    """Buat chart confidence untuk prediksi"""
    # Ambil top 10 atau semua jika kurang dari 10
    n_top = min(10, len(class_names))
    top_indices = np.argsort(predictions[0])[-n_top:][::-1]
    
    top_classes = [class_names[i] for i in top_indices]
    top_confidences = [predictions[0][i] for i in top_indices]
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_confidences,
            y=top_classes,
            orientation='h',
            marker=dict(
                color=top_confidences,
                colorscale='Greens',
                showscale=True,
                colorbar=dict(title="Confidence")
            ),
            text=[f'{val*100:.2f}%' for val in top_confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"Top {n_top} Confidence Scores",
        xaxis_title="Confidence",
        yaxis_title="Class",
        height=400,
        template="plotly_white",
        showlegend=False,
        xaxis=dict(range=[0, 1])
    )
    
    return fig

def create_comparison_chart(model_results):
    """Buat chart perbandingan antar model"""
    models = list(model_results.keys())
    confidences = [results['confidence'] for results in model_results.values()]
    predictions = [results['prediction'] for results in model_results.values()]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=confidences,
            marker=dict(
                color=confidences,
                colorscale='Greens',
                showscale=False
            ),
            text=[f'{pred}<br>{conf:.2f}%' for pred, conf in zip(predictions, confidences)],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Model Confidence Comparison",
        xaxis_title="Model",
        yaxis_title="Confidence (%)",
        height=400,
        template="plotly_white",
        yaxis=dict(range=[0, 100])
    )
    
    return fig

# Header
st.markdown("""
<div style='background: white; padding: 30px; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
    <h1 style='text-align: center; color: #27ae60; margin: 0;'>
        🌿 Herbal Leaf Classification
    </h1>
    <p style='text-align: center; color: #666; margin-top: 10px; font-size: 18px;'>
        Klasifikasi Daun Herbal menggunakan Deep Learning
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar untuk konfigurasi
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi Model")
    
    # Pilih model
    available_models = {
        "CNN Baseline": "cnn_baseline_model.h5",
        "MobileNetV2": "mobilenetv2.keras",
        "ResNet": "resnet.keras",
        "VGG16": "vgg16.keras"
    }
    
    selected_model = st.selectbox(
        "Pilih Model",
        options=list(available_models.keys()),
        help="Pilih arsitektur model yang akan digunakan"
    )
    
    # Multi-model comparison
    compare_models = st.checkbox(
        "Bandingkan Semua Model",
        help="Jalankan prediksi dengan semua model secara bersamaan"
    )
    
    st.markdown("---")
    
    # Class names untuk herbal leaves
    st.markdown("### 🌱 Konfigurasi Kelas")
    
    # Default herbal leaf classes (sesuaikan dengan dataset Anda)
    default_herbal_classes = [
        "Basil", "Mint", "Oregano", "Rosemary", "Sage",
        "Thyme", "Cilantro", "Parsley", "Dill", "Lavender"
    ]
    
    num_classes = st.number_input(
        "Jumlah Kelas", 
        min_value=2, 
        max_value=100, 
        value=len(default_herbal_classes)
    )
    
    # Generate default class names
    if num_classes == len(default_herbal_classes):
        default_classes = default_herbal_classes
    else:
        default_classes = [f"Class {i+1}" for i in range(num_classes)]
    
    class_names_input = st.text_area(
        "Nama Kelas (pisahkan dengan koma)",
        value=", ".join(default_classes),
        help="Masukkan nama kelas sesuai urutan output model",
        height=150
    )
    class_names = [name.strip() for name in class_names_input.split(",")]
    
    # Validasi jumlah kelas
    if len(class_names) != num_classes:
        st.warning(f"⚠️ Jumlah nama kelas ({len(class_names)}) tidak sesuai dengan jumlah kelas ({num_classes})")
    
    st.markdown("---")
    
    # Model info
    st.markdown("### 📊 Info Model")
    model_info = {
        "CNN Baseline": "🏗️ Custom CNN Architecture",
        "MobileNetV2": "📱 Lightweight & Efficient",
        "ResNet": "🔄 Deep Residual Network",
        "VGG16": "🎯 Classic Deep Architecture"
    }
    st.info(model_info[selected_model])
    
    # Tips
    with st.expander("💡 Tips Penggunaan"):
        st.markdown("""
        - Upload gambar daun dengan latar belakang yang jelas
        - Pastikan daun terlihat jelas dan tidak blur
        - Gunakan pencahayaan yang baik
        - Format yang didukung: JPG, JPEG, PNG, BMP
        - Ukuran gambar akan otomatis disesuaikan
        """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Gambar Daun")
    
    uploaded_file = st.file_uploader(
        "Pilih gambar daun untuk diklasifikasi",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload gambar dalam format JPG, PNG, atau BMP"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_container_width=True)
        
        # Image info
        with st.expander("ℹ️ Info Gambar"):
            st.write(f"*Nama File:* {uploaded_file.name}")
            st.write(f"*Ukuran:* {image.size[0]} x {image.size[1]} pixels")
            st.write(f"*Mode:* {image.mode}")
            st.write(f"*Format:* {uploaded_file.type}")
            st.write(f"*Ukuran File:* {uploaded_file.size / 1024:.2f} KB")

with col2:
    st.markdown("### 🎯 Hasil Prediksi")
    
    if uploaded_file is not None:
        if st.button("🚀 Mulai Klasifikasi", type="primary", use_container_width=True):
            
            if compare_models:
                # Multi-model comparison
                st.markdown("#### 🔄 Membandingkan semua model...")
                model_results = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, (model_name, model_file) in enumerate(available_models.items()):
                    status_text.text(f"⏳ Memproses dengan {model_name}...")
                    
                    # Load model
                    model = load_model(model_file)
                    
                    if model is not None:
                        try:
                            # Preprocess and predict
                            processed_img = preprocess_image(image)
                            predictions = model.predict(processed_img, verbose=0)
                            
                            # Get top prediction
                            predicted_class_idx = np.argmax(predictions[0])
                            confidence = predictions[0][predicted_class_idx] * 100
                            
                            model_results[model_name] = {
                                'prediction': class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else f"Class {predicted_class_idx}",
                                'confidence': confidence,
                                'all_predictions': predictions
                            }
                        except Exception as e:
                            st.error(f"Error saat prediksi dengan {model_name}: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(available_models))
                
                status_text.empty()
                progress_bar.empty()
                
                if model_results:
                    # Display results
                    st.success("✅ Prediksi selesai!")
                    
                    # Show comparison chart
                    st.plotly_chart(create_comparison_chart(model_results), use_container_width=True)
                    
                    # Show detailed results
                    st.markdown("#### 📊 Hasil Detail per Model:")
                    for model_name, results in model_results.items():
                        with st.expander(f"🔍 {model_name}", expanded=True):
                            st.markdown(f"""
                            <div class='prediction-box'>
                                {results['prediction']}<br>
                                <span style='font-size: 18px;'>Confidence: {results['confidence']:.2f}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Top 5 predictions
                            top_5_idx = np.argsort(results['all_predictions'][0])[-5:][::-1]
                            st.markdown("*Top 5 Prediksi:*")
                            for i, idx in enumerate(top_5_idx, 1):
                                if idx < len(class_names):
                                    conf = results['all_predictions'][0][idx] * 100
                                    st.write(f"{i}. *{class_names[idx]}*: {conf:.2f}%")
                else:
                    st.error("❌ Tidak ada model yang berhasil dimuat. Periksa folder 'models/' di repository Anda.")
            
            else:
                # Single model prediction
                with st.spinner(f"⏳ Memproses dengan {selected_model}..."):
                    # Load model
                    model = load_model(available_models[selected_model])
                    
                    if model is not None:
                        try:
                            # Preprocess image
                            processed_img = preprocess_image(image)
                            
                            # Make prediction
                            start_time = time.time()
                            predictions = model.predict(processed_img, verbose=0)
                            inference_time = time.time() - start_time
                            
                            # Get results
                            predicted_class_idx = np.argmax(predictions[0])
                            predicted_class = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else f"Class {predicted_class_idx}"
                            confidence = predictions[0][predicted_class_idx] * 100
                            
                            # Display main prediction
                            st.markdown(f"""
                            <div class='prediction-box'>
                                🌿 {predicted_class}<br>
                                <span style='font-size: 18px;'>Confidence: {confidence:.2f}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Metrics
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            with metric_col1:
                                st.metric("🏷️ Kelas Prediksi", predicted_class)
                            with metric_col2:
                                st.metric("📊 Confidence", f"{confidence:.2f}%")
                            with metric_col3:
                                st.metric("⚡ Waktu Inference", f"{inference_time*1000:.2f} ms")
                            
                            # Confidence chart
                            st.plotly_chart(
                                create_confidence_chart(predictions, class_names),
                                use_container_width=True
                            )
                            
                            # Top 5 predictions
                            st.markdown("#### 🏆 Top 5 Prediksi:")
                            top_5_idx = np.argsort(predictions[0])[-5:][::-1]
                            
                            for i, idx in enumerate(top_5_idx, 1):
                                if idx < len(class_names):
                                    conf = predictions[0][idx] * 100
                                    st.progress(conf/100, text=f"{i}. *{class_names[idx]}*: {conf:.2f}%")
                        
                        except Exception as e:
                            st.error(f"❌ Error saat melakukan prediksi: {str(e)}")
                            st.info("💡 Pastikan gambar sesuai dengan format yang diharapkan model")
    else:
        st.info("👆 Upload gambar daun terlebih dahulu untuk memulai klasifikasi")
        
        # Contoh gambar
        st.markdown("#### 📸 Contoh Gambar yang Baik:")
        st.markdown("""
        - ✅ Gambar daun yang jelas dan fokus
        - ✅ Latar belakang yang kontras
        - ✅ Pencahayaan yang cukup
        - ✅ Daun terlihat utuh
        - ❌ Hindari gambar blur atau gelap
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px; background: white; border-radius: 10px;'>
    <p style='margin: 0; font-size: 14px;'>
        🌿 <strong>Herbal Leaf Classification System</strong><br>
        🚀 Powered by TensorFlow & Streamlit | Built with ❤️ for Deep Learning
    </p>
    <p style='margin-top: 10px; font-size: 12px; color: #999;'>
        © 2024 | All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)