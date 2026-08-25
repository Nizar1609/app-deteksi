import streamlit as st
from PIL import Image
import pathlib
import pandas as pd

# --- PENAMBAL UNTUK MENGATASI WINERROR 1337 ---
original_exists = pathlib.Path.exists
def safe_exists(self):
    try:
        return original_exists(self)
    except OSError:
        return False
pathlib.Path.exists = safe_exists
# ----------------------------------------------

from ultralytics import YOLO

# 1. Konfigurasi Halaman Streamlit (Harus paling atas)
st.set_page_config(
    page_title="Sistem Rekognisi Varietas Biji Jagung",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS KUSTOM UNTUK MEMPERCANTIK UI ---
st.markdown("""
    <style>
    /* Menyembunyikan menu default Streamlit dan footer untuk kesan aplikasi mandiri */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    
    /* Mengubah radius border pada gambar */
    img {
        border-radius: 10px;
    }
    
    /* Modifikasi warna header metrik */
    [data-testid="stMetricLabel"] {
        font-size: 18px;
        font-weight: bold;
        color: #2E7D32;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Judul dan Deskripsi dengan Layout yang Lebih Rapi
with st.container():
    st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🌽 Sistem Rekognisi Varietas Biji Jagung Multi-Class</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Platform Deteksi Berbasis Deep Learning Menggunakan YOLOv8n</p>", unsafe_allow_html=True)
    st.divider()

# 3. Memuat Model
@st.cache_resource
def load_model():
    model_path = 'best.pt' 
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file '{model_path}' ada di direktori yang sama dengan aplikasi. Error: {e}")
        return None

model = load_model()

if model:
    # 4. Pengaturan Sidebar yang Lebih Informatif
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3014/3014420.png", width=100) # Ikon ilustrasi
        st.header("⚙️ Panel Kontrol")
        st.write("Sesuaikan parameter deteksi di bawah ini:")
        
        confidence = st.slider(
            "Ambang Batas Keyakinan (Confidence)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.25, 
            step=0.05,
            help="Tingkatkan nilai ini jika model mendeteksi terlalu banyak objek yang salah (False Positives)."
        )
        
        st.divider()
        st.subheader("ℹ️ Tentang Sistem")
        st.info("Sistem ini menggunakan arsitektur YOLOv8 Nano (YOLOv8n) untuk mengklasifikasikan dan mendeteksi varietas biji jagung secara real-time pada gambar statis.")

    # 5. Area Utama untuk Upload dan Hasil
    st.markdown("### 📤 Unggah Citra Biji Jagung")
    uploaded_file = st.file_uploader("Seret dan lepas file di sini, atau klik untuk memilih file...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Membuat area yang memisahkan gambar input dan output
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("#### Citra Asli (Input)")
            # Menambahkan border atau shadow tipis via markdown/css otomatis dari Streamlit
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("#### Citra Hasil Deteksi (Output)")
            with st.spinner('🔍 Model sedang menganalisis citra...'):
                # 6. Menjalankan Prediksi
                results = model.predict(image, conf=confidence)
                res_plotted = results[0].plot()
                res_plotted_rgb = res_plotted[:, :, ::-1]
                
            st.image(res_plotted_rgb, use_container_width=True)

        st.divider()
        
        # 7. Menampilkan Detail Deteksi Bergaya Dashboard
        st.markdown("### 📊 Ringkasan Hasil Deteksi")
        
        boxes = results[0].boxes
        if len(boxes) > 0:
            st.success(f"✅ Analisis selesai! Ditemukan total **{len(boxes)}** objek pada citra.")
            
            # Menggunakan Metric untuk tampilan visual angka yang menonjol
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric(label="Total Objek", value=len(boxes))
            
            # Menyiapkan data untuk ditampilkan dalam tabel yang rapi
            deteksi_list = []
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                conf = box.conf[0].item()
                deteksi_list.append({
                    "No": i + 1,
                    "Varietas Terdeteksi": class_name,
                    "Akurasi / Confidence": f"{conf*100:.1f}%"
                })
            
            # Menampilkan data dalam bentuk DataFrame interaktif
            df = pd.DataFrame(deteksi_list)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
        else:
            st.warning("⚠️ Tidak ada objek yang terdeteksi. Silakan coba turunkan nilai Ambang Batas Keyakinan di panel kiri atau unggah citra dengan pencahayaan yang lebih baik.")