import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

from utils.preprocessing import preprocess_image
# from utils.gradcam import make_gradcam_heatmap  # GradCAM dinonaktifkan

st.set_page_config(page_title="Klasifikasi Pelanggaran K3 dengan ResNet50", layout="centered", page_icon="🛡️")

@st.cache_resource
def load_classification_model():
    """Load model dengan caching untuk performance"""
    try:
        model = tf.keras.models.load_model("model/model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("🛡️ Deteksi Pelanggaran K3 dengan ResNet50")
    st.markdown("---")
    st.markdown("📋 **Upload gambar dan lihat klasifikasi pelanggaran!**")

    # Load model
    model = load_classification_model()
    if model is None:
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "Pilih gambar", 
        type=["jpg", "jpeg", "png"],
        help="Upload gambar workplace untuk analisis keamanan"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.subheader("🖼️ Gambar Asli")
        st.image(image, use_column_width=True)

        try:
            # ✅ Pastikan preprocess_image mengembalikan shape (1, 224, 224, 3)
            image_array = preprocess_image(image)
            if image_array.shape != (1, 224, 224, 3):
                st.warning(f"Shape tidak sesuai: {image_array.shape}. Harusnya (1, 224, 224, 3)")
                st.stop()

            with st.spinner("🔄 Menganalisis gambar..."):
                preds = model.predict(image_array, verbose=0)

            confidence_unsafe = float(preds[0][0])
            confidence_safe = 1.0 - confidence_unsafe

            if confidence_unsafe > 0.5:
                pred_label = "Unsafe"
                pred_confidence = confidence_unsafe
                pred_color = "🔴"
            else:
                pred_label = "Safe" 
                pred_confidence = confidence_safe
                pred_color = "🟢"

            # Display prediction results
            st.markdown("---")
            st.subheader("📊 Hasil Prediksi")
            st.markdown(f"### {pred_color} **{pred_label}** ({pred_confidence:.1%} confidence)")

            col_safe, col_unsafe = st.columns(2)
            with col_safe:
                st.metric("🟢 Safe", f"{confidence_safe:.1%}")
                st.progress(confidence_safe)

            with col_unsafe:
                st.metric("🔴 Unsafe", f"{confidence_unsafe:.1%}")
                st.progress(confidence_unsafe)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("💡 Pastikan fungsi preprocessing dan model kompatibel")

if __name__ == "__main__":
    main()
