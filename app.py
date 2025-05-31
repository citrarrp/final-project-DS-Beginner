import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from utils.preprocessing import preprocess_image
from utils.gradcam import make_gradcam_heatmap 
import cv2

st.set_page_config(page_title="Klasifikasi Pelanggaran K3 dengan ResNet50", layout="centered", page_icon="🛡️")

@st.cache_resource
def load_classification_model():
    try:
        model = load_model("modelku.keras")
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

    # Upload file
    uploaded_file = st.file_uploader(
        "Pilih gambar", 
        type=["jpg", "jpeg", "png"],
        help="Upload gambar workplace untuk analisis keamanan"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.subheader("🖼️ Gambar Asli")
        st.image(image, use_container_width=True)

        try:
            image_array = preprocess_image(image)

            with st.spinner("🔄 Menganalisis gambar..."):
                preds = model.predict({'input_layer_1': image_array})

            confidence = float(preds[0][0])

            if confidence > 0.5:
                pred_label = "Unsafe"
                pred_confidence = confidence
                pred_color = "🔴"
            else:
                pred_label = "Safe" 
                pred_confidence = 1.0 - confidence
                pred_color = "🟢"

            last_conv_layer_name = "conv5_block3_out"  # Sesuaikan dengan nama layer conv terakhir ResNet50
            heatmap = make_gradcam_heatmap(image_array, model, last_conv_layer_name)

            # Overlay heatmap ke gambar asli
            img = np.array(image.resize((224, 224)))
            # Resize heatmap ke ukuran gambar asli (224x224)
            heatmap = cv2.resize(heatmap, (224, 224))
            heatmap = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = heatmap_color * 0.2 + img
            superimposed_img = np.uint8(superimposed_img)
            # Tampilkan hasil prediksi
            st.markdown("---")
            st.markdown(f"### {pred_color} **{pred_label}** Detected")
            st.metric(label="Confidence", value=f"{pred_confidence:.2%}")

            st.progress(pred_confidence)

            st.subheader("🌡️ Grad-CAM Heatmap")
            st.image(superimposed_img, use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("💡 Pastikan fungsi preprocessing dan model kompatibel")

if __name__ == "__main__":
    main()
