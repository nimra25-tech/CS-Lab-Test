import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# 1. Page Configuration (UI ko behtar banane ke liye)
st.set_page_config(
    page_title="AI Image Classifier", 
    page_icon="🤖", 
    layout="centered"
)

# 2. Model Loading (Cache use kar rahe hain taake har bar page refresh par model dobara load na ho)
@st.cache_resource
def load_model():
    # MobileNetV2 ek free, pre-trained model hai jo image classification ke liye use hota hai
    model = MobileNetV2(weights='imagenet')
    return model

# Model ko memory mein load karna
model = load_model()

# 3. App Header aur Description
st.title("🤖 AI Image Classifier")
st.markdown("""
Yeh app **Computer Vision** ka pre-trained model (MobileNetV2) use karti hai. 
Is mein koi API ya payment shamil nahi hai. Bas ek tasveer upload karein aur AI batayega ke is mein kya hai!
""")

st.divider()

# 4. Image Upload Widget
uploaded_file = st.file_uploader("Apni tasveer yahan upload karein (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tasveer ko screen par show karna
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    with st.spinner("🔍 AI is tasveer ko analyze kar raha hai..."):
        # 5. Image Preprocessing (Model ke mutabiq image ko resize aur format karna)
        # MobileNetV2 ko 224x224 size ki image chahiye hoti hai
        img = image.resize((224, 224))
        
        # Image ko array mein convert karna
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Model ke liye input ko preprocess karna
        img_array = preprocess_input(img_array)

        # 6. Prediction (Model se result lena)
        predictions = model.predict(img_array)
        
        # Top 3 results nikalna
        decoded_predictions = decode_predictions(predictions, top=3)[0]

    st.success("✅ Analysis Complete!")

    # 7. Results ko UI par beautifully show karna
    st.subheader("AI ki Predictions:")
    
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        # Label ko clean karna (e.g., 'golden_retriever' -> 'Golden Retriever')
        clean_label = label.replace('_', ' ').title()
        confidence = float(score)
        
        st.write(f"**{i+1}. {clean_label}** (Confidence: {confidence*100:.2f}%)")
        # Progress bar se confidence level show karna
        st.progress(confidence)

st.divider()
st.caption("Built with ❤️ using Streamlit & TensorFlow")
