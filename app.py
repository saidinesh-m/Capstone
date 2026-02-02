import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io

from models.densenet_cxr import DenseNet121
from preprocessing.transforms import get_valid_transforms
from explainability.gradcam import GradCAMExplainer

# Configuration
st.set_page_config(
    page_title="CXR Cardiopathy Detection",
    page_icon="🫀",
    layout="wide"
)

# Constants
CLASSES = [
    'slvh', 
    'dlv', 
    'composite_slvh_dlv', 
    'heart_transplant', 
    'lung_transplant', 
    'pacemaker_or_icd'
]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'checkpoints/best_model.pth'

# 1. Load Model (Cached)
@st.cache_resource
def load_model():
    model = DenseNet121(num_classes=len(CLASSES), is_trained=True)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        st.error(f"Model checkpoint not found at {MODEL_PATH}. Please train the model first.")
        return None
        
    model.to(DEVICE)
    model.eval()
    
    # Initialize Explainer (target layer logic)
    target_layer = model.densenet121.features[-1]
    explainer = GradCAMExplainer(model, target_layer)
    
    return model, explainer

# 2. Inference Function
def predict_and_explain(model, explainer, image_bytes):
    # Convert bytes to cv2 image
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    if image is None:
        return None, None, None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    transforms = get_valid_transforms()
    augmented = transforms(image=image_rgb)
    img_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
    # Explain (Grad-CAM)
    rgb_norm = cv2.resize(image_rgb, (224, 224)) / 255.0
    heatmap = explainer.explain(img_tensor, rgb_norm, target_class=None)
    
    return image_rgb, probs, heatmap

# --- UI ---
st.title("🫀 Cardiovascular Disease Detection System")
st.markdown("""
This system uses a **DenseNet-121** deep learning model to detect cardiovascular abnormalities from Chest X-Rays.
It also uses **Grad-CAM** to visualize the areas of the image that led to the prediction.
""")

st.sidebar.header("About")
st.sidebar.info("Upload a Chest X-Ray (CXR) image to get a prediction.")

model_data = load_model()

if model_data:
    model, explainer = model_data
    
    uploaded_file = st.file_uploader("Choose a Chest X-Ray Image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original X-Ray")
            
        with st.spinner("Analyzing image..."):
            original_img, probs, heatmap = predict_and_explain(model, explainer, uploaded_file)
            
            if original_img is not None:
                # Show Original
                with col1:
                    st.image(original_img, channels="RGB", use_column_width=True)
                
                # Show Heatmap
                with col2:
                    st.subheader("Grad-CAM Activation")
                    st.image(heatmap, channels="RGB", use_column_width=True, caption="Model Attention Map")
                
                # Show Predictions
                st.write("---")
                st.subheader("Prediction Probabilities")
                
                # Format metrics nicely
                res = {cls: float(p) for cls, p in zip(CLASSES, probs)}
                
                # Highlight high probs
                cols = st.columns(len(CLASSES))
                for i, (cls, p) in enumerate(res.items()):
                    with cols[i]:
                        st.metric(label=cls.replace('_', ' ').upper(), value=f"{p*100:.1f}%")
                        if p > 0.5:
                            st.write("⚠️ **Detected**")
            else:
                st.error("Error processing image. Please try another file.")
