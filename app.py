import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path

# =====================================================
# Path Setup
# =====================================================
BASE_DIR = Path(__file__).resolve().parent

CLASSES_PATH = BASE_DIR / "data" / "food-101" / "meta" / "classes.txt"
MODEL_PATH = BASE_DIR / "checkpoints" / "best_model.pth"

class_names = [line.strip() for line in open(CLASSES_PATH, "r")]


# =====================================================
# Load Model
# =====================================================
@st.cache_resource
def load_model():
    num_classes = len(class_names)

    model = torch.hub.load("pytorch/vision:v0.14.0", "efficientnet_b0", pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model

model = load_model()


# =====================================================
# Image Transform
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================================
# MODERN UI CSS
# =====================================================
st.set_page_config(page_title="Food-101 Classifier", page_icon="🍽️", layout="centered")

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.title {
    text-align: center;
    font-size: 48px !important;
    font-weight: 900;
    background: linear-gradient(90deg, #fc4a1a, #f7b733);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    color: #555;
}

.glass-card {
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(12px);
    box-shadow: 0px 8px 30px rgba(0,0,0,0.12);
    padding: 25px;
    border-radius: 18px;
    margin-top: 20px;
    border: 1px solid rgba(255,255,255,0.4);
}

.pred-box {
    background: rgba(255, 244, 230, 0.9);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
}

.top5 {
    background: rgba(240, 240, 240, 0.9);
    padding: 15px;
    border-radius: 15px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================
st.markdown('<p class="title">🍽️ Food-101 Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A modern deep learning app with camera support</p>', unsafe_allow_html=True)


# =====================================================
# TABS: Upload | Camera
# =====================================================
tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Camera Mode"])


# =====================================================
# HELPER: Prediction Function
# =====================================================
def predict(img):

    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)

    pred_label = class_names[top_class.item()]
    confidence = float(top_prob.item() * 100)

    top5_prob, top5_idx = torch.topk(probs, 5)

    return pred_label, confidence, top5_prob, top5_idx


# =====================================================
# TAB 1: UPLOAD IMAGE
# =====================================================
with tab1:
    st.write("### Upload a food image for prediction:")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        pred_label, confidence, top5_prob, top5_idx = predict(img)

        # Display Prediction Card
        st.markdown(f"""
            <div class="pred-box">
                <h2>🍛 Prediction: <b>{pred_label}</b></h2>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
        """, unsafe_allow_html=True)

        # Top 5 Results
        st.markdown("<div class='top5'><h4>🔝 Top 5 Predictions</h4>", unsafe_allow_html=True)
        for i in range(5):
            label = class_names[top5_idx[0][i].item()]
            prob = top5_prob[0][i].item() * 100
            st.write(f"**{i+1}. {label} — {prob:.2f}%**")
        st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# TAB 2: CAMERA MODE
# =====================================================
with tab2:
    st.write("### Take a photo using your camera:")

    camera_image = st.camera_input("Capture image")

    if camera_image:
        img = Image.open(camera_image).convert("RGB")
        st.image(img, caption="Captured Image", use_column_width=True)

        pred_label, confidence, top5_prob, top5_idx = predict(img)

        st.markdown(f"""
            <div class="pred-box">
                <h2>🍛 Prediction: <b>{pred_label}</b></h2>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='top5'><h4>🔝 Top 5 Predictions</h4>", unsafe_allow_html=True)
        for i in range(5):
            label = class_names[top5_idx[0][i].item()]
            prob = top5_prob[0][i].item() * 100
            st.write(f"**{i+1}. {label} — {prob:.2f}%**")
        st.markdown("</div>", unsafe_allow_html=True)


# FOOTER
st.write("---")
st.caption("Built with ❤️ using Streamlit · PyTorch · EfficientNet-B0 · Food-101")
