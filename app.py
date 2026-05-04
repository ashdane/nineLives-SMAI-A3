import os
import json
import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# Set paths
BASE_DIR = os.path.dirname(__file__)
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
DISEASE_INFO_PATH = os.path.join(BASE_DIR, "disease_info.json")

# Streamlit config
st.set_page_config(
    page_title="Tomato Disease App",
    page_icon="🍅"
)

# A bit of CSS for style but much simpler
st.markdown("""
<style>
    .result-box {
        padding: 15px; 
        border-radius: 8px; 
        background-color: #f0f2f6; 
        border-left: 5px solid #28a745;
        margin-bottom: 20px;
    }
    .result-box.bad {
        border-left-color: #dc3545;
        background-color: #fdf2f2;
    }
</style>
""", unsafe_allow_html=True)

# ─── Stage 1: CLIP-based leaf gate ──────────────────────────────────────────

@st.cache_resource
def load_clip_gate():
    """Load a pre-trained CLIP model for zero-shot tomato-leaf verification."""
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    return clip_model, clip_processor

def is_tomato_leaf(clip_model, clip_processor, image):
    """
    Zero-shot check: is this image specifically a tomato leaf?
    Returns (bool, float) — whether it passed the gate and the tomato-leaf probability.
    """
    candidate_labels = [
        "a photo of a tomato plant leaf, with or without disease",
        "a photo of a non-tomato plant leaf such as banana, papaya, mango, or grape leaf",
        "a photo of something that is not a plant leaf, such as a notebook, person, animal, or object",
    ]

    inputs = clip_processor(
        text=candidate_labels,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image.squeeze()  # shape: [3]

    probs = logits.softmax(dim=0)

    # Only the first label (tomato leaf) counts as passing the gate
    tomato_leaf_prob = probs[0].item()
    return tomato_leaf_prob > 0.20, tomato_leaf_prob

# ─── Stage 2: Disease classifier ────────────────────────────────────────────

@st.cache_resource
def load_trained_model():
    if not os.path.exists(CHECKPOINT_PATH):
        st.error("No model found. Did you run train.py first?")
        st.stop()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    classes = checkpoint["class_names"]
    n_classes = checkpoint["num_classes"]

    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=n_classes)
    m.load_state_dict(checkpoint["model_state_dict"])
    m.eval()
    
    return m, classes

@st.cache_data
def get_disease_info():
    if not os.path.exists(DISEASE_INFO_PATH):
        return {}
    with open(DISEASE_INFO_PATH, "r") as f:
        data = json.load(f)
    return data

def preprocess_image(img):
    t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return t(img).unsqueeze(0)

def do_prediction(model, tensor_img, classes):
    with torch.no_grad():
        out = model(tensor_img)
        probs = torch.softmax(out, dim=1).squeeze()

    top_p, top_class_idx = probs.topk(5)
    
    res = []
    for i in range(len(top_p)):
        idx = top_class_idx[i].item()
        res.append((classes[idx], top_p[i].item()))
        
    return res

# ─── Streamlit UI ────────────────────────────────────────────────────────────

def main():
    st.title("🍅 Tomato Leaf Disease Detector")
    st.write("Upload a picture of a tomato leaf to see what disease it has (or if it is healthy).")

    with st.sidebar:
        st.header("Info")
        st.write("This app uses a trained EfficientNet-B0 model to predict diseases from the PlantVillage dataset.")
        st.write("Classes supported include Early Blight, Late Blight, Leaf Mold, and more.")
        st.markdown("---")
        st.caption("A CLIP model is used to first verify that the uploaded image is a tomato leaf before running the disease classifier.")

    # Load both models
    clip_model, clip_processor = load_clip_gate()
    model, class_names = load_trained_model()
    disease_info = get_disease_info()

    uploaded = st.file_uploader("Choose a tomato leaf image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(img, caption="Your Uploaded Image", width="stretch")

        with col2:
            st.write("### Prediction Results")

            # ── Stage 1: Leaf verification ──
            with st.spinner("Verifying image..."):
                passed_gate, leaf_prob = is_tomato_leaf(clip_model, clip_processor, img)

            if not passed_gate:
                st.error(
                    "🚫 **This does not appear to be a tomato leaf.**\n\n"
                    f"Leaf confidence: {leaf_prob:.0%}\n\n"
                    "Please upload a clear photo of a tomato leaf for disease diagnosis."
                )
                st.stop()

            # ── Stage 2: Disease classification ──
            tensor_img = preprocess_image(img)
            preds = do_prediction(model, tensor_img, class_names)
            
            top_disease, top_prob = preds[0]
            is_healthy = "healthy" in top_disease.lower()

            info = disease_info.get(top_disease, {})
            pretty_name = info.get("disease_name", str(top_disease))
            
            if is_healthy:
                st.success(f"**Diagnosis:** {pretty_name}")
            else:
                if top_prob > 0.6:
                    st.error(f"**Diagnosis:** {pretty_name} (Warning)")
                else:
                    st.warning(f"**Diagnosis:** {pretty_name} (Low Confidence)")

            st.write("Top 5 matches:")
            for c, p in preds:
                val = int(p * 100)
                st.progress(val, text=f"{c} - {val}%")
        
        st.markdown("---")
        
        if info:
            st.subheader("About the Condition")
            bad_class = "" if is_healthy else "bad"
            
            desc = info.get('description', 'No desc')
            st.markdown(f"<div class='result-box {bad_class}'>{desc}</div>", unsafe_allow_html=True)
            
            st.subheader("What to do")
            st.info(info.get('recommendation', 'No recommendation'))
            
        else:
            st.write("Could not find extra info on", top_disease)

    else:
        st.info("Waiting for an image to be uploaded...")

if __name__ == "__main__":
    main()
