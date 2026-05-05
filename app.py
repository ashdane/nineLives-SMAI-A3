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

# clip confidence thresholds for three-tier gate
CLIP_REJECT = 0.20    # below this -> reject
CLIP_ACCEPT = 0.50    # above this -> accept directly
# between them -> ask user to confirm

# prediction confidence — below this, suggest more photos
PREDICTION_UNCERTAIN = 0.40

# supported languages
LANGUAGES = {
    "English": "en",
    "हिन्दी": "hi",
    "தமிழ்": "ta",
    "తెలుగు": "te",
    "മലയാളം": "ml",
}

# Streamlit config
st.set_page_config(
    page_title="Tomato Disease App",
    page_icon="🍅"
)

# ── loading animation shown during model init ───────────────────────────────
LOADING_HTML = """
<div style="display:flex; flex-direction:column; align-items:center;
    justify-content:center; min-height:55vh; text-align:center;">
    <style>
        @keyframes pulse { 0%,100%{transform:scale(1);opacity:0.6} 50%{transform:scale(1.2);opacity:1} }
        @keyframes slide { 0%{transform:translateX(-100%)} 50%{transform:translateX(250%)} 100%{transform:translateX(-100%)} }
    </style>
    <div style="font-size:3.5rem; animation:pulse 1.8s ease-in-out infinite;">🌱</div>
    <div style="color:#333; font-size:1.1rem; font-weight:500; margin-top:1rem;">
        Initializing backend</div>
    <div style="color:#888; font-size:0.85rem; margin-top:0.4rem;">
        Loading disease detection models, this may take a minute on first run</div>
    <div style="width:180px; height:4px; background:#e0e0e0; border-radius:4px;
        margin-top:1.2rem; overflow:hidden;">
        <div style="width:40%; height:100%; background:#28a745; border-radius:4px;
            animation:slide 1.5s ease-in-out infinite;"></div>
    </div>
</div>
"""

# simple CSS — same style as the original
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
    .gate-warn {
        padding: 15px;
        border-radius: 8px;
        background-color: #fff8e1;
        border-left: 5px solid #ffa000;
        margin-bottom: 15px;
    }
    .gate-reject {
        padding: 15px;
        border-radius: 8px;
        background-color: #fdf2f2;
        border-left: 5px solid #dc3545;
        margin-bottom: 15px;
    }
    .uncertain-notice {
        padding: 15px;
        border-radius: 8px;
        background-color: #fffde7;
        border-left: 5px solid #f9a825;
        margin-bottom: 15px;
    }
    /* circular camera button — vertically centered with the upload box */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button {
        width: 52px !important;
        height: 52px !important;
        min-width: 52px !important;
        min-height: 52px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        font-size: 1.4rem;
        line-height: 52px;
        border: 2px solid #ccc !important;
        background: white !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button:hover {
        background: #f5f5f5 !important;
        border-color: #999 !important;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
        transform: scale(1.05);
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button p {
        font-size: 1.4rem !important;
        margin: 0 !important;
        padding: 0 !important;
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

def check_tomato_leaf(clip_model, clip_processor, image):
    """
    Zero-shot check: is this image a tomato leaf?
    Returns the tomato-leaf probability (0-1).
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
        logits = outputs.logits_per_image.squeeze()

    probs = logits.softmax(dim=0)
    return probs[0].item()

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


# ─── Translation helper — reads pre-generated translations from JSON ────────

def get_translated_field(info, field, lang_code):
    """
    Get a field in the requested language from disease info.
    Looks up translations -> lang_code -> field, falls back to english.
    """
    if lang_code == "en":
        return info.get(field, "")

    translations = info.get("translations", {})
    lang_data = translations.get(lang_code, {})

    if isinstance(lang_data, dict):
        val = lang_data.get(field)
        if val:
            return val

    # fallback to english
    return info.get(field, "")


# ─── Streamlit UI ────────────────────────────────────────────────────────────

def main():
    # session state for the three-tier gate
    if "force_continue" not in st.session_state:
        st.session_state.force_continue = False
    if "prev_upload_name" not in st.session_state:
        st.session_state.prev_upload_name = None

    st.title("🍅 Tomato Leaf Disease Detector")
    st.write("Upload a picture of a tomato leaf to see what disease it has (or if it is healthy).")

    with st.sidebar:
        st.header("Info")
        st.write("This app uses a trained EfficientNet-B0 model to predict diseases from the PlantVillage dataset.")
        st.markdown("---")
        st.caption("A CLIP model is used to first verify that the uploaded image is a tomato leaf before running the disease classifier.")
        st.markdown("---")

        # language selector in sidebar
        st.subheader("🌐 Language / भाषा")
        lang = st.selectbox("Choose language", list(LANGUAGES.keys()),
                          index=0, label_visibility="collapsed")
        lang_code = LANGUAGES[lang]
        st.markdown("---")

        # diseases we detect
        st.subheader("Diseases We Detect")
        disease_names = [
            "🟢 Healthy Leaf",
            "🔴 Bacterial Spot",
            "🔴 Early Blight",
            "🔴 Late Blight",
            "🟡 Leaf Mold",
            "🔴 Septoria Leaf Spot",
            "🟡 Spider Mites",
            "🟡 Target Spot",
            "🔴 Yellow Leaf Curl Virus",
            "🔴 Tomato Mosaic Virus",
        ]
        for d in disease_names:
            st.markdown(d)

    # ── Load models with loading animation ──
    loader = st.empty()
    loader.markdown(LOADING_HTML, unsafe_allow_html=True)

    clip_model, clip_processor = load_clip_gate()
    model, class_names = load_trained_model()
    disease_info = get_disease_info()

    loader.empty()

    # ── Image input: upload bar with camera button alongside ──
    img = None

    upload_col, cam_btn_col = st.columns([5, 1])

    with upload_col:
        uploaded = st.file_uploader("Choose a tomato leaf image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            if uploaded.name != st.session_state.prev_upload_name:
                st.session_state.force_continue = False
                st.session_state.prev_upload_name = uploaded.name

    with cam_btn_col:
        st.markdown('<div style="height:40px"></div>', unsafe_allow_html=True)
        if st.button("📷", help="Open camera to take a photo"):
            st.session_state.show_camera = not st.session_state.get("show_camera", False)

    # camera only appears when user clicks the camera button
    if st.session_state.get("show_camera", False):
        # force back camera on mobile devices
        st.markdown("""
        <script>
        // Override getUserMedia to prefer back camera (environment facing)
        const origGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
        navigator.mediaDevices.getUserMedia = function(constraints) {
            if (constraints && constraints.video && typeof constraints.video === 'object') {
                constraints.video.facingMode = { ideal: "environment" };
            } else if (constraints && constraints.video === true) {
                constraints.video = { facingMode: { ideal: "environment" } };
            }
            return origGetUserMedia(constraints);
        };
        </script>
        """, unsafe_allow_html=True)
        camera_photo = st.camera_input("Point your camera at the leaf and click capture")
        if camera_photo is not None:
            img = Image.open(camera_photo).convert("RGB")
            st.session_state.force_continue = False

    if img is None:
        st.info("Waiting for an image to be uploaded...")
        return

    # ── Main layout ──
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Your Uploaded Image", use_container_width=True)

    with col2:
        st.write("### Prediction Results")

        # ── Stage 1: Three-tier leaf verification ──
        with st.spinner("Verifying image..."):
            leaf_prob = check_tomato_leaf(clip_model, clip_processor, img)

        # Tier 1: clearly not a tomato leaf
        if leaf_prob < CLIP_REJECT:
            st.markdown(f"""
            <div class="gate-reject">
                <strong>🚫 This does not appear to be a tomato leaf.</strong><br>
                Leaf confidence: {leaf_prob:.0%}<br><br>
                Please upload a clear photo of a tomato leaf for disease diagnosis.
            </div>
            """, unsafe_allow_html=True)
            return

        # Tier 2: uncertain — ask user
        if leaf_prob < CLIP_ACCEPT and not st.session_state.force_continue:
            st.markdown(f"""
            <div class="gate-warn">
                <strong>⚠️ Not sure about this image</strong><br>
                Tomato leaf confidence: {leaf_prob:.0%}<br><br>
                The image might be blurry, taken from far away, or it could be a 
                different plant. Do you want to continue anyway?
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Continue anyway", type="primary", use_container_width=True):
                    st.session_state.force_continue = True
                    st.rerun()
            with c2:
                if st.button("📷 Re-upload photo", use_container_width=True):
                    st.session_state.force_continue = False
                    st.session_state.prev_upload_name = None
                    st.rerun()
            return

        # Tier 3: confident / user forced continue → classify
        # ── Stage 2: Disease classification ──
        tensor_img = preprocess_image(img)
        preds = do_prediction(model, tensor_img, class_names)
        
        top_disease, top_prob = preds[0]
        is_healthy = "healthy" in top_disease.lower()

        info = disease_info.get(top_disease, {})
        pretty_name = get_translated_field(info, "disease_name", lang_code)

        # check if top 2 predictions are very close (within 10%)
        tied = False
        if len(preds) >= 2:
            second_disease, second_prob = preds[1]
            if (top_prob - second_prob) < 0.10 and top_prob < 0.6:
                tied = True

        # uncertain prediction notice
        if top_prob < PREDICTION_UNCERTAIN and not is_healthy:
            st.markdown(f"""
            <div class="uncertain-notice">
                <strong>Result isn't very clear</strong> (confidence: {top_prob:.0%})<br>
                It looks like it could be <em>{pretty_name}</em>, but for a better result,
                try sharing photos from different angles in good daylight.
            </div>
            """, unsafe_allow_html=True)

        # tied prediction notice
        if tied and not is_healthy:
            second_info = disease_info.get(second_disease, {})
            second_name = get_translated_field(second_info, "disease_name", lang_code)
            st.markdown(f"""
            <div class="gate-warn">
                <strong>⚠️ Multiple possible matches</strong><br>
                The leaf shows signs that could match either 
                <strong>{pretty_name}</strong> ({top_prob:.0%}) or 
                <strong>{second_name}</strong> ({second_prob:.0%}).<br><br>
                For a clearer diagnosis, try uploading another photo from a different angle.
            </div>
            """, unsafe_allow_html=True)

        if is_healthy:
            st.success(f"**Diagnosis:** {pretty_name}")
        else:
            if top_prob > 0.6:
                st.error(f"**Diagnosis:** {pretty_name} (Warning)")
            else:
                st.warning(f"**Diagnosis:** {pretty_name} (Low Confidence)")

        st.write("Top 5 matches:")
        for c, p in preds:
            c_info = disease_info.get(c, {})
            c_name = get_translated_field(c_info, "disease_name", lang_code)
            val = int(p * 100)
            st.progress(val, text=f"{c_name} - {val}%")
    
    st.markdown("---")
    
    # ── Disease details — show for tied diseases or single top disease ──
    # figure out which diseases to show details for
    show_diseases = [(top_disease, info)]
    if tied and not is_healthy:
        second_info = disease_info.get(second_disease, {})
        show_diseases.append((second_disease, second_info))

    for disease_key, d_info in show_diseases:
        if not d_info:
            st.write("Could not find extra info on", disease_key)
            continue

        d_name = get_translated_field(d_info, "disease_name", lang_code)
        d_healthy = "healthy" in disease_key.lower()
        desc = get_translated_field(d_info, "description", lang_code)
        cause = get_translated_field(d_info, "cause", lang_code)
        symptoms = get_translated_field(d_info, "symptoms", lang_code)
        rec = get_translated_field(d_info, "recommendation", lang_code)

        # if showing multiple diseases, label each section
        if len(show_diseases) > 1:
            st.subheader(f"📋 {d_name}")

        if len(show_diseases) == 1:
            st.subheader("About the Condition")
        bad_class = "" if d_healthy else "bad"
        st.markdown(f"<div class='result-box {bad_class}'>{desc}</div>", unsafe_allow_html=True)

        if cause:
            st.subheader("What causes it")
            st.write(cause)

        if symptoms and isinstance(symptoms, list):
            st.subheader("Symptoms to look for")
            for s in symptoms:
                st.write(f"• {s}")

        st.subheader("What to do")
        if isinstance(rec, dict):
            imm = rec.get("immediate", "")
            treat = rec.get("treatment", "")
            prev = rec.get("prevention", "")
            
            if imm:
                st.error(f"**🚨 Do this now:** {imm}")
            if treat:
                st.info(f"**💊 Treatment:** {treat}")
            if prev:
                st.success(f"**🛡️ Prevention:** {prev}")
        elif isinstance(rec, str) and rec:
            st.info(rec)
        else:
            st.write("No recommendation available.")

        # divider between diseases if showing multiple
        if len(show_diseases) > 1:
            st.markdown("---")

if __name__ == "__main__":
    main()
