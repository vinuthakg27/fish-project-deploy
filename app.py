import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fish Weight Predictor", page_icon="🐟", layout="centered")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Full-page underwater fish background ── */
.stApp {
    background: url("https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=1800&q=90")
                center/cover no-repeat fixed;
}

/* ── Dark teal overlay for readability ── */
.stApp::before {
    content: "";
    position: fixed; inset: 0;
    background: linear-gradient(
        160deg,
        rgba(0, 30, 60, 0.60) 0%,
        rgba(0, 80, 100, 0.45) 50%,
        rgba(0, 20, 50, 0.65) 100%
    );
    z-index: 0;
}

/* ── Main card ── */
.card {
    position: relative; z-index: 1;
    background: rgba(255, 255, 255, 0.10);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 24px;
    padding: 40px 44px 36px;
    max-width: 680px;
    margin: 0 auto;
    box-shadow: 0 12px 48px rgba(0,0,0,0.45);
}

/* ── Page title ── */
.page-title {
    text-align: center;
    color: #ffffff;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 2px;
    text-shadow: 0 2px 12px rgba(0,0,0,0.6);
}
.page-sub {
    text-align: center;
    color: rgba(200,230,255,0.85);
    font-size: 1rem;
    margin-bottom: 28px;
    text-shadow: 0 1px 6px rgba(0,0,0,0.5);
}

/* ── Section label ── */
.section-label {
    color: rgba(200,230,255,0.9);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* ── Streamlit input labels ── */
label, .stNumberInput label, .stSelectbox label {
    color: rgba(220,240,255,0.95) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* ── Input boxes ── */
input[type="number"], .stSelectbox > div > div {
    background: rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* ── Predict button ── */
div.stButton > button {
    width: 100%;
    padding: 14px 0;
    border-radius: 14px;
    background: linear-gradient(135deg, #0077b6, #00b4d8);
    color: white;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    border: none;
    box-shadow: 0 4px 20px rgba(0,180,216,0.45);
    transition: all 0.2s ease;
    margin-top: 8px;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #023e8a, #0096c7);
    box-shadow: 0 6px 28px rgba(0,150,200,0.55);
    transform: translateY(-1px);
}

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%);
    border-radius: 18px;
    padding: 28px 24px;
    text-align: center;
    margin-top: 24px;
    box-shadow: 0 8px 32px rgba(0,120,200,0.5);
    border: 1px solid rgba(255,255,255,0.2);
    animation: pop 0.35s cubic-bezier(.175,.885,.32,1.275);
}
@keyframes pop {
    0%   { transform: scale(0.85); opacity: 0; }
    100% { transform: scale(1);    opacity: 1; }
}
.result-label {
    color: rgba(200,240,255,0.85);
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.result-weight {
    color: #ffffff;
    font-size: 3rem;
    font-weight: 900;
    line-height: 1.1;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
.result-kg {
    color: rgba(200,240,255,0.8);
    font-size: 1.1rem;
    margin-top: 4px;
}
.result-species {
    display: inline-block;
    margin-top: 14px;
    background: rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 4px 16px;
    color: white;
    font-size: 0.88rem;
    font-weight: 600;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.15);
    margin: 22px 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(0, 20, 50, 0.75) !important;
    backdrop-filter: blur(14px);
}
section[data-testid="stSidebar"] * { color: rgba(200,230,255,0.9) !important; }

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fish_model.pkl")
COLS_PATH  = os.path.join(BASE_DIR, "model_columns.pkl")

@st.cache_resource
def get_model():
    model = pickle.load(open(MODEL_PATH, "rb"))
    cols  = pickle.load(open(COLS_PATH,  "rb"))
    return model, cols

model, model_columns = get_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🐟 About")
    st.markdown("""
This app predicts fish weight using a
**Linear Regression** model trained on
morphometric measurements.

---
**Engineered Features**
- Volumetric Proxy (L × H × W)
- Log Volume
- Polynomial: L², L³
- Interaction: L × Girth
- One-Hot Encoded Species

---
**Supported Species**
Bream · Parkki · Perch · Pike
Roach · Smelt · Whitefish
""")

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">🐟 Fish Weight Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">biomass estimator — enter measurements below</p>', unsafe_allow_html=True)

# ── Main card ─────────────────────────────────────────────────────────────────
# st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<p class="section-label">📐 Measurements</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    length1 = st.number_input("Length 1 (cm)",  min_value=0.0, value=25.0, step=0.1)
    height  = st.number_input("Height (cm)",    min_value=0.0, value=8.0,  step=0.1)
with col2:
    width   = st.number_input("Width (cm)",     min_value=0.0, value=5.0,  step=0.1)
    girth   = st.number_input("Girth (cm)",     min_value=0.0, value=10.0, step=0.1)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p class="section-label">🐠 Species</p>', unsafe_allow_html=True)

species = st.selectbox(
    "Select species",
    ["Bream", "Parkki", "Perch", "Pike", "Roach", "Smelt", "Whitefish"],
    label_visibility="collapsed"
)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Weight"):

    volume = length1 * height * width

    input_dict = {
        "length1_cm":      length1,
        "height_cm":       height,
        "width_cm":        width,
        "girth_cm":        girth,
        "volume_proxy":    volume,
        "log_volume":      np.log(volume) if volume > 0 else 0,
        "length_sq":       length1 ** 2,
        "length_cu":       length1 ** 3,
        "length_x_girth":  length1 * girth,
    }

    input_df = pd.DataFrame([input_dict])
    input_df["species"] = species
    input_df = pd.get_dummies(input_df, columns=["species"], drop_first=True)

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]

    log_pred     = model.predict(input_df)[0]
    final_weight = max(np.exp(log_pred), 1.0)

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Estimated Weight</div>
        <div class="result-weight">{final_weight:,.1f} <span style="font-size:1.4rem">grams</span></div>
        <div class="result-kg">≈ {final_weight/1000:.3f} kg</div>
        <div class="result-species">🐟 {species}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Fish Weight Predictor",
    page_icon="🐟",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM BACKGROUND + STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Main card styling */
    .main-card {
        background: rgba(255, 255, 255, 0.88);
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0px 8px 32px rgba(0,0,0,0.3);
    }

    /* Button styling */
    div.stButton > button {
        background-color: #003366;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }

    div.stButton > button:hover {
        background-color: #0059b3;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = pickle.load(open("fish_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))
