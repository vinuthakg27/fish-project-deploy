import streamlit as st
import pandas as pd
import numpy as np
import pickle

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

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown(
    """
    <div style='text-align:center; margin-bottom:30px; color:white;'>
        <h1>🐟 Fish Weight Prediction</h1>
        <h4>AI Powered Biomass Estimator</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# MAIN CARD CONTAINER
# --------------------------------------------------
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    st.subheader("Enter Fish Measurements")

    col1, col2 = st.columns(2)

    with col1:
        length1 = st.number_input("Length1 (cm)", min_value=0.0, value=0.0)
        height = st.number_input("Height (cm)", min_value=0.0, value=0.0)

    with col2:
        width = st.number_input("Width (cm)", min_value=0.0, value=0.0)
        girth = st.number_input("Girth (cm)", min_value=0.0, value=0.0)

    species = st.selectbox(
        "Select Species",
        ["Bream", "Roach", "Pike", "Smelt", "Parkki", "Perch"]
    )

    # --------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------
    if st.button("Predict Weight"):

        # Base input
        input_dict = {
            "length1_cm": length1,
            "height_cm": height,
            "width_cm": width,
            "girth_cm": girth
        }

        input_df = pd.DataFrame([input_dict])

        # Feature Engineering (Same as training)
        input_df["volume_proxy"] = length1 * height * width
        input_df["length_sq"] = length1 ** 2
        input_df["length_cu"] = length1 ** 3
        input_df["length_x_girth"] = length1 * girth

        # One-hot encoding
        input_df["species"] = species
        input_df = pd.get_dummies(input_df, drop_first=True)

        # Align columns exactly as training
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Prediction
        prediction = model.predict(input_df)[0]

        # If you trained log model, use this:
        # prediction = np.exp(model.predict(input_df)[0])

        st.markdown(
            f"""
            <div style="
                background-color:#003366;
                padding:20px;
                border-radius:10px;
                text-align:center;
                color:white;
                font-size:22px;
                margin-top:20px;">
                Predicted Fish Weight: {prediction:.2f} grams
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("About")

st.sidebar.info(
    """
    This application predicts fish weight using
    Linear Regression with advanced feature engineering:

    • Volumetric Proxy  
    • Polynomial Features  
    • Interaction Terms  
    • One-Hot Encoding  

    Developed using Streamlit.
    """
)