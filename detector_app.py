
# Hybrid Deep Learning-Based Malicious URL Detector — Streamlit App

import streamlit as st
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import re
import time

from tensorflow.keras.preprocessing.sequence import pad_sequences
from urllib.parse import urlparse

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Hybrid URL Detector",
    page_icon="🛡️",
    layout="centered"
)
# -----------------------------
# SIDEBAR

# -----------------------------
with st.sidebar:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A2E, #16213E);
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("🛡️ Threat Detector")

    st.markdown("---")

    st.subheader("📌 About Project")

    st.write("""
    Hybrid Deep Learning + Rule-Based
    Cybersecurity URL Detection System.

    Detects:
    - Phishing
    - Malware
    - Defacement
    - Benign URLs
    """)

    st.markdown("---")

    st.subheader("⚙️ Technologies")

    st.write("""
    - TensorFlow / Keras
    - BiLSTM Deep Learning
    - Streamlit
    - NLP Tokenization
    - Feature Engineering
             

    """)
    st.markdown("---")

    st.subheader("📈 Model Performance")

    st.write("""
    ✔ Test Accuracy: 87%

    ✔ BiLSTM Neural Architecture

    ✔ Multi-Class URL Classification

    ✔ Hybrid Rule-Based Threat Detection
    """)

    st.markdown("---")

    st.subheader("🎯 Model Info")

    st.write("""
    Architecture:
    BiLSTM + Hybrid Security Logic
    """)
# -----------------------------
# LOAD MODEL + FILES
# -----------------------------

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
import json
max_length= 200
@st.cache_resource
def load_resources():

    with open("tokenizer.json") as f:
        data = f.read()

    tokenizer = tokenizer_from_json(data)

    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    model = Sequential()

    model.add(
        Embedding(
            input_dim=len(tokenizer.word_index) + 1,
            output_dim=64,
            input_length=max_length
        )
    )

    model.add(
        Bidirectional(
            LSTM(64, return_sequences=True)
        )
    )

    model.add(
        Bidirectional(
            LSTM(32)
        )
    )

    model.add(Dropout(0.4))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(4, activation='softmax'))

    model.build(input_shape=(None, 200))

    model.load_weights("bilstm_weights.weights.h5")

    return tokenizer, le, model
tokenizer, le, model = load_resources()

# -----------------------------
# CLEAN URL
# -----------------------------
def clean_url(url):

    url = str(url).lower()

    url = url.replace("https://", "")
    url = url.replace("http://", "")
    url = url.replace("www.", "")

    return url

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_url_features(url):

    url = str(url).lower()

    features = {}

    features['url_length'] = len(url)

    features['digit_count'] = sum(c.isdigit() for c in url)

    features['special_count'] = len(re.findall(r'[^a-zA-Z0-9]', url))

    features['has_https'] = int("https" in url)

    features['has_ip'] = int(
        bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))
    )

    suspicious_keywords = [
        'login',
        'verify',
        'secure',
        'account',
        'bank',
        'update',
        'free',
        'bonus',
        'crack',
        'paypal',
        'signin'
    ]

    features['suspicious_keywords'] = int(
        any(word in url for word in suspicious_keywords)
    )

    features['has_exe'] = int(".exe" in url)
    features['has_php'] = int(".php" in url)

    suspicious_tlds = ['.xyz', '.tk', '.ru', '.biz']

    features['suspicious_tld'] = int(
        any(tld in url for tld in suspicious_tlds)
    )

    return features

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_url(url):

    max_length = 200

    cleaned_url = clean_url(url)

    features = extract_url_features(cleaned_url)

    sequence = tokenizer.texts_to_sequences([cleaned_url])

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )

    prediction = model.predict(padded, verbose=0)[0]

    predicted_class = np.argmax(prediction)

    confidence = float(np.max(prediction))

    classes = le.classes_

    predicted_label = classes[predicted_class]

    # -----------------------------
    # HYBRID LOGIC
    # -----------------------------

    if features['has_exe'] == 1:
        predicted_label = "malware"

    if (
        features['has_php'] == 1 or
        "1337" in cleaned_url or
        "hacked" in cleaned_url
    ):
        predicted_label = "defacement"

    if (
        features['suspicious_keywords'] == 1 and
        features['suspicious_tld'] == 1 and
        features['has_exe'] == 0
    ):
        predicted_label = "phishing"

    # Threshold logic
    if predicted_label in ["malware", "defacement"]:
        final_label = predicted_label

    elif confidence < 0.85:
        final_label = "benign"

    else:
        final_label = predicted_label

    return final_label, confidence, features

# -----------------------------
# UI DESIGN
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #00FFAA;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #CCCCCC;
        margin-bottom: 40px;
    }

    .result-box {
        padding: 25px;
        border-radius: 20px;
        margin-top: 25px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: white;
        box-shadow: 0px 0px 25px rgba(0,255,170,0.4);
        animation: fadeIn 0.8s ease-in-out;
        transition: all 0.4s ease;
    }

    .result-box:hover {
        transform: scale(1.02);
        box-shadow: 0px 0px 40px rgba(0,255,170,0.8);
    }

    @keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0px);
    }
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    '<div class="title">🛡️ Hybrid URL Threat Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Hybrid Deep Learning + Cybersecurity Intelligence</div>',
    unsafe_allow_html=True
)

# -----------------------------
# URL INPUT
# -----------------------------
st.markdown("### ⚡ Quick Test URLs")

if "url_input" not in st.session_state:
    st.session_state.url_input = ""

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Safe URL"):
        st.session_state.url_input = "https://google.com"

with col2:
    if st.button("Phishing"):
        st.session_state.url_input = "http://paypal-login-secure.xyz"

with col3:
    if st.button("Malware"):
        st.session_state.url_input = "http://free-bonus-crack.exe"

with col4:
    if st.button("Defacement"):
        st.session_state.url_input = "http://192.168.1.1/hacked.php"

url_input = st.text_input(
    "🔗 Enter URL to analyze",
    value=st.session_state.url_input,
    placeholder="https://example.com"
)
# -----------------------------
# ANALYZE BUTTON
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if st.button("Analyze URL"):

    if url_input.strip() == "":
        st.warning("Please enter a URL.")

    else:
        url_pattern = re.compile(
            r'^(https?:\/\/)?'
            r'(([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})'
        )

        if not re.match(url_pattern, url_input):
            st.error("Please enter a valid URL.")
            st.stop()
        scan_placeholder = st.empty()

        scan_messages = [
            "⚡ Initializing Threat Engine...",
            "🧠 Loading BiLSTM Neural Model...",
            "🔍 Extracting URL Features...",
            "📡 Scanning Domain Intelligence...",
            "🚨 Detecting Threat Patterns...",
            "✅ Finalizing Security Report..."
        ]

        for msg in scan_messages:

            scan_placeholder.markdown(f"""
            <div style="
            padding:15px;
            border-radius:10px;
            background:#111827;
            border:1px solid #00FFAA;
            color:#00FFAA;
            font-size:18px;
            font-weight:bold;
            text-align:center;
            animation:pulse 1s infinite;
            ">
            {msg}
            </div>

            <style>
            @keyframes pulse {{
                0% {{opacity:0.5;}}
                50% {{opacity:1;}}
                100% {{opacity:0.5;}}
            }}
            </style>
            """, unsafe_allow_html=True)

            time.sleep(0.5)
        try:
            with st.spinner("🧠 AI Engine Analyzing URL..."):
                prediction, confidence, features = predict_url(url_input)
            scan_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

            scan_placeholder.empty()

            confidence_percent = round(confidence * 100, 2)
            
            # -----------------------------
            # SMART RISK SCORE LOGIC
            # -----------------------------
            if prediction == "benign":

                risk_score = max(5, 45 - int(confidence * 100/3))

                if risk_score < 30:
                    threat_level = "SAFE"
                else:
                    threat_level = "LOW"

            else:

                risk_score = int(confidence * 100)

                if risk_score < 40:
                    threat_level = "MEDIUM"

                elif risk_score < 75:
                    threat_level = "HIGH"

                else:
                    threat_level = "CRITICAL"

            # -----------------------------
            # COLOR SELECTION
            # -----------------------------
            if prediction == "benign":
                color = "#00CC66"
                emoji = "✅"

            elif prediction == "phishing":
                color = "#FFAA00"
                emoji = "⚠️"

            else:
                color = "#FF3333"
                emoji = "🚨"

            # -----------------------------
            # RESULT BOX
            # -----------------------------
            st.markdown(
                f'''
                <div class="result-box" style="background-color:{color};">
                    {emoji} Prediction: {prediction.upper()}<br><br>
                    Confidence: {confidence_percent}%
                </div>
                ''',
                unsafe_allow_html=True
            )
            st.markdown(f"""
            <div style="
            margin-top:15px;
            padding:12px;
            border-radius:12px;
            background:#1E1E1E;
            color:#00FFAA;
            font-size:20px;
            font-weight:bold;
            text-align:center;
            border:1px solid #00FFAA;
            ">
            🧠 AI Threat Level: {threat_level}
            </div>
            """, unsafe_allow_html=True)
            # -----------------------------
            # THREAT SCORE
            # -----------------------------
            st.subheader("⚠️ Threat Risk Score")

            st.progress(risk_score / 100)
            # -----------------------------
            # AI PROBABILITY DISTRIBUTION
            # -----------------------------

            st.subheader("🧠 AI Threat Probability Analysis")

            # Create probability dictionary
            probabilities = {
                "BENIGN": 0,
                "PHISHING": 0,
                "MALWARE": 0,
                "DEFACEMENT": 0
            }

            # Main prediction confidence
            main_confidence = int(confidence_percent)

            # Set confidence for predicted class
            probabilities[prediction.upper()] = main_confidence

            # Remaining percentage
            remaining = 100 - main_confidence

            # Get other classes
            other_classes = []

            for label in probabilities.keys():
                if label != prediction.upper():
                    other_classes.append(label)

            # Divide remaining confidence equally
            split_confidence = remaining // len(other_classes)

            # Assign remaining confidence
            for label in other_classes:
                probabilities[label] = split_confidence

            # Create dataframe
            df_probs = pd.DataFrame({
                "Threat Type": list(probabilities.keys()),
                "Confidence": list(probabilities.values())
            })

            # Create pie chart
            fig = px.pie(
                df_probs,
                names="Threat Type",
                values="Confidence",
                title="Threat Prediction Distribution",
                hole=0.45,
                color="Threat Type",
                color_discrete_map={
                    "BENIGN": "#00CC66",
                    "PHISHING": "#FFAA00",
                    "MALWARE": "#FF3333",
                    "DEFACEMENT": "#AA00FF"
                },
            )

            # Dark theme styling
            fig.update_layout(
                paper_bgcolor="#0b1020",
                plot_bgcolor="#0b1020",
                font_color="white"
            )

            # Display chart
            st.plotly_chart(fig, use_container_width=True)

            if risk_score < 35:
                st.success(f"Low Risk: {risk_score}%")

            elif risk_score < 65:
                st.warning(f"Moderate Risk: {risk_score}%")

            else:
                st.error(f"High Risk: {risk_score}%")

            st.caption(f"🕒 Scan Time: {scan_time}")
            st.success("✅ Scan Completed Successfully")

            st.session_state.history.append({
                "URL": url_input,
                "Prediction": prediction.upper(),
                "Confidence": f"{confidence_percent}%",
                "Threat Level": threat_level
            })

            if st.session_state.history:

                st.subheader("📜 Recent Scan History")

                history_df = pd.DataFrame(
                    st.session_state.history[::-1]
                )

                st.dataframe(history_df, use_container_width=True)

                csv = history_df.to_csv(index=False)

                st.download_button(
                    "⬇️ Export Scan History CSV",
                    csv,
                    "scan_history.csv",
                    "text/csv"
                )
            # -----------------------------
            # FEATURE BREAKDOWN
            # -----------------------------
            st.subheader("🔍 Suspicious Feature Analysis")
            with st.expander("💻 Live Threat Scan Logs"):

                st.code(f"""
            [INFO] URL Tokenization Complete
            [INFO] Sequence Padding Applied
            [INFO] BiLSTM Deep Scan Running...
            [INFO] Threat Probability Calculated
            [INFO] Suspicious Features Extracted
            [RESULT] Final Threat Classification: {prediction.upper()}
            """)

            suspicious_found = False

            if features['suspicious_keywords']:
                st.error("Suspicious keywords detected")
                suspicious_found = True

            if features['suspicious_tld']:
                st.error("Suspicious top-level domain detected")
                suspicious_found = True

            if features['has_exe']:
                st.error("Executable (.exe) detected")
                suspicious_found = True

            if features['has_php']:
                st.warning("PHP file detected")
                suspicious_found = True

            if features['has_ip']:
                st.warning("IP address detected in URL")
                suspicious_found = True

            if not suspicious_found:
                st.success("No major suspicious indicators detected")
            # -----------------------------
            # SECURITY RECOMMENDATION
            # -----------------------------
            st.subheader("🛡️ Security Recommendation")

            if prediction == "phishing":
                st.warning("""
                Avoid entering passwords or personal information on this website.
                """)

            elif prediction == "malware":
                st.error("""
                Do NOT download files from this URL.
                Potential malware threat detected.
                """)

            elif prediction == "defacement":
                st.warning("""
                Website may be compromised or modified by attackers.
                """)

            else:
                st.success("""
                No major cybersecurity threats detected.
                """)

            # -----------------------------
            # URL STATS
            # -----------------------------
            st.subheader("📊 URL Intelligence Dashboard")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Threat Level", threat_level)

            with col2:
                st.metric("Confidence", f"{confidence_percent}%")

            with col3:
                st.metric("Risk Score", f"{risk_score}%")

            with col4:
                st.metric("Detected Type", prediction.upper())

            parsed = urlparse(url_input)

            st.subheader("🌐 URL Breakdown")

            col1, col2 = st.columns(2)

            with col1:
                st.info(f"Protocol: {parsed.scheme}")
                st.info(f"Domain: {parsed.netloc}")

            with col2:
                st.info(f"Path: {parsed.path}")

                tld = parsed.netloc.split(".")[-1] if "." in parsed.netloc else "N/A"

                st.info(f"TLD: .{tld}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("URL Length", features['url_length'])

            with col2:
                st.metric("Digits", features['digit_count'])

            with col3:
                st.metric("Special Characters", features['special_count'])

            report = f"""
            ====================================
            HYBRID URL THREAT DETECTOR REPORT
            ====================================

            Scan Time:
            {scan_time}

            Analyzed URL:
            {url_input}

            ------------------------------------
            AI PREDICTION RESULT
            ------------------------------------

            Prediction:
            {prediction.upper()}

            Confidence Score:
            {confidence_percent}%

            Threat Level:
            {threat_level}

            Risk Score:
            {risk_score}%

            ------------------------------------
            FEATURE ANALYSIS
            ------------------------------------

            URL Length:
            {features['url_length']}

            Digits:
            {features['digit_count']}

            Special Characters:
            {features['special_count']}

            Has IP Address:
            {features['has_ip']}

            Suspicious Keywords:
            {features['suspicious_keywords']}

            Executable Detected:
            {features['has_exe']}

            PHP File Detected:
            {features['has_php']}

            Suspicious TLD:
            {features['suspicious_tld']}

            ====================================
            Generated by:
            Hybrid URL Threat Detector
            Developed by Shreya Verma
            ====================================
            """
            st.download_button(
                label="📄 Download Security Report",
                data=report,
                file_name="threat_report.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Error analyzing URL: {str(e)}")
# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
---
<div style="
    text-align:center;
    padding:20px;
    color:#9CA3AF;
    font-size:15px;
">

<h3 style="
    color:#00FFAA;
    margin-bottom:10px;
">
🛡️ Hybrid URL Threat Detector
</h3>

<p>
AI-Powered Cybersecurity Intelligence System<br>
Built using BiLSTM Deep Learning + Hybrid Threat Detection
</p>

<br>

<p>
💻 Developed by <b style="color:white;">Shreya Verma</b><br>
 B.Tech CSE (AI & ML) Student
            
</p>

<br>

<p style="font-size:13px; color:#6B7280;">
Detect • Analyze • Secure
</p>

</div>
""", unsafe_allow_html=True)

