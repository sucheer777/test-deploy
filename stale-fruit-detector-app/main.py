import streamlit as st
import base64
from translation import TRANSLATIONS
from style import apply_style

# Set page config
st.set_page_config(page_title="Stale Fruit Detector", layout="wide")

# Apply shared styles
apply_style()

# --- Language Selection ---
lang = st.sidebar.selectbox("🌐 Select Language", ["English", "Telugu", "Hindi"])
tr = TRANSLATIONS["main"][lang]

# Main Content
st.markdown(
    '''
    <div class="page-header">
        <h1>Welcome to Stale Fruit Detector 🍎</h1>
        <p>Your AI-powered assistant for fruit freshness analysis</p>
    </div>
    ''',
    unsafe_allow_html=True
)

# Feature Cards
st.markdown(
    '''
    <div class="card-grid">
        <div class="glass-card">
            <h3>🤖 AI-Powered Detection</h3>
            <p>Advanced machine learning models analyze your fruit images with high accuracy</p>
        </div>
        <div class="glass-card">
            <h3>⚡ Instant Results</h3>
            <p>Get immediate feedback on fruit freshness and shelf life estimates</p>
        </div>
        <div class="glass-card">
            <h3>📊 Detailed Analysis</h3>
            <p>Receive comprehensive reports with confidence scores and recommendations</p>
        </div>
    </div>
    ''',
    unsafe_allow_html=True
)

# How It Works Section
st.markdown(
    '''
    <div class="glass-card">
        <h2>How It Works</h2>
        <div style="margin: 20px 0;">
            <h4>1. Upload Your Image 📸</h4>
            <p>Take a clear photo of your fruit or upload an existing image</p>
        </div>
        <div style="margin: 20px 0;">
            <h4>2. AI Analysis 🔍</h4>
            <p>Our advanced AI models analyze the image for signs of freshness</p>
        </div>
        <div style="margin: 20px 0;">
            <h4>3. Get Results 📋</h4>
            <p>Receive instant feedback and storage recommendations</p>
        </div>
    </div>
    ''',
    unsafe_allow_html=True
)

# Call to Action
if 'email' not in st.session_state:
    st.markdown(
        '''
        <div class="glass-card" style="text-align: center;">
            <h2>Get Started Today</h2>
            <p style="margin: 20px 0;">Sign up or log in to start detecting fruit freshness</p>
            <div style="margin: 20px 0;">
                <a href="/Login" class="custom-button">Login</a>
                <a href="/Signup" class="custom-button" style="margin-left: 20px;">Sign Up</a>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '''
        <div class="glass-card" style="text-align: center;">
            <h2>Start Scanning</h2>
            <p style="margin: 20px 0;">Upload an image to check fruit freshness</p>
            <div style="margin: 20px 0;">
                <a href="/app" class="custom-button">Go to Scanner</a>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )