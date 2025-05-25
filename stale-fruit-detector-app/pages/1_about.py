import streamlit as st
from translation import TRANSLATIONS
from style import apply_style

# Set page config
st.set_page_config(page_title="About - Stale Fruit Detection", layout="wide")

# Apply shared styles
apply_style()

# --- Language Selection ---
lang = st.sidebar.selectbox("🌐 Select Language", ["English", "Telugu", "Hindi"])
tr = TRANSLATIONS["about"][lang]

# --- Title Section ---
st.markdown(
    f'''
    <div class="page-header fade-in">
        <h1>{tr['title']}</h1>
        <p>{tr['desc']}</p>
    </div>
    ''',
    unsafe_allow_html=True
)

# --- Features Section ---
st.markdown('<h2 class="section-title">✨ Key Features</h2>', unsafe_allow_html=True)

# Create three columns for features
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        '''
        <div class="glass-card fade-in">
            <h3>🤖 AI-Powered Detection</h3>
            <p>Advanced machine learning models analyze your fruits with high accuracy and reliability.</p>
        </div>
        ''',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        '''
        <div class="glass-card fade-in">
            <h3>⚡ Instant Results</h3>
            <p>Get immediate feedback on fruit freshness and storage recommendations.</p>
        </div>
        ''',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        '''
        <div class="glass-card fade-in">
            <h3>📊 Detailed Analysis</h3>
            <p>Comprehensive reports with freshness scores and shelf-life predictions.</p>
        </div>
        ''',
        unsafe_allow_html=True
    )

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# --- How It Works Section ---
st.markdown(
    '''
    <div class="glass-card slide-up">
        <h2>🔍 How It Works</h2>
        <div class="card-grid">
            <div class="glass-card">
                <h3>1. Upload Your Image 📸</h3>
                <p>Take a clear photo of your fruit and upload it to our platform.</p>
            </div>
            <div class="glass-card">
                <h3>2. AI Analysis 🧠</h3>
                <p>Our advanced AI models analyze the image for signs of freshness or deterioration.</p>
            </div>
            <div class="glass-card">
                <h3>3. Get Results 📋</h3>
                <p>Receive instant feedback about fruit freshness and storage recommendations.</p>
            </div>
        </div>
    </div>
    ''',
    unsafe_allow_html=True
)

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown(
    f'''
    <div class="glass-card" style="text-align: center;">
        <p>{tr["footer"]}</p>
    </div>
    ''',
    unsafe_allow_html=True
)
