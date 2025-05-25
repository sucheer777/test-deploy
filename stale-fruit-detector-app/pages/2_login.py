import streamlit as st
from utils.db_utils import db
import base64
from translation import TRANSLATIONS
from pathlib import Path
from style import apply_style
import json
import hashlib
import hmac
import time

# Set page config
st.set_page_config(page_title="Login - Stale Fruit Detection", layout="centered")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'email' not in st.session_state:
    st.session_state.email = None

# Secret key for cookie signing (in a real app, this should be in a secure config)
SECRET_KEY = "your-secret-key-12345"

def create_login_cookie(email):
    """Create a secure cookie for persistent login"""
    timestamp = str(int(time.time()))
    message = f"{email}:{timestamp}"
    signature = hmac.new(
        SECRET_KEY.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    cookie = f"{message}:{signature}"
    return cookie

def verify_login_cookie():
    """Verify the login cookie and return the email if valid"""
    try:
        cookie = st.session_state.get('login_cookie')
        if not cookie:
            return None
        
        email, timestamp, signature = cookie.split(':')
        message = f"{email}:{timestamp}"
        expected_signature = hmac.new(
            SECRET_KEY.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if hmac.compare_digest(signature, expected_signature):
            # Check if cookie is not too old (e.g., 30 days)
            if int(time.time()) - int(timestamp) < 30 * 24 * 60 * 60:
                return email
        return None
    except Exception:
        return None

# Check for existing login cookie
if not st.session_state.logged_in:
    email = verify_login_cookie()
    if email:
        st.session_state.logged_in = True
        st.session_state.email = email

# Apply shared styles
apply_style()

# --- Language Selection ---
lang = st.sidebar.selectbox("🌐 Select Language", ["English", "Telugu", "Hindi"])
tr = TRANSLATIONS["login"][lang]

# Add logout option in sidebar if logged in
if st.session_state.logged_in:
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.email = None
            if 'login_cookie' in st.session_state:
                del st.session_state.login_cookie
            st.rerun()

# Show current login status
if st.session_state.logged_in:
    st.info(f"Currently logged in as: {st.session_state.email}")
    if st.button("Go to App"):
        st.switch_page("pages/app.py")
    st.markdown("---")

# Main Content
st.markdown(
    '''
    <div class="page-header fade-in">
        <h1>Welcome Back 👋</h1>
        <p>Log in to your Stale Fruit Detector account</p>
    </div>
    ''',
    unsafe_allow_html=True
)

# Custom CSS for dark input fields
st.markdown(
    '''
    <style>
    .dark-input {
        background-color: #1a1a1a !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }
    
    .dark-input::placeholder {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    
    /* Override Streamlit's default input styling */
    .stTextInput > div > div > input {
        background-color: #1a1a1a !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px !important;
        border-radius: 8px !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# Login Form Container
st.markdown(
    '''
    <div class="auth-card slide-up" style="max-width: 450px; margin: 2rem auto; padding: 2rem;">
    ''',
    unsafe_allow_html=True
)

# Login Form
with st.form("login_form", clear_on_submit=True):
    email = st.text_input("Email", placeholder="Enter your email", label_visibility="collapsed")
    password = st.text_input("Password", type="password", placeholder="Enter your password", label_visibility="collapsed")
    remember_me = st.checkbox("Remember me", value=True)
    
    # Create columns for the buttons
    col1, col2 = st.columns([4, 1])
    with col1:
        submit = st.form_submit_button("Login", use_container_width=True)
    with col2:
        if st.form_submit_button("Sign Up"):
            st.switch_page("pages/3_signup.py")

    if submit:
        if not email or not password:
            st.error("Please fill in all fields.")
        else:
            try:
                if db.verify_user(email, password):
                    st.session_state.logged_in = True
                    st.session_state.email = email
                    
                    # Create and store login cookie if remember me is checked
                    if remember_me:
                        cookie = create_login_cookie(email)
                        st.session_state.login_cookie = cookie
                    
                    st.success(f"{tr['welcome']}, {email}!")
                    st.rerun()
                else:
                    st.error(tr["incorrect_password"])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please try again later or contact support if the issue persists.")

# Close the glass card container
st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    pass




