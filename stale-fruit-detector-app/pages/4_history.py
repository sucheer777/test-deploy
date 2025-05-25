import streamlit as st
import pandas as pd
from datetime import datetime
import base64
from translation import TRANSLATIONS
from uploads.db_utils import get_scan_history, get_user_scans
from db_utils import get_predictions_by_user
from style import apply_style

# Set page config
st.set_page_config(page_title="History - Stale Fruit Detection", layout="centered")

# --- Language Selection ---
lang = st.sidebar.selectbox("🌐 Select Language", ["English", "Telugu", "Hindi"])
tr = TRANSLATIONS["history"][lang]

# --- Set Background Image Function ---
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Modern UI Enhancements */
        .stApp {{
            background: linear-gradient(135deg, rgba(0,0,0,0.7), rgba(0,0,0,0.5));
        }}

        /* Enhanced Glass Card */
        .glass-card {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            color: white;
            transition: all 0.3s ease;
        }}

        .glass-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 48px 0 rgba(31, 38, 135, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }}

        /* Enhanced Heading Container */
        .glass-heading-container {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 25px;
            margin: 25px 0;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.4);
            display: flex;
            justify-content: space-between;
            align-items: center;
            animation: float-in 1s ease-out forwards;
        }}

        .glass-heading {{
            color: white;
            font-size: 2.8rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}

        /* Enhanced Data Table */
        .stDataFrame {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
        }}

        .stDataFrame th {{
            background: rgba(255, 255, 255, 0.15);
            color: white;
            font-weight: bold;
        }}

        .stDataFrame td {{
            color: white;
        }}

        /* Enhanced Selectbox */
        .stSelectbox [data-baseweb="select"] {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 12px;
            color: white;
            transition: all 0.3s ease;
        }}

        .stSelectbox [data-baseweb="select"]:hover {{
            background: rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 20px rgba(31, 38, 135, 0.4);
            border-color: rgba(255, 255, 255, 0.5);
            transform: translateY(-2px);
        }}

        /* Animations */
        @keyframes float-in {{
            0% {{
                opacity: 0;
                transform: translateY(-30px);
            }}
            100% {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes fade-in {{
            0% {{
                opacity: 0;
            }}
            100% {{
                opacity: 1;
            }}
        }}

        /* Sidebar Enhancement */
        .css-1d391kg {{
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }}

        /* Text Enhancements */
        p, h1, h2, h3, h4, h5, h6 {{
            color: white;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
        }}

        /* Enhanced Messages */
        .stSuccess {{
            background: rgba(40, 167, 69, 0.2);
            border: 1px solid rgba(40, 167, 69, 0.3);
            border-radius: 12px;
            padding: 15px;
            margin: 10px 0;
            animation: fade-in 0.5s ease-out;
        }}

        .stError {{
            background: rgba(220, 53, 69, 0.2);
            border: 1px solid rgba(220, 53, 69, 0.3);
            border-radius: 12px;
            padding: 15px;
            margin: 10px 0;
            animation: fade-in 0.5s ease-out;
        }}

        /* Enhanced Stats Cards */
        .stats-card {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 20px;
            margin: 10px;
            text-align: center;
            transition: all 0.3s ease;
        }}

        .stats-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }}

        .stats-value {{
            font-size: 2rem;
            font-weight: bold;
            color: white;
            margin: 10px 0;
        }}

        .stats-label {{
            color: rgba(255, 255, 255, 0.8);
            font-size: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set Background Image
set_background("background_images/about page background.jpg")

# Main Content
st.markdown(f"""
    <div class="glass-heading-container">
        <h1 class="glass-heading">{tr['title']}</h1>
    </div>
""", unsafe_allow_html=True)

# Apply shared styles
apply_style()

def format_timestamp(timestamp):
    try:
        if isinstance(timestamp, str):
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        else:
            dt = timestamp
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except:
        return str(timestamp)

def main():
    # Apply shared styles
    apply_style()

    if 'email' not in st.session_state:
        st.error(tr["login_required"])
        if st.button(tr["go_to_login"]):
            st.switch_page("pages/2_login.py")
        return

    st.markdown(
        '''
        <div class="page-header fade-in">
            <h1>Prediction History 📊</h1>
            <p>View your past fruit freshness predictions</p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Get predictions
    try:
        predictions = get_predictions_by_user(st.session_state["email"])
        
        if not predictions:
            st.info("No predictions found. Try analyzing some fruits!")
            return
            
        # Convert predictions to DataFrame
        df = pd.DataFrame(predictions, columns=[
            "Result", "Confidence (%)", "Fruit Type", "Timestamp",
            "Condition", "Storage Recommendation", "Shelf Life Confidence (%)"
        ])
        
        # Format timestamp
        df["Timestamp"] = df["Timestamp"].apply(format_timestamp)
        
        # Format confidence values
        df["Confidence (%)"] = df["Confidence (%)"].round(1)
        df["Shelf Life Confidence (%)"] = df["Shelf Life Confidence (%)"].fillna(0).round(1)
        
        # Replace NaN values with "Not Available"
        df = df.fillna("Not Available")
        
        # Display the predictions in a glass card
        st.markdown(
            '''
            <div class="glass-card fade-in">
            ''',
            unsafe_allow_html=True
        )
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error loading prediction history: {str(e)}")

if __name__ == "__main__":
    main() 