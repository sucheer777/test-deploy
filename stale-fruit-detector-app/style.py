import streamlit as st

def load_css():
    return """
        <style>
            /* Global Styles */
            .stApp {
                background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }

            /* Modern Card Styles */
            .glass-card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                padding: 2rem;
                margin: 1.5rem 0;
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
                color: white;
                transition: all 0.3s ease;
            }

            /* Login/Signup Card Style */
            .auth-card {
                background: #1a1a1a;
                border-radius: 10px;
                padding: 2rem;
                margin: 1.5rem 0;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                color: white;
                transition: all 0.3s ease;
            }

            .auth-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 48px rgba(0, 0, 0, 0.4);
            }

            .glass-card h2, .glass-card h3, .glass-card h4 {
                color: #1a1a1a;
                margin-bottom: 1rem;
            }

            .glass-card h2 {
                font-size: 1.8rem;
            }

            .glass-card h3 {
                font-size: 1.3rem;
            }

            .glass-card h4 {
                font-size: 1.1rem;
            }

            .glass-card p {
                color: #333333;
                font-size: 1rem;
                line-height: 1.5;
            }

            /* Form Input Styles */
            .stTextInput > div > div > input {
                background-color: #2d2d2d !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 12px !important;
            }

            .stTextInput > div > div > input::placeholder {
                color: rgba(255, 255, 255, 0.6) !important;
            }

            /* Button Styles */
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

            /* Header Styles */
            .page-header {
                text-align: center;
                padding: 2rem;
                background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
                border-radius: 15px;
                margin-bottom: 2rem;
                animation: fadeIn 0.5s ease-in;
            }

            .page-header h1 {
                color: white;
                font-size: 2.5rem;
                margin-bottom: 1rem;
            }

            .page-header p {
                color: white;
                font-size: 1.2rem;
                opacity: 0.9;
            }

            /* Alert and Message Styles */
            .success-message {
                background: rgba(34, 197, 94, 0.2);
                border: 1px solid rgba(34, 197, 94, 0.3);
                color: #22c55e;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
                animation: fadeIn 0.5s ease-in;
            }

            .error-message {
                background: rgba(239, 68, 68, 0.2);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: #ef4444;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
                animation: fadeIn 0.5s ease-in;
            }

            /* Card Grid */
            .card-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
                margin: 1.5rem 0;
                animation: fadeIn 0.5s ease-in;
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .page-header h1 {
                    font-size: 2rem;
                }

                .glass-card, .auth-card {
                    padding: 1.5rem;
                }

                .card-grid {
                    grid-template-columns: 1fr;
                }
            }

            /* Animation Classes */
            .fade-in {
                animation: fadeIn 0.5s ease-in;
            }

            .slide-up {
                animation: slideUp 0.5s ease-out;
            }

            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            @keyframes slideUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            /* Custom Scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }

            ::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.1);
            }

            ::-webkit-scrollbar-thumb {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 4px;
            }

            ::-webkit-scrollbar-thumb:hover {
                background: rgba(255, 255, 255, 0.4);
            }

            /* Loading Spinner */
            .loading-spinner {
                border: 4px solid rgba(255, 255, 255, 0.1);
                border-left: 4px solid white;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    """

def apply_style():
    st.markdown(load_css(), unsafe_allow_html=True) 