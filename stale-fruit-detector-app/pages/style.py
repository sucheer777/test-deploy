import streamlit as st

def load_css():
    return """
        <style>
            /* Global Styles */
            .stApp {
                background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(12,74,110,0.95));
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }

            /* Modern Card Styles */
            .glass-card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                padding: 2rem;
                margin: 1.5rem 0;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                color: white;
                transition: all 0.3s ease;
            }

            .glass-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 48px 0 rgba(31, 38, 135, 0.5);
                border: 1px solid rgba(255, 255, 255, 0.3);
            }

            /* Header Styles */
            .page-header {
                background: rgba(255, 255, 255, 0.1);
                padding: 2rem;
                border-radius: 20px;
                margin-bottom: 2rem;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }

            .page-header h1 {
                color: white;
                font-size: 2.5rem;
                margin: 0;
                font-weight: 600;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            }

            .page-header p {
                color: rgba(255, 255, 255, 0.8);
                margin-top: 1rem;
                font-size: 1.1rem;
            }

            /* Form Styles */
            .form-container {
                background: rgba(255, 255, 255, 0.1);
                padding: 2rem;
                border-radius: 20px;
                margin: 1rem 0;
            }

            .form-field {
                margin: 1rem 0;
            }

            .form-field label {
                color: white;
                font-size: 1rem;
                margin-bottom: 0.5rem;
                display: block;
            }

            .form-field input {
                width: 100%;
                padding: 0.75rem;
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                background: rgba(255, 255, 255, 0.1);
                color: white;
                font-size: 1rem;
            }

            /* Button Styles */
            .custom-button {
                background: linear-gradient(135deg, #6366f1, #4f46e5);
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: 10px;
                border: none;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.3s ease;
                text-align: center;
                display: inline-block;
                margin: 0.5rem 0;
                text-decoration: none;
            }

            .custom-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
                background: linear-gradient(135deg, #4f46e5, #4338ca);
            }

            /* Alert and Message Styles */
            .success-message {
                background: rgba(34, 197, 94, 0.2);
                border: 1px solid rgba(34, 197, 94, 0.3);
                color: #22c55e;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
            }

            .error-message {
                background: rgba(239, 68, 68, 0.2);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: #ef4444;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
            }

            /* Navigation Styles */
            .nav-link {
                color: white;
                text-decoration: none;
                padding: 0.5rem 1rem;
                border-radius: 8px;
                transition: all 0.3s ease;
            }

            .nav-link:hover {
                background: rgba(255, 255, 255, 0.1);
            }

            /* Card Grid */
            .card-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
                margin: 1.5rem 0;
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .page-header h1 {
                    font-size: 2rem;
                }

                .glass-card {
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