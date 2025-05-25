import streamlit as st
from utils.db_utils import db
from style import apply_style

# Set page config
st.set_page_config(page_title="Sign Up - Stale Fruit Detection", layout="centered")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'email' not in st.session_state:
    st.session_state.email = None

# Apply shared styles
apply_style()

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

def main():
    st.markdown(
        '''
        <div class="page-header fade-in">
            <h1>Create Account 🚀</h1>
            <p>Join us to start detecting fruit freshness with AI</p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Sign Up Form Container
    st.markdown(
        '''
        <div class="auth-card slide-up" style="max-width: 450px; margin: 2rem auto; padding: 2rem;">
        ''',
        unsafe_allow_html=True
    )

    with st.form("signup_form"):
        email = st.text_input("Email", placeholder="Enter your email", label_visibility="collapsed")
        password = st.text_input("Password", type="password", placeholder="Create a password", label_visibility="collapsed")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", label_visibility="collapsed")

        # Create columns for the buttons
        col1, col2 = st.columns([4, 1])
        with col1:
            submit = st.form_submit_button("Sign Up", use_container_width=True)
        with col2:
            login = st.form_submit_button("Login")

        if submit:
            if not email or not password or not confirm_password:
                st.error("Please fill in all fields.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                if db.add_user(email, password):
                    st.session_state.logged_in = True
                    st.session_state.email = email
                    st.success("Account created successfully!")
                    st.markdown("""
                        <div style='text-align: center; margin-top: 1rem;'>
                            <p>Click below to start using the app:</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Email already exists. Please use a different email or login instead.")

    st.markdown('</div>', unsafe_allow_html=True)  # Close auth-card div

    # Add Go to App button outside the form
    if st.session_state.logged_in:
        if st.button("Go to App", key="goto_app"):
            st.switch_page("pages/app.py")

    # Add Login button outside the form
    if login:
        st.switch_page("pages/2_login.py")

if __name__ == "__main__":
    main()