import streamlit as st
import psycopg2
import hashlib
import uuid
from datetime import datetime
import time

# Database configuration - UPDATE WITH YOUR ACTUAL CREDENTIALS
DB_CONFIG = {
    'host': 'localhost',          # Your PostgreSQL host
    'database': 'Users',  # Your actual database name
    'user': 'postgres',      # Your PostgreSQL username
    'password': 'Shiti09',  # Your PostgreSQL password
    'port': '5432'               # Default PostgreSQL port
}

def create_connection():
    """Create a database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_user_id():
    """Generate a unique user ID - Not needed since ID is auto-increment"""
    return str(uuid.uuid4())

def create_user(email, password):
    """Create a new user in the database"""
    conn = create_connection()
    if conn is None:
        return False, "Database connection failed"
    
    try:
        cursor = conn.cursor()
        
        # Check if email already exists (updated column name)
        cursor.execute("SELECT email FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return False, "Email already exists"
        
        # Generate user data
        hashed_password = hash_password(password)
        signup_date = datetime.now().date()
        signup_time = datetime.now().time()
        
        # Insert new user (updated to match your schema - ID will auto-increment)
        insert_query = """
            INSERT INTO users (email, password, signup_date, signup_time)
            VALUES (%s, %s, %s, %s)
            """
        cursor.execute(insert_query, (email, hashed_password, signup_date, signup_time))
        conn.commit()
        
        return True, "User created successfully"
        
    except Exception as e:
        return False, f"Error creating user: {e}"
    finally:
        cursor.close()
        conn.close()

def verify_user(email, password):
    """Verify user credentials"""
    conn = create_connection()
    if conn is None:
        return False, "Database connection failed"
    
    try:
        cursor = conn.cursor()
        hashed_password = hash_password(password)
        
        # Updated to match your table schema
        cursor.execute(
                "SELECT id, email FROM users WHERE email = %s AND password = %s", (email, hashed_password))

        user = cursor.fetchone()
        
        if user:
            return True, f"Login successful! Welcome, {user[1]}"
        else:
            return False, "Invalid email or password"
            
    except Exception as e:
        return False, f"Error verifying user: {e}"
    finally:
        cursor.close()
        conn.close()

def main():
    st.set_page_config(page_title="User Authentication", page_icon="🔐", layout="centered")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .tab-header {
        text-align: center;
        color: #333;
        margin-bottom: 1rem;
    }
    .success-msg {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-msg {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔐 User Authentication System</h1>', unsafe_allow_html=True)
    
    # Create tabs for Login and Signup
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
    
    with tab1:
        st.markdown('<h2 class="tab-header">Login to Your Account</h2>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("📧 Email", placeholder="Enter your email")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            login_button = st.form_submit_button("Login", use_container_width=True)
            
            if login_button:
                if email and password:
                    success, message = verify_user(email, password)
                    if success:
                        st.success(message)
                        st.balloons()
                        # Here you can redirect to main app or set session state
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                    else:
                        st.error(message)
                else:
                    st.warning("Please fill in all fields")
    
    with tab2:
        st.markdown('<h2 class="tab-header">Create New Account</h2>', unsafe_allow_html=True)
        
        with st.form("signup_form"):
            new_email = st.text_input("📧 Email", placeholder="Enter your email")
            new_password = st.text_input("🔒 Password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")
            signup_button = st.form_submit_button("Sign Up", use_container_width=True)
            
            if signup_button:
                if new_email and new_password and confirm_password:
                    if new_password == confirm_password:
                        if len(new_password) >= 6:
                            success, message = create_user(new_email, new_password)
                            if success:
                                st.success(message)
                                st.balloons()
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error(message)
                        else:
                            st.warning("Password must be at least 6 characters long")
                    else:
                        st.error("Passwords do not match")
                else:
                    st.warning("Please fill in all fields")
    
    # Display login status
    if 'logged_in' in st.session_state and st.session_state.logged_in:
        st.sidebar.success(f"Logged in as: {st.session_state.user_email}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_email = None
            st.rerun()

    # Database setup instructions
    with st.expander("📋 Database Setup Instructions"):
        st.markdown("""
        **Your Current PostgreSQL Table Schema:**
        ```sql
        -- Your table schema (based on the screenshot):
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password TEXT NOT NULL,
            signup_date DATE,
            signup_time TIME WITH TIME ZONE
);
        ```
        
        **Configuration Steps:**
        1. Update the `DB_CONFIG` dictionary with your PostgreSQL credentials
        2. Make sure your PostgreSQL server is running
        3. Your table schema is already set up correctly
        4. Install required packages: `pip install streamlit psycopg2-binary`
        
        **Notes:**
        - ID will auto-increment automatically
        - Email and Password will be stored as provided
        - Signup_Date and Signup_Time will be automatically generated
        """)

if __name__ == "__main__":
    main()