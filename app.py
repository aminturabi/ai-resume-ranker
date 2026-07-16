import streamlit as st
import pdfplumber
import pandas as pd
import re
import io
import sqlite3
from datetime import datetime, timezone, timedelta
import uuid
from collections import Counter
from werkzeug.security import generate_password_hash, check_password_hash

# Optional imports
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Resume Ranker ATS",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "ats_database.db"

# SaaS / Plan configuration
PLANS = {
    7:  {"label": "7-Day Pass",  "price": 500},
    15: {"label": "15-Day Pass", "price": 900},
    30: {"label": "30-Day Pass", "price": 1500},
}

FREE_PDF_LIMIT = 10

PAYMENT_INSTRUCTIONS = {
    "easypaisa_number": "0311-8036997",
    "account_title": "Amin Khan",
}

ATS_STAGES = ["New", "Screening", "Shortlisted", "Interview", "Offer", "Hired", "Rejected"]

STAGE_COLORS = {
    "New": "#6B7280",
    "Screening": "#4F46E5",
    "Shortlisted": "#059669",
    "Interview": "#D97706",
    "Offer": "#65A30D",
    "Hired": "#0284C7",
    "Rejected": "#DC2626",
}

DEFAULT_SKILLS = [
    "python", "machine learning", "deep learning", "sql", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch", "nlp", "computer vision",
    "docker", "fastapi", "flask", "streamlit", "git", "aws", "azure",
    "data analysis", "statistics", "matplotlib", "seaborn", "transformers",
    "langchain", "rag", "llm"
]

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Font & App Styling overrides */
    html, body, .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    p, h1, h2, h3, h4, h5, h6, label, button, input, select, textarea {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    .stApp {
        background-color: #F8FAFC !important;
    }

    /* Force dark color on headings & body text in the main content container to prevent dark theme conflicts */
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] h5,
    [data-testid="stAppViewContainer"] h6 {
        color: #0F172A !important;
    }
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] li {
        color: #1E293B !important;
    }
    
    /* Reset button and link inner text color to inherit parent button color */
    [data-testid="stAppViewContainer"] button p,
    [data-testid="stAppViewContainer"] a p,
    [data-testid="stAppViewContainer"] button span,
    [data-testid="stAppViewContainer"] a span {
        color: inherit !important;
    }

    /* Sidebar styling overrides */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        border-right: 1px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1E293B !important;
    }

    /* Streamlit tabs customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 2px solid #E2E8F0 !important;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
        border: none !important;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"] div,
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        font-weight: 600 !important;
        color: #64748B !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #EEF2F6 !important;
    }
    .stTabs [data-baseweb="tab"]:hover div,
    .stTabs [data-baseweb="tab"]:hover p,
    .stTabs [data-baseweb="tab"]:hover span {
        color: #4F46E5 !important;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid #4F46E5 !important;
    }
    .stTabs [aria-selected="true"] div,
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        color: #4F46E5 !important;
        font-weight: 700 !important;
    }

    /* Primary and default buttons styling */
    .stButton>button, div[data-testid="stLinkButton"] a {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
        color: white !important;
        text-decoration: none !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.15), 0 2px 4px -2px rgba(79, 70, 229, 0.15) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100%;
    }
    .stButton>button:hover, div[data-testid="stLinkButton"] a:hover {
        background: linear-gradient(135deg, #4338CA 0%, #2563EB 100%) !important;
        color: white !important;
        text-decoration: none !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.25), 0 4px 6px -4px rgba(79, 70, 229, 0.25) !important;
    }
    .stButton>button:active, div[data-testid="stLinkButton"] a:active {
        transform: translateY(0px) !important;
    }

    /* Header styling */
    .main-title {
        font-size: 2.85rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }
    .subtitle {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 30px;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        padding: 20px 16px;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.02);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0px 12px 24px rgba(0,0,0,0.05);
        border-color: #CBD5E1;
    }

    /* Recommendations / Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        border: 1px solid transparent;
        line-height: 1;
    }
    .badge-strong {
        background-color: #ECFDF5;
        color: #047857;
        border-color: #A7F3D0;
    }
    .badge-good {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border-color: #BFDBFE;
    }
    .badge-possible {
        background-color: #F8FAFC;
        color: #475569;
        border-color: #E2E8F0;
    }
    .badge-missing {
        background-color: #FFFBEB;
        color: #B45309;
        border-color: #FDE68A;
    }
    .badge-weak {
        background-color: #FEF2F2;
        color: #B91C1C;
        border-color: #FCA5A5;
    }

    /* Custom progress bar styles inside expanders */
    .scores-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-bottom: 20px;
        background: #F8FAFC;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }
    .score-progress-container {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .score-progress-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        font-weight: 600;
        color: #475569;
    }
    .score-progress-bar {
        background-color: #E2E8F0;
        border-radius: 9999px;
        height: 8px;
        overflow: hidden;
        position: relative;
        width: 100%;
    }
    .score-progress-fill {
        height: 100%;
        border-radius: 9999px;
    }

    /* Pipeline Status boxes */
    .pipeline-stage-box {
        text-align: center;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 14px 8px;
        background-color: #FFFFFF;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.01);
        transition: all 0.2s ease;
    }
    .pipeline-stage-box:hover {
        transform: translateY(-2px);
        box-shadow: 0px 8px 16px rgba(0,0,0,0.04);
    }

    /* Expander card custom outline styling targeting data-testid for compatibility */
    [data-testid="stExpander"] {
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        margin-bottom: 8px !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stExpander"] summary {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary div,
    [data-testid="stExpander"] summary svg,
    [data-testid="stExpander"] summary svg path {
        color: #1E293B !important;
        fill: #1E293B !important;
    }
    [data-testid="stExpander"]:hover {
        border-color: #CBD5E1 !important;
    }
    
    /* File Uploader area */
    div[data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        border: 2px dashed #CBD5E1;
        border-radius: 14px;
        padding: 16px;
        transition: border-color 0.2s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #4F46E5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# DATABASE
# =====================================================
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            required_skills TEXT,
            required_years REAL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            candidate_id TEXT NOT NULL,
            name TEXT,
            email TEXT,
            phone TEXT,
            filename TEXT,
            final_score REAL,
            semantic_score REAL,
            skill_score REAL,
            exp_score REAL,
            years_exp REAL,
            skills_found TEXT,
            skills_missing TEXT,
            auto_summary TEXT,
            recommendation TEXT,
            extraction_method TEXT,
            stage TEXT DEFAULT 'New',
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (job_id) REFERENCES jobs(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT,
            job_id INTEGER,
            action TEXT,
            old_value TEXT,
            new_value TEXT,
            timestamp TEXT NOT NULL
        )
    """)

    # Schema migration: check and dynamically add any missing columns due to older DB schemas
    try:
        # Check jobs table
        c.execute("PRAGMA table_info(jobs)")
        jobs_cols = [col[1] for col in c.fetchall()]
        if "required_years" not in jobs_cols:
            c.execute("ALTER TABLE jobs ADD COLUMN required_years REAL DEFAULT 0")
        if "user_id" not in jobs_cols:
            c.execute("ALTER TABLE jobs ADD COLUMN user_id INTEGER")

        # Check candidates table
        c.execute("PRAGMA table_info(candidates)")
        candidates_cols = [col[1] for col in c.fetchall()]
        missing_candidate_cols = {
            "auto_summary": "TEXT",
            "recommendation": "TEXT",
            "extraction_method": "TEXT",
            "stage": "TEXT DEFAULT 'New'",
            "notes": "TEXT DEFAULT ''"
        }
        for col_name, col_type in missing_candidate_cols.items():
            if col_name not in candidates_cols:
                c.execute(f"ALTER TABLE candidates ADD COLUMN {col_name} {col_type}")
    except sqlite3.OperationalError:
        pass

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            free_pdfs_used INTEGER DEFAULT 0,
            plan_expiry TEXT,
            active_plan_days INTEGER,
            created_at TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS payment_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            plan_days INTEGER NOT NULL,
            amount REAL NOT NULL,
            method TEXT NOT NULL,
            transaction_id TEXT NOT NULL,
            sender_number TEXT,
            status TEXT DEFAULT 'pending',
            admin_note TEXT,
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    c.execute("CREATE INDEX IF NOT EXISTS idx_candidates_job_id ON candidates(job_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_candidates_candidate_id ON candidates(candidate_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_audit_job_id ON audit_log(job_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_payments_user_id ON payment_requests(user_id)")

    # Bootstrap default admin if none exists
    c.execute("SELECT id FROM users WHERE email = ? LIMIT 1", ("aminturabi594@gmail.com",))
    if not c.fetchone():
        hashed_pw = generate_password_hash("changeme123")
        c.execute(
            "INSERT INTO users (email, password_hash, is_admin, created_at) VALUES (?, ?, ?, ?)",
            ("aminturabi594@gmail.com", hashed_pw, 1, datetime.now(timezone.utc).isoformat())
        )

    conn.commit()
    conn.close()


def get_user_by_email(email):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, email, password_hash, is_admin, free_pdfs_used, plan_expiry, active_plan_days FROM users WHERE email = ?", (email.strip().lower(),))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "email": row[1],
            "password_hash": row[2],
            "is_admin": bool(row[3]),
            "free_pdfs_used": row[4],
            "plan_expiry": row[5],
            "active_plan_days": row[6],
        }
    return None


def get_user_by_id(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, email, password_hash, is_admin, free_pdfs_used, plan_expiry, active_plan_days FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "email": row[1],
            "password_hash": row[2],
            "is_admin": bool(row[3]),
            "free_pdfs_used": row[4],
            "plan_expiry": row[5],
            "active_plan_days": row[6],
        }
    return None


def create_user(email, password):
    conn = get_connection()
    c = conn.cursor()
    hashed = generate_password_hash(password)
    try:
        c.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email.strip().lower(), hashed, datetime.now(timezone.utc).isoformat())
        )
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        return None


def verify_user(email, password):
    user = get_user_by_email(email)
    if user and check_password_hash(user["password_hash"], password):
        return user
    return None


def increment_user_free_count(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE users SET free_pdfs_used = free_pdfs_used + 1 WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


def submit_payment_request(user_id, plan_days, amount, method, transaction_id, sender_number):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO payment_requests 
        (user_id, plan_days, amount, method, transaction_id, sender_number, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, plan_days, amount, method, transaction_id, sender_number, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()


def get_pending_payment_requests():
    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT pr.id, pr.user_id, u.email, pr.plan_days, pr.amount, pr.method, 
               pr.transaction_id, pr.sender_number, pr.status, pr.created_at
        FROM payment_requests pr
        JOIN users u ON pr.user_id = u.id
        WHERE pr.status = 'pending'
        ORDER BY pr.created_at DESC
        """,
        conn
    )
    conn.close()
    return df


def approve_payment_request(req_id):
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT user_id, plan_days FROM payment_requests WHERE id = ?", (req_id,))
    req = c.fetchone()
    if not req:
        conn.close()
        return False
        
    user_id, plan_days = req
    
    c.execute("SELECT plan_expiry FROM users WHERE id = ?", (user_id,))
    user_row = c.fetchone()
    now = datetime.now(timezone.utc)
    
    plan_expiry_str = user_row[0] if user_row else None
    base_time = now
    if plan_expiry_str:
        try:
            exp_time = datetime.fromisoformat(plan_expiry_str)
            if exp_time.tzinfo is None:
                exp_time = exp_time.replace(tzinfo=timezone.utc)
            if exp_time > now:
                base_time = exp_time
        except ValueError:
            pass
            
    new_expiry = base_time + timedelta(days=plan_days)
    
    c.execute(
        "UPDATE users SET plan_expiry = ?, active_plan_days = ? WHERE id = ?",
        (new_expiry.isoformat(), plan_days, user_id)
    )
    c.execute(
        "UPDATE payment_requests SET status = 'approved', reviewed_at = ? WHERE id = ?",
        (now.isoformat(), req_id)
    )
    conn.commit()
    conn.close()
    return True


def reject_payment_request(req_id, note=""):
    conn = get_connection()
    c = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    c.execute(
        "UPDATE payment_requests SET status = 'rejected', admin_note = ?, reviewed_at = ? WHERE id = ?",
        (note, now, req_id)
    )
    conn.commit()
    conn.close()
    return True


def check_user_access(user_id):
    user = get_user_by_id(user_id)
    if not user:
        return False, "User not found"
    if user["is_admin"]:
        return True, "Admin access"
        
    expiry_str = user["plan_expiry"]
    if expiry_str:
        try:
            exp_time = datetime.fromisoformat(expiry_str)
            now = datetime.now(timezone.utc)
            if exp_time.tzinfo is None:
                exp_time = exp_time.replace(tzinfo=timezone.utc)
            if exp_time > now:
                return True, f"Paid Plan Active until {exp_time.strftime('%Y-%m-%d %H:%M')} UTC"
        except ValueError:
            pass
            
    used = user["free_pdfs_used"]
    if used < FREE_PDF_LIMIT:
        return True, f"Free Tier: {FREE_PDF_LIMIT - used} free screenings remaining"
        
    return False, "Free limit reached. Please upgrade your plan."


def update_user_password(user_id, new_password):
    conn = get_connection()
    c = conn.cursor()
    hashed = generate_password_hash(new_password)
    c.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hashed, user_id))
    conn.commit()
    conn.close()


def save_job(user_id, title, description, required_skills, required_years):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO jobs (user_id, title, description, required_skills, required_years, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, title, description, required_skills, required_years, datetime.now(timezone.utc).isoformat()),
    )
    job_id = c.lastrowid
    conn.commit()
    conn.close()
    return job_id


def save_candidates_batch(job_id, rows):
    conn = get_connection()
    c = conn.cursor()
    c.executemany(
        """
        INSERT INTO candidates
        (job_id, candidate_id, name, email, phone, filename,
         final_score, semantic_score, skill_score, exp_score,
         years_exp, skills_found, skills_missing, auto_summary,
         recommendation, extraction_method, stage, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                job_id,
                r["candidate_id"],
                r["name"],
                r["email"],
                r["phone"],
                r["filename"],
                r["final_score"],
                r["semantic_score"],
                r["skill_score"],
                r["exp_score"],
                r["years_exp"],
                r["skills_found"],
                r["skills_missing"],
                r["auto_summary"],
                r["recommendation"],
                r["extraction_method"],
                "New",
                datetime.now(timezone.utc).isoformat(),
            )
            for r in rows
        ],
    )
    conn.commit()
    conn.close()


@st.cache_data(ttl=10)
def get_jobs(user_id):
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM jobs WHERE user_id=? ORDER BY created_at DESC", conn, params=(user_id,))
    conn.close()
    return df


@st.cache_data(ttl=10)
def get_candidates(user_id, job_id):
    conn = get_connection()
    # Verify ownership
    c = conn.cursor()
    c.execute("SELECT user_id FROM jobs WHERE id = ?", (job_id,))
    row = c.fetchone()
    if not row or row[0] != user_id:
        conn.close()
        return pd.DataFrame()
        
    df = pd.read_sql(
        "SELECT * FROM candidates WHERE job_id=? ORDER BY final_score DESC",
        conn,
        params=(job_id,),
    )
    conn.close()
    return df


def get_audit_log(user_id, job_id):
    conn = get_connection()
    # Verify ownership
    c = conn.cursor()
    c.execute("SELECT user_id FROM jobs WHERE id = ?", (job_id,))
    row = c.fetchone()
    if not row or row[0] != user_id:
        conn.close()
        return pd.DataFrame()
        
    df = pd.read_sql(
        "SELECT * FROM audit_log WHERE job_id=? ORDER BY timestamp DESC",
        conn,
        params=(job_id,),
    )
    conn.close()
    return df


def update_stage(user_id, candidate_id, job_id, new_stage, old_stage):
    conn = get_connection()
    c = conn.cursor()
    # Verify ownership
    c.execute("SELECT user_id FROM jobs WHERE id = ?", (job_id,))
    row = c.fetchone()
    if not row or row[0] != user_id:
        conn.close()
        return False
        
    c.execute(
        "UPDATE candidates SET stage=? WHERE candidate_id=? AND job_id=?",
        (new_stage, candidate_id, job_id),
    )
    c.execute(
        """
        INSERT INTO audit_log (candidate_id, job_id, action, old_value, new_value, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (candidate_id, job_id, "stage_change", old_stage, new_stage, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return True


def save_notes(user_id, candidate_id, job_id, notes):
    conn = get_connection()
    c = conn.cursor()
    # Verify ownership
    c.execute("SELECT user_id FROM jobs WHERE id = ?", (job_id,))
    row = c.fetchone()
    if not row or row[0] != user_id:
        conn.close()
        return False
        
    c.execute(
        "UPDATE candidates SET notes=? WHERE candidate_id=? AND job_id=?",
        (notes, candidate_id, job_id),
    )
    conn.commit()
    conn.close()
    return True


init_db()

# =====================================================
# MODEL
# =====================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


@st.cache_data(show_spinner=False)
def encode_texts(texts):
    return model.encode(texts, batch_size=16, show_progress_bar=False)


# =====================================================
# PDF EXTRACTION
# =====================================================
@st.cache_data(show_spinner=False)
def extract_text_cached(file_name, file_size, file_bytes):
    if PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if doc.is_encrypted:
                doc.authenticate("")
            text = "\n".join(page.get_text("text") for page in doc).strip()
            if len(text) > 80:
                return clean_text(text), "PyMuPDF"
        except Exception:
            pass

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if text:
                    pages.append(text)
            text = "\n".join(pages).strip()
            if len(text) > 80:
                return clean_text(text), "pdfplumber"
    except Exception:
        pass

    if PYMUPDF_AVAILABLE and OCR_AVAILABLE:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            parts = []
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                parts.append(pytesseract.image_to_string(img, config="--psm 6"))
            text = "\n".join(parts).strip()
            if len(text) > 80:
                return clean_text(text), "OCR"
        except Exception:
            pass

    return "", "Failed"


def extract_text(file):
    file_bytes = file.read()
    file.seek(0)
    return extract_text_cached(file.name, len(file_bytes), file_bytes)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


# =====================================================
# RESUME PARSING
# =====================================================
def extract_candidate_info(text):
    email_match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    email = email_match.group(0) if email_match else "Not found"

    phone_match = re.search(
        r"(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?(\d{3,4}[\s\-]?\d{3,4})",
        text,
    )
    phone = phone_match.group(0).strip() if phone_match else "Not found"

    name = "Unknown"
    possible_lines = re.split(r"[\n\|•]", text[:600])
    for line in possible_lines:
        line = line.strip()
        if 2 <= len(line.split()) <= 4 and len(line) < 60:
            if not re.search(r"@|http|www|github|linkedin|\d", line.lower()):
                name = line
                break

    return {"name": name, "email": email, "phone": phone}


def extract_years_experience(text):
    text_lower = text.lower()
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s+years?\s+of\s+experience",
        r"(\d+(?:\.\d+)?)\+?\s+years?\s+experience",
        r"experience\s+of\s+(\d+(?:\.\d+)?)\+?\s+years?",
        r"(\d+(?:\.\d+)?)\+?\s+yrs?\s+experience",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return float(match.group(1))

    years = re.findall(r"\b(20[0-2][0-9]|19[89][0-9])\b", text)
    years = sorted([int(y) for y in years])
    if len(years) >= 2:
        return float(min(years[-1] - years[0], 25))
    return 0.0


def extract_known_skills(text):
    text_lower = text.lower()
    found = []
    for skill in DEFAULT_SKILLS:
        if re.search(rf"\b{re.escape(skill)}\b", text_lower):
            found.append(skill)
    return found


# =====================================================
# SCORING
# =====================================================
def parse_skills(skills_input):
    return [s.strip().lower() for s in skills_input.split(",") if s.strip()]


def parse_skill_weights(raw_text):
    weights = {}
    for line in raw_text.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            try:
                weights[key.strip().lower()] = float(value.strip())
            except ValueError:
                continue
    return weights


def skill_match_score(resume_text, skills_input, weights):
    required_skills = parse_skills(skills_input)
    if not required_skills:
        return 100.0, [], []

    resume_lower = resume_text.lower()
    found = []
    missing = []
    weighted_score = 0.0
    total_weight = 0.0

    for skill in required_skills:
        weight = weights.get(skill, 1.0)
        total_weight += weight

        exact_match = re.search(rf"\b{re.escape(skill)}\b", resume_lower)
        soft_match = skill in resume_lower

        if exact_match:
            found.append(skill)
            weighted_score += weight
        elif soft_match:
            found.append(skill + " (partial)")
            weighted_score += weight * 0.7
        else:
            missing.append(skill)

    score = (weighted_score / total_weight) * 100 if total_weight else 0
    return round(score, 2), found, missing


def experience_score(years, required_years):
    if required_years <= 0:
        return 100.0
    return round(min(years / required_years, 1.0) * 100, 2)


def hybrid_score(semantic, skill, exp, w_semantic, w_skill, w_exp):
    total_weight = w_semantic + w_skill + w_exp
    if total_weight == 0:
        return 0.0
    return round((semantic * w_semantic + skill * w_skill + exp * w_exp) / total_weight, 2)


def recommendation_label(final_score, skill_score, semantic_score):
    if final_score >= 80 and skill_score >= 70:
        return "Strong Match"
    if final_score >= 65:
        return "Good Match"
    if semantic_score >= 70 and skill_score < 50:
        return "Relevant Background, Missing Key Skills"
    if final_score >= 50:
        return "Possible Match"
    return "Weak Match"


def generate_candidate_summary(name, final_score, semantic_score, skill_score, exp_score, found, missing, years):
    strengths = []
    concerns = []

    if semantic_score >= 75:
        strengths.append("strong job-description alignment")
    elif semantic_score < 50:
        concerns.append("low semantic similarity with the job description")

    if skill_score >= 75:
        strengths.append("good required-skill coverage")
    elif skill_score < 50:
        concerns.append("missing several required skills")

    if exp_score >= 75:
        strengths.append("experience level appears suitable")
    elif exp_score < 50:
        concerns.append("experience may be below requirement")

    if not strengths:
        strengths.append("some relevant profile signals")
    if not concerns:
        concerns.append("no major weakness detected from extracted text")

    missing_text = ", ".join(missing[:5]) if missing else "none"
    found_text = ", ".join(found[:5]) if found else "none"

    return (
        f"Score: {final_score}%. Strengths: {', '.join(strengths)}. "
        f"Concerns: {', '.join(concerns)}. Years detected: {years}. "
        f"Top skills found: {found_text}. Missing skills: {missing_text}."
    )


# =====================================================
# UI HELPERS
# =====================================================
def score_class(score):
    if score >= 75:
        return "score-high"
    if score >= 50:
        return "score-mid"
    return "score-low"


def get_score_color(score):
    if score >= 75:
        return "#059669"  # Emerald
    if score >= 50:
        return "#D97706"  # Amber
    return "#DC2626"  # Red


def get_score_gradient(score):
    if score >= 75:
        return "linear-gradient(90deg, #10B981 0%, #34D399 100%)"
    if score >= 50:
        return "linear-gradient(90deg, #F59E0B 0%, #FBBF24 100%)"
    return "linear-gradient(90deg, #EF4444 0%, #F87171 100%)"


def render_scores_grid(final, semantic, skill, exp):
    st.markdown(
        f"""
        <div class="scores-grid">
            <div class="score-progress-container">
                <div class="score-progress-header">
                    <span>Overall Score</span>
                    <span style="color: {get_score_color(final)}; font-weight: 700;">{final}%</span>
                </div>
                <div class="score-progress-bar">
                    <div class="score-progress-fill" style="width: {final}%; background: {get_score_gradient(final)};"></div>
                </div>
            </div>
            <div class="score-progress-container">
                <div class="score-progress-header">
                    <span>Semantic Similarity</span>
                    <span style="color: {get_score_color(semantic)}; font-weight: 700;">{semantic}%</span>
                </div>
                <div class="score-progress-bar">
                    <div class="score-progress-fill" style="width: {semantic}%; background: {get_score_gradient(semantic)};"></div>
                </div>
            </div>
            <div class="score-progress-container">
                <div class="score-progress-header">
                    <span>Skill Match</span>
                    <span style="color: {get_score_color(skill)}; font-weight: 700;">{skill}%</span>
                </div>
                <div class="score-progress-bar">
                    <div class="score-progress-fill" style="width: {skill}%; background: {get_score_gradient(skill)};"></div>
                </div>
            </div>
            <div class="score-progress-container">
                <div class="score-progress-header">
                    <span>Experience Score</span>
                    <span style="color: {get_score_color(exp)}; font-weight: 700;">{exp}%</span>
                </div>
                <div class="score-progress-bar">
                    <div class="score-progress-fill" style="width: {exp}%; background: {get_score_gradient(exp)};"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title, value, subtitle="", border_color="#4F46E5"):
    st.markdown(
        f"""
        <div class="metric-card" style="border-top: 4px solid {border_color};">
            <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">{title}</div>
            <div style="font-size: 2.1rem; font-weight: 800; color: #0F172A; line-height: 1.2;">{value}</div>
            {f'<div style="font-size: 0.75rem; color: #94A3B8; margin-top: 6px; font-weight: 500;">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_download_excel(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Candidates")
    return buffer.getvalue()


def show_pricing_and_payment_flow():
    st.markdown("### 💎 Unlock Full Access")
    st.write("You have used your 10 free screenings. Choose a subscription pass to keep screening resumes.")

    # Pricing plans cards
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.markdown(
            f"""
            <div style="border: 1px solid #E2E8F0; padding: 20px; border-radius: 16px; text-align: center; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.01);">
                <div style="font-size: 1.1rem; font-weight: 700; color: #4F46E5;">7-Day Pass</div>
                <div style="font-size: 1.8rem; font-weight: 800; margin: 10px 0; color: #0F172A;">500 PKR</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 12px;">Full access for 1 week</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with p_col2:
        st.markdown(
            f"""
            <div style="border: 2px solid #3B82F6; padding: 20px; border-radius: 16px; text-align: center; background-color: white; box-shadow: 0 6px 12px rgba(59,130,246,0.08); position: relative;">
                <div style="position: absolute; top: -12px; left: 50%; transform: translateX(-50%); background-color: #3B82F6; color: white; padding: 2px 10px; border-radius: 20px; font-size: 9px; font-weight: 700; text-transform: uppercase;">Best Value</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #3B82F6;">15-Day Pass</div>
                <div style="font-size: 1.8rem; font-weight: 800; margin: 10px 0; color: #0F172A;">900 PKR</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 12px;">Full access for 15 days</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with p_col3:
        st.markdown(
            f"""
            <div style="border: 1px solid #E2E8F0; padding: 20px; border-radius: 16px; text-align: center; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.01);">
                <div style="font-size: 1.1rem; font-weight: 700; color: #8B5CF6;">30-Day Pass</div>
                <div style="font-size: 1.8rem; font-weight: 800; margin: 10px 0; color: #0F172A;">1500 PKR</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 12px;">Full access for 1 month</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr style='margin: 24px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
    
    st.markdown("### 💸 Payment Instructions")
    st.markdown(
        f"""
        Please transfer the amount for your chosen plan to our Easypaisa mobile wallet:
        - **Easypaisa Wallet:** {PAYMENT_INSTRUCTIONS["easypaisa_number"]}
        - **Account Title:** {PAYMENT_INSTRUCTIONS["account_title"]}
        """
    )
    
    st.markdown("**Need another payment method?**")
    st.write("If you wish to pay via Bank Transfer or any other payment method, please contact us directly:")
    st.link_button("💬 Chat on WhatsApp", "https://wa.me/923118036997?text=Hi%2C%20I%20would%20like%20to%20buy%20a%20subscription%20plan%20for%20AI%20Resume%20Ranker%20ATS.", use_container_width=True)
    
    st.divider()
    st.markdown("### 📝 Submit Payment Proof")
    
    pay_col1, pay_col2 = st.columns(2)
    with pay_col1:
        plan_sel = st.selectbox("Select Subscription Plan", ["7-Day Pass (500 PKR)", "15-Day Pass (900 PKR)", "30-Day Pass (1500 PKR)"])
        st.text_input("Transfer Method", value="Easypaisa", disabled=True)
    with pay_col2:
        tx_id = st.text_input("Transaction ID (TxID)", placeholder="e.g. 12345678901")
        sender_num = st.text_input("Sender Mobile Number", placeholder="e.g. 03001234567")
        
    if st.button("Submit Payment Proof", use_container_width=True):
        if not tx_id.strip() or not sender_num.strip():
            st.error("Please fill in all transaction proof fields.")
        else:
            plan_days = 7 if "7-Day" in plan_sel else 15 if "15-Day" in plan_sel else 30
            amount = PLANS[plan_days]["price"]
            
            submit_payment_request(
                st.session_state["user_id"],
                plan_days,
                amount,
                "easypaisa",
                tx_id.strip(),
                sender_num.strip()
            )
            st.success("Payment submitted successfully! We will verify it and activate your plan shortly.")


# =====================================================
# AUTHENTICATION GATE
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["user_id"] = None
    st.session_state["user_email"] = None
    st.session_state["is_admin"] = False

# If not logged in, show Auth Screen and stop
if not st.session_state["logged_in"]:
    # Simplified sidebar for unauthenticated users
    with st.sidebar:
        st.title("🔒 Access Restricted")
        st.write("Please sign in or create an account to start screening resumes.")
        st.caption("AI-powered ATS Resume Ranker & screening workflow platform.")

    st.markdown('<div class="main-title">AI Resume Ranker ATS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Screen resumes using transformer embeddings, skill matching, experience scoring, and ATS pipeline tracking.</div>',
        unsafe_allow_html=True,
    )
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1.6, 1])
    with col2:
        st.markdown(
            """
            <div style="background-color: white; padding: 24px; border-radius: 16px; border: 1px solid #E2E8F0; box-shadow: 0px 4px 20px rgba(0,0,0,0.03); margin-bottom: 20px;">
                <h3 style="margin: 0 0 8px 0; text-align: center; color: #0F172A; font-weight: 800; font-size: 1.4rem;">🔑 Welcome Back</h3>
                <p style="margin: 0; text-align: center; color: #64748B; font-size: 0.9rem;">Log in to access your dashboard, pricing plans, and screenings.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        auth_tab_login, auth_tab_reg = st.tabs(["🚀 Login", "✨ Register"])
        
        with auth_tab_login:
            st.write("")
            email_in = st.text_input("Email Address", key="login_email", placeholder="e.g. user@example.com")
            pass_in = st.text_input("Password", type="password", key="login_pass", placeholder="••••••••")
            if st.button("Sign In to Workspace", key="login_btn", use_container_width=True):
                if not email_in.strip() or not pass_in.strip():
                    st.error("Please enter both email and password.")
                else:
                    user = verify_user(email_in, pass_in)
                    if user:
                        st.session_state["logged_in"] = True
                        st.session_state["user_id"] = user["id"]
                        st.session_state["user_email"] = user["email"]
                        st.session_state["is_admin"] = user["is_admin"]
                        st.success("Successfully logged in!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
                        
        with auth_tab_reg:
            st.write("")
            email_reg = st.text_input("Email Address", key="reg_email", placeholder="e.g. user@example.com")
            pass_reg = st.text_input("Password", type="password", key="reg_pass", placeholder="••••••••")
            pass_confirm = st.text_input("Confirm Password", type="password", key="reg_pass_confirm", placeholder="••••••••")
            if st.button("Create Account & Start", key="reg_btn", use_container_width=True):
                if not email_reg.strip() or not pass_reg.strip() or not pass_confirm.strip():
                    st.error("Please fill in all fields.")
                elif pass_reg != pass_confirm:
                    st.error("Passwords do not match.")
                elif len(pass_reg) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    user_exist = get_user_by_email(email_reg)
                    if user_exist:
                        st.error("An account with that email already exists.")
                    else:
                        user_id = create_user(email_reg, pass_reg)
                        if user_id:
                            st.session_state["logged_in"] = True
                            st.session_state["user_id"] = user_id
                            st.session_state["user_email"] = email_reg.lower().strip()
                            st.session_state["is_admin"] = False
                            st.success("Account created successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to create account.")
    st.stop()


# =====================================================
# SIDEBAR (Authenticated Users)
# =====================================================
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.markdown(
        f"""
        <div style="background-color: #1E293B; padding: 14px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 15px;">
            <div style="font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px;">Logged in as</div>
            <div style="font-size: 13px; font-weight: 600; color: #F8FAFC; word-break: break-all; margin-bottom: 8px;">{st.session_state["user_email"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    has_access, access_msg = check_user_access(st.session_state["user_id"])
    if st.session_state["is_admin"]:
        st.success("👑 Admin Account")
    elif "Paid Plan Active" in access_msg:
        st.success(f"⭐ {access_msg}")
    else:
        st.warning(f"⏳ {access_msg}")
        
    if st.button("🚪 Logout", key="logout_btn"):
        st.session_state.clear()
        st.rerun()
        
    st.divider()
    st.caption("Adjust ranking weights based on hiring priorities.")

    w_semantic = st.slider("Semantic similarity", 0, 100, 40, 5)
    w_skill = st.slider("Skill match", 0, 100, 40, 5)
    w_exp = st.slider("Experience", 0, 100, 20, 5)

    st.divider()
    req_years = st.number_input("Required years of experience", min_value=0.0, value=2.0, step=0.5)

    st.divider()
    st.subheader("PDF Support")
    st.caption(f"{'✅' if PYMUPDF_AVAILABLE else '❌'} PyMuPDF")
    st.caption("✅ pdfplumber")
    st.caption(f"{'✅' if OCR_AVAILABLE else '❌'} OCR / Tesseract")

    st.divider()
    st.info("Tip: For best results, paste a detailed job description and comma-separated skills.")


# =====================================================
# MAIN HEADER
# =====================================================
st.markdown('<div class="main-title">AI Resume Ranker ATS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Screen resumes using transformer embeddings, skill matching, experience scoring, and ATS pipeline tracking.</div>',
    unsafe_allow_html=True,
)

jobs_df = get_jobs(st.session_state["user_id"])

tabs_list = [
    "🎯 Screen & Rank",
    "📊 Dashboard",
    "🧩 ATS Pipeline",
    "📝 Audit Log",
]
if st.session_state["is_admin"]:
    tabs_list.append("👑 Admin Panel")

tabs = st.tabs(tabs_list)

tab_screen = tabs[0]
tab_dashboard = tabs[1]
tab_ats = tabs[2]
tab_audit = tabs[3]
if st.session_state["is_admin"]:
    tab_admin = tabs[4]

# =====================================================
# TAB 1: SCREEN AND RANK
# =====================================================
with tab_screen:
    has_access, access_msg = check_user_access(st.session_state["user_id"])
    if not has_access:
        st.warning(f"🔒 {access_msg}")
        show_pricing_and_payment_flow()
    else:
        left, right = st.columns([1.1, 0.9])

        with left:
            st.subheader("Job Information")
            job_title = st.text_input("Job title", placeholder="Example: Junior Machine Learning Engineer")
            jd_text = st.text_area("Job description", height=220, placeholder="Paste the complete job description here...")
            required_skills_input = st.text_input(
                "Required skills",
                placeholder="Python, Machine Learning, SQL, Pandas, Scikit-learn",
            )
            skill_weights_raw = st.text_area(
                "Optional skill weights",
                placeholder="Python: 3\nMachine Learning: 3\nSQL: 2",
                height=100,
            )

        with right:
            st.subheader("Upload Resumes")
            uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

            if uploaded_files:
                st.success(f"{len(uploaded_files)} resume(s) uploaded")
                with st.expander("Uploaded files"):
                    for file in uploaded_files:
                        st.write(f"📄 {file.name}")

            st.markdown("### Ranking Formula")
            st.write("Final score = semantic similarity + skill match + experience score")
            st.caption("Weights are controlled from the sidebar.")

        skill_weights = parse_skill_weights(skill_weights_raw)

        rank_button = st.button("🚀 Rank Candidates", type="primary", use_container_width=True)

        if rank_button:
            if not job_title.strip():
                st.warning("Please enter a job title.")
            elif not jd_text.strip():
                st.warning("Please paste a job description.")
            elif not uploaded_files:
                st.warning("Please upload at least one PDF resume.")
            else:
                # Check remaining quota
                user = get_user_by_id(st.session_state["user_id"])
                is_paid = False
                if user["plan_expiry"]:
                    try:
                        exp_time = datetime.fromisoformat(user["plan_expiry"])
                        if exp_time.tzinfo is None:
                            exp_time = exp_time.replace(tzinfo=timezone.utc)
                        if exp_time > datetime.now(timezone.utc):
                            is_paid = True
                    except ValueError:
                        pass

                if not user["is_admin"] and not is_paid:
                    rem = FREE_PDF_LIMIT - user["free_pdfs_used"]
                    if len(uploaded_files) > rem:
                        st.error(f"Quota exceeded. You only have {rem} free resume checks remaining, but you uploaded {len(uploaded_files)} resumes. Please upgrade your plan.")
                        st.stop()

                progress = st.progress(0)
                status = st.empty()

                resume_texts = []
                valid_files = []
                extraction_methods = []

                for i, file in enumerate(uploaded_files):
                    status.info(f"Reading {file.name}...")
                    text, method = extract_text(file)
                    if text:
                        resume_texts.append(text)
                        valid_files.append(file)
                        extraction_methods.append(method)
                    else:
                        st.warning(f"Could not read {file.name}. It was skipped.")
                    progress.progress((i + 1) / max(len(uploaded_files), 1))

                if not resume_texts:
                    st.error("No readable resumes found.")
                    st.stop()

                status.info("Generating embeddings and ranking candidates...")

                all_texts = [jd_text] + resume_texts
                embeddings = encode_texts(all_texts)
                jd_embedding = embeddings[0:1]
                resume_embeddings = embeddings[1:]
                semantic_scores = cosine_similarity(jd_embedding, resume_embeddings)[0]

                job_id = save_job(st.session_state["user_id"], job_title, jd_text, required_skills_input, req_years)

                results = []
                for i, (file, text) in enumerate(zip(valid_files, resume_texts)):
                    info = extract_candidate_info(text)
                    years = extract_years_experience(text)
                    semantic = round(float(semantic_scores[i]) * 100, 2)
                    skill_pct, found_skills, missing_skills = skill_match_score(text, required_skills_input, skill_weights)
                    exp_pct = experience_score(years, req_years)
                    final = hybrid_score(semantic, skill_pct, exp_pct, w_semantic, w_skill, w_exp)
                    rec = recommendation_label(final, skill_pct, semantic)
                    summary = generate_candidate_summary(
                        info["name"], final, semantic, skill_pct, exp_pct, found_skills, missing_skills, years
                    )

                    results.append({
                        "candidate_id": f"J{job_id}_{uuid.uuid4().hex[:8].upper()}",
                        "name": info["name"],
                        "email": info["email"],
                        "phone": info["phone"],
                        "filename": file.name,
                        "final_score": final,
                        "semantic_score": semantic,
                        "skill_score": skill_pct,
                        "exp_score": exp_pct,
                        "years_exp": years,
                        "skills_found": ", ".join(found_skills) if found_skills else "None",
                        "skills_missing": ", ".join(missing_skills) if missing_skills else "None",
                        "auto_summary": summary,
                        "recommendation": rec,
                        "extraction_method": extraction_methods[i],
                    })

                results.sort(key=lambda x: x["final_score"], reverse=True)
                save_candidates_batch(job_id, results)

                # Increment free checks count in DB
                if not user["is_admin"] and not is_paid:
                    for _ in range(len(results)):
                        increment_user_free_count(st.session_state["user_id"])

                st.session_state["last_job_id"] = job_id
                st.session_state["last_results"] = results
                st.cache_data.clear()
                progress.empty()
                status.empty()
                st.success(f"Ranked {len(results)} candidate(s) for {job_title}.")

        if "last_results" in st.session_state:
            results = st.session_state["last_results"]
            st.divider()

            blind_mode = st.toggle("Bias-blind mode", help="Hide name, email, and phone during review.")

            top_score = max(r["final_score"] for r in results)
            avg_score = round(sum(r["final_score"] for r in results) / len(results), 2)
            strong_matches = sum(1 for r in results if r["final_score"] >= 75)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("Candidates", len(results), "total screened", border_color="#3B82F6")
            with c2:
                render_metric_card("Top Score", f"{top_score}%", "best match", border_color="#10B981")
            with c3:
                render_metric_card("Average Score", f"{avg_score}%", "overall quality", border_color="#06B6D4")
            with c4:
                render_metric_card("Strong Matches", strong_matches, "score ≥ 75%", border_color="#8B5CF6")

            st.subheader("Ranked Candidates")

            rows = []
            for idx, r in enumerate(results, start=1):
                rows.append({
                    "Rank": idx,
                    "Candidate ID": f"#{r['candidate_id'][-8:]}",
                    "Name": "Hidden" if blind_mode else r["name"],
                    "Email": "Hidden" if blind_mode else r["email"],
                    "Final Score": r["final_score"],
                    "Semantic": r["semantic_score"],
                    "Skill Match": r["skill_score"],
                    "Experience": r["exp_score"],
                    "Years": r["years_exp"],
                    "Recommendation": r["recommendation"],
                    "PDF Method": r["extraction_method"],
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.subheader("Candidate Details")
            for idx, r in enumerate(results, start=1):
                display_name = f"Candidate #{r['candidate_id'][-8:]}" if blind_mode else r["name"]
                medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"#{idx}"

                with st.expander(f"{medal} {display_name} — {r['final_score']}% — {r['recommendation']}"):
                    # Render beautiful custom score grid with progress bars
                    render_scores_grid(r['final_score'], r['semantic_score'], r['skill_score'], r['exp_score'])

                    # Details split into columns
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("<p style='font-size: 0.95rem; font-weight: 700; color: #1E293B; margin-bottom: 8px;'>👤 Contact & Profile Info</p>", unsafe_allow_html=True)
                        if not blind_mode:
                            st.markdown(f"**Email:** `{r['email']}`")
                            st.markdown(f"**Phone:** `{r['phone']}`")
                        st.markdown(f"**Filename:** `{r['filename']}`")
                        st.markdown(f"**Years detected:** `{r['years_exp']}`")

                    with col2:
                        st.markdown("<p style='font-size: 0.95rem; font-weight: 700; color: #1E293B; margin-bottom: 8px;'>🛠️ Required Skill Coverage</p>", unsafe_allow_html=True)
                        
                        # Found skills badges
                        found_badges = ""
                        if r['skills_found'] and r['skills_found'] != "None":
                            for sk in r['skills_found'].split(","):
                                sk = sk.strip()
                                if "partial" in sk.lower():
                                    found_badges += f'<span class="badge badge-missing" style="margin: 2px 4px 2px 0;">{sk}</span>'
                                else:
                                    found_badges += f'<span class="badge badge-strong" style="margin: 2px 4px 2px 0;">{sk}</span>'
                        else:
                            found_badges = '<span class="badge badge-possible">None</span>'
                        
                        # Missing skills badges
                        missing_badges = ""
                        if r['skills_missing'] and r['skills_missing'] != "None":
                            for sk in r['skills_missing'].split(","):
                                sk = sk.strip()
                                missing_badges += f'<span class="badge badge-weak" style="margin: 2px 4px 2px 0;">{sk}</span>'
                        else:
                            missing_badges = '<span class="badge badge-strong">None</span>'

                        st.markdown(f"<div style='margin-bottom: 8px;'><b>Skills Found:</b><br>{found_badges}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div><b>Skills Missing:</b><br>{missing_badges}</div>", unsafe_allow_html=True)

                    st.markdown("<hr style='margin: 12px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
                    st.markdown("<p style='font-size: 0.95rem; font-weight: 700; color: #1E293B; margin-bottom: 8px;'>📝 AI Candidate Summary</p>", unsafe_allow_html=True)
                    st.info(r["auto_summary"])

            export_df = pd.DataFrame(results)
            if blind_mode:
                for col in ["name", "email", "phone"]:
                    export_df[col] = "REDACTED"

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download CSV",
                    export_df.to_csv(index=False).encode("utf-8"),
                    "ranked_candidates.csv",
                    "text/csv",
                    use_container_width=True,
                )
            with d2:
                try:
                    st.download_button(
                        "Download Excel",
                        create_download_excel(export_df),
                        "ranked_candidates.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                except Exception:
                    st.caption("Install openpyxl to enable Excel export: pip install openpyxl")

# =====================================================
# TAB 2: DASHBOARD
# =====================================================
with tab_dashboard:
    st.subheader("Analytics Dashboard")

    if jobs_df.empty:
        st.info("No jobs yet. Run a screening first.")
    else:
        job_options = {f"{row['title']} (#{row['id']})": row["id"] for _, row in jobs_df.iterrows()}
        selected_job_label = st.selectbox("Select job", list(job_options.keys()), key="dashboard_job")
        selected_job_id = job_options[selected_job_label]
        candidates = get_candidates(st.session_state["user_id"], selected_job_id)

        if candidates.empty:
            st.warning("No candidates found for this job.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("Candidates", len(candidates), border_color="#3B82F6")
            with c2:
                render_metric_card("Average Score", f"{round(candidates['final_score'].mean(), 2)}%", border_color="#06B6D4")
            with c3:
                render_metric_card("Top Score", f"{round(candidates['final_score'].max(), 2)}%", border_color="#10B981")
            with c4:
                render_metric_card("Strong Matches", int((candidates["final_score"] >= 75).sum()), border_color="#8B5CF6")

            st.divider()

            if PLOTLY_AVAILABLE:
                # Sleek bar chart with red-orange-emerald gradient based on scores
                score_fig = px.bar(
                    candidates,
                    x="name",
                    y="final_score",
                    title="Candidate Final Scores",
                    labels={"name": "Candidate", "final_score": "Final Score"},
                    color="final_score",
                    color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
                )
                score_fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    title_font=dict(size=16, family="Plus Jakarta Sans, sans-serif", color="#0F172A", weight="bold"),
                    font=dict(family="Plus Jakarta Sans, sans-serif", color="#64748B"),
                    margin=dict(t=50, b=30, l=40, r=40),
                )
                score_fig.update_yaxes(gridcolor="#E2E8F0")
                st.plotly_chart(score_fig, use_container_width=True)

                # Modern donut chart with colors synchronized with stage colors
                stage_counts = candidates["stage"].value_counts().reset_index()
                stage_counts.columns = ["stage", "count"]
                stage_fig = px.pie(
                    stage_counts,
                    names="stage",
                    values="count",
                    title="Pipeline Stage Distribution",
                    hole=0.4,
                    color="stage",
                    color_discrete_map=STAGE_COLORS,
                )
                stage_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    title_font=dict(size=16, family="Plus Jakarta Sans, sans-serif", color="#0F172A", weight="bold"),
                    font=dict(family="Plus Jakarta Sans, sans-serif", color="#64748B"),
                    margin=dict(t=50, b=30, l=40, r=40),
                )
                st.plotly_chart(stage_fig, use_container_width=True)
            else:
                st.info("Install Plotly for charts: pip install plotly")

            all_found = []
            for item in candidates["skills_found"].fillna("None"):
                if item != "None":
                    all_found.extend([x.strip().replace(" (partial)", "") for x in item.split(",")])

            if all_found:
                st.subheader("Most Common Found Skills")
                skill_df = pd.DataFrame(Counter(all_found).most_common(15), columns=["Skill", "Count"])
                st.dataframe(skill_df, use_container_width=True, hide_index=True)

# =====================================================
# TAB 3: ATS PIPELINE
# =====================================================
with tab_ats:
    st.subheader("ATS Pipeline")

    if jobs_df.empty:
        st.info("No jobs yet. Run a screening first.")
    else:
        job_options = {f"{row['title']} (#{row['id']})": row["id"] for _, row in jobs_df.iterrows()}
        selected_job_label = st.selectbox("Select job", list(job_options.keys()), key="ats_job")
        selected_job_id = job_options[selected_job_label]
        candidates = get_candidates(st.session_state["user_id"], selected_job_id)

        if candidates.empty:
            st.warning("No candidates found.")
        else:
            stage_counts = candidates["stage"].value_counts()
            cols = st.columns(len(ATS_STAGES))
            for i, stage in enumerate(ATS_STAGES):
                count = stage_counts.get(stage, 0)
                color = STAGE_COLORS.get(stage, "#6B7280")
                cols[i].markdown(
                    f"""
                    <div class="pipeline-stage-box" style="border-top: 4px solid {color};">
                        <div style="font-size: 26px; font-weight: 800; color: {color}; line-height: 1.2;">{count}</div>
                        <div style="font-size: 10px; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">{stage}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.divider()
            stage_filter = st.multiselect("Filter by stage", ATS_STAGES, default=ATS_STAGES)
            filtered = candidates[candidates["stage"].isin(stage_filter)]
            blind_ats = st.toggle("Bias-blind mode", key="blind_ats")

            for _, row in filtered.iterrows():
                cid = row["candidate_id"]
                display_name = f"Candidate #{cid[-8:]}" if blind_ats else row.get("name", "Unknown")
                score = row.get("final_score", 0)
                current_stage = row.get("stage", "New")

                with st.expander(f"{display_name} — {score}% — {current_stage}"):
                    # Beautiful custom score grid
                    render_scores_grid(
                        row.get("final_score", 0),
                        row.get("semantic_score", 0),
                        row.get("skill_score", 0),
                        row.get("exp_score", 0)
                    )

                    # Split content into info and skills columns
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("<p style='font-size: 0.95rem; font-weight: 700; color: #1E293B; margin-bottom: 8px;'>👤 Contact & Profile Info</p>", unsafe_allow_html=True)
                        if not blind_ats:
                            st.markdown(f"**Email:** `{row.get('email', 'N/A')}`")
                            st.markdown(f"**Phone:** `{row.get('phone', 'N/A')}`")
                        st.markdown(f"**Filename:** `{row.get('filename', 'N/A')}`")
                        st.markdown(f"**Years detected:** `{row.get('years_exp', 0.0)}`")
                        st.markdown(f"**Recommendation:** **{row.get('recommendation', 'N/A')}**")

                    with col2:
                        st.markdown("<p style='font-size: 0.95rem; font-weight: 700; color: #1E293B; margin-bottom: 8px;'>🛠️ Required Skill Coverage</p>", unsafe_allow_html=True)
                        
                        # Found skills badges
                        skills_found_str = row.get('skills_found', 'None')
                        found_badges = ""
                        if skills_found_str and skills_found_str != "None":
                            for sk in skills_found_str.split(","):
                                sk = sk.strip()
                                if "partial" in sk.lower():
                                    found_badges += f'<span class="badge badge-missing" style="margin: 2px 4px 2px 0;">{sk}</span>'
                                else:
                                    found_badges += f'<span class="badge badge-strong" style="margin: 2px 4px 2px 0;">{sk}</span>'
                        else:
                            found_badges = '<span class="badge badge-possible">None</span>'
                        
                        # Missing skills badges
                        skills_missing_str = row.get('skills_missing', 'None')
                        missing_badges = ""
                        if skills_missing_str and skills_missing_str != "None":
                            for sk in skills_missing_str.split(","):
                                sk = sk.strip()
                                missing_badges += f'<span class="badge badge-weak" style="margin: 2px 4px 2px 0;">{sk}</span>'
                        else:
                            missing_badges = '<span class="badge badge-strong">None</span>'

                        st.markdown(f"<div style='margin-bottom: 8px;'><b>Skills Found:</b><br>{found_badges}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div><b>Skills Missing:</b><br>{missing_badges}</div>", unsafe_allow_html=True)

                    st.markdown("<hr style='margin: 12px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
                    st.markdown("<p style='font-size: 0.95rem; font-weight: 700; color: #1E293B; margin-bottom: 8px;'>📝 AI Candidate Summary</p>", unsafe_allow_html=True)
                    st.info(row.get("auto_summary", "No summary available."))

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        try:
                            current_index = ATS_STAGES.index(current_stage)
                        except ValueError:
                            current_index = 0
                        new_stage = st.selectbox(
                            "Move to stage",
                            ATS_STAGES,
                            index=current_index,
                            key=f"stage_{cid}_{selected_job_id}",
                        )
                    with col2:
                        st.write(" ")
                        st.write(" ")
                        if new_stage != current_stage:
                            if st.button("🔄 Confirm Move", key=f"confirm_{cid}_{selected_job_id}"):
                                update_stage(st.session_state["user_id"], cid, selected_job_id, new_stage, current_stage)
                                st.cache_data.clear()
                                st.success(f"Moved to {new_stage}")
                                st.rerun()

                    notes = st.text_area(
                        "Reviewer notes",
                        value=row.get("notes", "") or "",
                        key=f"notes_{cid}_{selected_job_id}",
                        height=100,
                    )
                    if st.button("💾 Save Notes", key=f"save_notes_{cid}_{selected_job_id}"):
                        save_notes(st.session_state["user_id"], cid, selected_job_id, notes)
                        st.cache_data.clear()
                        st.success("Notes saved.")

            st.download_button(
                "Export Pipeline CSV",
                filtered.to_csv(index=False).encode("utf-8"),
                f"pipeline_job_{selected_job_id}.csv",
                "text/csv",
                use_container_width=True,
            )

# =====================================================
# TAB 4: AUDIT LOG
# =====================================================
with tab_audit:
    st.subheader("Audit Log")

    if jobs_df.empty:
        st.info("No jobs available.")
    else:
        job_options = {f"{row['title']} (#{row['id']})": row["id"] for _, row in jobs_df.iterrows()}
        selected_job_label = st.selectbox("Select job", list(job_options.keys()), key="audit_job")
        selected_job_id = job_options[selected_job_label]
        audit_df = get_audit_log(st.session_state["user_id"], selected_job_id)

        if audit_df.empty:
            st.info("No audit history yet.")
        else:
            audit_df["Candidate"] = audit_df["candidate_id"].apply(lambda x: f"#{str(x)[-8:]}")
            audit_df["timestamp"] = pd.to_datetime(audit_df["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M UTC")
            display_cols = ["timestamp", "Candidate", "action", "old_value", "new_value"]
            st.dataframe(audit_df[display_cols], use_container_width=True, hide_index=True)


# =====================================================
# TAB 5: ADMIN PANEL
# =====================================================
if st.session_state["is_admin"]:
    with tab_admin:
        st.subheader("👑 Pending Payment Requests")
        st.write("Review transaction proofs from users claiming JazzCash/Easypaisa payments.")
        
        pending_df = get_pending_payment_requests()
        if pending_df.empty:
            st.info("No pending payment requests to verify.")
        else:
            for _, row in pending_df.iterrows():
                req_id = row["id"]
                user_email = row["email"]
                plan_days = row["plan_days"]
                amount = row["amount"]
                method = row["method"].upper()
                tx_id = row["transaction_id"]
                sender = row["sender_number"]
                created = row["created_at"]
                
                with st.container(border=True):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**User:** `{user_email}`")
                        st.markdown(f"**Plan Tier:** `{plan_days} Days` | **Expected Amount:** `{amount} PKR`")
                        st.markdown(f"**Method:** `{method}` | **TxID:** `{tx_id}` | **Sender:** `{sender}`")
                        st.caption(f"Submitted on: {created}")
                        
                    with col2:
                        st.write("")
                        if st.button("✅ Approve Access", key=f"approve_{req_id}", use_container_width=True):
                            if approve_payment_request(req_id):
                                st.success("Access approved and subscription plan activated!")
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.error("Failed to approve payment request.")
                                
                        reject_note = st.text_input("Optional Reject Reason", key=f"note_{req_id}", placeholder="e.g. Invalid TxID")
                        if st.button("❌ Reject Request", key=f"reject_{req_id}", use_container_width=True):
                            if reject_payment_request(req_id, reject_note):
                                st.warning("Request rejected.")
                                st.cache_data.clear()
                                st.rerun()

        st.divider()
        st.subheader("🔐 Change Admin Password")
        with st.form("change_password_form", clear_on_submit=True):
            new_pw = st.text_input("New Password", type="password", placeholder="••••••••")
            confirm_pw = st.text_input("Confirm New Password", type="password", placeholder="••••••••")
            submit_pw = st.form_submit_button("Update Password", use_container_width=True)
            
            if submit_pw:
                if not new_pw.strip() or not confirm_pw.strip():
                    st.error("Please fill in both password fields.")
                elif new_pw != confirm_pw:
                    st.error("Passwords do not match.")
                elif len(new_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    update_user_password(st.session_state["user_id"], new_pw)
                    st.success("Password updated successfully!")


# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("Built with Streamlit, Sentence Transformers, SQLite, PyMuPDF, pdfplumber, OCR, and Scikit-learn.")
