import streamlit as st
import pdfplumber
import pandas as pd
import re
import io
import sqlite3
from datetime import datetime, timezone
import uuid
from collections import Counter

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
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0px;
    }
    .subtitle {
        color: #6B7280;
        font-size: 17px;
        margin-bottom: 25px;
    }
    .metric-card {
        background: #FFFFFF;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #E5E7EB;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.04);
        text-align: center;
    }
    .score-high {
        color: #059669;
        font-weight: 800;
    }
    .score-mid {
        color: #D97706;
        font-weight: 800;
    }
    .score-low {
        color: #DC2626;
        font-weight: 800;
    }
    .candidate-card {
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 16px;
        background: #FFFFFF;
        margin-bottom: 12px;
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

    c.execute("CREATE INDEX IF NOT EXISTS idx_candidates_job_id ON candidates(job_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_candidates_candidate_id ON candidates(candidate_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_audit_job_id ON audit_log(job_id)")

    conn.commit()
    conn.close()


def save_job(title, description, required_skills, required_years):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO jobs (title, description, required_skills, required_years, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (title, description, required_skills, required_years, datetime.now(timezone.utc).isoformat()),
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
def get_jobs():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM jobs ORDER BY created_at DESC", conn)
    conn.close()
    return df


@st.cache_data(ttl=10)
def get_candidates(job_id):
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM candidates WHERE job_id=? ORDER BY final_score DESC",
        conn,
        params=(job_id,),
    )
    conn.close()
    return df


def get_audit_log(job_id):
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM audit_log WHERE job_id=? ORDER BY timestamp DESC",
        conn,
        params=(job_id,),
    )
    conn.close()
    return df


def update_stage(candidate_id, job_id, new_stage, old_stage):
    conn = get_connection()
    c = conn.cursor()
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


def save_notes(candidate_id, job_id, notes):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "UPDATE candidates SET notes=? WHERE candidate_id=? AND job_id=?",
        (notes, candidate_id, job_id),
    )
    conn.commit()
    conn.close()


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


def render_metric_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:14px;color:#6B7280;">{title}</div>
            <div style="font-size:30px;font-weight:800;color:#111827;">{value}</div>
            <div style="font-size:12px;color:#9CA3AF;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_download_excel(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Candidates")
    return buffer.getvalue()


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.title("⚙️ Settings")
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

jobs_df = get_jobs()

tab_screen, tab_dashboard, tab_ats, tab_audit = st.tabs([
    "🎯 Screen & Rank",
    "📊 Dashboard",
    "🧩 ATS Pipeline",
    "📝 Audit Log",
])

# =====================================================
# TAB 1: SCREEN AND RANK
# =====================================================
with tab_screen:
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

            job_id = save_job(job_title, jd_text, required_skills_input, req_years)

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
            render_metric_card("Candidates", len(results), "total screened")
        with c2:
            render_metric_card("Top Score", f"{top_score}%", "best match")
        with c3:
            render_metric_card("Average Score", f"{avg_score}%", "overall quality")
        with c4:
            render_metric_card("Strong Matches", strong_matches, "score ≥ 75%")

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
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Final", f"{r['final_score']}%")
                m2.metric("Semantic", f"{r['semantic_score']}%")
                m3.metric("Skill", f"{r['skill_score']}%")
                m4.metric("Experience", f"{r['exp_score']}%")

                if not blind_mode:
                    st.write(f"Email: `{r['email']}`")
                    st.write(f"Phone: `{r['phone']}`")

                st.write(f"Filename: `{r['filename']}`")
                st.write(f"Years detected: `{r['years_exp']}`")
                st.write(f"Skills found: {r['skills_found']}")
                st.write(f"Missing skills: {r['skills_missing']}")
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
        candidates = get_candidates(selected_job_id)

        if candidates.empty:
            st.warning("No candidates found for this job.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("Candidates", len(candidates))
            with c2:
                render_metric_card("Average Score", f"{round(candidates['final_score'].mean(), 2)}%")
            with c3:
                render_metric_card("Top Score", f"{round(candidates['final_score'].max(), 2)}%")
            with c4:
                render_metric_card("Strong Matches", int((candidates["final_score"] >= 75).sum()))

            st.divider()

            if PLOTLY_AVAILABLE:
                score_fig = px.bar(
                    candidates,
                    x="name",
                    y="final_score",
                    title="Candidate Final Scores",
                    labels={"name": "Candidate", "final_score": "Final Score"},
                )
                st.plotly_chart(score_fig, use_container_width=True)

                stage_counts = candidates["stage"].value_counts().reset_index()
                stage_counts.columns = ["stage", "count"]
                stage_fig = px.pie(stage_counts, names="stage", values="count", title="Pipeline Stage Distribution")
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
        candidates = get_candidates(selected_job_id)

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
                    <div style="text-align:center;border:1px solid {color};border-radius:14px;padding:12px;">
                        <div style="font-size:24px;font-weight:800;color:{color};">{count}</div>
                        <div style="font-size:12px;color:{color};">{stage}</div>
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
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Final", f"{score}%")
                    m2.metric("Semantic", f"{row.get('semantic_score', 0)}%")
                    m3.metric("Skill", f"{row.get('skill_score', 0)}%")
                    m4.metric("Experience", f"{row.get('exp_score', 0)}%")

                    if not blind_ats:
                        st.write(f"Email: `{row.get('email', 'N/A')}`")
                        st.write(f"Phone: `{row.get('phone', 'N/A')}`")

                    st.write(f"Recommendation: **{row.get('recommendation', 'N/A')}**")
                    st.write(f"Skills found: {row.get('skills_found', 'None')}")
                    st.write(f"Missing skills: {row.get('skills_missing', 'None')}")
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
                            if st.button("Confirm", key=f"confirm_{cid}_{selected_job_id}"):
                                update_stage(cid, selected_job_id, new_stage, current_stage)
                                st.cache_data.clear()
                                st.success(f"Moved to {new_stage}")
                                st.rerun()

                    notes = st.text_area(
                        "Reviewer notes",
                        value=row.get("notes", "") or "",
                        key=f"notes_{cid}_{selected_job_id}",
                        height=100,
                    )
                    if st.button("Save Notes", key=f"save_notes_{cid}_{selected_job_id}"):
                        save_notes(cid, selected_job_id, notes)
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
        audit_df = get_audit_log(selected_job_id)

        if audit_df.empty:
            st.info("No audit history yet.")
        else:
            audit_df["Candidate"] = audit_df["candidate_id"].apply(lambda x: f"#{str(x)[-8:]}")
            audit_df["timestamp"] = pd.to_datetime(audit_df["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M UTC")
            display_cols = ["timestamp", "Candidate", "action", "old_value", "new_value"]
            st.dataframe(audit_df[display_cols], use_container_width=True, hide_index=True)

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("Built with Streamlit, Sentence Transformers, SQLite, PyMuPDF, pdfplumber, OCR, and Scikit-learn.")
