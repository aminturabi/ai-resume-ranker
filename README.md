# AI Resume Ranker & ATS 🎯

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://airesume-ranker.streamlit.app/)

An AI-powered Applicant Tracking System (ATS) designed to automate, parse, score, and rank candidate resumes using NLP models, custom skill weightings, and experience matching. It also features a fully-fledged recruitment pipeline dashboard to manage candidates through stages (from Screening to Offered/Hired).

> 🚀 **Live Demo:** The application is live and accessible at [https://airesume-ranker.streamlit.app/](https://airesume-ranker.streamlit.app/)

---

## ✨ Features

* **Professional UI Redesign**: Modern, polished layout with Google Fonts (`Plus Jakarta Sans`), sleek settings sidebar, gradient buttons, custom metric cards, visual grids, and custom donut charts.
* **User Authentication**: Secure user registration and login forms with password hashing (`werkzeug.security`) to protect access to the dashboard.
* **SaaS Subscription Paywall**: Free tier users are capped at **10 resume screenings**. Gating checks are performed automatically. Upon reaching the limit, the dashboard locks and prompts users to upgrade.
* **Mobile Payments Integration**: Visual selection cards for 7-Day, 15-Day, and 30-Day passes. Simple payment instructions for Easypaisa transfers (with a WhatsApp contact option for other payment channels) and transaction proof submissions (TxID and sender number).
* **👑 Admin Verification Board**: Administrative panel (visible only to admins) to approve or reject pending subscription proofs, and a secure password change form.
* **Multi-Method PDF Text Extraction**: Uses PyMuPDF and pdfplumber, with a fallback to **Tesseract OCR** for scanned/image-based PDF resumes.
* **Semantic Similarity Matching**: Uses Sentence Transformers (`all-MiniLM-L6-v2`) and Cosine Similarity to compare resume contents against the job description.
* **Weighted Skill Matching**: Input required skills and custom weights (e.g. Python: 3, SQL: 2) to score candidate expertise matches.
* **Experience Matching**: Automatically detects candidates' years of experience and scores them against the position's requirements.
* **Bias-Blind Review Mode**: Toggle to hide candidate name, email, and phone number to prevent hiring bias during review.
* **ATS Pipeline & Stages**: Move candidates through pipeline columns (`New`, `Screening`, `Shortlisted`, `Interview`, `Offer`, `Hired`, `Rejected`), write reviewer notes, and keep an active audit log.
* **Reports Export**: Export ranked lists and pipeline stages to `.csv` or `.xlsx` files.

---

## 🛠️ Technology Stack

* **Frontend**: [Streamlit](https://streamlit.io/) (with modern custom CSS styling and responsive layout).
* **Security & Auth**: `werkzeug` (for secure password hashing and verification).
* **NLP / AI Models**: [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`), [Scikit-Learn](https://scikit-learn.org/) (`cosine_similarity`).
* **PDF Extractors**: `pdfplumber`, `PyMuPDF (fitz)`, `pytesseract (Tesseract OCR)`, and `Pillow`.
* **Database & Tables**: `SQLite3` (relational database with automatic migration and admin bootstrapping).
* **Data Processing & Export**: `pandas` and `openpyxl`.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8 or higher.
* (Optional) [Tesseract OCR engine](https://github.com/UB-Mannheim/tesseract/wiki) installed on your system to support scanned resumes.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aminturabi/ai-resume-ranker.git
   cd ai-resume-ranker
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Open your web browser to `http://localhost:8501`.
   * **Default Admin Account**: Log in using `aminturabi594@gmail.com` and password `changeme123`.

---

## 📁 Database Schema

The SQLite database (`ats_database.db`) is automatically initialized and migrates itself on launch:
* **`users`**: Stores user email addresses, hashed passwords, admin flags, free screenings usage count, and active subscription plan durations/expiries.
* **`payment_requests`**: Stores pending, approved, and rejected mobile wallet transaction proofs submitted by users.
* **`jobs`**: Stores job titles, descriptions, required skills, and required experience levels.
* **`candidates`**: Stores candidate details, extracted contact info, calculated metric scores (semantic, skill, experience), status stages, and notes.
* **`audit_log`**: Records changes (like moving a candidate to a new stage) for full audit capabilities.

---

## 📄 License
This project is open-source and available under the MIT License.
