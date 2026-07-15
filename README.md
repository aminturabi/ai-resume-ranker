# AI Resume Ranker & ATS 🎯

An AI-powered Applicant Tracking System (ATS) designed to automate, parse, score, and rank candidate resumes using NLP models, custom skill weightings, and experience matching. It also features a fully-fledged recruitment pipeline dashboard to manage candidates through stages (from Screening to Offered/Hired).

---

## ✨ Features

* **Multi-Method PDF Text Extraction**: Uses PyMuPDF and pdfplumber, with a fallback to **Tesseract OCR** for scanned/image-based PDF resumes.
* **Semantic Similarity Matching**: Uses Sentence Transformers (`all-MiniLM-L6-v2`) and Cosine Similarity to compare resume contents against the job description.
* **Weighted Skill Matching**: Input required skills and custom weights (e.g. Python: 3, SQL: 2) to score candidate expertise matches.
* **Experience Matching**: Automatically detects candidates' years of experience and scores them against the position's requirements.
* **Bias-Blind Review Mode**: Toggle to hide candidate name, email, and phone number to prevent hiring bias during review.
* **ATS Pipeline & Stages**: Move candidates through pipeline columns (`New`, `Screening`, `Shortlisted`, `Interview`, `Offer`, `Hired`, `Rejected`), write reviewer notes, and keep an active audit log.
* **Reports Export**: Export ranked lists and pipeline stages to `.csv` or `.xlsx` files.

---

## 🛠️ Technology Stack

* **Frontend**: [Streamlit](https://streamlit.io/) (including customized HTML/CSS styling).
* **NLP / AI Models**: [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2` via Hugging Face), [Scikit-Learn](https://scikit-learn.org/) (`cosine_similarity`).
* **PDF Extractors**: `pdfplumber`, `PyMuPDF (fitz)`, `pytesseract (Tesseract OCR)`, and `Pillow`.
* **Database & Tables**: `SQLite3` (relational database with `jobs`, `candidates`, and `audit_log` tables).
* **Data Processing & Export**: `pandas` and `openpyxl`.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8 or higher.
* (Optional) [Tesseract OCR engine](https://github.com/UB-Mannheim/tesseract/wiki) installed on your system if you want to support scanned resumes.

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

---

## 📁 Database Schema

The SQLite database (`ats_database.db`) is automatically initialized and migrates itself on launch:
* **`jobs`**: Stores job titles, descriptions, required skills, and required experience levels.
* **`candidates`**: Stores candidate details, extracted contact info, calculated metric scores (semantic, skill, experience), status stages, and notes.
* **`audit_log`**: Records changes (like moving a candidate to a new stage) for full audit capabilities.

---

## 📄 License
This project is open-source and available under the MIT License.
