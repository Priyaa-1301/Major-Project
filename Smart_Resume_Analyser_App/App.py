import streamlit as st
import nltk
try:
    import spacy
except Exception:
    spacy = None

nltk.download('stopwords', quiet=True)

nlp = None
if spacy is not None:
    try:
        nlp = spacy.load('en_core_web_sm')
    except Exception:
        import os
        os.system('python -m spacy download en_core_web_sm')
        try:
            nlp = spacy.load('en_core_web_sm')
        except Exception:
            pass

from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import base64
import random
import hashlib
import time
import datetime
import sqlite3
import pdfplumber
import re
import io
from streamlit_tags import st_tags
from PIL import Image
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
import yt_dlp
import plotly.express as px
from collections import Counter
import ast

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Resume Analyzer",
    page_icon='./Logo/SRA_Logo.ico',
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS / THEMING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---------- Main background ---------- */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e0e0e0;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-right: 1px solid rgba(130, 80, 255, 0.3);
}
[data-testid="stSidebar"] * {
    color: #f0eeff !important;
}

/* ---------- Hero title ---------- */
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #818cf8, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.25rem;
}
.hero-sub {
    color: #e2e8f0;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* ---------- Section card ---------- */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(130,80,255,0.25);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(6px);
}

/* ---------- Score badge ---------- */
.score-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: #fff;
    font-size: 2rem;
    font-weight: 700;
    border-radius: 50%;
    width: 90px; height: 90px;
    line-height: 90px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(124,58,237,0.5);
}

/* ---------- Tip row ---------- */
.tip-ok  { color: #4ade80; font-size: 1rem; margin: 6px 0; }
.tip-bad { color: #fbbf24; font-size: 1rem; margin: 6px 0; }

/* ---------- Metric cards ---------- */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(130,80,255,0.25);
    border-radius: 12px;
    padding: 0.75rem 1rem;
}

/* ---------- Tabs ---------- */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #a78bfa !important;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 2px solid #a78bfa !important;
}

/* ---------- Buttons ---------- */
.stButton>button {
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    transition: opacity 0.2s;
}
.stButton>button:hover { opacity: 0.85; }

/* ---------- Progress bar colour ---------- */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #38bdf8);
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1a1a2e; }
::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 3px; }

/* ---------- Dataframe ---------- */
.stDataFrame { background: rgba(255,255,255,0.03); border-radius: 12px; }

/* ---------- File uploader ---------- */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(130,80,255,0.4);
    border-radius: 12px;
    background: rgba(255,255,255,0.03);
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS_HASH = os.getenv("ADMIN_PASS_HASH", hash_password("1234"))


@st.cache_data(show_spinner=False)
def fetch_yt_video(link: str) -> str:
    try:
        ydl_opts = {"quiet": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(link, download=False)
            return info.get('title', '')
    except Exception:
        return ''


def show_pdf(file_path: str):
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = (
        f'<iframe src="data:application/pdf;base64,{b64}" '
        f'width="100%" height="800" type="application/pdf" '
        f'style="border-radius:12px; border:1px solid rgba(130,80,255,0.3);"></iframe>'
    )
    st.markdown(pdf_display, unsafe_allow_html=True)


def course_recommender(course_list: list, key_suffix: str = ""):
    st.markdown("#### 🎓 Courses & Certificates Recommendations")
    no_of_reco = st.slider('Number of course recommendations:', 1, 10, 4, key=f'course_slider_{key_suffix}')
    random.shuffle(course_list)
    rec_course = []
    for idx, (c_name, c_link) in enumerate(course_list[:no_of_reco]):
        st.markdown(
            f'<div class="card" style="padding:0.75rem 1rem; margin-bottom:0.5rem;">'
            f'<span style="color:#a78bfa; font-weight:600;">#{idx+1}</span> '
            f'<a href="{c_link}" target="_blank" style="color:#38bdf8; text-decoration:none;">{c_name}</a>'
            f'</div>',
            unsafe_allow_html=True,
        )
        rec_course.append(c_name)
    return rec_course


# ─────────────────────────────────────────────────────────────
# EXPANDED SKILL LIBRARY  (130 + keywords)
# ─────────────────────────────────────────────────────────────
SKILL_KEYWORDS = {
    # Data Science / ML / AI
    "python", "r", "sql", "nosql", "pandas", "numpy", "scipy",
    "scikit-learn", "tensorflow", "keras", "pytorch", "xgboost",
    "lightgbm", "catboost", "hugging face", "transformers", "nlp",
    "machine learning", "deep learning", "computer vision",
    "data analysis", "data visualization", "tableau", "power bi",
    "matplotlib", "seaborn", "plotly", "statistics", "probability",
    "data mining", "web scraping", "mlflow", "airflow", "spark",
    "hadoop", "hive", "kafka",
    # Web
    "html", "css", "javascript", "typescript", "react", "react js",
    "angular", "angular js", "vue", "vue js", "next.js", "nuxt",
    "node", "node js", "express", "django", "flask", "fastapi",
    "laravel", "php", "ruby on rails", "graphql", "rest api",
    "soap", "webpack", "tailwind", "bootstrap", "sass",
    # Mobile
    "android", "kotlin", "java", "swift", "ios", "flutter",
    "dart", "react native", "xamarin", "objective-c",
    # DevOps / Cloud / Infra
    "docker", "kubernetes", "aws", "azure", "gcp", "ci/cd",
    "jenkins", "github actions", "terraform", "ansible",
    "linux", "bash", "shell scripting", "nginx", "apache",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "sqlite",
    "oracle", "sql server", "dynamodb", "cassandra", "firebase",
    # UI/UX & Design
    "figma", "adobe xd", "sketch", "photoshop", "illustrator",
    "after effects", "indesign", "zeplin", "balsamiq",
    "wireframing", "prototyping", "user research", "ux writing",
    # General / Tools
    "git", "github", "gitlab", "jira", "confluence", "agile",
    "scrum", "kanban", "excel", "powerpoint", "word",
    "c", "c++", "c#", ".net", "go", "rust", "scala",
}

# Keyword sets for field detection
DS_KW     = {"python","tensorflow","keras","pytorch","machine learning","deep learning","nlp",
             "pandas","numpy","scikit-learn","spark","hadoop","data analysis","data visualization",
             "statistics","tableau","power bi","mlflow","airflow","xgboost","hugging face",
             "computer vision","matplotlib","seaborn"}
WEB_KW    = {"react","django","node","flask","fastapi","html","css","javascript","typescript",
             "angular","vue","next.js","laravel","php","graphql","rest api","bootstrap","tailwind",
             "express","webpack","ruby on rails"}
ANDROID_KW= {"android","kotlin","flutter","dart","react native","java","xamarin"}
IOS_KW    = {"ios","swift","objective-c","xcode","cocoa"}
UIUX_KW   = {"figma","adobe xd","sketch","photoshop","illustrator","wireframing","prototyping",
             "user research","zeplin","balsamiq","ux writing","after effects","indesign"}


# ─────────────────────────────────────────────────────────────
# RESUME PARSER  – multi-format, multi-strategy
# ─────────────────────────────────────────────────────────────

# Section header aliases → canonical name
SECTION_ALIASES = {
    # Summary / Objective
    'summary': 'summary', 'professional summary': 'summary',
    'career summary': 'summary', 'objective': 'summary',
    'career objective': 'summary', 'about me': 'summary',
    'profile': 'summary', 'professional profile': 'summary',
    'personal statement': 'summary',
    # Experience
    'experience': 'experience', 'work experience': 'experience',
    'professional experience': 'experience', 'employment history': 'experience',
    'work history': 'experience', 'career history': 'experience',
    'internship': 'experience', 'internships': 'experience',
    # Education
    'education': 'education', 'educational background': 'education',
    'academic background': 'education', 'qualifications': 'education',
    'academic qualifications': 'education', 'scholastic details': 'education',
    # Skills
    'skills': 'skills', 'technical skills': 'skills',
    'core competencies': 'skills', 'competencies': 'skills',
    'key skills': 'skills', 'areas of expertise': 'skills',
    'expertise': 'skills', 'technologies': 'skills',
    'tools & technologies': 'skills', 'tools and technologies': 'skills',
    # Projects
    'projects': 'projects', 'academic projects': 'projects',
    'personal projects': 'projects', 'key projects': 'projects',
    'project work': 'projects', 'portfolio': 'projects',
    # Achievements
    'achievements': 'achievements', 'awards': 'achievements',
    'honors': 'achievements', 'honours': 'achievements',
    'accomplishments': 'achievements', 'certifications': 'achievements',
    'certificates': 'achievements',
    # Misc
    'languages': 'languages', 'hobbies': 'hobbies',
    'interests': 'hobbies', 'activities': 'hobbies',
    'references': 'references', 'declaration': 'declaration',
}

# Compiled regex for fast section-header detection
_SECTION_RE = re.compile(
    r'^\s*(' + '|'.join(re.escape(k) for k in SECTION_ALIASES) + r')\s*[:\-–—]?\s*$',
    re.IGNORECASE,
)


def _words_to_text(words: list) -> str:
    """Reconstruct text from pdfplumber word dicts sorted by (top, x0).
    Groups words into lines with a Y-tolerance of 4 pt, handling
    multi-column layouts by using spatial order rather than raw order.
    """
    if not words:
        return ""
    words_sorted = sorted(words, key=lambda w: (round(w['top'] / 4) * 4, w['x0']))
    lines, current_line, prev_top = [], [], None
    for w in words_sorted:
        top = round(w['top'] / 4) * 4
        if prev_top is not None and abs(top - prev_top) > 4:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = []
        current_line.append(w['text'])
        prev_top = top
    if current_line:
        lines.append(" ".join(current_line))
    return "\n".join(lines)


def _extract_text_multimethod(file_path: str):
    """Return (pages, per_page_texts, full_text) using the best available method."""
    try:
        with pdfplumber.open(file_path) as pdf:
            pages = len(pdf.pages)
            layout_texts, word_texts = [], []
            for page in pdf.pages:
                # Method 1: standard layout-aware extraction
                layout_texts.append(page.extract_text(x_tolerance=3, y_tolerance=3) or "")
                # Method 2: word-sort reconstruction (handles multi-column)
                words = page.extract_words(x_tolerance=3, y_tolerance=3)
                word_texts.append(_words_to_text(words))

            # Pick the method that produces more text per page on average
            layout_avg = sum(len(t) for t in layout_texts) / max(pages, 1)
            word_avg   = sum(len(t) for t in word_texts)   / max(pages, 1)
            texts = word_texts if word_avg > layout_avg else layout_texts
            full_text = "\n".join(texts)
            return pages, texts, full_text
    except Exception:
        return None, [], ""


def _detect_sections(full_text: str) -> dict:
    """Identify which canonical sections are present in the resume."""
    present = set()
    for line in full_text.splitlines():
        m = _SECTION_RE.match(line.strip())
        if m:
            canonical = SECTION_ALIASES.get(m.group(1).lower().strip())
            if canonical:
                present.add(canonical)
    return present


def _extract_name(first_page_text: str, email: str, phone: str) -> str:
    """Multi-heuristic name extractor."""
    # 1. spaCy NER
    if nlp and first_page_text:
        try:
            doc = nlp(first_page_text[:1200])
            for ent in doc.ents:
                if ent.label_ == 'PERSON' and 1 <= len(ent.text.split()) <= 5:
                    return ent.text.strip()
        except Exception:
            pass

    # 2. Line scan heuristics – top of first page
    blacklist_words = {'resume', 'curriculum', 'vitae', 'cv', 'profile',
                       'contact', 'address', 'email', 'phone', 'mobile',
                       'objective', 'summary', 'linkedin', 'github'}
    for line in first_page_text.splitlines()[:30]:   # only top 30 lines
        line = line.strip()
        if not line or len(line) < 2:
            continue
        if email and email.lower() in line.lower():
            continue
        if phone and phone in line:
            continue
        if re.search(r'@|http|www\.|\d{4}', line):
            continue
        words = line.split()
        if not (1 <= len(words) <= 6):
            continue
        low = line.lower()
        if any(bw in low for bw in blacklist_words):
            continue

        # Accept lines that are title-case, ALL-CAPS, or pure alpha+spaces
        is_title  = line.istitle()
        is_allcap = line.isupper() and line.replace(" ", "").isalpha()
        is_alpha  = line.replace(" ", "").isalpha()

        if is_alpha or is_title or is_allcap:
            # Convert ALL-CAPS names to Title Case for display
            return line.title() if is_allcap else line

    return ""


def _extract_contact_urls(full_text: str) -> dict:
    """Extract LinkedIn and GitHub profile URLs if present."""
    linkedin = re.search(
        r'(linkedin\.com/in/[\w\-]+)', full_text, re.IGNORECASE
    )
    github = re.search(
        r'(github\.com/[\w\-]+)', full_text, re.IGNORECASE
    )
    return {
        'linkedin': linkedin.group(1) if linkedin else '',
        'github':   github.group(1)   if github   else '',
    }


def _extract_phone(full_text: str) -> str:
    """Try multiple phone patterns in priority order."""
    patterns = [
        r'(\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4})',  # intl
        r'(\(?\d{3}\)?[\s\-\.]\d{3}[\s\-\.]\d{4})',                        # US/CA
        r'(\d{5}[\s\-]\d{5})',                                              # IN 5+5
        r'(\+?\d[\d\s\-()]{8,}\d)',                                         # generic
    ]
    for pat in patterns:
        m = re.search(pat, full_text)
        if m:
            phone = re.sub(r'\s+', ' ', m.group(1)).strip()
            # sanity-check: must have 7+ digits
            if len(re.sub(r'\D', '', phone)) >= 7:
                return phone
    return ""


def parse_resume(file_path: str) -> dict | None:
    """
    Format-robust resume parser.
    Handles: single/multi-column layouts, all-caps / mixed-case headers,
    tabular skill sections, varied contact formats, LinkedIn/GitHub URLs.
    """
    pages, texts, full_text = _extract_text_multimethod(file_path)
    if not full_text.strip():
        return None

    first_page = texts[0] if texts else full_text

    # ── Contact info ────────────────────────────────────────
    email_m = re.search(r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9.\-]+", full_text)
    email   = email_m.group(0) if email_m else ""
    phone   = _extract_phone(full_text)
    urls    = _extract_contact_urls(full_text)

    # ── Name ────────────────────────────────────────────────
    name = _extract_name(first_page, email, phone)

    # ── Sections present ───────────────────────────────────
    detected_sections = _detect_sections(full_text)

    # ── Education keyword fallback ──────────────────────────
    degree_re = re.compile(
        r'\b(b\.?tech|b\.?e|b\.?sc|b\.?com|bca|mca|m\.?sc|m\.?tech|mba'
        r'|phd|ph\.?d|bachelor|master|doctorate|b\.?s|m\.?s|b\.?a|m\.?a'
        r'|associate degree|high school|secondary school)\b',
        re.IGNORECASE,
    )
    has_education = ('education' in detected_sections) or bool(degree_re.search(full_text))

    # ── Skills ──────────────────────────────────────────────
    lower_text = full_text.lower()
    found_skills = set()
    for kw in SKILL_KEYWORDS:
        if kw[-1] in ('+', '#', '.'):
            pattern = r'\b' + re.escape(kw) + r'(?!\w)'
        else:
            pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, lower_text):
            found_skills.add(kw)

    return {
        'name':            name or '',
        'email':           email,
        'mobile_number':   phone,
        'linkedin':        urls['linkedin'],
        'github':          urls['github'],
        'no_of_pages':     pages,
        'skills':          list(found_skills),
        'has_education':   has_education,
        'detected_sections': detected_sections,
        'full_text':       full_text,
    }


# ─────────────────────────────────────────────────────────────
# FIELD DETECTION
# ─────────────────────────────────────────────────────────────
def detect_field(skills: list) -> str:
    skill_set = {s.lower() for s in skills}
    counts = {
        'Data Science':          len(skill_set & DS_KW),
        'Web Development':       len(skill_set & WEB_KW),
        'Android Development':   len(skill_set & ANDROID_KW),
        'IOS Development':       len(skill_set & IOS_KW),
        'UI-UX Development':     len(skill_set & UIUX_KW),
    }
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else 'General'


FIELD_RECOMMENDED_SKILLS = {
    'Data Science': ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling',
                     'Data Mining', 'Clustering & Classification', 'Data Analytics',
                     'Quantitative Analysis', 'Web Scraping', 'ML Algorithms', 'Keras',
                     'Pytorch', 'Probability', 'Scikit-learn', 'Tensorflow', 'Flask', 'Streamlit'],
    'Web Development': ['React', 'Django', 'Node JS', 'React JS', 'PHP', 'Laravel',
                        'WordPress', 'JavaScript', 'Angular JS', 'C#', 'Flask', 'SDK',
                        'TypeScript', 'GraphQL'],
    'Android Development': ['Android', 'Android Development', 'Flutter', 'Kotlin', 'Dart',
                             'XML', 'Java', 'Kivy', 'Git', 'SDK', 'SQLite'],
    'IOS Development': ['iOS', 'Swift', 'Cocoa', 'Cocoa Touch', 'Xcode', 'Objective-C',
                        'SQLite', 'Plist', 'StoreKit', 'UI-Kit', 'AV Foundation', 'Auto-Layout'],
    'UI-UX Development': ['UI', 'User Experience', 'Adobe XD', 'Figma', 'Zeplin', 'Balsamiq',
                          'Prototyping', 'Wireframes', 'Adobe Photoshop', 'Illustrator',
                          'After Effects', 'Wireframe', 'User Research'],
    'General': ['Communication', 'Problem Solving', 'Time Management', 'Leadership',
                'Teamwork', 'Critical Thinking', 'Project Management'],
}

FIELD_COURSES = {
    'Data Science': ds_course,
    'Web Development': web_course,
    'Android Development': android_course,
    'IOS Development': ios_course,
    'UI-UX Development': uiux_course,
    'General': ds_course + web_course,
}

FIELD_EMOJI = {
    'Data Science': '🤖',
    'Web Development': '🌐',
    'Android Development': '🤖',
    'IOS Development': '🍎',
    'UI-UX Development': '🎨',
    'General': '💼',
}


# ─────────────────────────────────────────────────────────────
# RESUME SCORING  – 14 criteria, 5 categories, 100 pts total
# ─────────────────────────────────────────────────────────────

# Common resume action verbs that signal well-written bullet points
ACTION_VERBS = {
    'developed', 'designed', 'implemented', 'built', 'created', 'managed',
    'led', 'improved', 'optimized', 'reduced', 'increased', 'launched',
    'delivered', 'achieved', 'collaborated', 'mentored', 'automated',
    'deployed', 'resolved', 'analysed', 'analyzed', 'spearheaded',
    'architected', 'migrated', 'integrated', 'researched', 'published',
    'awarded', 'coordinated', 'streamlined', 'engineered', 'contributed',
    'established', 'executed', 'facilitated', 'generated', 'supervised',
}

# Each entry: key -> (max_points, category, emoji_label, tip_if_missing)
SCORE_CRITERIA = {
    # Contact & Profile (20 pts)
    'has_email': (5, 'Contact & Profile', '📧 Email Address', 'Add your email address so recruiters can contact you.'),
    'has_phone': (5, 'Contact & Profile', '📞 Phone Number', 'Include a phone number for easy reachability.'),
    'has_linkedin': (5, 'Contact & Profile', '🔗 LinkedIn Profile', 'Add your LinkedIn URL to boost professional credibility.'),
    'has_github': (5, 'Contact & Profile', '🐙 GitHub / Portfolio', 'A GitHub or portfolio link showcases your real work.'),
    # Resume Sections (40 pts)
    'has_summary': (10, 'Resume Sections', '📝 Summary / Objective', 'A professional summary tells recruiters who you are at a glance.'),
    'has_education': (10, 'Resume Sections', '🎓 Education', 'Include your degree(s), institution, and graduation year.'),
    'has_skills': (10, 'Resume Sections', '🛠️ Skills Section', 'A dedicated Skills section makes your expertise immediately visible.'),
    'has_projects': (10, 'Resume Sections', '💻 Projects', 'Projects demonstrate hands-on ability beyond coursework.'),
    # Content Quality (20 pts)
    'has_achievements': (10, 'Content Quality', '🏅 Achievements / Certifications', 'Awards and certifications add credibility and stand out to recruiters.'),
    'has_quantified_impact': (10, 'Content Quality', '📊 Quantified Achievements', 'Use numbers (e.g. "improved speed by 30%") to make your impact concrete.'),
    # Structure & Formatting (20 pts)
    'has_action_verbs': (5, 'Structure & Formatting', '🚀 Action Verbs', 'Start bullet points with strong verbs (built, led, optimized) to sound impactful.'),
    'uses_bullet_points': (5, 'Structure & Formatting', '• Bullet Points Used', 'Use bullet points to organize information — easier for recruiters to scan.'),
    'well_structured': (5, 'Structure & Formatting', '🗂️ Clear Section Headers', 'Use clear section headers (Education, Skills, Projects…) for easy navigation.'),
    'adequate_length': (5, 'Structure & Formatting', '📄 Concise Length (1–2 pages)', 'Keep your resume to 1–2 pages — concise, focused resumes are preferred.'),
}

CATEGORY_ORDER = ['Contact & Profile', 'Resume Sections', 'Content Quality', 'Structure & Formatting']

CATEGORY_MAX = {cat: 0 for cat in CATEGORY_ORDER}
for key, (pts, cat, *_rest) in SCORE_CRITERIA.items():
    CATEGORY_MAX[cat] += pts


def compute_resume_score(resume_data: dict) -> tuple[int, dict]:
    """
    14-criteria weighted resume scorer (100 pts total):
      Contact & Profile      (20 pts) - email, phone, LinkedIn, GitHub
      Resume Sections        (40 pts) - summary, education, skills, projects
      Content Quality        (20 pts) - achievements, quantified impact
      Structure & Formatting (20 pts) - action verbs, bullets, clear headers, concise length
    """
    sections = resume_data.get('detected_sections', set())
    text     = resume_data.get('full_text', '')
    lower    = text.lower()
    skills   = resume_data.get('skills', [])
    lines    = text.splitlines()
    results  = {}

    # Contact & Profile
    results['has_email']    = bool(resume_data.get('email'))
    results['has_phone']    = bool(resume_data.get('mobile_number'))
    results['has_linkedin'] = bool(resume_data.get('linkedin'))
    results['has_github']   = bool(resume_data.get('github'))

    # Resume Sections
    results['has_summary'] = ('summary' in sections) or bool(re.search(
        r'\b(summary|objective|about me|profile|career objective|personal statement)\b',
        text, re.IGNORECASE
    ))
    results['has_education'] = (
        'education' in sections
        or resume_data.get('has_education', False)
        or bool(re.search(r'\b(education|academic|qualification|degree|university|college)\b',
                          text, re.IGNORECASE))
    )
    results['has_skills']   = ('skills' in sections) or (len(skills) > 0)
    results['has_projects'] = ('projects' in sections) or bool(re.search(
        r'\b(project|projects|portfolio|case study)\b', text, re.IGNORECASE
    ))

    # Content Quality
    results['has_achievements'] = ('achievements' in sections) or bool(re.search(
        r'\b(achievement|award|accomplishment|certification|honour|honor|recogni[sz]ed)\b',
        text, re.IGNORECASE
    ))
    results['has_quantified_impact'] = bool(re.search(
        r'(\d+\s*%|\d+\s*x\b|\$\s*\d+|\d+\s*[kKmMbB]\b|\d+\+)', text
    ))

    # Structure & Formatting
    words_in_text = set(re.findall(r'\b[a-z]+\b', lower))
    results['has_action_verbs'] = len(words_in_text & ACTION_VERBS) >= 3

    bullet_lines = sum(
        1 for line in lines
        if re.match(r'^\s*[\u2022\u2023\u25e6\u2043\-\*\>]', line.strip())
    )
    results['uses_bullet_points'] = bullet_lines >= 3

    # Clear section headers: 3+ canonical sections detected
    results['well_structured'] = len(sections) >= 3

    # Concise length: 1 or 2 pages
    results['adequate_length'] = 1 <= resume_data.get('no_of_pages', 1) <= 2

    score = sum(
        pts
        for key, (pts, *_rest) in SCORE_CRITERIA.items()
        if results.get(key)
    )
    return score, results



# ─────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────
DB_PATH = 'resume_analyzer.db'
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_cur = _conn.cursor()
_cur.execute('''
    CREATE TABLE IF NOT EXISTS user_data (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Name TEXT NOT NULL,
        Email_ID TEXT NOT NULL,
        resume_score TEXT NOT NULL,
        Timestamp TEXT NOT NULL,
        Page_no TEXT NOT NULL,
        Predicted_Field TEXT NOT NULL,
        User_level TEXT NOT NULL,
        Actual_skills TEXT NOT NULL,
        Recommended_skills TEXT NOT NULL,
        Recommended_courses TEXT NOT NULL
    )
''')
_conn.commit()


def insert_data(name, email, res_score, timestamp, no_of_pages,
                reco_field, cand_level, skills, recommended_skills, courses):
    try:
        _cur.execute('''
            INSERT INTO user_data
            (Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field,
             User_level, Actual_skills, Recommended_skills, Recommended_courses)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, email, str(res_score), timestamp,
              str(no_of_pages), reco_field, cand_level,
              str(skills), str(recommended_skills), str(courses)))
        _conn.commit()
    except Exception:
        pass


def delete_records(ids: list):
    """Delete rows by their ID list."""
    if not ids:
        return
    placeholders = ','.join('?' * len(ids))
    try:
        _cur.execute(f'DELETE FROM user_data WHERE ID IN ({placeholders})', ids)
        _conn.commit()
    except Exception as e:
        st.error(f"Delete error: {e}")


def delete_all_records():
    """Wipe all rows from user_data."""
    try:
        _cur.execute('DELETE FROM user_data')
        _conn.commit()
    except Exception as e:
        st.error(f"Delete-all error: {e}")


# DB column name mapping  (display name → DB column)
_DB_COL_MAP = {
    'Name':                'Name',
    'Email':               'Email_ID',
    'Resume Score':        'resume_score',
    'Timestamp':           'Timestamp',
    'Total Pages':         'Page_no',
    'Predicted Field':     'Predicted_Field',
    'User Level':          'User_level',
    'Actual Skills':       'Actual_skills',
    'Recommended Skills':  'Recommended_skills',
    'Recommended Courses': 'Recommended_courses',
}


def update_record(row_id: int, display_col: str, value):
    """Persist a single cell change back to SQLite."""
    db_col = _DB_COL_MAP.get(display_col)
    if db_col is None:
        return
    try:
        _cur.execute(f'UPDATE user_data SET "{db_col}" = ? WHERE ID = ?',
                     (str(value), row_id))
        _conn.commit()
    except Exception as e:
        st.error(f"Update error: {e}")


# ─────────────────────────────────────────────────────────────
# CANDIDATE LEVEL
# ─────────────────────────────────────────────────────────────
def infer_cand_level(pages: int) -> tuple[str, str, str]:
    if pages == 1:
        return "Fresher", "#a78bfa", "🌱 You appear to be a Fresher — keep building those skills!"
    elif pages == 2:
        return "Intermediate", "#38bdf8", "🚀 You're at an Intermediate level — great progress!"
    else:
        return "Experienced", "#4ade80", "🏆 You're an Experienced professional — impressive!"


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def run():
    # ── Sidebar ──────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <div style="text-align:center; padding:1rem 0;">
            <div style="font-size:2.5rem;">📄</div>
            <div style="font-size:1.2rem; font-weight:700; color:#a78bfa;">Resume Analyzer</div>
            <div style="font-size:0.78rem; color:#b8b8e0; margin-top:0.25rem;">AI-Powered Career Insights</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    choice = st.sidebar.selectbox("🔀 Mode", ["👤 Normal User", "🔐 Admin"])

    # ── Hero ─────────────────────────────────────────────────
    st.markdown(
        '<div class="hero-title">Smart Resume Analyzer</div>'
        '<div class="hero-sub">Upload your resume and get instant AI-powered feedback, '
        'skill gap analysis, and career recommendations.</div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════
    # NORMAL USER
    # ══════════════════════════════════════════════════════════
    if choice == "👤 Normal User":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Your Resume")
        pdf_file = st.file_uploader("Supported format: PDF", type=["pdf"])
        st.markdown('</div>', unsafe_allow_html=True)

        if pdf_file is None:
            return

        save_path = os.path.join('./Uploaded_Resumes/', pdf_file.name)
        with open(save_path, "wb") as f:
            f.write(pdf_file.getbuffer())

        with st.spinner("🔍 Analyzing your resume…"):
            resume_data = parse_resume(save_path)

        if resume_data is None:
            st.error("❌ Could not extract text from your PDF. Please try a text-based PDF.")
            if os.path.exists(save_path):
                os.remove(save_path)
            return

        # ── PDF Preview ──────────────────────────────────────
        with st.expander("📄 Preview Resume", expanded=False):
            show_pdf(save_path)

        # ── Basic Info ───────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 👤 Basic Information")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Name",  resume_data['name']          or "—")
        c2.metric("Email", resume_data['email']         or "—")
        c3.metric("Phone", resume_data['mobile_number'] or "—")
        c4.metric("Pages", str(resume_data['no_of_pages']))

        # LinkedIn / GitHub links (only show if found)
        link_parts = []
        if resume_data.get('linkedin'):
            link_parts.append(
                f'<a href="https://{resume_data["linkedin"]}" target="_blank" '
                f'style="color:#38bdf8; margin-right:1.5rem;">🔗 LinkedIn</a>'
            )
        if resume_data.get('github'):
            link_parts.append(
                f'<a href="https://{resume_data["github"]}" target="_blank" '
                f'style="color:#a78bfa;">🐙 GitHub</a>'
            )
        if link_parts:
            st.markdown(
                '<div style="margin-top:0.6rem; font-size:0.95rem;">' + ''.join(link_parts) + '</div>',
                unsafe_allow_html=True
            )

        cand_level, lvl_color, lvl_msg = infer_cand_level(resume_data['no_of_pages'])
        st.markdown(
            f'<p style="color:{lvl_color}; font-weight:600; font-size:1rem; margin-top:0.5rem;">{lvl_msg}</p>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Skills ───────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🛠️ Skills Detected")
        if resume_data['skills']:
            st_tags(label='', text='Extracted from your resume',
                    value=resume_data['skills'], key='detected_skills')
        else:
            st.warning("No technical skills were detected. Consider adding a dedicated Skills section.")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Field Detection & Recommendations ────────────────
        reco_field = detect_field(resume_data['skills'])
        rec_skills = FIELD_RECOMMENDED_SKILLS.get(reco_field, FIELD_RECOMMENDED_SKILLS['General'])

        st.markdown('<div class="card">', unsafe_allow_html=True)
        field_emoji = FIELD_EMOJI.get(reco_field, '💼')
        st.markdown(f"### {field_emoji} Career Field Detection")
        if reco_field != 'General':
            st.success(f"**Our analysis suggests you're targeting: {reco_field}**")
        else:
            st.info("Could not detect a specific field — showing general recommendations.")

        st.markdown("#### 💡 Recommended Skills to Add")
        st_tags(label='', text='Adding these will boost your resume',
                value=rec_skills, key='recommended_skills')
        st.markdown(
            '<p style="color:#4ade80; font-weight:500; margin-top:0.5rem;">'
            '🚀 Adding these skills to your resume will significantly improve your chances of getting hired!</p>',
            unsafe_allow_html=True
        )
        rec_course = course_recommender(FIELD_COURSES.get(reco_field, ds_course), key_suffix=reco_field)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Resume Score ─────────────────────────────────────
        resume_score, score_breakdown = compute_resume_score(resume_data)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Resume Score & Tips")

        col_score, col_cats = st.columns([1, 2])
        with col_score:
            st.markdown(
                f'<div style="text-align:center; padding:1.5rem;">'
                f'<div class="score-badge">{resume_score}</div>'
                f'<div style="color:#e2e8f0; margin-top:0.75rem; font-size:0.9rem;">out of 100</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            my_bar = st.progress(0)
            for i in range(resume_score + 1):
                time.sleep(0.012)
                my_bar.progress(i)

            # Category mini-scorecard
            st.markdown("<br>", unsafe_allow_html=True)
            for cat in CATEGORY_ORDER:
                cat_max = CATEGORY_MAX[cat]
                cat_got = sum(
                    pts
                    for key, (pts, c, *_r) in SCORE_CRITERIA.items()
                    if c == cat and score_breakdown.get(key)
                )
                pct = int(cat_got / cat_max * 100) if cat_max else 0
                color = "#4ade80" if pct == 100 else "#38bdf8" if pct >= 60 else "#fbbf24"
                st.markdown(
                    f'<div style="font-size:0.78rem; color:#e2e8f0; margin-bottom:2px;">'
                    f'{cat} &nbsp;<span style="color:{color}; float:right;">{cat_got}/{cat_max}</span></div>',
                    unsafe_allow_html=True
                )
                st.progress(pct)

        with col_cats:
            st.markdown("**Detailed Checklist**")
            current_cat = None
            for key, (pts, cat, label, advice) in SCORE_CRITERIA.items():
                if cat != current_cat:
                    current_cat = cat
                    st.markdown(
                        f'<p style="color:#a78bfa; font-weight:700; font-size:0.85rem; '
                        f'margin:0.8rem 0 0.2rem 0; text-transform:uppercase; letter-spacing:0.05em;">'
                        f'{cat}</p>',
                        unsafe_allow_html=True
                    )
                passed = score_breakdown.get(key, False)
                pts_text = f"+{pts}pt{'s' if pts > 1 else ''}"
                if passed:
                    st.markdown(
                        f'<p class="tip-ok">✅ {label} '
                        f'<span style="color:#4ade80; font-size:0.78rem; opacity:0.85;">{pts_text}</span></p>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<p class="tip-bad">⚠️ {label} — {advice}</p>',
                        unsafe_allow_html=True
                    )

        if resume_score == 100:
            st.success("🎉 Perfect score! Your resume is outstanding.")
        elif resume_score >= 80:
            st.success("👏 Great resume! A few small tweaks and you're there.")
        elif resume_score >= 60:
            st.warning("🔧 Solid start — address the ⚠️ items above to stand out.")
        elif resume_score >= 40:
            st.warning("📋 Needs improvement — work through the checklist above.")
        else:
            st.error("🚨 Your resume needs significant work. Follow the tips above carefully.")

        st.markdown('</div>', unsafe_allow_html=True)

        # ── DB Insert ────────────────────────────────────────
        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        insert_data(
            name=resume_data['name'],
            email=resume_data['email'],
            res_score=str(resume_score),
            timestamp=ts,
            no_of_pages=str(resume_data['no_of_pages']),
            reco_field=reco_field,
            cand_level=cand_level,
            skills=str(resume_data['skills']),
            recommended_skills=str(rec_skills),
            courses=str(rec_course),
        )

        # ── Video Recommendations ────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎬 Bonus Video Recommendations")
        no_of_videos = st.slider('Number of videos:', 1, 15, 4, key='vid_slider')
        all_videos = resume_videos + interview_videos
        random.shuffle(all_videos)
        cols = st.columns(2)
        for idx, vid in enumerate(all_videos[:no_of_videos]):
            vid_title = fetch_yt_video(vid) or f"Video #{idx+1}"
            with cols[idx % 2]:
                st.markdown(f"**✅ {vid_title}**")
                st.video(vid)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Cleanup ──────────────────────────────────────────
        if os.path.exists(save_path):
            os.remove(save_path)

    # ══════════════════════════════════════════════════════════
    # ADMIN
    # ══════════════════════════════════════════════════════════
    else:
        # ── Persist admin auth across re-runs (e.g. filter/widget interactions) ──
        if not st.session_state.get('admin_authenticated', False):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🔐 Admin Login")
            ad_user = st.text_input("Username", key="admin_user")
            ad_pass = st.text_input("Password", type='password', key="admin_pass")
            login_btn = st.button("Login")
            st.markdown('</div>', unsafe_allow_html=True)

            if not login_btn:
                return

            if ad_user != ADMIN_USER or hash_password(ad_pass) != ADMIN_PASS_HASH:
                st.error("❌ Wrong username or password.")
                return

            st.session_state['admin_authenticated'] = True

        col_welcome, col_logout = st.columns([8, 1])
        col_welcome.success("✅ Welcome, Admin!")
        if col_logout.button("Logout", key="admin_logout"):
            st.session_state['admin_authenticated'] = False
            st.rerun()

        try:
            _cur.execute('SELECT * FROM user_data')
            data = _cur.fetchall()
        except Exception as e:
            st.error(f"Database error: {e}")
            return

        df = pd.DataFrame(data, columns=[
            'ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Pages',
            'Predicted Field', 'User Level', 'Actual Skills',
            'Recommended Skills', 'Recommended Courses'
        ])
        df['Resume Score'] = pd.to_numeric(df['Resume Score'], errors='coerce')

        if df.empty:
            st.info("No data yet — analyze some resumes first.")
            return

        # KPI row
        st.markdown("### 📊 Key Metrics")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Resumes",   len(df))
        k2.metric("Avg Score",       f"{df['Resume Score'].mean():.0f} / 100")
        k3.metric("Top Field",       df['Predicted Field'].mode()[0])
        k4.metric("Top Level",       df['User Level'].mode()[0])

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📋 Data View", "📈 Analytics", "🛠️ Skills Analysis"])

        # ── Tab 1: Data View ─────────────────────────────────
        with tab1:

            # ── Filters row ──────────────────────────────────
            fcol1, fcol2, fcol3, fcol4 = st.columns([2, 2, 2, 1])
            with fcol1:
                field_opt = ["All"] + sorted(df['Predicted Field'].unique().tolist())
                field_filter = st.selectbox("Filter by Field", field_opt, key="f_field")
            with fcol2:
                level_opt = ["All"] + sorted(df['User Level'].unique().tolist())
                level_filter = st.selectbox("Filter by Level", level_opt, key="f_level")
            with fcol3:
                sort_col = st.selectbox(
                    "Sort by",
                    ['ID', 'Name', 'Email', 'Resume Score', 'Timestamp',
                     'Total Pages', 'Predicted Field', 'User Level'],
                    key="f_sort"
                )
            with fcol4:
                sort_asc = st.radio("Order", ["⬆ Asc", "⬇ Desc"],
                                    key="f_order", horizontal=False) == "⬆ Asc"

            fdf = df.copy()
            if field_filter != "All":
                fdf = fdf[fdf['Predicted Field'] == field_filter]
            if level_filter != "All":
                fdf = fdf[fdf['User Level'] == level_filter]

            # Convert object columns to string for proper display
            for col in ['Name', 'Email', 'Actual Skills',
                        'Recommended Skills', 'Recommended Courses',
                        'Predicted Field', 'User Level', 'Timestamp', 'Total Pages']:
                fdf[col] = fdf[col].astype(str)

            fdf = fdf.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

            # ── Selection column for deletion ─────────────────
            fdf.insert(0, '_del', False)

            edited = st.data_editor(
                fdf,
                use_container_width=True,
                num_rows="fixed",
                key="admin_data_editor",
                column_config={
                    "_del": st.column_config.CheckboxColumn(
                        "🗑 Select", help="Check to mark for deletion", default=False
                    ),
                    "ID": st.column_config.NumberColumn("ID", disabled=True),
                    "Resume Score": st.column_config.NumberColumn(
                        "Resume Score", min_value=0, max_value=100, step=1
                    ),
                    "Total Pages": st.column_config.TextColumn("Total Pages"),
                    "Name": st.column_config.TextColumn("Name"),
                    "Email": st.column_config.TextColumn("Email"),
                    "Timestamp": st.column_config.TextColumn("Timestamp"),
                    "Predicted Field": st.column_config.SelectboxColumn(
                        "Predicted Field",
                        options=['Data Science', 'Web Development',
                                 'Android Development', 'IOS Development',
                                 'UI-UX Development', 'General']
                    ),
                    "User Level": st.column_config.SelectboxColumn(
                        "User Level",
                        options=['Fresher', 'Intermediate', 'Experienced']
                    ),
                    "Actual Skills": st.column_config.TextColumn(
                        "Actual Skills", help="Comma-separated skill list"
                    ),
                    "Recommended Skills": st.column_config.TextColumn(
                        "Recommended Skills"
                    ),
                    "Recommended Courses": st.column_config.TextColumn(
                        "Recommended Courses"
                    ),
                },
                hide_index=True,
            )



            # ── Action buttons row ────────────────────────────
            ba1, ba2, ba3, ba4 = st.columns([2, 2, 2, 2])

            # ── Save edits ────────────────────────────────────
            with ba1:
                if st.button("💾 Save Changes", key="btn_save"):
                    editable_cols = [
                        'Name', 'Email', 'Resume Score', 'Timestamp',
                        'Total Pages', 'Predicted Field', 'User Level',
                        'Actual Skills', 'Recommended Skills', 'Recommended Courses'
                    ]
                    changes = 0
                    for idx, orig_row in fdf.iterrows():
                        row_id = int(orig_row['ID'])
                        for col in editable_cols:
                            orig_val = orig_row[col]
                            new_val  = edited.at[idx, col]
                            if str(orig_val) != str(new_val):
                                update_record(row_id, col, new_val)
                                changes += 1
                    if changes:
                        st.success(f"✅ Saved {changes} change(s) to the database.")
                        st.rerun()
                    else:
                        st.info("No changes detected.")

            # ── Delete selected button ────────────────────────
            # Compute which rows are checked RIGHT NOW (before any rerun)
            with ba2:
                # `== True` is required — data_editor returns numpy.bool_, not Python bool
                # so `is True` always evaluates to False and the list stays empty
                selected_now = [
                    int(edited.at[i, 'ID'])
                    for i in edited.index
                    if edited.at[i, '_del'] == True  # noqa: E712
                ]
                if st.button(
                    f"🗑️ Delete Selected ({len(selected_now)})",
                    key="btn_del_sel",
                    disabled=(len(selected_now) == 0),
                ):
                    # Save IDs to session_state NOW — on the next rerun the
                    # data_editor resets its checkboxes, so selected_now will
                    # be empty again; we need the IDs to persist.
                    st.session_state['pending_delete_ids'] = selected_now
                    st.session_state['confirm_del_sel'] = True
                    st.rerun()

            # ── Delete selected confirmation ──────────────────
            # MUST be outside the `with ba2:` block — nested columns inside a
            # column cause rendering issues and the sub-buttons won't fire.
            pending_ids = st.session_state.get('pending_delete_ids', [])
            if st.session_state.get('confirm_del_sel', False) and pending_ids:
                st.warning(
                    f"⚠️ About to permanently delete **{len(pending_ids)}** record(s). Confirm?"
                )
                c_yes, c_no, _ = st.columns([1, 1, 6])
                with c_yes:
                    if st.button("Yes, Delete", key="btn_del_sel_yes"):
                        delete_records(pending_ids)
                        st.session_state['confirm_del_sel'] = False
                        st.session_state['pending_delete_ids'] = []
                        st.success(f"✅ Deleted {len(pending_ids)} record(s).")
                        st.rerun()
                with c_no:
                    if st.button("Cancel", key="btn_del_sel_no"):
                        st.session_state['confirm_del_sel'] = False
                        st.session_state['pending_delete_ids'] = []
                        st.rerun()

            # ── Delete ALL button ─────────────────────────────
            with ba3:
                if st.button(" Delete All", key="btn_del_all"):
                    st.session_state['confirm_del_all'] = True
                    st.rerun()

            # ── Delete ALL confirmation ───────────────────────
            if st.session_state.get('confirm_del_all', False):
                st.warning("⚠️ This will **permanently erase ALL records**. Cannot be undone.")
                d_yes, d_no, _ = st.columns([1, 1, 6])
                with d_yes:
                    if st.button("Yes, Wipe All", key="btn_del_all_yes"):
                        delete_all_records()
                        st.session_state['confirm_del_all'] = False
                        st.success("All records deleted.")
                        st.rerun()
                with d_no:
                    if st.button("Cancel", key="btn_del_all_no"):
                        st.session_state['confirm_del_all'] = False
                        st.rerun()

            # ── Download CSV ──────────────────────────────────
            with ba4:
                csv_bytes = fdf.drop(columns=['_del']).to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv_bytes, "resume_data.csv", "text/csv")

        # ── Tab 2: Analytics ─────────────────────────────────
        with tab2:
            c_l, c_r = st.columns(2)
            with c_l:
                fig1 = px.pie(
                    df, names='Predicted Field',
                    title='Predicted Field Distribution',
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Purples_r,
                )
                fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
                st.plotly_chart(fig1, use_container_width=True)
            with c_r:
                fig2 = px.pie(
                    df, names='User Level',
                    title="Candidate Experience Level",
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                )
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
                st.plotly_chart(fig2, use_container_width=True)

            fig3 = px.histogram(
                df, x='Resume Score', nbins=10,
                title='Resume Score Distribution',
                color_discrete_sequence=['#7c3aed'],
            )
            fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0',
                               bargap=0.1)
            st.plotly_chart(fig3, use_container_width=True)

        # ── Tab 3: Skills Analysis ────────────────────────────
        with tab3:
            all_skills = []
            for s in df['Actual Skills']:
                if isinstance(s, str):
                    try:
                        lst = ast.literal_eval(s)
                        if isinstance(lst, list):
                            all_skills.extend(lst)
                    except Exception:
                        all_skills.extend([x.strip() for x in
                                           s.replace('[','').replace(']','').replace("'",'').split(',') if x.strip()])

            if all_skills:
                top_skills = dict(Counter(all_skills).most_common(15))
                sdf = pd.DataFrame(list(top_skills.items()), columns=['Skill', 'Count'])
                fig4 = px.bar(
                    sdf, x='Count', y='Skill', orientation='h',
                    title='Top 15 Most Common Skills',
                    color='Count', color_continuous_scale='Viridis',
                )
                fig4.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0',
                    yaxis={'categoryorder': 'total ascending'},
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("No skills data to display yet.")


run()