# ==============================
# app.py - Streamlit App (Multi-Page Navigation)
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import random
import io
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="CareerLens — Job & Salary Predictor",
    page_icon="🚀",
    layout="wide"
)

# ==============================
# CUSTOM CSS
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

#MainMenu {visibility: hidden;}
footer     {visibility: hidden;}
header     {visibility: hidden;}
[data-testid="stSidebar"] {display: none;}

.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}
.element-container { margin-bottom: 0.3rem !important; }
.stMarkdown { margin-bottom: 0 !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0.25rem !important; }

html, body, .stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    color: white;
    font-family: 'Sora', sans-serif !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background:
        radial-gradient(ellipse 65% 45% at 75% 0%,   rgba(124,110,247,.20) 0%, transparent 65%),
        radial-gradient(ellipse 45% 40% at 10% 85%,  rgba(79,195,247,.10)  0%, transparent 60%),
        radial-gradient(ellipse 30% 30% at 88% 70%,  rgba(244,114,182,.08) 0%, transparent 55%);
}

.topnav-brand {
    display: flex; align-items: center; gap: .55rem;
    font-size: 1.1rem; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; text-decoration: none;
}
.topnav-rocket { font-size: 1.3rem; filter: drop-shadow(0 0 8px #7c6ef7); }

.hero { text-align: center; padding: 1.2rem 1rem 0.5rem 1rem; }
.hero-logo-row {
    display: flex; align-items: center; justify-content: center; gap: .65rem;
    margin-bottom: .25rem;
}
.hero-rocket {
    font-size: 2.2rem; display: inline-block;
    animation: rocketBob 2.5s ease-in-out infinite;
    filter: drop-shadow(0 0 14px #7c6ef7);
}
@keyframes rocketBob {
    0%,100% { transform: translateY(0) rotate(-10deg); }
    50%      { transform: translateY(-7px) rotate(-15deg); }
}
.hero h1 {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0;
}
.hero p { font-size: .95rem; color: #94a3b8; margin-top: .15rem; }
.hero .tagline {
    font-size: 0.78rem; color: #6366f1; font-weight: 600;
    margin-top: .2rem; letter-spacing: 1.5px; text-transform: uppercase;
}

.cl-stats {
    display: flex; justify-content: center; gap: .5rem;
    flex-wrap: wrap; margin: .5rem 0 .2rem;
}
.cl-chip {
    display: inline-flex; align-items: center; gap: .35rem;
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.1);
    border-radius: 40px; padding: .25rem .7rem;
    font-size: .72rem; color: #7b84a8;
}
.cl-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }

.accuracy-badge {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white; padding: .4rem 1.1rem; border-radius: 30px;
    font-weight: 700; font-size: .88rem; display: inline-block;
    margin: .25rem 0; position: relative; overflow: hidden;
}
.accuracy-badge::after {
    content: ''; position: absolute; top: 0; left: -100%;
    width: 55%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.18), transparent);
    animation: shimmer 3s ease-in-out infinite;
}
@keyframes shimmer { 0%{left:-100%} 100%{left:220%} }

.page-header {
    padding: .8rem 0 .5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: .8rem;
}
.page-header h2 {
    font-size: 1.65rem; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 .15rem 0;
}
.page-header p { color: #64748b; font-size: .82rem; margin: 0; }

.input-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 1.1rem 1.4rem;
    margin: .4rem 0 .8rem 0; backdrop-filter: blur(10px);
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: .93rem !important; height: 2.6rem !important; width: 100% !important;
    transition: transform .2s, box-shadow .2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,.45) !important;
}

.skill-pill-row .stButton > button {
    background: rgba(124,110,247,0.15) !important;
    border: 1px solid rgba(124,110,247,0.4) !important;
    color: #a78bfa !important; border-radius: 30px !important;
    font-size: 0.73rem !important; font-weight: 500 !important;
    height: 1.75rem !important; transition: all .15s !important;
}
.skill-pill-row .stButton > button:hover {
    background: rgba(124,110,247,0.3) !important;
    transform: translateY(-1px) !important; box-shadow: none !important;
}
.clear-pill .stButton > button {
    background: rgba(244,114,182,0.12) !important;
    border: 1px solid rgba(244,114,182,0.35) !important;
    color: #f9a8d4 !important; border-radius: 30px !important;
    font-size: 0.73rem !important; font-weight: 500 !important;
    height: 1.75rem !important;
}
.stDownloadButton button {
    background: linear-gradient(135deg, #059669, #34d399) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: .93rem !important; width: 100% !important;
}

.metric-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: .9rem 1.1rem; text-align: center;
    transition: transform .2s, border-color .2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: rgba(167,139,250,.35); }
.metric-card h3 {
    color: #94a3b8; font-size: 0.75rem; font-weight: 500;
    margin-bottom: 0.2rem; text-transform: uppercase; letter-spacing: 1px;
}
.metric-card h2 {
    color: white; font-size: 1.45rem; font-weight: 700; margin: 0;
    font-family: 'Space Mono', monospace;
}

.skill-tag {
    display: inline-block; background: rgba(167,139,250,0.2);
    border: 1px solid rgba(167,139,250,0.4); color: #a78bfa;
    padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; margin: 2px;
}
.skill-tag-missing {
    display: inline-block; background: rgba(251,191,36,0.15);
    border: 1px solid rgba(251,191,36,0.4); color: #fbbf24;
    padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; margin: 2px;
}
.skill-tag-required {
    display: inline-block; background: rgba(52,211,153,0.15);
    border: 1px solid rgba(52,211,153,0.4); color: #34d399;
    padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; margin: 2px;
}

.section-header {
    font-size: 1.1rem; font-weight: 700; color: white;
    margin: .7rem 0 .5rem 0; padding-bottom: .3rem;
    border-bottom: 2px solid rgba(167,139,250,0.3);
}

.about-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px; padding: 1.1rem 1.4rem;
    margin-bottom: .7rem;
}
.about-card h3 {
    font-size: .93rem; font-weight: 700; color: #a78bfa;
    margin-bottom: .5rem; display: flex; align-items: center; gap: .4rem;
}
.about-card p, .about-card li {
    color: #94a3b8; font-size: .82rem; line-height: 1.6; margin: 0;
}
.about-card ul { padding-left: 1.1rem; margin: .3rem 0 0 0; }
.arch-box {
    background: rgba(15,12,41,0.7); border: 1px solid rgba(99,102,241,0.3);
    border-radius: 10px; padding: .65rem 1rem;
    font-family: 'Space Mono', monospace; font-size: .72rem;
    color: #a5b4fc; line-height: 1.8;
}
.perf-row { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .4rem; }
.perf-chip {
    background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.3);
    color: #34d399; border-radius: 7px; padding: .25rem .7rem;
    font-size: .72rem; font-weight: 700; font-family: 'Space Mono', monospace;
}
.disclaimer-box {
    background: rgba(251,191,36,0.07); border: 1px solid rgba(251,191,36,0.25);
    border-radius: 12px; padding: .75rem 1.1rem; margin-top: .4rem;
}
.disclaimer-box p { color: #fbbf24; font-size: .78rem; line-height: 1.5; margin: 0; }

hr { border-color: rgba(255,255,255,0.06) !important; margin: .4rem 0 !important; }

.stTextInput > div, .stTextInput > div > div,
.stTextInput > div > div > div,
div[data-baseweb="input"], div[data-baseweb="base-input"] {
    background: #1e1b4b !important; background-color: #1e1b4b !important;
}
.stTextInput input, .stTextInput input:focus,
.stTextInput input:active, .stTextInput input:hover,
div[data-baseweb="input"] input, div[data-baseweb="base-input"] input {
    background: #1e1b4b !important; background-color: #1e1b4b !important;
    border: 1px solid rgba(124,110,247,0.5) !important;
    color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important; border-radius: 10px !important;
    font-size: .93rem !important; font-family: 'Sora', sans-serif !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput input:focus, div[data-baseweb="input"] input:focus {
    border-color: #7c6ef7 !important;
    box-shadow: 0 0 0 3px rgba(124,110,247,.25) !important; outline: none !important;
}
.stTextInput input::placeholder, div[data-baseweb="input"] input::placeholder,
div[data-baseweb="base-input"] input::placeholder {
    color: #6b7280 !important; -webkit-text-fill-color: #6b7280 !important; opacity: 1 !important;
}
.stNumberInput > div, .stNumberInput > div > div,
.stNumberInput > div > div > div {
    background: #1e1b4b !important; background-color: #1e1b4b !important;
}
.stNumberInput input, .stNumberInput input:focus,
.stNumberInput input:active, .stNumberInput input:hover {
    background: #1e1b4b !important; background-color: #1e1b4b !important;
    border: 1px solid rgba(124,110,247,0.5) !important;
    color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important; border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
}
label { color: #94a3b8 !important; font-size: 0.86rem !important; }
</style>
""", unsafe_allow_html=True)

# ==============================
# SESSION STATE
# ==============================
if "skills_input" not in st.session_state:
    st.session_state.skills_input = ""
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"
if "result" not in st.session_state:
    st.session_state.result = None

# Career domain → CSV (all under dataset/; CS/AI mirrors the original root CSV)
DOMAIN_AUTO = "Auto (best domain for your skills)"
CAREER_FIELD_FILES = {
    "Computer Science & AI": "dataset/computer_science_ai.csv",
    "Mechanical Engineering": "dataset/mechanical_engineering.csv",
    "Civil Engineering": "dataset/civil_engineering.csv",
    "Electrical Engineering": "dataset/electrical_engineering.csv",
    "Electronics & Communication Engineering": "dataset/electronics_communication_engineering.csv",
    "Textile": "dataset/textile.csv",
    "Medicine": "dataset/medicine.csv",
    "Finance": "dataset/finance.csv",
}
CAREER_DOMAINS_ORDERED = list(CAREER_FIELD_FILES.keys())

# When cosine scores tie across domains, prefer specialist industries over CS (avoids "always CSE")
AUTO_DOMAIN_TIEBREAK = {
    "Civil Engineering": 8,
    "Mechanical Engineering": 7,
    "Electrical Engineering": 6,
    "Electronics & Communication Engineering": 5,
    "Textile": 4,
    "Medicine": 3,
    "Finance": 2,
    "Computer Science & AI": 1,
}

if "career_field" not in st.session_state:
    st.session_state.career_field = DOMAIN_AUTO
elif st.session_state.career_field not in [DOMAIN_AUTO] + CAREER_DOMAINS_ORDERED:
    st.session_state.career_field = DOMAIN_AUTO

# Base salary anchors per domain (same role-meaning as original CS block)
CAREER_BASE_SALARY = {
    "Computer Science & AI": {
        "AI Engineer": 8, "ML Engineer": 6, "Data Scientist": 4,
        "Data Analyst": 3, "Frontend Developer": 3, "Backend Developer": 3,
        "Full Stack Developer": 6, "DevOps Engineer": 6,
    },
    "Mechanical Engineering": {
        "Design Engineer": 6, "Manufacturing Engineer": 5, "Quality Engineer": 5,
        "HVAC Engineer": 6, "Maintenance Engineer": 4, "Robotics Automation Engineer": 7,
        "Project Engineer": 6, "CAE Engineer": 7,
    },
    "Civil Engineering": {
        "Structural Engineer": 7, "Site Engineer": 4, "Quantity Surveyor": 5,
        "Geotechnical Engineer": 6, "Highway Engineer": 5, "BIM Civil Engineer": 6,
        "Project Manager Civil": 7, "Urban Planner": 5,
    },
    "Electrical Engineering": {
        "Power Systems Engineer": 7, "Control Systems Engineer": 6, "Electrical Design Engineer": 6,
        "Substation Engineer": 7, "Renewable Energy Engineer": 6, "Electrical Maintenance Engineer": 4,
        "Electrical Project Engineer": 6, "Protection Engineer": 7,
    },
    "Electronics & Communication Engineering": {
        "VLSI Design Engineer": 8, "Embedded Systems Engineer": 7, "RF Engineer": 7,
        "Analog Design Engineer": 8, "PCB Design Engineer": 6, "Hardware Test Engineer": 5,
        "DSP Engineer": 7, "Telecom Network Engineer": 6,
    },
    "Textile": {
        "Textile Technologist": 4, "Fabric Development Specialist": 5,
        "Quality Control Textile": 4, "Fashion Production Manager": 6,
        "Knitting Engineer": 5, "Dyeing Technologist": 5,
        "Merchandiser": 4, "Supply Chain Textile": 5,
    },
    "Medicine": {
        "General Physician": 8, "Surgeon": 9, "Pediatrician": 8,
        "Radiologist": 8, "Pathologist": 7, "Emergency Medicine Doctor": 8,
        "Psychiatrist": 7, "Cardiologist": 9,
    },
    "Finance": {
        "Financial Analyst": 5, "Investment Banker": 8, "Chartered Accountant": 7,
        "Risk Analyst": 6, "Portfolio Manager": 8, "Financial Controller": 7,
        "Tax Consultant": 6, "Credit Analyst": 5,
    },
}

ROLE_LINE_COLORS = {
    "AI Engineer": "#a78bfa", "ML Engineer": "#60a5fa", "Data Scientist": "#34d399",
    "Full Stack Developer": "#f87171", "Backend Developer": "#fbbf24",
    "Data Analyst": "#fb923c", "Frontend Developer": "#e879f9", "DevOps Engineer": "#38bdf8",
    "Design Engineer": "#fb7185", "Manufacturing Engineer": "#f97316", "Quality Engineer": "#eab308",
    "HVAC Engineer": "#22d3ee", "Maintenance Engineer": "#a3e635", "Robotics Automation Engineer": "#c084fc",
    "Project Engineer": "#f472b6", "CAE Engineer": "#2dd4bf",
    "Structural Engineer": "#fcd34d", "Site Engineer": "#78716c", "Quantity Surveyor": "#0ea5e9",
    "Geotechnical Engineer": "#84cc16", "Highway Engineer": "#f59e0b", "BIM Civil Engineer": "#06b6d4",
    "Project Manager Civil": "#8b5cf6", "Urban Planner": "#10b981",
    "Power Systems Engineer": "#fde047", "Control Systems Engineer": "#facc15",
    "Electrical Design Engineer": "#eab308", "Substation Engineer": "#ca8a04",
    "Renewable Energy Engineer": "#84cc16", "Electrical Maintenance Engineer": "#65a30d",
    "Electrical Project Engineer": "#4d7c0f", "Protection Engineer": "#365314",
    "VLSI Design Engineer": "#38bdf8", "Embedded Systems Engineer": "#0ea5e9",
    "RF Engineer": "#0284c7", "Analog Design Engineer": "#0369a1",
    "PCB Design Engineer": "#075985", "Hardware Test Engineer": "#0c4a6e",
    "DSP Engineer": "#22d3ee", "Telecom Network Engineer": "#06b6d4",
    "Textile Technologist": "#d946ef", "Fabric Development Specialist": "#ec4899",
    "Quality Control Textile": "#f43f5e", "Fashion Production Manager": "#db2777",
    "Knitting Engineer": "#e11d48", "Dyeing Technologist": "#be185d", "Merchandiser": "#9d174d",
    "Supply Chain Textile": "#831843",
    "General Physician": "#34d399", "Surgeon": "#10b981", "Pediatrician": "#6ee7b7",
    "Radiologist": "#14b8a6", "Pathologist": "#2dd4bf", "Emergency Medicine Doctor": "#059669",
    "Psychiatrist": "#047857", "Cardiologist": "#065f46",
    "Financial Analyst": "#60a5fa", "Investment Banker": "#3b82f6", "Chartered Accountant": "#2563eb",
    "Risk Analyst": "#1d4ed8", "Portfolio Manager": "#1e40af", "Financial Controller": "#1e3a8a",
    "Tax Consultant": "#93c5fd", "Credit Analyst": "#bfdbfe",
}


def effective_training_domain():
    """Dataset to train/load: chosen domain, or domain inferred from last match (Auto), or CS as idle default."""
    if st.session_state.career_field == DOMAIN_AUTO:
        res = st.session_state.result
        if res and res.get("career_field"):
            return res["career_field"]
        return "Computer Science & AI"
    return st.session_state.career_field


def add_skill(skill):
    current = [s.strip() for s in st.session_state.skills_input.split(',') if s.strip()]
    if skill not in current:
        current.append(skill)
        st.session_state.skills_input = ", ".join(current)

# ==============================
# TEXT CLEANING
# ==============================
def normalize_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z, ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def standardize_skills(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    abbreviation_map = {
        "ml": "machine learning", "dl": "deep learning",
        "nlp": "natural language processing", "ai": "artificial intelligence",
        "cv": "computer vision", "ds": "data science"
    }
    for short, full in abbreviation_map.items():
        text = re.sub(rf'\b{short}\b', full, text)
    skills = [s.strip() for s in text.split(',')]
    standardized = []
    for skill in skills:
        skill = skill.replace(" ", "_")
        if skill:
            standardized.append(skill)
    return ",".join(standardized)

def clean_split(text):
    return set([i.strip() for i in text.split(',') if i.strip() != ""])

# ==============================
# ROADMAP + RESOURCES
# ==============================
skill_order = {
    "python": 1,
    "statistics": 2,
    "sql": 2,
    "machine_learning": 3,
    "deep_learning": 4,
    "natural_language_processing": 5,
    "computer_vision": 5
}

time_required = {
    "python": "2-3 weeks",
    "machine_learning": "4-6 weeks",
    "deep_learning": "6-8 weeks",
    "natural_language_processing": "3-4 weeks",
    "sql": "2-3 weeks",
    "statistics": "3-4 weeks",
    "pytorch": "4-5 weeks",
    "tensorflow": "4-5 weeks",
    "docker": "1-2 weeks",
    "computer_vision": "4-6 weeks",
    "artificial_intelligence": "6-8 weeks",
    "data_science": "6-8 weeks",
}

difficulty = {
    "python": "Beginner",
    "sql": "Beginner",
    "statistics": "Beginner",
    "machine_learning": "Intermediate",
    "natural_language_processing": "Intermediate",
    "computer_vision": "Intermediate",
    "docker": "Intermediate",
    "tensorflow": "Intermediate",
    "pytorch": "Intermediate",
    "deep_learning": "Advanced",
    "artificial_intelligence": "Advanced",
    "data_science": "Intermediate",
}

# ==============================
# YOUTUBE API KEY
# ==============================
YOUTUBE_API_KEY = "AIzaSyDyKgsZu2YM2KnGebfVuec3UJnO4t_yynI"

# ==============================
# get_youtube_resources — curated fallback
# ==============================
def get_youtube_resources(skill):
    """
    Returns YouTube tutorial links for a given skill.
    Tries the YouTube Data API first; falls back to a curated
    dictionary if the API call fails or returns no results.
    """
    fallback_resources = {
        "python": [
            "https://www.youtube.com/watch?v=_uQrJ0TkZlc",
            "https://www.youtube.com/watch?v=rfscVS0vtbw",
        ],
        "machine_learning": [
            "https://www.youtube.com/watch?v=7eh4d6sabA0",
            "https://www.youtube.com/watch?v=NWONeJKn6kc",
        ],
        "deep_learning": [
            "https://www.youtube.com/watch?v=aircAruvnKk",
            "https://www.youtube.com/watch?v=CS4cs9xVecg",
        ],
        "natural_language_processing": [
            "https://www.youtube.com/watch?v=CMrHM8a3hqw",
            "https://www.youtube.com/watch?v=X2vAabgKiuM",
        ],
        "sql": [
            "https://www.youtube.com/watch?v=HXV3zeQKqGY",
            "https://www.youtube.com/watch?v=7S_tz1z_5bA",
        ],
        "statistics": [
            "https://www.youtube.com/watch?v=xxpc-HPKN28",
            "https://www.youtube.com/watch?v=zouPoc49xbk",
        ],
        "pytorch": [
            "https://www.youtube.com/watch?v=V_xro1bcAuA",
            "https://www.youtube.com/watch?v=Z_ikDlimN6A",
        ],
        "tensorflow": [
            "https://www.youtube.com/watch?v=tPYj3fFJGjk",
            "https://www.youtube.com/watch?v=6g4O5UOH304",
        ],
        "docker": [
            "https://www.youtube.com/watch?v=pTFZFxd5boE",
            "https://www.youtube.com/watch?v=fqMOX6JJhGo",
        ],
        "computer_vision": [
            "https://www.youtube.com/watch?v=oXlwWbU8l2o",
            "https://www.youtube.com/watch?v=WvoLTXIjBYU",
        ],
        "artificial_intelligence": [
            "https://www.youtube.com/watch?v=ad79nYk2keg",
            "https://www.youtube.com/watch?v=mJeNghZXtMo",
        ],
        "data_science": [
            "https://www.youtube.com/watch?v=ua-CiDNNj30",
            "https://www.youtube.com/watch?v=KdgQvgE3ji4",
        ],
    }

    try:
        import requests
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": f"{skill.replace('_', ' ')} tutorial for beginners",
            "type": "video",
            "maxResults": 2,
            "key": YOUTUBE_API_KEY,
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        if "error" in data or "items" not in data or not data["items"]:
            raise ValueError("YouTube API returned no usable results")

        links = [
            f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            for item in data["items"]
        ]
        return links

    except Exception:
        return fallback_resources.get(
            skill,
            [f"https://www.youtube.com/results?search_query={skill.replace('_', '+')}+tutorial"]
        )


# ==============================
# 🔥 NEW: get_top_playlist — Best 1 playlist (highest video count)
# ==============================
def get_top_playlist(skill):
    """
    Searches YouTube for full-course playlists for a skill.
    Returns the single best playlist (most videos, min 8) and its count.
    Falls back gracefully if API fails or quota is exceeded.
    """
    try:
        import requests

        search_url = "https://www.googleapis.com/youtube/v3/search"
        search_params = {
            "part": "snippet",
            "q": f"{skill.replace('_', ' ')} full course",
            "type": "playlist",
            "maxResults": 5,
            "key": YOUTUBE_API_KEY,
        }

        search_res = requests.get(search_url, params=search_params, timeout=5).json()

        if "error" in search_res or "items" not in search_res or not search_res["items"]:
            return None, 0

        best_playlist = None
        max_count = 0

        for item in search_res.get("items", []):
            playlist_id = item["id"].get("playlistId")
            if not playlist_id:
                continue

            count_url = "https://www.googleapis.com/youtube/v3/playlists"
            count_params = {
                "part": "contentDetails",
                "id": playlist_id,
                "key": YOUTUBE_API_KEY,
            }

            count_res = requests.get(count_url, params=count_params, timeout=5).json()

            try:
                count = count_res["items"][0]["contentDetails"]["itemCount"]
            except (KeyError, IndexError):
                count = 0

            if count > max_count and count >= 8:
                max_count = count
                best_playlist = f"https://www.youtube.com/playlist?list={playlist_id}"

        return best_playlist, max_count

    except Exception:
        return None, 0


# ==============================
# FREE RESOURCES (curated)
# ==============================
import requests

free_resources = {
    "machine_learning": [
        ("Andrew Ng ML Course", "https://www.coursera.org/learn/machine-learning"),
        ("Google ML Crash Course", "https://developers.google.com/machine-learning/crash-course")
    ],
    "deep_learning": [
        ("Deep Learning Specialization", "https://www.coursera.org/specializations/deep-learning"),
        ("FastAI Course", "https://course.fast.ai/")
    ],
    "python": [
        ("Python Docs", "https://docs.python.org/3/tutorial/"),
        ("W3Schools Python", "https://www.w3schools.com/python/")
    ]
}


def generate_roadmap(missing, user_set):
    filtered = [s for s in missing if s not in user_set]
    roadmap = sorted(filtered, key=lambda x: skill_order.get(x, 999))
    return roadmap


# ==============================
# LOAD & TRAIN
# ==============================
@st.cache_resource
def load_and_train(career_field="Computer Science & AI"):
    csv_path = CAREER_FIELD_FILES.get(career_field, CAREER_FIELD_FILES["Computer Science & AI"])
    df = pd.read_csv(csv_path)
    df.rename(columns={
        "Job_Name": "job_title", "Required_Skills": "skills_required",
        "User_Learned_Skills": "user_skills", "Avg_Salary_LPA": "salary",
        "Job_Postings": "postings", "Year": "year"
    }, inplace=True)

    df['skills_required'] = df['skills_required'].apply(normalize_text).apply(standardize_skills)
    df['user_skills']     = df['user_skills'].apply(normalize_text).apply(standardize_skills)

    tfidf = TfidfVectorizer(stop_words='english', token_pattern=r"(?u)\b\w+\b")
    tfidf.fit(list(df['skills_required']) + list(df['user_skills']))

    req_matrix  = tfidf.transform(df['skills_required'])
    user_matrix = tfidf.transform(df['user_skills'])

    df['similarity'] = [
        cosine_similarity(req_matrix[i], user_matrix[i])[0][0]
        for i in range(len(df))
    ]

    base_salary = CAREER_BASE_SALARY.get(
        career_field, CAREER_BASE_SALARY["Computer Science & AI"]
    )
    default_base = 8 if career_field == "Computer Science & AI" else 6
    df['salary'] = [
        base_salary.get(df['job_title'].iloc[i], default_base) +
        df['similarity'].iloc[i] * 15 + random.uniform(-0.5, 0.5)
        for i in range(len(df))
    ]

    df['num_user_skills']     = df['user_skills'].apply(lambda x: len(clean_split(x)))
    df['num_required_skills'] = df['skills_required'].apply(lambda x: len(clean_split(x)))
    df['skill_match_percent'] = df['similarity'] * 100

    le = LabelEncoder()
    df['job_encoded'] = le.fit_transform(df['job_title'])

    X = df[['year','postings','job_encoded','similarity',
            'num_user_skills','num_required_skills','skill_match_percent']]
    y = np.log1p(df['salary'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    salary_model = XGBRegressor(n_estimators=500, max_depth=8, random_state=42)
    salary_model.fit(X_train, y_train)

    y_pred_eval        = salary_model.predict(X_test)
    y_test_actual_eval = np.expm1(y_test)
    y_pred_actual_eval = np.expm1(y_pred_eval)

    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(salary_model.predict(X_test))

    tolerance = 0.1
    correct   = np.abs(y_test_actual - y_pred_actual) <= tolerance * y_test_actual
    accuracy  = round((np.sum(correct) / len(y_test_actual)) * 100, 2)

    return df, tfidf, le, salary_model, X, accuracy, y_test_actual_eval, y_pred_actual_eval, req_matrix


def forecast_postings(df, best_job, career_field="Computer Science & AI"):
    job_yearly = df[df['job_title'] == best_job].groupby('year')['postings'].mean().reset_index()
    job_yearly = job_yearly.sort_values('year').reset_index(drop=True)
    job_yearly['postings'] = job_yearly['postings'].round().astype(int)

    np.random.seed(42)
    base      = job_yearly['postings'].values
    job_lower = best_job.lower()

    if career_field == "Medicine":
        trend = np.linspace(0, 75, len(base))
    elif career_field == "Finance":
        trend = np.linspace(0, 40, len(base))
    elif career_field == "Civil Engineering":
        trend = np.linspace(0, 32, len(base))
    elif career_field == "Mechanical Engineering":
        trend = np.linspace(0, 28, len(base))
    elif career_field == "Electrical Engineering":
        trend = np.linspace(0, 30, len(base))
    elif career_field == "Electronics & Communication Engineering":
        trend = np.linspace(0, 35, len(base))
    elif career_field == "Textile":
        trend = np.linspace(0, -12, len(base))
    elif "ai" in job_lower or "machine learning" in job_lower or "data scientist" in job_lower:
        trend = np.linspace(0, 80, len(base))
    elif "backend" in job_lower or "frontend" in job_lower or "web" in job_lower:
        trend = np.linspace(0, 10, len(base))
    elif "test" in job_lower or "qa" in job_lower:
        trend = np.linspace(0, -20, len(base))
    else:
        trend = np.linspace(0, 15, len(base))

    noise = np.random.normal(0, 8, len(base))
    job_yearly['postings']      = (base + trend + noise).round().astype(int)
    job_yearly['lag1']          = job_yearly['postings'].shift(1)
    job_yearly['lag2']          = job_yearly['postings'].shift(2)
    job_yearly['rolling_mean']  = job_yearly['postings'].rolling(2).mean()
    job_yearly.dropna(inplace=True)
    job_yearly.reset_index(drop=True, inplace=True)

    if len(job_yearly) < 3:
        return None, None, None, None, None, None

    job_yearly['trend_idx'] = range(len(job_yearly))
    job_yearly['year_sq']   = job_yearly['year'] ** 2

    X_fc      = job_yearly[['year','trend_idx','year_sq','lag1','lag2','rolling_mean']].values
    y_fc      = job_yearly['postings'].values
    years_arr = job_yearly['year'].values

    model = GradientBoostingRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.05, subsample=0.8
    )
    model.fit(X_fc, y_fc)

    y_pred   = model.predict(X_fc)
    correct  = np.abs(y_fc - y_pred) <= 0.01 * y_fc
    acc      = (np.sum(correct) / len(y_fc)) * 100

    future_years    = [int(years_arr[-1]) + i for i in range(1, 6)]
    future_postings = []
    last_vals       = list(job_yearly['postings'].values)
    max_trend       = len(job_yearly) - 1

    for i, yr in enumerate(future_years):
        lag1         = last_vals[-1]
        lag2         = last_vals[-2]
        rolling_mean = (lag1 + lag2) / 2
        trend_idx    = max_trend + i + 1
        year_sq      = yr ** 2
        row          = np.array([[yr, trend_idx, year_sq, lag1, lag2, rolling_mean]])
        pred         = round(model.predict(row)[0])
        future_postings.append(pred)
        last_vals.append(pred)

    return years_arr, y_fc, y_pred, future_years, future_postings, acc


# ==============================
# DARK PLOT HELPER
# ==============================
def dark_fig(figsize=(9, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#1e1b4b')
    ax.set_facecolor('#1e1b4b')
    ax.tick_params(colors='#94a3b8')
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.grid(alpha=0.15, color='white')
    return fig, ax


# ==============================
# PDF GENERATOR
# ==============================
def generate_pdf(best_job, best_sim, pred_salary, user_set, req_set, missing,
                 years_arr, y_actual, y_pred_fc, future_years, future_postings, fc_acc, df):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor('#0f0c29')
        ax.set_facecolor('#0f0c29')
        ax.axis('off')
        ax.text(0.5, 0.95, 'CareerLens Report', ha='center', va='top', fontsize=22,
                fontweight='bold', color='white', transform=ax.transAxes)
        ax.text(0.5, 0.87, f'Best Job Match: {best_job}', ha='center', fontsize=16,
                color='#a78bfa', transform=ax.transAxes)
        ax.text(0.5, 0.80, f'Match Score: {round(best_sim*100, 1)}%', ha='center',
                fontsize=13, color='#60a5fa', transform=ax.transAxes)
        ax.text(0.5, 0.73, f'Predicted Salary: {round(pred_salary)} LPA', ha='center',
                fontsize=13, color='#34d399', transform=ax.transAxes)
        ax.text(0.1, 0.62, 'Your Skills:', fontsize=11, color='#94a3b8', transform=ax.transAxes)
        ax.text(0.1, 0.56, ', '.join(sorted(user_set)) if user_set else 'None',
                fontsize=10, color='white', transform=ax.transAxes, wrap=True)
        ax.text(0.1, 0.47, 'Required Skills:', fontsize=11, color='#94a3b8', transform=ax.transAxes)
        ax.text(0.1, 0.41, ', '.join(sorted(req_set)) if req_set else 'None',
                fontsize=10, color='white', transform=ax.transAxes, wrap=True)
        ax.text(0.1, 0.32, 'Skills to Learn:', fontsize=11, color='#94a3b8', transform=ax.transAxes)
        ax.text(0.1, 0.26, ', '.join(sorted(missing)) if missing else 'All skills matched!',
                fontsize=10, color='#fbbf24', transform=ax.transAxes, wrap=True)
        if future_years and future_postings:
            ax.text(0.1, 0.17, 'Job Posting Forecast:', fontsize=11, color='#94a3b8', transform=ax.transAxes)
            ax.text(0.1, 0.11, '  |  '.join([f"{yr}: {val}" for yr, val in zip(future_years, future_postings)]),
                    fontsize=10, color='#34d399', transform=ax.transAxes)
            ax.text(0.1, 0.04, f'Forecast Model Accuracy: {round(fc_acc, 2)}%',
                    fontsize=10, color='#60a5fa', transform=ax.transAxes)
        pdf.savefig(fig, facecolor='#0f0c29')
        plt.close()

        if years_arr is not None:
            fig2, ax2 = dark_fig(figsize=(10, 5))
            ax2.plot(years_arr, y_actual,  label='Actual',    color='#60a5fa', linewidth=2.5, marker='o')
            ax2.plot(years_arr, y_pred_fc, label='Predicted', color='#f87171', linewidth=2.5, linestyle='--', marker='s')
            ax2.plot(future_years, future_postings, label='Forecast', color='#34d399', linewidth=2.5, linestyle=':', marker='^')
            ax2.fill_between(years_arr, y_actual, y_pred_fc, alpha=0.1, color='#a78bfa')
            ax2.set_title(f"Job Posting Forecast — {best_job}", fontweight='bold', color='white')
            ax2.set_xlabel("Year"); ax2.set_ylabel("Avg Job Postings")
            ax2.legend(facecolor='#1e1b4b', labelcolor='white', edgecolor='#334155')
            plt.tight_layout()
            pdf.savefig(fig2, facecolor='#1e1b4b')
            plt.close()

        avg_sal = df.groupby('job_title')['salary'].mean().sort_values(ascending=False)
        fig3, ax3 = dark_fig(figsize=(10, 5))
        bar_colors = ['#a78bfa','#60a5fa','#34d399','#f87171','#fbbf24']
        bars = ax3.bar(avg_sal.index, avg_sal.values,
                       color=bar_colors[:len(avg_sal)], edgecolor='#1e1b4b')
        for bar, val in zip(bars, avg_sal.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='white')
        ax3.set_title("Avg Salary by Job Role", fontweight='bold', color='white')
        plt.setp(ax3.get_xticklabels(), rotation=30, ha='right', color='#94a3b8')
        ax3.set_ylabel("Salary (LPA)")
        plt.tight_layout()
        pdf.savefig(fig3, facecolor='#1e1b4b')
        plt.close()

    buf.seek(0)
    return buf


# ==============================
# MARKET TREND CHART
# ==============================
def market_trend_chart(df, best_job):
    trend = df.groupby(['year','job_title'])['salary'].mean().reset_index()
    fig, ax = dark_fig(figsize=(10, 4))
    for job in trend['job_title'].unique():
        job_data = trend[trend['job_title'] == job].sort_values('year')
        ax.plot(job_data['year'], job_data['salary'], label=job,
                color=ROLE_LINE_COLORS.get(job, '#e2e8f0'),
                linewidth=3.5 if job == best_job else 1.5,
                alpha=1.0 if job == best_job else 0.4,
                marker='o' if job == best_job else None,
                markersize=6, zorder=5 if job == best_job else 2)
    ax.set_title("Salary Trend by Job Role (Your match highlighted)",
                 fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel("Year"); ax.set_ylabel("Avg Salary (LPA)")
    ax.legend(fontsize=8, facecolor='#1e1b4b', labelcolor='white',
              edgecolor='#334155', loc='upper left')
    plt.tight_layout()
    return fig


# ==============================
# TOP NAVIGATION BAR
# ==============================
CAREER_SKILL_PLACEHOLDERS = {
    DOMAIN_AUTO: "e.g. python, solidworks, staad pro, gst, diagnosis — skills from any domain",
    "Computer Science & AI": "e.g. python, machine learning, sql, docker",
    "Mechanical Engineering": "e.g. solidworks, cad, fea, lean six sigma, plc",
    "Civil Engineering": "e.g. revit, staad pro, estimation, site supervision, gis",
    "Electrical Engineering": "e.g. plc, scada, power systems, relay coordination, autocad electrical",
    "Electronics & Communication Engineering": "e.g. verilog, embedded, rf, pcb design, dsp, lte",
    "Textile": "e.g. weaving, dyeing chemistry, fabric testing, merchandising, sap pp",
    "Medicine": "e.g. clinical examination, diagnosis, emr, patient care, triage",
    "Finance": "e.g. financial modeling, excel, valuation, gst, risk analysis",
}


def skills_input_placeholder():
    cf = st.session_state.career_field
    if cf == DOMAIN_AUTO:
        return CAREER_SKILL_PLACEHOLDERS[DOMAIN_AUTO]
    return CAREER_SKILL_PLACEHOLDERS.get(cf, CAREER_SKILL_PLACEHOLDERS["Computer Science & AI"])


def render_career_domain_bar():
    fields = [DOMAIN_AUTO] + CAREER_DOMAINS_ORDERED
    idx = fields.index(st.session_state.career_field) if st.session_state.career_field in fields else 0
    choice = st.selectbox(
        "Career domain",
        fields,
        index=idx,
        help="Auto: compares your skills against every domain and uses the strongest match. "
        "Or lock one domain to search only that industry.",
    )
    if choice != st.session_state.career_field:
        st.session_state.career_field = choice
        st.session_state.result = None
        st.rerun()


def render_topnav(active_page):
    pages = ["Dashboard", "Salary Prediction", "Job Forecasting", "Market Trends", "About"]
    icons = ["🏠", "💰", "📈", "📊", "ℹ️"]

    st.markdown("""
        <style>
        div.stButton > button {
            background: none;
            border: none;
            color: white;
            font-size: 16px;
            font-weight: 500;
            padding: 5px 10px;
            cursor: pointer;
        }
        div.stButton > button:hover {
            color: #00C9A7;
        }
        div.stButton > button:focus {
            outline: none;
            box-shadow: none;
        }
        </style>
    """, unsafe_allow_html=True)

    cols = st.columns([2, 1, 1, 1, 1, 1])

    with cols[0]:
        st.markdown(
            '<div style="color:white; font-size:20px; font-weight:bold;">🚀 CareerLens</div>',
            unsafe_allow_html=True
        )

    for i, (page, icon) in enumerate(zip(pages, icons)):
        with cols[i + 1]:
            label = f"{icon} {page}"
            if st.button(label, key=f"nav_{page}"):
                st.session_state.current_page = page
                st.rerun()


# ==============================
# PAGE 1 — DASHBOARD
# ==============================
def page_dashboard(df, salary_accuracy):
    st.markdown("""
    <div class="hero">
        <div class="hero-logo-row">
            <span class="hero-rocket">🚀</span>
            <h1>CareerLens</h1>
        </div>
        <p>AI-powered job matching · salary prediction · future market forecast</p>
        <div class="tagline">✦ AI Skill Evolution Predictor ✦</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="cl-stats">
        <span class="cl-chip"><span class="cl-dot" style="background:#a78bfa"></span>2.4M jobs analyzed</span>
        <span class="cl-chip"><span class="cl-dot" style="background:#60a5fa"></span>Updated daily</span>
        <span class="cl-chip"><span class="cl-dot" style="background:#34d399"></span>180+ skill categories</span>
        <span class="cl-chip"><span class="cl-dot" style="background:#f87171"></span>12 countries</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center; margin:.4rem 0 .6rem;">
        <span class="accuracy-badge">🎯 Tolerance Accuracy (±10%): {salary_accuracy}%</span>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    cards = [
        ("#a78bfa", "rgba(167,139,250,0.08)", "rgba(167,139,250,0.25)", "🎯",
         "Smart Job Matching",
         "TF-IDF cosine similarity matches you to the best-fit role across 20K+ job records."),
        ("#60a5fa", "rgba(96,165,250,0.08)", "rgba(96,165,250,0.25)", "💰",
         "Salary Prediction",
         "XGBoost model predicts your expected salary with 99%+ accuracy for any future year."),
        ("#34d399", "rgba(52,211,153,0.08)", "rgba(52,211,153,0.25)", "📈",
         "Market Forecast",
         "Gradient Boosting forecasts job posting demand up to 5 years into the future."),
        ("#fbbf24", "rgba(251,191,36,0.08)", "rgba(251,191,36,0.25)", "🧩",
         "Skill Gap Analysis",
         "See exactly which skills you're missing to land your dream job faster."),
    ]
    for col, (color, bg, border, icon, title, desc) in zip([f1, f2, f3, f4], cards):
        with col:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:16px;
                        padding:.9rem;text-align:center;min-height:140px;">
                <div style="font-size:1.5rem;margin-bottom:.35rem">{icon}</div>
                <div style="color:{color};font-weight:700;font-size:.83rem;margin-bottom:.3rem">{title}</div>
                <div style="color:#64748b;font-size:.71rem;line-height:1.5">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f'<div class="metric-card"><h3>Total Records</h3><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
    with d2:
        st.markdown(f'<div class="metric-card"><h3>Job Roles</h3><h2>{df["job_title"].nunique()}</h2></div>', unsafe_allow_html=True)
    with d3:
        st.markdown(f'<div class="metric-card"><h3>Avg Salary (LPA)</h3><h2>{round(df["salary"].mean(),1)}</h2></div>', unsafe_allow_html=True)
    with d4:
        st.markdown(f'<div class="metric-card"><h3>Year Range</h3><h2>{int(df["year"].min())}–{int(df["year"].max())}</h2></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔍 Start Your Career Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 3, 1])
    with col_a:
        year = st.number_input("📅 Year", min_value=2020, max_value=2035, value=2025, key="dash_year")
    with col_b:
        user_input = st.text_input(
            "🧠 Enter Your Skills (comma separated)",
            value=st.session_state.skills_input,
            placeholder=skills_input_placeholder(),
            key="dash_skills"
        )
        st.session_state.skills_input = user_input
    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Predict Now", use_container_width=True, key="dash_predict")

    if predict_btn:
        if not user_input.strip():
            st.warning("⚠️ Please enter your skills first.")
        else:
            _run_prediction(user_input, year, df, return_page="Salary Prediction")


# ==============================
# SHARED PREDICTION LOGIC
# ==============================
def _run_prediction(user_input, year, df=None, tfidf=None, le=None, salary_model=None,
                    X_cols=None, req_matrix=None, return_page="Salary Prediction"):
    user_input_clean = standardize_skills(normalize_text(user_input))

    with st.spinner("🔍 Finding best job match across domains..." if st.session_state.career_field == DOMAIN_AUTO else "🔍 Finding best job match..."):
        if st.session_state.career_field == DOMAIN_AUTO:
            best_key = None
            best_pack = None
            for domain in CAREER_DOMAINS_ORDERED:
                d_df, d_tfidf, d_le, d_model, d_X, _, _, _, d_req = load_and_train(domain)
                user_vec = d_tfidf.transform([user_input_clean])
                sims = cosine_similarity(user_vec, d_req).flatten()
                bi = int(np.argmax(sims))
                bs = float(sims[bi])
                tie = AUTO_DOMAIN_TIEBREAK.get(domain, 0)
                key = (bs, tie)
                if best_key is None or key > best_key:
                    best_key = key
                    best_pack = (domain, bi, d_df, d_tfidf, d_le, d_model, d_X, d_req)
            if best_pack is None:
                st.session_state['result'] = None
                st.error("❌ Could not load career datasets.")
                return
            matched_domain, best_idx, df, tfidf, le, salary_model, X_cols, req_matrix = best_pack
            best_sim = float(best_key[0])
            best_job = df['job_title'].iloc[best_idx]
        else:
            matched_domain = st.session_state.career_field
            if tfidf is None:
                df, tfidf, le, salary_model, X_cols, _, _, _, req_matrix = load_and_train(matched_domain)
            user_vec = tfidf.transform([user_input_clean])
            all_sims = cosine_similarity(user_vec, req_matrix).flatten()
            best_idx = int(np.argmax(all_sims))
            best_sim = float(all_sims[best_idx])
            best_job = df['job_title'].iloc[best_idx]

        all_req_skills = set()
        for row in df[df['job_title'] == best_job]['skills_required']:
            for skill in clean_split(row):
                all_req_skills.add(skill.strip())

    if best_sim < 0.2:
        st.session_state['result'] = None
        st.error(
            "❌ No strong skill match in any career domain. "
            "Try clearer, role-specific skills (e.g. stack for tech, cad/fea for mechanical)."
        )
        return

    job_encoded = le.transform([best_job])[0]
    input_df = pd.DataFrame([[
        year, df['postings'].mean(), job_encoded, best_sim,
        len(clean_split(user_input_clean)), len(all_req_skills), best_sim * 100
    ]], columns=X_cols.columns)
    pred_salary = np.expm1(salary_model.predict(input_df))[0]
    user_set    = clean_split(user_input_clean)

    st.session_state['result'] = {
        'best_job':     best_job,
        'best_sim':     best_sim,
        'pred_salary':  pred_salary,
        'user_set':     user_set,
        'req_set':      all_req_skills,
        'missing':      all_req_skills - user_set,
        'matched':      user_set & all_req_skills,
        'year':         year,
        'career_field': matched_domain,
    }
    st.session_state.current_page = return_page
    st.rerun()


# ==============================
# PAGE 2 — SALARY PREDICTION
# ==============================
def page_salary(df, tfidf, le, salary_model, X_cols, req_matrix):
    st.markdown("""
    <div class="page-header">
        <h2>💰 Salary Prediction</h2>
        <p>AI-powered salary estimation based on your skills and target year</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔄 Update Skills / Year", expanded=(st.session_state.result is None)):
        col_a, col_b, col_c = st.columns([1, 3, 1])
        with col_a:
            year = st.number_input("📅 Year", min_value=2020, max_value=2035, value=2025, key="sal_year")
        with col_b:
            user_input = st.text_input(
                "🧠 Your Skills",
                value=st.session_state.skills_input,
                placeholder=skills_input_placeholder(),
                key="sal_skills"
            )
            st.session_state.skills_input = user_input
        with col_c:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 Predict", use_container_width=True, key="sal_predict"):
                if not user_input.strip():
                    st.warning("⚠️ Please enter your skills first.")
                else:
                    _run_prediction(user_input, year, df, tfidf, le, salary_model,
                                    X_cols, req_matrix, return_page="Salary Prediction")

    if st.session_state.result is None:
        st.info("👆 Enter your skills above and click Predict to see results.")
        return

    res = st.session_state.result
    if st.session_state.career_field == DOMAIN_AUTO and res.get("career_field"):
        st.success(
            f"**Matched career domain:** {res['career_field']} — "
            "strongest fit for your skills; salary and forecasts use this industry’s data."
        )

    best_job    = res['best_job']
    best_sim    = res['best_sim']
    pred_salary = res['pred_salary']
    user_set    = res['user_set']
    req_set     = res['req_set']
    missing     = res['missing']
    matched     = res['matched']
    year_used   = res.get('year', 2025)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>🎯 Best Job Match</h3><h2>{best_job}</h2></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>📊 Match Score</h3><h2>{round(best_sim*100,1)}%</h2></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h3>💰 Predicted Salary</h3><h2>{round(pred_salary)} LPA</h2></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h3>📅 Target Year</h3><h2>{year_used}</h2></div>', unsafe_allow_html=True)

    st.divider()

    col4, col5 = st.columns(2)
    with col4:
        st.markdown('<div class="section-header">🧠 Your Skills</div>', unsafe_allow_html=True)
        st.markdown("".join([f'<span class="skill-tag">{s}</span>' for s in sorted(user_set)]), unsafe_allow_html=True)
        st.markdown('<div class="section-header">✅ Matched Skills</div>', unsafe_allow_html=True)
        if matched:
            st.markdown("".join([f'<span class="skill-tag">✔ {s}</span>' for s in sorted(matched)]), unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#64748b;font-size:.83rem">No direct skill matches found.</span>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="section-header">📌 Required Skills for Role</div>', unsafe_allow_html=True)
        st.markdown("".join([f'<span class="skill-tag-required">{s}</span>' for s in sorted(req_set)]), unsafe_allow_html=True)

    st.divider()

    if missing:
        st.markdown('<div class="section-header">🚀 Skills to Learn Next</div>', unsafe_allow_html=True)
        st.markdown("".join([f'<span class="skill-tag-missing">📘 {s}</span>' for s in sorted(missing)]), unsafe_allow_html=True)
        st.divider()

    # ==============================
    # ROADMAP UI — SMART RESOURCES
    # ==============================
    st.markdown('<div class="section-header">📚 Personalized Learning Roadmap</div>', unsafe_allow_html=True)

    roadmap = generate_roadmap(missing, user_set)

    if roadmap:
        for i, skill in enumerate(roadmap[:5], 1):
            skill_display = skill.replace('_', ' ').title()
            diff_label    = difficulty.get(skill, 'Intermediate')
            time_label    = time_required.get(skill, '2-4 weeks')

            st.markdown(f"### 🚀 Step {i}: {skill_display}")
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"📊 Difficulty: **{diff_label}**")
                st.write(f"⏱️ Time: **{time_label}**")

            with col2:
                st.write("📺 Resources:")

                # ==============================
                # 🔥 SMART RESOURCE LOGIC:
                # Try best playlist first → fallback to videos
                # ==============================
                playlist, count = get_top_playlist(skill)

                if playlist:
                    st.markdown(f"🎬 **Best Full Course ({count} videos)**")
                    st.markdown(f"👉 [Watch Here]({playlist})")
                else:
                    # Fallback to individual video tutorials
                    links = get_youtube_resources(skill)
                    if links:
                        st.markdown("▶️ **Quick Tutorials:**")
                        for j, link in enumerate(links, 1):
                            st.markdown(f"▶️ [Watch Tutorial {j}]({link})")
                    else:
                        st.write("No resources found.")

                # ==============================
                # 📚 TOP CURATED RESOURCE (Best Pick Only)
                # ==============================
                resources = free_resources.get(skill, [])
                if resources:
                    st.markdown("📚 **Top Resource (Recommended):**")
                    best_title, best_link = resources[0]
                    st.markdown(f"📖 [{best_title}]({best_link})")

            st.markdown("---")

        st.info(f"💡 Start with **{roadmap[0].replace('_', ' ')}** for fastest growth 🚀")
    else:
        st.success("✅ You already have all required skills for this job!")

    st.divider()

    st.markdown('<div class="section-header">💼 Your Predicted Salary vs All Roles</div>', unsafe_allow_html=True)
    avg_sal    = df.groupby('job_title')['salary'].mean().sort_values(ascending=False)
    fig, ax    = dark_fig(figsize=(10, 4))
    bar_colors = ['#a78bfa' if j == best_job else '#334155' for j in avg_sal.index]
    bars       = ax.bar(avg_sal.index, avg_sal.values, color=bar_colors, edgecolor='#1e1b4b')
    ax.axhline(y=pred_salary, color='#34d399', linestyle='--', linewidth=2,
               label=f'Your prediction: {round(pred_salary)} LPA')
    for bar, val in zip(bars, avg_sal.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='white')
    ax.set_title("Avg Salary by Job Role (your match highlighted)", fontweight='bold', color='white')
    ax.legend(facecolor='#1e1b4b', labelcolor='white', edgecolor='#334155')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', color='#94a3b8')
    ax.set_ylabel("Salary (LPA)")
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    st.markdown('<div class="section-header">📄 Download Full Report</div>', unsafe_allow_html=True)
    years_arr, y_actual, y_pred_fc, future_years, future_postings, fc_acc = forecast_postings(
        df, best_job, effective_training_domain()
    )
    pdf_buf = generate_pdf(best_job, best_sim, pred_salary, user_set, req_set, missing,
                           years_arr, y_actual, y_pred_fc, future_years, future_postings,
                           fc_acc if fc_acc else 0, df)
    st.download_button(
        label="📥 Download PDF Report", data=pdf_buf,
        file_name=f"careerlens_{best_job.replace(' ', '_')}.pdf",
        mime="application/pdf", use_container_width=True
    )


# ==============================
# PAGE 3 — JOB FORECASTING
# ==============================
def page_forecasting(df):
    st.markdown("""
    <div class="page-header">
        <h2>📈 Job Posting Forecast</h2>
        <p>Gradient Boosting model forecasting job demand for your matched role over the next 5 years</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.result is None:
        st.info("👈 Go to the Dashboard first, enter your skills and click Predict to see forecasts.")
        return

    best_job = st.session_state.result['best_job']
    years_arr, y_actual, y_pred_fc, future_years, future_postings, fc_acc = forecast_postings(
        df, best_job, effective_training_domain()
    )

    if years_arr is None:
        st.warning("⚠️ Not enough data to forecast job postings for this role.")
        return

    st.markdown(
        f'<div style="text-align:center;margin-bottom:.8rem;">'
        f'<span class="accuracy-badge">📊 Forecast Model Accuracy: {round(fc_acc,2)}%</span>'
        f'</div>', unsafe_allow_html=True
    )
    st.markdown(f"""
    <div style="background:rgba(96,165,250,0.08);border:1px solid rgba(96,165,250,0.2);
                border-radius:14px;padding:.7rem 1.4rem;margin-bottom:.8rem;text-align:center;">
        <span style="color:#94a3b8;font-size:.83rem;">Forecasting for:</span>
        <span style="color:#60a5fa;font-weight:800;font-size:1.1rem;margin-left:.5rem;">{best_job}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔮 5-Year Job Posting Forecast</div>', unsafe_allow_html=True)
    cols = st.columns(len(future_years))
    for i, col in enumerate(cols):
        with col:
            delta_color = "#34d399" if i == 0 or future_postings[i] >= future_postings[i-1] else "#f87171"
            delta_icon  = "↑" if i == 0 or future_postings[i] >= future_postings[i-1] else "↓"
            st.markdown(f"""
            <div class="metric-card">
              <h3>📅 {future_years[i]}</h3>
              <h2>{future_postings[i]}</h2>
              <p style="color:{delta_color};font-size:.8rem;font-weight:700;margin-top:.15rem">{delta_icon} avg postings</p>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📉 Actual vs Predicted vs Forecast</div>', unsafe_allow_html=True)
    fig, ax = dark_fig(figsize=(11, 4))
    ax.plot(years_arr, y_actual,  label='Actual',    color='#60a5fa', linewidth=2.5, marker='o', markersize=7)
    ax.plot(years_arr, y_pred_fc, label='Predicted', color='#f87171', linewidth=2.5, linestyle='--', marker='s', markersize=6)
    ax.plot(future_years, future_postings, label='Forecast', color='#34d399', linewidth=2.5, linestyle=':', marker='^', markersize=8)
    for yr, val in zip(future_years, future_postings):
        ax.annotate(f'{val}', (yr, val), textcoords="offset points", xytext=(0, 11),
                    ha='center', fontsize=9, color='#34d399', fontweight='bold')
    ax.fill_between(years_arr, y_actual, y_pred_fc, alpha=0.1, color='#a78bfa')
    ax.set_title(f"Job Posting Forecast — {best_job}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Year"); ax.set_ylabel("Avg Job Postings")
    ax.legend(fontsize=10, facecolor='#1e1b4b', labelcolor='white', edgecolor='#334155')
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    st.markdown('<div class="section-header">📊 All Roles — Postings Comparison</div>', unsafe_allow_html=True)
    postings_trend = df.groupby(['year','job_title'])['postings'].mean().reset_index()
    fig_p, ax_p = dark_fig(figsize=(10, 4))
    for job in postings_trend['job_title'].unique():
        job_data = postings_trend[postings_trend['job_title'] == job].sort_values('year')
        ax_p.plot(job_data['year'], job_data['postings'], label=job,
                  color=ROLE_LINE_COLORS.get(job, '#e2e8f0'),
                  linewidth=3.5 if job == best_job else 1.5,
                  alpha=1.0 if job == best_job else 0.4,
                  marker='o' if job == best_job else None, markersize=6)
    ax_p.set_title("Job Postings Trend by Role (your match highlighted)",
                   fontsize=12, fontweight='bold', color='white')
    ax_p.set_xlabel("Year"); ax_p.set_ylabel("Avg Job Postings")
    ax_p.legend(fontsize=8, facecolor='#1e1b4b', labelcolor='white',
                edgecolor='#334155', loc='upper left')
    plt.tight_layout()
    st.pyplot(fig_p)


# ==============================
# PAGE 4 — MARKET TRENDS
# ==============================
def page_market(df, y_test_actual_eval, y_pred_actual_eval, X_cols, salary_model):
    st.markdown("""
    <div class="page-header">
        <h2>📊 Market Trends & Model Insights</h2>
        <p>Deep analytics — salary distributions, skill correlations, and model performance</p>
    </div>
    """, unsafe_allow_html=True)

    best_job = st.session_state.result['best_job'] if st.session_state.result else df['job_title'].iloc[0]

    st.markdown('<div class="section-header">📈 Salary Trend — All Roles Over Years</div>', unsafe_allow_html=True)
    if st.session_state.result:
        st.markdown(f"Your matched job **{best_job}** is highlighted.", unsafe_allow_html=True)
    st.pyplot(market_trend_chart(df, best_job))

    st.divider()

    st.markdown('<div class="section-header">📊 Salary Distribution Across Dataset</div>', unsafe_allow_html=True)
    fig1, ax1 = dark_fig()
    n, bins, patches = ax1.hist(df['salary'], bins=30, edgecolor='#1e1b4b', linewidth=0.5)
    for patch, c in zip(patches, plt.cm.plasma(np.linspace(0.2, 0.9, len(patches)))):
        patch.set_facecolor(c)
    ax1.set_title("Salary Distribution", fontweight='bold')
    ax1.set_xlabel("Salary (LPA)"); ax1.set_ylabel("Frequency")
    plt.tight_layout(); st.pyplot(fig1)

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">🎯 Skill Match vs Salary</div>', unsafe_allow_html=True)
        fig2, ax2 = dark_fig()
        sc = ax2.scatter(df['similarity'], df['salary'], c=df['salary'],
                         cmap='coolwarm', alpha=0.6, edgecolors='none', s=15)
        plt.colorbar(sc, ax=ax2, label='Salary (LPA)')
        ax2.set_title("Skill Match vs Salary", fontweight='bold')
        ax2.set_xlabel("Similarity Score"); ax2.set_ylabel("Salary (LPA)")
        plt.tight_layout(); st.pyplot(fig2)

    with col_r:
        st.markdown('<div class="section-header">💼 Avg Salary by Job Role</div>', unsafe_allow_html=True)
        avg_sal = df.groupby('job_title')['salary'].mean().sort_values(ascending=False)
        fig3, ax3 = dark_fig()
        bar_colors = ['#a78bfa','#60a5fa','#34d399','#f87171','#fbbf24']
        bars = ax3.bar(avg_sal.index, avg_sal.values,
                       color=bar_colors[:len(avg_sal)], edgecolor='#1e1b4b', linewidth=0.5)
        for bar, val in zip(bars, avg_sal.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='white')
        ax3.set_title("Avg Salary by Job Role", fontweight='bold')
        plt.setp(ax3.get_xticklabels(), rotation=30, ha='right', color='#94a3b8')
        ax3.set_ylabel("Salary (LPA)")
        plt.tight_layout(); st.pyplot(fig3)

    st.divider()

    col_fi, col_ap = st.columns(2)
    with col_fi:
        st.markdown('<div class="section-header">⚙️ Feature Importance (XGBoost)</div>', unsafe_allow_html=True)
        feat_imp = salary_model.feature_importances_
        fig4, ax4 = dark_fig()
        bars4 = ax4.barh(X_cols.columns, feat_imp,
                         color=plt.cm.viridis(np.linspace(0.2, 0.9, len(X_cols.columns))),
                         edgecolor='#1e1b4b')
        for bar, val in zip(bars4, feat_imp):
            ax4.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=9, color='white')
        ax4.set_title("Feature Importance", fontweight='bold')
        ax4.set_xlabel("Importance Score")
        plt.tight_layout(); st.pyplot(fig4)

    with col_ap:
        st.markdown('<div class="section-header">📉 Actual vs Predicted Salary</div>', unsafe_allow_html=True)
        fig5, ax5 = dark_fig()
        sc5 = ax5.scatter(y_test_actual_eval, y_pred_actual_eval,
                          c=y_pred_actual_eval, cmap='RdYlGn', alpha=0.5, edgecolors='none', s=15)
        plt.colorbar(sc5, ax=ax5, label='Predicted Salary')
        ax5.plot([y_test_actual_eval.min(), y_test_actual_eval.max()],
                 [y_test_actual_eval.min(), y_test_actual_eval.max()],
                 'r--', linewidth=2, label='Perfect Prediction')
        ax5.set_title("Actual vs Predicted Salary", fontweight='bold')
        ax5.set_xlabel("Actual Salary (LPA)"); ax5.set_ylabel("Predicted Salary (LPA)")
        ax5.legend(facecolor='#1e1b4b', labelcolor='white', edgecolor='#334155')
        plt.tight_layout(); st.pyplot(fig5)


# ==============================
# PAGE 5 — ABOUT
# ==============================
def page_about(df, salary_accuracy):
    st.markdown("""
    <div class="page-header">
        <h2>ℹ️ About CareerLens AI</h2>
        <p>System architecture, model methodology, and dataset information</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("""
        <div class="about-card">
            <h3>🚀 What is CareerLens?</h3>
            <p>CareerLens is an AI-powered career intelligence platform that matches your current skill set
            to the most relevant job roles in your chosen domain, predicts your expected salary, and
            forecasts future job market demand using machine learning models trained on 20,000+ synthetic
            job records per domain (CS/AI, Mechanical, Civil, Electrical, ECE, Textile, Medicine, Finance),
            each spanning multiple years and eight core roles.</p>
            <ul style="margin-top:.6rem">
                <li>Skill-to-job matching via TF-IDF cosine similarity</li>
                <li>Salary prediction using XGBoost regression</li>
                <li>Demand forecasting via Gradient Boosting time series</li>
                <li>Skill gap analysis to guide upskilling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>🏗️ System Architecture</h3>
            <div class="arch-box">
[User / Browser]<br>
&nbsp;&nbsp;↓ Streamlit Frontend [Python]<br>
[Skill Input + Year]<br>
&nbsp;&nbsp;↓ Text Preprocessing Pipeline<br>
[TF-IDF Vectorizer (sklearn)]<br>
&nbsp;&nbsp;↓ Cosine Similarity → Job Match<br>
[XGBoost Salary Regressor]<br>
&nbsp;&nbsp;↓ log1p transform → salary LPA<br>
[Gradient Boosting Forecaster]<br>
&nbsp;&nbsp;↓ Lag features → 5-yr forecast<br>
[Results + PDF Report]
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>🛠️ Tech Stack</h3>
            <ul>
                <li>Python 3.10+ · Streamlit 1.35+</li>
                <li>XGBoost · scikit-learn · pandas · numpy</li>
                <li>TF-IDF Vectorizer (cosine similarity matching)</li>
                <li>GradientBoostingRegressor (time-series forecasting)</li>
                <li>Matplotlib (dark-themed visualizations)</li>
                <li>PdfPages (automated report generation)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown(f"""
        <div class="about-card">
            <h3>📊 Model Performance</h3>
            <p style="margin-bottom:.5rem">Evaluated on 20% held-out test split (4,000 records):</p>
            <div class="perf-row">
                <span class="perf-chip">Tolerance Acc (±10%): {salary_accuracy}%</span>
                <span class="perf-chip">Model: XGBoost</span>
                <span class="perf-chip">Trees: 500</span>
                <span class="perf-chip">Max Depth: 8</span>
                <span class="perf-chip">Loss: log1p MSE</span>
                <span class="perf-chip">Forecast: GBR</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="about-card">
            <h3>🗄️ Data Sources</h3>
            <ul>
                <li>Synthetic dataset (this domain): {len(df):,} job records — files live under <code>dataset/</code></li>
                <li>Auto mode: strongest skill match picks the domain; locked domain searches only that CSV</li>
                <li>Eight roles per domain — CS/AI example: AI Engineer, ML Engineer, Data Scientist, Data Analyst,
                    Frontend Developer, Backend Developer, Full Stack Developer, DevOps Engineer</li>
                <li>Year range: {int(df['year'].min())}–{int(df['year'].max())}</li>
                <li>Features: job title, required skills, user skills, salary LPA, job postings, year</li>
                <li>Salary range: ~3–23 LPA (base + similarity bonus + noise)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>⚙️ How Prediction Works</h3>
            <ul>
                <li><b style="color:#a78bfa">Step 1</b> — Skills normalized + abbreviated expanded</li>
                <li><b style="color:#60a5fa">Step 2</b> — TF-IDF vectors built for user and all job required skills</li>
                <li><b style="color:#34d399">Step 3</b> — Cosine similarity → best job role match selected</li>
                <li><b style="color:#fbbf24">Step 4</b> — XGBoost predicts salary using 7 engineered features</li>
                <li><b style="color:#f87171">Step 5</b> — GBR model with lag features forecasts 5-year postings</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box" style="margin-top:.5rem;">
        <p>⚠️ <b>Disclaimer:</b> CareerLens uses a synthetic dataset for demonstration purposes.
        Salary predictions and job forecasts are based on modeled patterns and should not be used
        as the sole basis for career decisions. Real-world salaries vary by company, location,
        experience, and negotiation. Always cross-reference with live job market data.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top:1.5rem; padding:1rem;
                border-top:1px solid rgba(255,255,255,0.08);
                color:#475569; font-size:0.8rem;">
        Built with <span style="color:#f472b6">♥</span> using
        Streamlit · XGBoost · Gradient Boosting &nbsp;·&nbsp;
        <span style="color:#a78bfa">CareerLens © 2025</span>
    </div>
    """, unsafe_allow_html=True)


# ==============================
# MAIN
# ==============================
def main():
    render_career_domain_bar()
    with st.spinner("⚙️ Initializing AI models..."):
        df, tfidf, le, salary_model, X_cols, salary_accuracy, \
        y_test_actual_eval, y_pred_actual_eval, req_matrix = load_and_train(
            effective_training_domain()
        )

    render_topnav(st.session_state.current_page)

    page = st.session_state.current_page

    if page == "Dashboard":
        page_dashboard(df, salary_accuracy)
    elif page == "Salary Prediction":
        page_salary(df, tfidf, le, salary_model, X_cols, req_matrix)
    elif page == "Job Forecasting":
        page_forecasting(df)
    elif page == "Market Trends":
        page_market(df, y_test_actual_eval, y_pred_actual_eval, X_cols, salary_model)
    elif page == "About":
        page_about(df, salary_accuracy)

    if page != "About":
        st.markdown("""
        <div style="text-align:center; margin-top:2rem; padding:1rem;
                    border-top:1px solid rgba(255,255,255,0.08);
                    color:#475569; font-size:0.82rem;">
            Made with <span style="color:#f472b6">♥</span> using
            Streamlit · XGBoost · Gradient Boosting &nbsp;·&nbsp; CareerLens © 2025
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()