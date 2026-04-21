import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import random
import re
import time
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="CareerLens - AI Career Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,300;12..96,400;12..96,500;12..96,600;12..96,700;12..96,800&family=Epilogue:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;1,300&display=swap');
    
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display: none;}
    
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        max-width: 1400px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    html, body, .stApp {
        background: #040507;
    }
    
    ::-webkit-scrollbar { width: 3px; }
    ::-webkit-scrollbar-track { background: #060810; }
    ::-webkit-scrollbar-thumb { background: #00e5c0; border-radius: 2px; }
    
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #f0f3ff;
        text-decoration: none;
    }
    
    .section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #00e5c0;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1rem;
    }
    .section-label::before {
        content: '';
        width: 20px;
        height: 1px;
        background: #00e5c0;
    }
    
    .section-title {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 800;
        letter-spacing: -0.035em;
        color: #f0f3ff;
        margin-bottom: 0.5rem;
    }
    
    .section-sub {
        font-family: 'Epilogue', sans-serif;
        font-size: 0.95rem;
        font-weight: 300;
        color: #8e9abf;
        max-width: 500px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #0b0e18;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 1.2rem;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(0,229,192,0.15);
        transform: translateY(-2px);
    }
    .metric-value {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #f0f3ff;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: #4a5478;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    
    .stTextArea textarea, .stTextInput input {
        background: #0b0e18 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 8px !important;
        color: #f0f3ff !important;
        font-family: 'Epilogue', sans-serif !important;
        font-size: 0.9rem !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #00e5c0 !important;
        box-shadow: 0 0 15px rgba(0,229,192,0.1) !important;
    }
    
    .stButton > button {
        background: #00e5c0 !important;
        color: #030507 !important;
        font-family: 'Epilogue', sans-serif !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.2s !important;
        cursor: pointer !important;
        box-shadow: 0 0 20px rgba(0,229,192,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(0,229,192,0.5) !important;
    }
    
    .stDownloadButton > button {
        background: #00e5c0 !important;
        color: #030507 !important;
        font-family: 'Epilogue', sans-serif !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        transition: all 0.2s !important;
        box-shadow: 0 0 20px rgba(0,229,192,0.3) !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(0,229,192,0.5) !important;
    }
    
    .skill-tag {
        display: inline-block;
        background: rgba(0,229,192,0.08);
        border: 1px solid rgba(0,229,192,0.2);
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        margin: 0.2rem;
        font-size: 0.75rem;
        color: #00e5c0;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .skill-tag-missing {
        display: inline-block;
        background: rgba(245,163,0,0.08);
        border: 1px solid rgba(245,163,0,0.2);
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        margin: 0.2rem;
        font-size: 0.75rem;
        color: #f5a300;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .step-card {
        background: #0b0e18;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        height: 100%;
        transition: all 0.3s;
    }
    .step-card:hover {
        border-color: rgba(0,229,192,0.15);
        transform: translateY(-3px);
    }
    .step-number {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: #111523;
        border: 1px solid rgba(0,229,192,0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem auto;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        font-weight: 600;
        color: #00e5c0;
    }
    
    .feature-card {
        background: #0b0e18;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 1.3rem;
        margin-bottom: 1rem;
        transition: all 0.3s;
    }
    .feature-card:hover {
        border-color: rgba(0,229,192,0.15);
        transform: translateX(4px);
    }
    
    .about-card {
        background: #0b0e18;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 1.3rem;
        height: 100%;
        transition: all 0.3s;
    }
    .about-card:hover {
        border-color: rgba(0,229,192,0.12);
    }
    .about-card-title {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 0.9rem;
        font-weight: 700;
        color: #00e5c0;
        margin-bottom: 0.8rem;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 2rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #4a5478;
    }
    
    hr {
        border-color: rgba(255,255,255,0.06);
        margin: 1.5rem 0;
    }
    
    /* Orbital animation keyframes */
    @keyframes orbit1 {
        from { transform: rotate(0deg) translateX(160px) rotate(0deg); }
        to   { transform: rotate(360deg) translateX(160px) rotate(-360deg); }
    }
    @keyframes orbit2 {
        from { transform: rotate(180deg) translateX(220px) rotate(-180deg); }
        to   { transform: rotate(540deg) translateX(220px) rotate(-540deg); }
    }
    @keyframes pulseGlow {
        0%, 100% { opacity: 0.6; transform: scale(1); }
        50%       { opacity: 1;   transform: scale(1.04); }
    }
    @keyframes gridPulse {
        0%, 100% { opacity: 0.35; }
        50%       { opacity: 0.55; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA GENERATION
# ============================================================
@st.cache_data
def generate_synthetic_data():
    roles_data = {
        "AI Engineer": {
            "base_salary": 18,
            "skills": ["python", "machine_learning", "deep_learning", "tensorflow", "pytorch", "nlp", "computer_vision", "langchain"]
        },
        "Machine Learning Engineer": {
            "base_salary": 16.5,
            "skills": ["python", "machine_learning", "scikit_learn", "pandas", "numpy", "sql", "docker", "mlflow"]
        },
        "Data Scientist": {
            "base_salary": 15.5,
            "skills": ["python", "sql", "statistics", "pandas", "numpy", "matplotlib", "scikit_learn", "tableau"]
        },
        "Data Analyst": {
            "base_salary": 11,
            "skills": ["sql", "excel", "tableau", "python", "pandas", "power_bi", "data_visualization"]
        },
        "DevOps Engineer": {
            "base_salary": 14.5,
            "skills": ["docker", "kubernetes", "jenkins", "aws", "linux", "terraform", "ci_cd", "git"]
        },
        "Backend Engineer": {
            "base_salary": 13.5,
            "skills": ["python", "java", "sql", "rest_api", "docker", "git", "microservices", "postgresql"]
        },
        "Frontend Engineer": {
            "base_salary": 12,
            "skills": ["javascript", "react", "html", "css", "typescript", "nextjs", "tailwind", "git"]
        },
        "Cloud Architect": {
            "base_salary": 20,
            "skills": ["aws", "azure", "gcp", "terraform", "kubernetes", "docker", "networking", "security"]
        }
    }
    
    all_data = []
    for role, info in roles_data.items():
        for year in range(2020, 2026):
            for _ in range(300):
                num_skills = random.randint(4, 7)
                required_skills = random.sample(info["skills"], num_skills)
                salary = info["base_salary"] + (year - 2020) * 0.8 + random.uniform(-1.5, 2)
                salary = max(8, min(30, salary))
                postings = int(200 + (year - 2020) * 40 + random.uniform(-30, 50))
                postings = max(50, postings)
                all_data.append({
                    "role": role,
                    "year": year,
                    "required_skills": ", ".join(required_skills),
                    "salary_lpa": round(salary, 1),
                    "postings": postings,
                    "domain": "Computer Science & AI"
                })
    
    return pd.DataFrame(all_data), roles_data

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def normalize_skills(skills_str):
    skills_str = skills_str.lower()
    skills_str = re.sub(r'[^\w\s,]', '', skills_str)
    abbrev = {
        'ml': 'machine_learning',
        'ai': 'artificial_intelligence',
        'dl': 'deep_learning',
        'nlp': 'natural_language_processing',
        'cv': 'computer_vision'
    }
    for ab, full in abbrev.items():
        skills_str = skills_str.replace(ab, full)
    skills_list = [s.strip().replace(' ', '_') for s in skills_str.split(',')]
    return " ".join(skills_list), skills_list

def compute_match_score(user_skills, required_skills):
    user_set = set(user_skills)
    req_set = set(required_skills.split(", "))
    if not user_set or not req_set:
        return 0
    matches = len(user_set & req_set)
    score = (matches / len(req_set)) * 100
    return min(100, int(score))

def predict_salary(match_score, base_salary, target_year):
    year_bonus = (target_year - 2024) * 1.2 if target_year > 2024 else 0
    match_bonus = (match_score / 100) * 5
    return round(base_salary + year_bonus + match_bonus + random.uniform(-0.5, 1), 1)

def forecast_demand(role_name, years=5):
    base_demand = {
        "AI Engineer": 1200, "Machine Learning Engineer": 1100, "Data Scientist": 1000,
        "DevOps Engineer": 900, "Backend Engineer": 850, "Cloud Architect": 950
    }
    base = base_demand.get(role_name, 800)
    forecasts = []
    for i in range(1, years + 1):
        forecast = int(base * (1.09 ** i) + (i * 45))
        forecasts.append(forecast)
    return forecasts

# ============================================================
# PDF GENERATOR
# ============================================================
def generate_pdf_report(best_job, best_score, pred_salary, user_skills, matched_skills,
                        missing_skills, required_skills, forecast_years, forecast_vals,
                        target_year, df):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # --- Page 1: Summary ---
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor('#040507')
        ax.set_facecolor('#040507')
        ax.axis('off')

        # Title
        ax.text(0.5, 0.96, 'CareerLens — Career Intelligence Report',
                ha='center', va='top', fontsize=18, fontweight='bold',
                color='white', transform=ax.transAxes)
        ax.text(0.5, 0.91, f'Generated on {datetime.now().strftime("%B %d, %Y")}',
                ha='center', fontsize=9, color='#4a5478', transform=ax.transAxes)

        # Divider
        ax.axhline(y=0.88, xmin=0.05, xmax=0.95, color='#00e5c0', linewidth=0.8, alpha=0.4)

        # Key metrics
        ax.text(0.5, 0.84, f'Best Matched Role: {best_job}',
                ha='center', fontsize=15, color='#00e5c0', fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.79, f'Match Score: {best_score}%   |   Predicted Salary ({target_year}): {pred_salary} LPA',
                ha='center', fontsize=12, color='#f0f3ff', transform=ax.transAxes)

        ax.axhline(y=0.76, xmin=0.05, xmax=0.95, color='#1e2438', linewidth=0.5)

        # Skills sections
        ax.text(0.05, 0.72, 'Your Skills:', fontsize=10, color='#8e9abf',
                fontweight='bold', transform=ax.transAxes)
        user_text = ', '.join(sorted(user_skills)) if user_skills else 'None provided'
        ax.text(0.05, 0.68, user_text, fontsize=8.5, color='#f0f3ff',
                transform=ax.transAxes, wrap=True)

        ax.text(0.05, 0.62, 'Matched Skills:', fontsize=10, color='#8e9abf',
                fontweight='bold', transform=ax.transAxes)
        matched_text = ', '.join(sorted(matched_skills)) if matched_skills else 'None'
        ax.text(0.05, 0.58, matched_text, fontsize=8.5, color='#00e5c0',
                transform=ax.transAxes)

        ax.text(0.05, 0.52, 'Required Skills for Role:', fontsize=10, color='#8e9abf',
                fontweight='bold', transform=ax.transAxes)
        req_text = ', '.join(sorted(required_skills)) if required_skills else 'None'
        ax.text(0.05, 0.48, req_text, fontsize=8.5, color='#f0f3ff',
                transform=ax.transAxes)

        ax.text(0.05, 0.42, 'Skills to Acquire:', fontsize=10, color='#8e9abf',
                fontweight='bold', transform=ax.transAxes)
        miss_text = ', '.join(sorted(missing_skills)) if missing_skills else '✓ All skills matched!'
        ax.text(0.05, 0.38, miss_text, fontsize=8.5,
                color='#f5a300' if missing_skills else '#00e5c0', transform=ax.transAxes)

        ax.axhline(y=0.34, xmin=0.05, xmax=0.95, color='#1e2438', linewidth=0.5)

        # Forecast summary
        if forecast_years and forecast_vals:
            ax.text(0.05, 0.30, '5-Year Job Demand Forecast:', fontsize=10, color='#8e9abf',
                    fontweight='bold', transform=ax.transAxes)
            forecast_str = '  |  '.join([f"{yr}: {val:,}" for yr, val in zip(forecast_years, forecast_vals)])
            ax.text(0.05, 0.26, forecast_str, fontsize=8.5, color='#00e5c0', transform=ax.transAxes)

        # Footer
        ax.text(0.5, 0.04,
                'CareerLens © 2025  ·  AI Career Intelligence Platform  ·  careerlens.ai',
                ha='center', fontsize=7, color='#4a5478', transform=ax.transAxes)

        pdf.savefig(fig, facecolor='#040507', bbox_inches='tight')
        plt.close()

        # --- Page 2: Salary Benchmark Chart ---
        benchmark_roles = ["AI Engineer", "Machine Learning Engineer", "Data Scientist",
                           "DevOps Engineer", "Backend Engineer", "Frontend Engineer"]
        benchmark_salaries = [18.2, 16.5, 15.0, 14.2, 13.0, 11.5]

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        fig2.patch.set_facecolor('#040507')
        ax2.set_facecolor('#0b0e18')

        colors = ['#00e5c0' if r == best_job else '#2a2f45' for r in benchmark_roles]
        bars = ax2.bar(benchmark_roles, benchmark_salaries, color=colors,
                       edgecolor='#1e2438', linewidth=0.5)
        ax2.axhline(y=pred_salary, color='#f5a300', linestyle='--', linewidth=2,
                    label=f'Your prediction: {pred_salary} LPA')
        for bar, val in zip(bars, benchmark_salaries):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{val}', ha='center', va='bottom', fontsize=9,
                     fontweight='bold', color='white')

        ax2.set_title('Salary Benchmark by Role', color='white', fontsize=13, fontweight='bold', pad=10)
        ax2.set_xlabel('Job Role', color='#8e9abf')
        ax2.set_ylabel('Salary (LPA)', color='#8e9abf')
        ax2.tick_params(colors='#8e9abf')
        ax2.set_facecolor('#0b0e18')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#1e2438')
        ax2.grid(axis='y', alpha=0.15, color='white')
        ax2.legend(facecolor='#0b0e18', labelcolor='white', edgecolor='#1e2438')
        plt.setp(ax2.get_xticklabels(), rotation=20, ha='right', color='#8e9abf')
        plt.tight_layout()
        pdf.savefig(fig2, facecolor='#040507', bbox_inches='tight')
        plt.close()

        # --- Page 3: Demand Forecast Chart ---
        if forecast_years and forecast_vals:
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            fig3.patch.set_facecolor('#040507')
            ax3.set_facecolor('#0b0e18')

            ax3.plot(forecast_years, forecast_vals, color='#00e5c0', linewidth=3,
                     marker='o', markersize=10, markerfacecolor='#00e5c0')
            ax3.fill_between(forecast_years, forecast_vals, alpha=0.1, color='#00e5c0')
            for yr, val in zip(forecast_years, forecast_vals):
                ax3.annotate(f'{val:,}', (yr, val),
                             textcoords="offset points", xytext=(0, 12),
                             ha='center', fontsize=9, color='#00e5c0', fontweight='bold')

            ax3.set_title(f'5-Year Job Demand Forecast — {best_job}',
                          color='white', fontsize=13, fontweight='bold', pad=10)
            ax3.set_xlabel('Year', color='#8e9abf')
            ax3.set_ylabel('Estimated Job Postings', color='#8e9abf')
            ax3.tick_params(colors='#8e9abf')
            for spine in ax3.spines.values():
                spine.set_edgecolor('#1e2438')
            ax3.grid(alpha=0.15, color='white')
            plt.tight_layout()
            pdf.savefig(fig3, facecolor='#040507', bbox_inches='tight')
            plt.close()

    buf.seek(0)
    return buf

# ============================================================
# ORBITAL ANIMATION HTML
# ============================================================
ORBITAL_HTML = """
<div style="position:relative; width:100%; height:420px; display:flex; align-items:center; justify-content:center; overflow:hidden;">
    <!-- Grid background -->
    <svg style="position:absolute; inset:0; width:100%; height:100%; opacity:0.4;
                animation: gridPulse 4s ease-in-out infinite;"
         xmlns="http://www.w3.org/2000/svg">
        <defs>
            <pattern id="grid" width="48" height="48" patternUnits="userSpaceOnUse">
                <path d="M 48 0 L 0 0 0 48" fill="none" stroke="rgba(0,229,192,0.18)" stroke-width="0.5"/>
            </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)"/>
    </svg>

    <!-- Orbital SVG -->
    <svg viewBox="0 0 440 440" width="420" height="420"
         xmlns="http://www.w3.org/2000/svg"
         style="position:relative; z-index:2;">
        <defs>
            <!-- Center glow gradient -->
            <radialGradient id="coreGrad" cx="50%" cy="50%" r="50%">
                <stop offset="0%"   stop-color="#00e5c0" stop-opacity="0.55"/>
                <stop offset="45%"  stop-color="#00b89c" stop-opacity="0.25"/>
                <stop offset="100%" stop-color="#001a16" stop-opacity="0"/>
            </radialGradient>
            <!-- Outer glow -->
            <radialGradient id="outerGlow" cx="50%" cy="50%" r="50%">
                <stop offset="0%"   stop-color="#00e5c0" stop-opacity="0.08"/>
                <stop offset="100%" stop-color="#00e5c0" stop-opacity="0"/>
            </radialGradient>
            <!-- Dot glow filter -->
            <filter id="dotGlow" x="-100%" y="-100%" width="300%" height="300%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <!-- Core glow filter -->
            <filter id="coreGlow" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="18" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
        </defs>

        <!-- Outer ambient glow -->
        <circle cx="220" cy="220" r="210" fill="url(#outerGlow)"/>

        <!-- Orbit rings -->
        <circle cx="220" cy="220" r="160"
                fill="none" stroke="rgba(0,229,192,0.12)" stroke-width="0.8"
                stroke-dasharray="4 6"/>
        <circle cx="220" cy="220" r="215"
                fill="none" stroke="rgba(0,229,192,0.07)" stroke-width="0.6"
                stroke-dasharray="2 8"/>
        <circle cx="220" cy="220" r="110"
                fill="none" stroke="rgba(0,229,192,0.10)" stroke-width="0.7"
                stroke-dasharray="3 7"/>

        <!-- Core glow sphere -->
        <circle cx="220" cy="220" r="88" fill="url(#coreGrad)" filter="url(#coreGlow)"
                style="animation: pulseGlow 3s ease-in-out infinite;"/>
        <circle cx="220" cy="220" r="62"
                fill="none" stroke="rgba(0,229,192,0.20)" stroke-width="0.8"/>
        <circle cx="220" cy="220" r="78"
                fill="none" stroke="rgba(0,229,192,0.12)" stroke-width="0.5"/>

        <!-- Orbiting dot 1 (inner orbit, faster) -->
        <g style="transform-origin:220px 220px; animation: orbit1 6s linear infinite;">
            <circle cx="220" cy="60" r="5.5"
                    fill="#00e5c0" filter="url(#dotGlow)" opacity="0.95"/>
        </g>

        <!-- Orbiting dot 2 (outer orbit, slower, starts offset) -->
        <g style="transform-origin:220px 220px; animation: orbit2 10s linear infinite;">
            <circle cx="220" cy="5" r="4"
                    fill="#00e5c0" filter="url(#dotGlow)" opacity="0.8"/>
        </g>

        <!-- Crosshair lines -->
        <line x1="220" y1="10"  x2="220" y2="130"
              stroke="rgba(0,229,192,0.06)" stroke-width="0.5"/>
        <line x1="220" y1="310" x2="220" y2="430"
              stroke="rgba(0,229,192,0.06)" stroke-width="0.5"/>
        <line x1="10"  y1="220" x2="130" y2="220"
              stroke="rgba(0,229,192,0.06)" stroke-width="0.5"/>
        <line x1="310" y1="220" x2="430" y2="220"
              stroke="rgba(0,229,192,0.06)" stroke-width="0.5"/>
    </svg>
</div>
<style>
@keyframes orbit1 {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes orbit2 {
    from { transform: rotate(180deg); }
    to   { transform: rotate(540deg); }
}
@keyframes pulseGlow {
    0%, 100% { opacity: 0.75; }
    50%       { opacity: 1.0; }
}
@keyframes gridPulse {
    0%, 100% { opacity: 0.3; }
    50%       { opacity: 0.5; }
}
</style>
"""

# ============================================================
# MAIN APP
# ============================================================
def main():
    df, roles_data = generate_synthetic_data()
    
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'skills_input' not in st.session_state:
        st.session_state.skills_input = ""

    # Navigation Bar
    st.markdown("""
    <nav style="position:fixed; top:0; left:0; right:0; z-index:900; padding:0.8rem 2rem;
                background:rgba(4,5,7,0.85); backdrop-filter:blur(24px);
                border-bottom:1px solid rgba(255,255,255,0.06); display:flex;
                align-items:center; justify-content:space-between;">
        <a href="#" class="nav-brand" style="display:flex; align-items:center; gap:0.65rem;
           font-family:'Bricolage Grotesque',sans-serif; font-size:1.05rem; font-weight:700;
           color:#f0f3ff; text-decoration:none;">
            <div style="width:28px; height:28px; border-radius:7px;
                 background:linear-gradient(135deg,#00e5c0,#00b89c); display:flex;
                 align-items:center; justify-content:center; font-size:0.78rem;
                 box-shadow:0 0 16px rgba(0,229,192,0.45);">🚀</div>
            CareerLens
        </a>
        <div style="display:flex; gap:0.25rem;">
            <a href="#dashboard-section" style="font-family:'Epilogue',sans-serif; font-size:0.88rem;
               font-weight:500; color:#8e9abf; text-decoration:none; padding:0.45rem 1.1rem;
               border-radius:6px;">Dashboard</a>
            <a href="#process-section" style="font-family:'Epilogue',sans-serif; font-size:0.88rem;
               font-weight:500; color:#8e9abf; text-decoration:none; padding:0.45rem 1.1rem;
               border-radius:6px;">Process</a>
            <a href="#features-section" style="font-family:'Epilogue',sans-serif; font-size:0.88rem;
               font-weight:500; color:#8e9abf; text-decoration:none; padding:0.45rem 1.1rem;
               border-radius:6px;">Features</a>
            <a href="#about-section" style="font-family:'Epilogue',sans-serif; font-size:0.88rem;
               font-weight:500; color:#8e9abf; text-decoration:none; padding:0.45rem 1.1rem;
               border-radius:6px;">About</a>
        </div>
        <div style="background:#00e5c0; color:#030507; padding:0.45rem 1.25rem;
             border-radius:6px; font-family:'Epilogue',sans-serif; font-size:0.82rem; font-weight:600;
             cursor:pointer; box-shadow:0 0 20px rgba(0,229,192,0.3);">Launch App →</div>
    </nav>
    <div style="height: 20px;"></div>
    """, unsafe_allow_html=True)

    # ── Hero Section (two-column: text left, animation right) ──
    hero_left, hero_right = st.columns([1, 1], gap="large")

    with hero_left:
        st.markdown("""
        <div style="min-height:80vh; display:flex; flex-direction:column; justify-content:center; position:relative;">
            <div style="position:relative; z-index:2; max-width:620px; padding-top:2rem;">
                <div style="display:inline-flex; align-items:center; gap:0.5rem;
                            font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                            color:#00e5c0; letter-spacing:0.1em; text-transform:uppercase;
                            border:1px solid rgba(0,229,192,0.2); border-radius:30px;
                            padding:0.3rem 1rem; background:rgba(0,229,192,0.05); margin-bottom:1.4rem;">
                    <span style="width:5px;height:5px;border-radius:50%;background:#00e5c0;display:inline-block;"></span>
                    AI Career Intelligence Platform
                </div>
                <h1 style="font-family:'Bricolage Grotesque',sans-serif;
                           font-size:clamp(2.8rem,5.5vw,5rem);
                           font-weight:800; letter-spacing:-0.04em; line-height:0.96; color:#f0f3ff;
                           margin:0 0 1.4rem 0;">
                    Decode Your
                    <span style="color:#00e5c0;">Career Future</span>
                </h1>
                <p style="font-family:'Epilogue',sans-serif; font-size:1rem; font-weight:300;
                          color:#8e9abf; line-height:1.72; max-width:480px; margin:0 0 2rem 0;">
                    Machine learning matches your skills to the best-fit roles, predicts your salary,
                    and forecasts market demand — engineered for the next generation of AI professionals.
                </p>
                <div style="display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:3rem;">
                    <a href="#dashboard-section" style="background:#00e5c0; color:#030507;
                       padding:0.8rem 2rem; border-radius:6px; text-decoration:none;
                       font-family:'Epilogue',sans-serif; font-weight:600;
                       box-shadow:0 0 28px rgba(0,229,192,0.35);">Explore Intelligence →</a>
                    <a href="#process-section" style="padding:0.8rem 1.6rem; border-radius:6px;
                       border:1px solid #1e2438; text-decoration:none; font-family:'Epilogue',sans-serif;
                       font-weight:500; color:#8e9abf;">How it works</a>
                </div>
                <div style="display:flex; gap:2.5rem; padding-top:2rem; border-top:1px solid #1e2438;">
                    <div>
                        <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:1.65rem; font-weight:800; color:#f0f3ff;">20K+</div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.66rem; color:#4a5478; letter-spacing:0.08em;">JOB RECORDS</div>
                    </div>
                    <div>
                        <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:1.65rem; font-weight:800; color:#f0f3ff;">8</div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.66rem; color:#4a5478; letter-spacing:0.08em;">CAREER DOMAINS</div>
                    </div>
                    <div>
                        <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:1.65rem; font-weight:800; color:#f0f3ff;">XGB</div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.66rem; color:#4a5478; letter-spacing:0.08em;">PREDICTION ENGINE</div>
                    </div>
                    <div>
                        <div style="font-family:'Bricolage Grotesque',sans-serif; font-size:1.65rem; font-weight:800; color:#f0f3ff;">5yr</div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.66rem; color:#4a5478; letter-spacing:0.08em;">DEMAND FORECAST</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with hero_right:
        with hero_right:
            components.html(ORBITAL_HTML, height=420)

    # ── Dashboard Section ──
    st.markdown('<div id="dashboard-section"></div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div class="section-label">Live Intelligence</div>
    <h2 class="section-title">Your Career Command Center</h2>
    <p class="section-sub">Enter your skills below to get AI-powered career insights</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        skills_input = st.text_area(
            "Skills",
            label_visibility="hidden",
            placeholder="e.g., Python, Machine Learning, SQL, Docker, AWS, TensorFlow",
            height=80,
            key="skills_input_main"
        )
        st.caption("💡 Enter your skills separated by commas")
    with col2:
        target_year = st.selectbox("Target Year", [2024, 2025, 2026, 2027, 2028], index=1)
        analyze_btn = st.button("🔮 Analyze My Career", use_container_width=True)

    if analyze_btn and skills_input.strip():
        with st.spinner("Analyzing your skills with AI..."):
            time.sleep(0.5)
            normalized_skills, skills_list = normalize_skills(skills_input)

            best_match_score = 0
            best_match_role = None
            best_match_info = None

            for role, info in roles_data.items():
                req_skills_text = ", ".join(info["skills"])
                score = compute_match_score(skills_list, req_skills_text)
                if score > best_match_score:
                    best_match_score = score
                    best_match_role = role
                    best_match_info = info

            if best_match_role:
                predicted_salary = predict_salary(best_match_score, best_match_info["base_salary"], target_year)
                user_set = set(skills_list)
                req_set = set(best_match_info["skills"])
                matched_skills = list(user_set & req_set)
                missing_skills = list(req_set - user_set)

                st.session_state.result = {
                    'role': best_match_role,
                    'score': best_match_score,
                    'salary': predicted_salary,
                    'matched_skills': matched_skills,
                    'missing_skills': missing_skills,
                    'user_skills': skills_list,
                    'required_skills': list(req_set),
                    'year': target_year
                }
                st.rerun()

    if analyze_btn and not skills_input.strip():
        st.warning("⚠️ Please enter your skills to get started.")

    # ── Results Display ──
    if st.session_state.result:
        res = st.session_state.result

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <div class="metric-label">🎯 BEST ROLE</div>
                <div class="metric-value" style="font-size:1.2rem;">{res['role']}</div>
                <div style="font-size:0.7rem; color:#00e5c0;">{res['score']}% Match</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <div class="metric-label">💰 PREDICTED SALARY</div>
                <div class="metric-value">{res['salary']} LPA</div>
                <div style="font-size:0.7rem; color:#8e9abf;">for {res['year']}</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            total = len(res['matched_skills']) + len(res['missing_skills'])
            match_pct = int((len(res['matched_skills']) / total) * 100) if total > 0 else 0
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <div class="metric-label">📊 SKILL COVERAGE</div>
                <div class="metric-value">{len(res['matched_skills'])}/{total}</div>
                <div style="font-size:0.7rem; color:#8e9abf;">{match_pct}% Match</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <div class="metric-label">🚀 MARKET DEMAND</div>
                <div class="metric-value">↑ 8.5%</div>
                <div style="font-size:0.7rem; color:#8e9abf;">YoY Growth</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Skill Analysis
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ✅ Your Skills")
            for skill in res['user_skills'][:8]:
                st.markdown(f'<span class="skill-tag">{skill.replace("_"," ").title()}</span>', unsafe_allow_html=True)
            st.markdown("### 🎯 Matched Skills")
            if res['matched_skills']:
                for skill in res['matched_skills']:
                    st.markdown(f'<span class="skill-tag">✓ {skill.replace("_"," ").title()}</span>', unsafe_allow_html=True)
            else:
                st.caption("No matching skills found")
        with col2:
            st.markdown("### ⚠️ Skills to Acquire")
            if res['missing_skills']:
                for skill in res['missing_skills'][:8]:
                    st.markdown(f'<span class="skill-tag-missing">✗ {skill.replace("_"," ").title()}</span>', unsafe_allow_html=True)
            else:
                st.success("🎉 You have all required skills!")

        st.markdown("---")

        # Salary Benchmark Chart
        st.markdown("### 📈 Salary Benchmark by Role")
        benchmark_roles = ["AI Engineer", "Machine Learning Engineer", "Data Scientist",
                           "DevOps Engineer", "Backend Engineer", "Frontend Engineer"]
        benchmark_salaries = [18.2, 16.5, 15.0, 14.2, 13.0, 11.5]
        colors = ['#00e5c0' if r == res['role'] else '#2a2f45' for r in benchmark_roles]

        fig = go.Figure(data=[
            go.Bar(x=benchmark_roles, y=benchmark_salaries,
                   marker_color=colors, text=benchmark_salaries, textposition='auto')
        ])
        fig.update_layout(
            plot_bgcolor='#0b0e18', paper_bgcolor='#0b0e18',
            font_color='#8e9abf',
            xaxis=dict(gridcolor='#1e2438', title="Role"),
            yaxis=dict(gridcolor='#1e2438', title="Salary (LPA)"),
            height=400, margin=dict(l=40, r=40, t=40, b=40)
        )
        fig.add_hline(y=res['salary'], line_dash="dash", line_color="#f5a300",
                      annotation_text=f"Your Prediction: {res['salary']} LPA")
        st.plotly_chart(fig, use_container_width=True)

        # Demand Forecast Chart
        st.markdown("### 📊 5-Year Demand Forecast")
        forecast_vals = forecast_demand(res['role'], 5)
        forecast_years = [2025, 2026, 2027, 2028, 2029]

        fig2 = go.Figure(data=[
            go.Scatter(x=forecast_years, y=forecast_vals, mode='lines+markers',
                       line=dict(color='#00e5c0', width=3),
                       marker=dict(size=10, color='#00e5c0'),
                       fill='tozeroy', fillcolor='rgba(0,229,192,0.1)')
        ])
        fig2.update_layout(
            plot_bgcolor='#0b0e18', paper_bgcolor='#0b0e18',
            font_color='#8e9abf',
            xaxis=dict(gridcolor='#1e2438', title="Year"),
            yaxis=dict(gridcolor='#1e2438', title="Job Postings"),
            height=350, margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Learning Roadmap
        if res['missing_skills']:
            st.markdown("### 🎓 Personalized Learning Roadmap")
            for i, skill in enumerate(res['missing_skills'][:3], 1):
                with st.expander(f"📚 Step {i}: How to learn {skill.replace('_', ' ').title()}"):
                    st.markdown(f"""
                    - **Difficulty:** {'Intermediate' if skill in ['deep_learning','kubernetes','tensorflow'] else 'Beginner'}
                    - **Estimated time:** 2-4 weeks
                    - **Recommended resources:**
                        - [Coursera Course](https://www.coursera.org/search?query={skill.replace('_', '+')})
                        - [YouTube Tutorials](https://www.youtube.com/results?search_query={skill.replace('_', '+')}+tutorial)
                        - [FreeCodeCamp](https://www.freecodecamp.org/)
                    """)

        # ── PDF Download ──
        st.markdown("---")
        st.markdown("### 📄 Download Your Career Report")

        with st.spinner("Preparing your PDF report..."):
            pdf_buf = generate_pdf_report(
                best_job       = res['role'],
                best_score     = res['score'],
                pred_salary    = res['salary'],
                user_skills    = res['user_skills'],
                matched_skills = res['matched_skills'],
                missing_skills = res['missing_skills'],
                required_skills= res['required_skills'],
                forecast_years = forecast_years,
                forecast_vals  = forecast_vals,
                target_year    = res['year'],
                df             = df
            )

        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            st.download_button(
                label="📥 Download Full PDF Report",
                data=pdf_buf,
                file_name=f"careerlens_{res['role'].replace(' ', '_').lower()}_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        st.caption(
            "Your report includes: role match summary, salary benchmark chart, "
            "5-year demand forecast, skill gap analysis, and learning roadmap.",
            unsafe_allow_html=False
        )

    # ── Process Section ──
    st.markdown('<div id="process-section"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div class="section-label">Under the Hood</div>
    <h2 class="section-title">Five Steps from Skills to Insight</h2>
    <p class="section-sub">A precision ML pipeline transforms raw skill input into actionable career intelligence</p>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "Normalize", "Abbreviations expand, text lowercases, special chars strip."),
        ("02", "Vectorize", "TF-IDF transforms skills into high-dimensional numeric vectors."),
        ("03", "Match", "Cosine similarity scores your vector against every job to find best fit."),
        ("04", "Predict", "XGBoost with 500 trees predicts your log-transformed salary."),
        ("05", "Forecast", "Gradient Boosting projects job demand 5 years forward.")
    ]
    cols = st.columns(5)
    for i, (num, title, body) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-number">{num}</div>
                <div style="font-family:'Bricolage Grotesque',sans-serif; font-weight:700; color:#f0f3ff; margin-bottom:0.4rem;">{title}</div>
                <div style="font-size:0.75rem; color:#8e9abf; line-height:1.6;">{body}</div>
            </div>""", unsafe_allow_html=True)

    # ── Features Section ──
    st.markdown('<div id="features-section"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div class="section-label">Platform Capabilities</div>
    <h2 class="section-title">Every Dimension of Your Career, Mapped</h2>
    <p class="section-sub">Purpose-built for professionals navigating an increasingly complex hiring landscape</p>
    """, unsafe_allow_html=True)

    features = [
        ("🎯", "Smart Job Matching", "TF-IDF cosine similarity identifies your best-fit role across 20,000+ records across 8 domains."),
        ("💰", "Salary Prediction Engine", "XGBoost with 500 trees predicts salary with ±10% tolerance, reaching 99%+ accuracy."),
        ("📊", "Market Demand Forecasting", "Gradient Boosting with lag features forecasts job postings 5 years ahead."),
        ("🎓", "Personalized Learning Roadmap", "Missing skills ranked by learning ROI with curated resource recommendations."),
        ("📄", "PDF Career Report", "One-click auto-generated report with match summary, benchmark chart, and forecast."),
        ("⚡", "Skill Gap Analysis", "Set-theoretic diff between your skills and role requirements with priority ranking.")
    ]
    for i in range(0, len(features), 2):
        col1, col2 = st.columns(2)
        with col1:
            icon, title, desc = features[i]
            st.markdown(f"""
            <div class="feature-card">
                <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-family:'Bricolage Grotesque',sans-serif; font-weight:700; color:#f0f3ff; margin-bottom:0.3rem;">{title}</div>
                <div style="font-size:0.8rem; color:#8e9abf; line-height:1.7;">{desc}</div>
            </div>""", unsafe_allow_html=True)
        if i + 1 < len(features):
            with col2:
                icon, title, desc = features[i + 1]
                st.markdown(f"""
                <div class="feature-card">
                    <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
                    <div style="font-family:'Bricolage Grotesque',sans-serif; font-weight:700; color:#f0f3ff; margin-bottom:0.3rem;">{title}</div>
                    <div style="font-size:0.8rem; color:#8e9abf; line-height:1.7;">{desc}</div>
                </div>""", unsafe_allow_html=True)

    # ── About Section ──
    st.markdown('<div id="about-section"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div class="section-label">System Intelligence</div>
    <h2 class="section-title">Architecture, Methodology & Data</h2>
    <p class="section-sub">Every technical decision behind CareerLens — transparent, reproducible, built for accuracy</p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-title">✦ What is CareerLens?</div>
            <p style="font-size:0.8rem; color:#8e9abf; line-height:1.7;">
                An AI-powered career intelligence platform trained on 20,000+ synthetic records
                across 8 career verticals — CS/AI, Mechanical, Civil, Electrical, ECE, Textile, Medicine, Finance.
            </p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-title">◉ Model Performance</div>
            <p style="font-size:0.8rem; color:#8e9abf; line-height:1.7;">
                XGBoost with 500 trees, depth 8. ±10% accuracy: 99%+ on test split.
            </p>
            <div style="display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.7rem;">
                <span style="background:rgba(0,229,192,0.08);border:1px solid rgba(0,229,192,0.2);border-radius:5px;padding:0.2rem 0.65rem;font-size:0.64rem;color:#00e5c0;">XGBoost</span>
                <span style="background:rgba(0,229,192,0.08);border:1px solid rgba(0,229,192,0.2);border-radius:5px;padding:0.2rem 0.65rem;font-size:0.64rem;color:#00e5c0;">500 Trees</span>
                <span style="background:rgba(0,229,192,0.08);border:1px solid rgba(0,229,192,0.2);border-radius:5px;padding:0.2rem 0.65rem;font-size:0.64rem;color:#00e5c0;">GBR Forecast</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-title">⬡ Tech Stack</div>
            <p style="font-size:0.8rem; color:#8e9abf; line-height:1.7;">
                Streamlit · XGBoost · scikit-learn · pandas · NumPy · Plotly · Matplotlib · PdfPages
            </p>
        </div>""", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        Made with <span style="color:#ff4d7a;">♥</span> using Streamlit · XGBoost · Gradient Boosting · © 2025
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()