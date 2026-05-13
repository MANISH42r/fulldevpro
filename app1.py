# ======================================================
# CareerLens — final_app.py  (FIXED: nav + multi-graph)
# ======================================================
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re, io
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

st.set_page_config(
    page_title="CareerLens — AI Career Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================================================
# CSS
# ======================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,300;12..96,400;12..96,500;12..96,600;12..96,700;12..96,800&family=Epilogue:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;1,300&display=swap');

#MainMenu,footer,header{visibility:hidden;}
.stAppDeployButton{display:none;}
[data-testid="stSidebar"]{display:none;}
*{box-sizing:border-box;}

a{text-decoration:none!important;}
a:hover{text-decoration:none!important;}
a:visited{text-decoration:none!important;}
a:active{text-decoration:none!important;}

:root{
  --bg:#040507;--bg2:#0b0e18;--surface2:#111523;
  --accent:#00e5c0;--accent2:#00b89c;--amber:#f5a300;
  --red:#ff4b5a;--green:#4cef9a;--text:#f0f3ff;
  --muted:#8e9abf;--dim:#4a5478;
  --border:rgba(255,255,255,0.06);--border2:rgba(255,255,255,0.09);
  --glow:rgba(0,229,192,0.35);--glow2:rgba(0,229,192,0.10);
}

html,body,.stApp{background:var(--bg)!important;}
[data-testid="stAppViewContainer"]{background:var(--bg)!important;}
.main .block-container{padding-top:0!important;padding-bottom:3rem!important;max-width:1400px!important;padding-left:2rem!important;padding-right:2rem!important;}
::-webkit-scrollbar{width:3px;}::-webkit-scrollbar-track{background:#060810;}::-webkit-scrollbar-thumb{background:var(--accent);border-radius:2px;}

/* ── Navbar ── */
.cl-topbar{position:fixed;top:0;left:0;right:0;z-index:9000;padding:0 2rem;background:rgba(4,5,7,0.92);backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);border-bottom:1px solid rgba(255,255,255,0.06);display:flex;align-items:center;justify-content:space-between;height:60px;}
.cl-brand{display:flex;align-items:center;gap:0.65rem;font-family:'Bricolage Grotesque',sans-serif;font-size:1.05rem;font-weight:700;color:var(--text);text-decoration:none;flex-shrink:0;}
.cl-brand-icon{width:28px;height:28px;border-radius:7px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:flex;align-items:center;justify-content:center;font-size:0.78rem;box-shadow:0 0 16px rgba(0,229,192,0.45);}
.cl-nav-links{display:flex;align-items:center;gap:0.1rem;}
.cl-nav-link{font-family:'Epilogue',sans-serif;font-size:0.88rem;font-weight:500;color:var(--muted);text-decoration:none!important;padding:0.45rem 1rem;border-radius:6px;transition:color 0.2s;white-space:nowrap;cursor:pointer;display:inline-block;}
.cl-nav-link:hover{color:var(--text);background:rgba(255,255,255,0.04);text-decoration:none!important;}
.cl-nav-link:visited{text-decoration:none!important;color:var(--muted);}
.cl-nav-link:active{text-decoration:none!important;}
.cl-nav-link.active{color:var(--text);text-decoration:none!important;background:rgba(0,229,192,0.08);}
.cl-cta{background:var(--accent);color:#030507!important;font-family:'Epilogue',sans-serif;font-size:0.82rem;font-weight:600;padding:0.45rem 1.25rem;border-radius:6px;text-decoration:none;box-shadow:0 0 20px rgba(0,229,192,0.3);white-space:nowrap;flex-shrink:0;display:flex;align-items:center;gap:0.4rem;}
.cl-dot{width:6px;height:6px;border-radius:50%;background:#030507;animation:pulseDot 1.5s infinite;}
@keyframes pulseDot{0%,100%{opacity:0.7;}50%{opacity:1.0;}}

/* ── Labels / titles ── */
.section-label{font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:var(--accent);letter-spacing:0.14em;text-transform:uppercase;display:flex;align-items:center;gap:0.6rem;margin-bottom:0.35rem;}
.section-label::before{content:'';width:20px;height:1px;background:var(--accent);}
.section-title{font-family:'Bricolage Grotesque',sans-serif;font-size:clamp(1.6rem,2.8vw,2.4rem);font-weight:800;letter-spacing:-0.035em;color:var(--text);margin-bottom:0.15rem;line-height:1.1;}
.section-sub{font-family:'Epilogue',sans-serif;font-size:0.88rem;font-weight:300;color:var(--muted);max-width:520px;margin-bottom:1rem;line-height:1.6;}
.cl-section{font-size:0.64rem;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:0.14em;margin:1.3rem 0 0.65rem 0;display:flex;align-items:center;gap:0.6rem;font-family:'JetBrains Mono',monospace;}
.cl-section::before{content:'//';color:var(--accent2);}
.cl-section::after{content:'';flex:1;height:1px;background:linear-gradient(to right,var(--border2),transparent);}

/* ── Metric cards ── */
.cl-metric{background:var(--bg2);border:1px solid rgba(255,255,255,0.05);border-radius:14px;padding:1.2rem;transition:all 0.3s;height:100%;}
.cl-metric:hover{border-color:rgba(0,229,192,0.15);transform:translateY(-2px);}
.cl-metric-label{font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:var(--dim);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;}
.cl-metric-value{font-family:'Bricolage Grotesque',sans-serif;font-size:1.75rem;font-weight:800;color:var(--text);line-height:1.05;margin-bottom:0.12rem;}
.cl-metric-value.teal{color:var(--accent);}.cl-metric-value.amber{color:var(--amber);}.cl-metric-value.green{color:var(--green);}
.cl-metric-sub{font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:var(--muted);}
.cl-metric-sub .up{color:var(--accent);}.cl-metric-sub .hot{color:var(--accent);}

/* ── Feature cards ── */
.cl-feature{background:var(--bg2);border:1px solid rgba(255,255,255,0.05);border-radius:14px;padding:1.3rem;margin-bottom:1rem;transition:all 0.3s;}
.cl-feature:hover{border-color:rgba(0,229,192,0.15);transform:translateX(3px);}
.cl-feature-icon{font-size:1.8rem;margin-bottom:0.4rem;}
.cl-feature-title{font-family:'Bricolage Grotesque',sans-serif;font-size:0.88rem;font-weight:700;color:var(--text);margin-bottom:0.22rem;}
.cl-feature-desc{font-size:0.77rem;color:var(--muted);line-height:1.6;font-family:'Epilogue',sans-serif;}

/* ── Alerts ── */
.cl-alert{display:flex;align-items:flex-start;gap:0.7rem;padding:0.7rem 1rem;border-radius:10px;font-size:0.8rem;margin-bottom:0.5rem;line-height:1.5;font-family:'Epilogue',sans-serif;}
.cl-alert.warn{background:rgba(245,163,0,0.06);border-left:3px solid var(--amber);color:rgba(245,163,0,0.85);}
.cl-alert.ok{background:rgba(0,229,192,0.05);border-left:3px solid var(--accent);color:var(--accent);}
.cl-alert.danger{background:rgba(255,75,90,0.06);border-left:3px solid var(--red);color:rgba(255,75,90,0.85);}
.cl-alert.info{background:rgba(0,229,192,0.05);border-left:3px solid var(--accent);color:var(--accent);}
.cl-alert-icon{font-size:0.88rem;flex-shrink:0;}

/* ── Input panel ── */
.cl-input-panel{background:var(--bg2);border:1px solid rgba(255,255,255,0.05);border-radius:14px;padding:1.4rem 1.6rem;margin-bottom:1rem;}

/* ── Streamlit input overrides ── */
label{color:var(--muted)!important;font-size:0.65rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:0.1em!important;font-family:'JetBrains Mono',monospace!important;}
.stTextInput>div>div>input{background:var(--bg)!important;border:1px solid rgba(255,255,255,0.08)!important;border-radius:8px!important;color:var(--text)!important;font-family:'Epilogue',sans-serif!important;font-size:0.9rem!important;}
.stTextInput>div>div>input:focus{border-color:var(--accent)!important;box-shadow:0 0 15px var(--glow2)!important;}
.stTextInput>div>div>input::placeholder{color:rgba(142,154,191,0.4)!important;}
.stNumberInput>div>div>input{background:var(--bg)!important;border:1px solid rgba(255,255,255,0.08)!important;border-radius:8px!important;color:var(--accent)!important;font-family:'JetBrains Mono',monospace!important;font-size:0.9rem!important;}
.stSelectbox>div>div{background:var(--bg)!important;border:1px solid rgba(255,255,255,0.08)!important;border-radius:8px!important;color:var(--text)!important;font-family:'Epilogue',sans-serif!important;font-size:0.85rem!important;}

/* ── Buttons ── */
.stButton>button{background:var(--accent)!important;color:#030507!important;border:none!important;border-radius:6px!important;font-family:'Epilogue',sans-serif!important;font-weight:700!important;font-size:0.88rem!important;height:2.5rem!important;width:100%!important;transition:all 0.2s!important;box-shadow:0 0 20px var(--glow2)!important;letter-spacing:0.04em!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 0 30px var(--glow)!important;}
.stDownloadButton>button{background:var(--accent)!important;color:#030507!important;border:none!important;border-radius:6px!important;font-family:'Epilogue',sans-serif!important;font-weight:700!important;width:100%!important;box-shadow:0 0 20px var(--glow2)!important;transition:all 0.2s!important;}
.stDownloadButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 0 30px var(--glow)!important;}

/* ── Skill tags ── */
.sk{display:inline-flex;align-items:center;gap:0.3rem;padding:0.22rem 0.75rem;border-radius:20px;font-size:0.72rem;font-weight:500;margin:3px;font-family:'JetBrains Mono',monospace;}
.sk-you{background:rgba(0,229,192,0.08);border:1px solid rgba(0,229,192,0.22);color:var(--accent);}
.sk-req{background:rgba(245,163,0,0.08);border:1px solid rgba(245,163,0,0.22);color:var(--amber);}
.sk-miss{background:rgba(255,75,90,0.07);border:1px solid rgba(255,75,90,0.22);color:var(--red);}
.sk-match{background:rgba(0,229,192,0.08);border:1px solid rgba(0,229,192,0.28);color:var(--accent);}

/* ── Roadmap ── */
.roadmap-step{background:var(--surface2);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:10px;padding:0.9rem 1.2rem;margin-bottom:0.55rem;}
.roadmap-step-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:0.32rem;}
.roadmap-step-title{font-weight:700;font-size:0.86rem;color:var(--text);font-family:'Bricolage Grotesque',sans-serif;text-transform:uppercase;letter-spacing:0.04em;}
.roadmap-badge{font-size:0.6rem;font-weight:700;padding:0.18rem 0.6rem;border-radius:20px;text-transform:uppercase;letter-spacing:0.06em;font-family:'JetBrains Mono',monospace;}
.badge-beginner{background:rgba(0,229,192,0.1);color:var(--accent);border:1px solid rgba(0,229,192,0.3);}
.badge-inter{background:rgba(245,163,0,0.1);color:var(--amber);border:1px solid rgba(245,163,0,0.3);}
.badge-advanced{background:rgba(255,75,90,0.1);color:var(--red);border:1px solid rgba(255,75,90,0.3);}
.roadmap-meta{font-size:0.63rem;color:var(--muted);margin-bottom:0.32rem;font-family:'JetBrains Mono',monospace;}
.roadmap-link{font-size:0.73rem;color:var(--accent);text-decoration:none;font-weight:600;font-family:'JetBrains Mono',monospace;}
.roadmap-link:hover{text-decoration:underline;}

/* ── About cards ── */
.about-card{background:var(--bg2);border:1px solid rgba(255,255,255,0.05);border-radius:14px;padding:1.3rem;margin-bottom:0.8rem;transition:all 0.3s;height:100%;position:relative;overflow:hidden;}
.about-card:hover{border-color:rgba(0,229,192,0.12);}
.about-card::before{content:'';position:absolute;top:0;left:0;width:100%;height:2px;background:linear-gradient(to right,var(--accent),transparent);}
.about-card h3{font-family:'Bricolage Grotesque',sans-serif;font-size:0.84rem;font-weight:700;color:var(--accent);margin-bottom:0.55rem;display:flex;align-items:center;gap:0.4rem;text-transform:uppercase;letter-spacing:0.06em;}
.about-card p,.about-card li{color:var(--muted);font-size:0.77rem;line-height:1.7;margin:0;font-family:'Epilogue',sans-serif;}
.about-card ul{padding-left:1.1rem;margin:0.3rem 0 0;}
.arch-box{background:var(--bg);border:1px solid var(--border2);border-radius:8px;padding:0.8rem 1rem;font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:rgba(255,255,255,0.4);line-height:2;}
.about-card-title{font-family:'Bricolage Grotesque',sans-serif;font-size:0.9rem;font-weight:700;color:var(--accent);margin-bottom:0.75rem;}
.domain-banner{background:rgba(0,229,192,0.05);border:1px solid rgba(0,229,192,0.15);border-radius:10px;padding:0.5rem 1rem;font-size:0.78rem;color:var(--accent);margin-bottom:0.9rem;font-weight:600;font-family:'Epilogue',sans-serif;}

/* ── Misc ── */
.disclaimer{background:rgba(245,163,0,0.04);border:1px solid rgba(245,163,0,0.15);border-radius:10px;padding:0.85rem 1.1rem;font-size:0.72rem;color:rgba(245,163,0,0.75);line-height:1.6;font-family:'Epilogue',sans-serif;}
.threat-card{background:var(--surface2);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:10px;padding:0.85rem 1.1rem;margin-bottom:0.5rem;}
.threat-title{font-size:0.8rem;font-weight:700;color:var(--text);font-family:'Bricolage Grotesque',sans-serif;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:0.18rem;}
.threat-meta{font-size:0.61rem;color:var(--muted);font-family:'JetBrains Mono',monospace;}
.threat-badge{display:inline-flex;align-items:center;padding:0.12rem 0.5rem;font-size:0.6rem;font-family:'JetBrains Mono',monospace;border-radius:20px;font-weight:700;letter-spacing:0.06em;border:1px solid;}
.about-table{width:100%;border-collapse:collapse;margin-top:0.7rem;font-family:'JetBrains Mono',monospace;font-size:0.67rem;}
.about-table thead tr{border-bottom:1px solid var(--border2);}
.about-table th{text-align:left;padding:0.38rem 0.5rem;color:var(--muted);font-weight:700;letter-spacing:0.06em;text-transform:uppercase;}
.about-table tbody tr{border-bottom:1px solid var(--border);}
.about-table tbody tr:last-child{border-bottom:none;}
.about-table td{padding:0.42rem 0.5rem;vertical-align:middle;}
.about-table td.label{color:var(--muted);}.about-table td.val-pink{color:var(--accent);}.about-table td.val-cyan{color:var(--accent);}.about-table td.val-green{color:var(--green);font-weight:700;}.about-table td.val-text{color:var(--text);}
.step-card{background:var(--bg2);border:1px solid rgba(255,255,255,0.05);border-radius:16px;padding:1.4rem;text-align:center;height:100%;transition:all 0.3s;}
.step-card:hover{border-color:rgba(0,229,192,0.15);transform:translateY(-3px);}
.step-number{width:44px;height:44px;border-radius:50%;background:var(--surface2);border:1px solid rgba(0,229,192,0.2);display:flex;align-items:center;justify-content:center;margin:0 auto 0.9rem auto;font-family:'JetBrains Mono',monospace;font-size:0.88rem;font-weight:600;color:var(--accent);}
.footer{text-align:center;padding:2rem;border-top:1px solid rgba(255,255,255,0.04);margin-top:2rem;font-family:'JetBrains Mono',monospace;font-size:0.67rem;color:var(--dim);}
hr{border-color:rgba(255,255,255,0.06)!important;margin:1.4rem 0!important;}
[data-testid="stExpander"]{background:var(--bg2)!important;border:1px solid rgba(255,255,255,0.05)!important;border-radius:10px!important;}
[data-testid="stExpander"] summary{color:var(--muted)!important;font-family:'JetBrains Mono',monospace!important;font-size:0.78rem!important;}
[data-testid="stImage"],.stPyplot{display:flex!important;justify-content:center!important;}
@keyframes orbit1{from{transform:rotate(0deg);}to{transform:rotate(360deg);}}
@keyframes orbit2{from{transform:rotate(180deg);}to{transform:rotate(540deg);}}
@keyframes pulseGlow{0%,100%{opacity:0.75;}50%{opacity:1.0;}}
@keyframes gridPulse{0%,100%{opacity:0.3;}50%{opacity:0.5;}}
</style>
""", unsafe_allow_html=True)

# ======================================================
# ORBITAL HTML
# ======================================================
ORBITAL_HTML = """
<div style="position:relative;width:100%;height:420px;display:flex;align-items:center;justify-content:center;overflow:hidden;">
  <svg style="position:absolute;inset:0;width:100%;height:100%;opacity:0.4;animation:gridPulse 4s ease-in-out infinite;" xmlns="http://www.w3.org/2000/svg">
    <defs><pattern id="grid" width="48" height="48" patternUnits="userSpaceOnUse"><path d="M 48 0 L 0 0 0 48" fill="none" stroke="rgba(0,229,192,0.18)" stroke-width="0.5"/></pattern></defs>
    <rect width="100%" height="100%" fill="url(#grid)"/>
  </svg>
  <svg viewBox="0 0 440 440" width="420" height="420" xmlns="http://www.w3.org/2000/svg" style="position:relative;z-index:2;">
    <defs>
      <radialGradient id="coreGrad" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="#00e5c0" stop-opacity="0.55"/><stop offset="45%" stop-color="#00b89c" stop-opacity="0.25"/><stop offset="100%" stop-color="#001a16" stop-opacity="0"/></radialGradient>
      <radialGradient id="outerGlow" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="#00e5c0" stop-opacity="0.08"/><stop offset="100%" stop-color="#00e5c0" stop-opacity="0"/></radialGradient>
      <filter id="dotGlow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      <filter id="coreGlow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur in="SourceGraphic" stdDeviation="18" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    </defs>
    <circle cx="220" cy="220" r="210" fill="url(#outerGlow)"/>
    <circle cx="220" cy="220" r="160" fill="none" stroke="rgba(0,229,192,0.12)" stroke-width="0.8" stroke-dasharray="4 6"/>
    <circle cx="220" cy="220" r="215" fill="none" stroke="rgba(0,229,192,0.07)" stroke-width="0.6" stroke-dasharray="2 8"/>
    <circle cx="220" cy="220" r="110" fill="none" stroke="rgba(0,229,192,0.10)" stroke-width="0.7" stroke-dasharray="3 7"/>
    <circle cx="220" cy="220" r="88" fill="url(#coreGrad)" filter="url(#coreGlow)" style="animation:pulseGlow 3s ease-in-out infinite;"/>
    <circle cx="220" cy="220" r="62" fill="none" stroke="rgba(0,229,192,0.20)" stroke-width="0.8"/>
    <circle cx="220" cy="220" r="78" fill="none" stroke="rgba(0,229,192,0.12)" stroke-width="0.5"/>
    <g style="transform-origin:220px 220px;animation:orbit1 6s linear infinite;"><circle cx="220" cy="60" r="5.5" fill="#00e5c0" filter="url(#dotGlow)" opacity="0.95"/></g>
    <g style="transform-origin:220px 220px;animation:orbit2 10s linear infinite;"><circle cx="220" cy="5" r="4" fill="#00e5c0" filter="url(#dotGlow)" opacity="0.8"/></g>
    <line x1="220" y1="10" x2="220" y2="130" stroke="rgba(0,229,192,0.06)" stroke-width="0.5"/>
    <line x1="220" y1="310" x2="220" y2="430" stroke="rgba(0,229,192,0.06)" stroke-width="0.5"/>
    <line x1="10" y1="220" x2="130" y2="220" stroke="rgba(0,229,192,0.06)" stroke-width="0.5"/>
    <line x1="310" y1="220" x2="430" y2="220" stroke="rgba(0,229,192,0.06)" stroke-width="0.5"/>
  </svg>
</div>
<style>
@keyframes orbit1{from{transform:rotate(0deg);}to{transform:rotate(360deg);}}
@keyframes orbit2{from{transform:rotate(180deg);}to{transform:rotate(540deg);}}
@keyframes pulseGlow{0%,100%{opacity:0.75;}50%{opacity:1.0;}}
@keyframes gridPulse{0%,100%{opacity:0.3;}50%{opacity:0.5;}}
</style>
"""

# ======================================================
# SESSION STATE
# ======================================================
for k,v in [("skills_input",""),("current_page","Dashboard"),("result",None),("market_domain",None)]:
    if k not in st.session_state: st.session_state[k] = v

# ======================================================
# CONSTANTS
# ======================================================
DOMAIN_AUTO = "Auto (best domain for your skills)"
CAREER_FIELD_FILES = {
    "Computer Science & AI":"dataset/computer_science_ai.csv",
    "Mechanical Engineering":"dataset/mechanical_engineering.csv",
    "Civil Engineering":"dataset/civil_engineering.csv",
    "Electrical Engineering":"dataset/electrical_engineering.csv",
    "Electronics & Communication Engineering":"dataset/electronics_communication_engineering.csv",
    "Textile":"dataset/textile.csv",
    "Medicine":"dataset/medicine.csv",
    "Finance":"dataset/finance.csv",
}
CAREER_DOMAINS_ORDERED = list(CAREER_FIELD_FILES.keys())
AUTO_DOMAIN_TIEBREAK = {"Civil Engineering":8,"Mechanical Engineering":7,"Electrical Engineering":6,"Electronics & Communication Engineering":5,"Textile":4,"Medicine":3,"Finance":2,"Computer Science & AI":1}
if "career_field" not in st.session_state: st.session_state.career_field = DOMAIN_AUTO
elif st.session_state.career_field not in [DOMAIN_AUTO]+CAREER_DOMAINS_ORDERED: st.session_state.career_field = DOMAIN_AUTO

CAREER_BASE_SALARY = {
    "Computer Science & AI":{"AI Engineer":8,"ML Engineer":6,"Data Scientist":4,"Data Analyst":3,"Frontend Developer":3,"Backend Developer":3,"Full Stack Developer":6,"DevOps Engineer":6},
    "Mechanical Engineering":{"Design Engineer":6,"Manufacturing Engineer":5,"Quality Engineer":5,"HVAC Engineer":6,"Maintenance Engineer":4,"Robotics Automation Engineer":7,"Project Engineer":6,"CAE Engineer":7},
    "Civil Engineering":{"Structural Engineer":7,"Site Engineer":4,"Quantity Surveyor":5,"Geotechnical Engineer":6,"Highway Engineer":5,"BIM Civil Engineer":6,"Project Manager Civil":7,"Urban Planner":5},
    "Electrical Engineering":{"Power Systems Engineer":7,"Control Systems Engineer":6,"Electrical Design Engineer":6,"Substation Engineer":7,"Renewable Energy Engineer":6,"Electrical Maintenance Engineer":4,"Electrical Project Engineer":6,"Protection Engineer":7},
    "Electronics & Communication Engineering":{"VLSI Design Engineer":8,"Embedded Systems Engineer":7,"RF Engineer":7,"Analog Design Engineer":8,"PCB Design Engineer":6,"Hardware Test Engineer":5,"DSP Engineer":7,"Telecom Network Engineer":6},
    "Textile":{"Textile Technologist":4,"Fabric Development Specialist":5,"Quality Control Textile":4,"Fashion Production Manager":6,"Knitting Engineer":5,"Dyeing Technologist":5,"Merchandiser":4,"Supply Chain Textile":5},
    "Medicine":{"General Physician":8,"Surgeon":9,"Pediatrician":8,"Radiologist":8,"Pathologist":7,"Emergency Medicine Doctor":8,"Psychiatrist":7,"Cardiologist":9},
    "Finance":{"Financial Analyst":5,"Investment Banker":8,"Chartered Accountant":7,"Risk Analyst":6,"Portfolio Manager":8,"Financial Controller":7,"Tax Consultant":6,"Credit Analyst":5},
}
ROLE_LINE_COLORS = {
    "AI Engineer":"#00e5c0","ML Engineer":"#00b89c","Data Scientist":"#4cef9a","Data Analyst":"#f5a300","Frontend Developer":"#ff4b5a","Full Stack Developer":"#00e5c0","Backend Developer":"#8e9abf","DevOps Engineer":"#f5a300",
    "Design Engineer":"#00e5c0","Manufacturing Engineer":"#f5a300","Quality Engineer":"#4cef9a","HVAC Engineer":"#00b89c","Maintenance Engineer":"#8e9abf","Robotics Automation Engineer":"#00e5c0","Project Engineer":"#f5a300","CAE Engineer":"#00b89c",
    "Structural Engineer":"#f5a300","Site Engineer":"#8e9abf","Quantity Surveyor":"#00e5c0","Geotechnical Engineer":"#4cef9a","Highway Engineer":"#00b89c","BIM Civil Engineer":"#8e9abf","Project Manager Civil":"#00e5c0","Urban Planner":"#ff4b5a",
    "Power Systems Engineer":"#f5a300","Control Systems Engineer":"#00b89c","Electrical Design Engineer":"#00e5c0","Substation Engineer":"#8e9abf","Renewable Energy Engineer":"#4cef9a","Electrical Maintenance Engineer":"#8e9abf","Electrical Project Engineer":"#8e9abf","Protection Engineer":"#ff4b5a",
    "VLSI Design Engineer":"#00e5c0","Embedded Systems Engineer":"#f5a300","RF Engineer":"#4cef9a","Analog Design Engineer":"#8e9abf","PCB Design Engineer":"#00b89c","Hardware Test Engineer":"#8e9abf","DSP Engineer":"#ff4b5a","Telecom Network Engineer":"#00e5c0",
    "Textile Technologist":"#8e9abf","Fabric Development Specialist":"#ff4b5a","Quality Control Textile":"#f5a300","Fashion Production Manager":"#00b89c","Knitting Engineer":"#00e5c0","Dyeing Technologist":"#4cef9a","Merchandiser":"#8e9abf","Supply Chain Textile":"#8e9abf",
    "General Physician":"#4cef9a","Surgeon":"#00e5c0","Pediatrician":"#00b89c","Radiologist":"#f5a300","Pathologist":"#8e9abf","Emergency Medicine Doctor":"#ff4b5a","Psychiatrist":"#8e9abf","Cardiologist":"#00e5c0",
    "Financial Analyst":"#00e5c0","Investment Banker":"#f5a300","Chartered Accountant":"#4cef9a","Risk Analyst":"#00b89c","Portfolio Manager":"#ff4b5a","Financial Controller":"#8e9abf","Tax Consultant":"#8e9abf","Credit Analyst":"#8e9abf",
}
CAREER_SKILL_PLACEHOLDERS = {
    DOMAIN_AUTO:"e.g. python, solidworks, staad pro, gst, diagnosis — any domain",
    "Computer Science & AI":"e.g. python, machine learning, sql, docker",
    "Mechanical Engineering":"e.g. solidworks, cad, fea, lean six sigma, plc",
    "Civil Engineering":"e.g. revit, staad pro, estimation, site supervision, gis",
    "Electrical Engineering":"e.g. plc, scada, power systems, relay coordination",
    "Electronics & Communication Engineering":"e.g. verilog, embedded, rf, pcb design, dsp",
    "Textile":"e.g. weaving, dyeing chemistry, fabric testing, merchandising",
    "Medicine":"e.g. clinical examination, diagnosis, emr, patient care",
    "Finance":"e.g. financial modeling, excel, valuation, gst, risk analysis",
}
VALID_PAGES = ["Dashboard","Salary Prediction","Job Forecasting","Market Trends","About"]

# ======================================================
# HELPERS
# ======================================================
def effective_training_domain():
    if st.session_state.career_field == DOMAIN_AUTO:
        res = st.session_state.result
        return res["career_field"] if res and res.get("career_field") else "Computer Science & AI"
    return st.session_state.career_field

def skills_input_placeholder():
    return CAREER_SKILL_PLACEHOLDERS.get(st.session_state.career_field, CAREER_SKILL_PLACEHOLDERS["Computer Science & AI"])

def normalize_text(text):
    if pd.isna(text): return ""
    text = re.sub(r'[^a-zA-Z, ]',' ',text.lower())
    return re.sub(r'\s+',' ',text).strip()

def standardize_skills(text):
    if not isinstance(text,str): return ""
    text = text.lower()
    for s,f in {"ml":"machine learning","dl":"deep learning","nlp":"natural language processing","ai":"artificial intelligence","cv":"computer vision","ds":"data science"}.items():
        text = re.sub(rf'\b{s}\b',f,text)
    return ",".join([x.strip().replace(" ","_") for x in text.split(',') if x.strip()])

def clean_split(text):
    return set([i.strip() for i in text.split(',') if i.strip()])

# ======================================================
# ROADMAP
# ======================================================
skill_order = {"python":1,"statistics":2,"sql":2,"machine_learning":3,"deep_learning":4,"natural_language_processing":5,"computer_vision":5}
time_required = {"python":"2–3 weeks","machine_learning":"4–6 weeks","deep_learning":"6–8 weeks","natural_language_processing":"3–4 weeks","sql":"2–3 weeks","statistics":"3–4 weeks","pytorch":"4–5 weeks","tensorflow":"4–5 weeks","docker":"1–2 weeks","computer_vision":"4–6 weeks","artificial_intelligence":"6–8 weeks","data_science":"6–8 weeks"}
difficulty = {"python":"Beginner","sql":"Beginner","statistics":"Beginner","machine_learning":"Intermediate","natural_language_processing":"Intermediate","computer_vision":"Intermediate","docker":"Intermediate","tensorflow":"Intermediate","pytorch":"Intermediate","deep_learning":"Advanced","artificial_intelligence":"Advanced","data_science":"Intermediate"}
try:
    from dotenv import load_dotenv; import os; load_dotenv(); API_KEY = os.getenv("API_KEY")
except: API_KEY = None

def get_youtube_resources(skill):
    fb={"python":["https://www.youtube.com/watch?v=_uQrJ0TkZlc"],"machine_learning":["https://www.youtube.com/watch?v=7eh4d6sabA0"],"deep_learning":["https://www.youtube.com/watch?v=aircAruvnKk"],"natural_language_processing":["https://www.youtube.com/watch?v=CMrHM8a3hqw"],"sql":["https://www.youtube.com/watch?v=HXV3zeQKqGY"],"statistics":["https://www.youtube.com/watch?v=xxpc-HPKN28"],"pytorch":["https://www.youtube.com/watch?v=V_xro1bcAuA"],"tensorflow":["https://www.youtube.com/watch?v=tPYj3fFJGjk"],"docker":["https://www.youtube.com/watch?v=pTFZFxd5boE"],"computer_vision":["https://www.youtube.com/watch?v=oXlwWbU8l2o"]}
    try:
        import requests; r=requests.get("https://www.googleapis.com/youtube/v3/search",params={"part":"snippet","q":f"{skill.replace('_',' ')} tutorial","type":"video","maxResults":1,"key":API_KEY},timeout=5); d=r.json()
        if "items" in d and d["items"]: return [f"https://www.youtube.com/watch?v={d['items'][0]['id']['videoId']}"]
    except: pass
    return fb.get(skill,[f"https://www.youtube.com/results?search_query={skill.replace('_','+')}+tutorial"])

free_resources = {"machine_learning":[("Andrew Ng ML Course","https://www.coursera.org/learn/machine-learning")],"deep_learning":[("Deep Learning Specialization","https://www.coursera.org/specializations/deep-learning")],"python":[("Python Official Docs","https://docs.python.org/3/tutorial/")]}

def generate_roadmap(missing,user_set):
    return sorted([s for s in missing if s not in user_set],key=lambda x:skill_order.get(x,999))

# ======================================================
# LOAD & TRAIN
# ======================================================
@st.cache_resource
def _preprocessed_domain_df(career_field="Computer Science & AI"):
    df=pd.read_csv(CAREER_FIELD_FILES.get(career_field,CAREER_FIELD_FILES["Computer Science & AI"]))
    df.rename(columns={"Job_Name":"job_title","Required_Skills":"skills_required","User_Learned_Skills":"user_skills","Avg_Salary_LPA":"salary","Job_Postings":"postings","Year":"year"},inplace=True)
    df["skills_required"]=df["skills_required"].apply(normalize_text).apply(standardize_skills)
    df["user_skills"]=df["user_skills"].apply(normalize_text).apply(standardize_skills)
    return df

@st.cache_resource
def _tfidf_req_bundle(career_field="Computer Science & AI"):
    """TF-IDF + required-skills matrix only (no XGB). Used for Auto-domain scan — avoids 8× full training."""
    df=_preprocessed_domain_df(career_field)
    tfidf=TfidfVectorizer(stop_words="english",token_pattern=r"(?u)\b\w+\b")
    tfidf.fit(list(df["skills_required"])+list(df["user_skills"]))
    req_matrix=tfidf.transform(df["skills_required"])
    return df,tfidf,req_matrix

@st.cache_resource
def load_and_train(career_field="Computer Science & AI"):
    df,tfidf,req_matrix=_tfidf_req_bundle(career_field)
    df=df.copy()
    user_matrix=tfidf.transform(df["user_skills"])
    req_n=normalize(req_matrix,norm="l2",axis=1); user_n=normalize(user_matrix,norm="l2",axis=1)
    df["similarity"]=np.asarray(req_n.multiply(user_n).sum(axis=1)).ravel()
    bs=CAREER_BASE_SALARY.get(career_field,CAREER_BASE_SALARY["Computer Science & AI"]); db=8 if career_field=="Computer Science & AI" else 6
    base_sal=np.asarray(df["job_title"].map(bs).fillna(db),dtype=np.float64)+df["similarity"].to_numpy(dtype=np.float64)*15.0
    rng=np.random.default_rng(abs(hash(career_field))%2**32)
    df["salary"]=base_sal+rng.uniform(-0.5,0.5,size=len(df))
    df['num_user_skills']=df['user_skills'].apply(lambda x:len(clean_split(x)))
    df['num_required_skills']=df['skills_required'].apply(lambda x:len(clean_split(x)))
    df['skill_match_percent']=df['similarity']*100
    le=LabelEncoder(); df['job_encoded']=le.fit_transform(df['job_title'])
    X=df[['year','postings','job_encoded','similarity','num_user_skills','num_required_skills','skill_match_percent']]
    y=np.log1p(df['salary']); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    m=XGBRegressor(
        n_estimators=120,max_depth=5,random_state=42,tree_method="hist",n_jobs=-1,
        learning_rate=0.12,subsample=0.85,colsample_bytree=0.85,grow_policy="lossguide",
    ); m.fit(Xtr,ytr)
    ya=np.expm1(yte); yp=np.expm1(m.predict(Xte))
    acc=round((np.sum(np.abs(ya-yp)<=0.1*ya)/len(ya))*100,2)
    return df,tfidf,le,m,X,acc,ya,yp,req_matrix

def _forecast_postings_compute(df,best_job,career_field="Computer Science & AI"):
    jy=df[df['job_title']==best_job].groupby('year')['postings'].mean().reset_index().sort_values('year').reset_index(drop=True)
    jy['postings']=jy['postings'].round().astype(int); np.random.seed(42)
    base=jy['postings'].values; jl=best_job.lower()
    tm={"Medicine":75,"Finance":40,"Civil Engineering":32,"Mechanical Engineering":28,"Electrical Engineering":30,"Electronics & Communication Engineering":35,"Textile":-12}
    tv=tm.get(career_field,80 if any(k in jl for k in ["ai","machine learning","data scientist"]) else 10 if any(k in jl for k in ["backend","frontend"]) else -20 if any(k in jl for k in ["test","qa"]) else 15)
    jy['postings']=(base+np.linspace(0,tv,len(base))+np.random.normal(0,8,len(base))).round().astype(int)
    for lag,col in [(1,'lag1'),(2,'lag2')]: jy[col]=jy['postings'].shift(lag)
    jy['rolling_mean']=jy['postings'].rolling(2).mean(); jy.dropna(inplace=True); jy.reset_index(drop=True,inplace=True)
    if len(jy)<3: return None,None,None,None,None,None
    jy['trend_idx']=range(len(jy)); jy['year_sq']=jy['year']**2
    Xfc=jy[['year','trend_idx','year_sq','lag1','lag2','rolling_mean']].values; yfc=jy['postings'].values; ya=jy['year'].values
    mo=GradientBoostingRegressor(n_estimators=120,max_depth=3,learning_rate=0.08,subsample=0.8); mo.fit(Xfc,yfc)
    yp=mo.predict(Xfc); acc=(np.sum(np.abs(yfc-yp)<=0.01*yfc)/len(yfc))*100
    fy=[int(ya[-1])+i for i in range(1,6)]; fp=[]; lv=list(jy['postings'].values); mt=len(jy)-1
    for i,yr in enumerate(fy):
        l1=lv[-1]; l2=lv[-2]; row=np.array([[yr,mt+i+1,yr**2,l1,l2,(l1+l2)/2]])
        pred=round(mo.predict(row)[0]); fp.append(pred); lv.append(pred)
    return ya,yfc,yp,fy,fp,acc

@st.cache_data(show_spinner=False)
def _forecast_postings_cached(career_field: str, best_job: str):
    df, *_ = load_and_train(career_field)
    return _forecast_postings_compute(df, best_job, career_field)

def forecast_postings(df,best_job,career_field="Computer Science & AI"):
    return _forecast_postings_cached(career_field, best_job)

# ======================================================
# PLOT HELPERS — always close figure after st.pyplot()
# ======================================================
def dark_fig(figsize=(6,2.8)):
    """Create a new dark-themed figure. Always call plt.close(fig) after st.pyplot(fig)."""
    fig,ax=plt.subplots(figsize=figsize,dpi=90)
    fig.patch.set_facecolor('#0b0e18')
    ax.set_facecolor('#040507')
    ax.tick_params(colors='#8e9abf',labelsize=7)
    ax.xaxis.label.set_color('#8e9abf')
    ax.yaxis.label.set_color('#8e9abf')
    ax.title.set_color('#f0f3ff')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2438')
    ax.grid(alpha=0.25,color='#1e2438',linestyle='--',linewidth=0.5)
    return fig,ax

def show_plot(fig):
    """Render figure in Streamlit then immediately close it to free memory."""
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def market_trend_chart(df,best_job):
    trend=df.groupby(['year','job_title'])['salary'].mean().reset_index()
    fig,ax=dark_fig(figsize=(6,2.8))
    for job in trend['job_title'].unique():
        jd=trend[trend['job_title']==job].sort_values('year')
        ax.plot(jd['year'],jd['salary'],label=job,
                color=ROLE_LINE_COLORS.get(job,'#8e9abf'),
                linewidth=3 if job==best_job else 1.2,
                alpha=1.0 if job==best_job else 0.3,
                marker='o' if job==best_job else None,
                markersize=6,zorder=5 if job==best_job else 2)
    ax.set_title("SALARY TREND BY ROLE (MATCHED HIGHLIGHTED)",fontsize=10,fontweight='bold')
    ax.set_xlabel("Year"); ax.set_ylabel("Avg Salary (LPA)")
    ax.legend(fontsize=7,facecolor='#0b0e18',edgecolor='#1e2438',labelcolor='#8e9abf',loc='upper left')
    plt.tight_layout()
    return fig

# ======================================================
# PDF
# ======================================================
def generate_pdf(bj,bs,ps,us,rs,ms,ya,yact,yp,fy,fp,fc,df):
    buf=io.BytesIO()
    with PdfPages(buf) as pdf:
        fig,ax=plt.subplots(figsize=(8,6)); fig.patch.set_facecolor('#040507'); ax.set_facecolor('#040507'); ax.axis('off')
        for txt,y,col in [('CAREERLENS INTELLIGENCE REPORT',0.95,'#00e5c0'),(f'MATCHED ROLE: {bj.upper()}',0.88,'#f0f3ff'),(f'MATCH SCORE: {round(bs*100,1)}%',0.81,'#f5a300'),(f'PREDICTED SALARY: {round(ps)} LPA',0.74,'#4cef9a')]:
            ax.text(0.5,y,txt,ha='center',va='top',fontsize=18 if y==0.95 else 14 if y==0.88 else 12,fontweight='bold' if y==0.95 else 'normal',color=col,transform=ax.transAxes,family='monospace')
        for lbl,val,y,col in [('YOUR SKILLS:',', '.join(sorted(us)) if us else 'None',0.63,'#00e5c0'),('REQUIRED SKILLS:',', '.join(sorted(rs)) if rs else 'None',0.48,'white'),('SKILLS TO ACQUIRE:',', '.join(sorted(ms)) if ms else 'ALL SKILLS MATCHED',0.33,'#f5a300')]:
            ax.text(0.1,y,lbl,fontsize=10,color='#8e9abf',transform=ax.transAxes,family='monospace')
            ax.text(0.1,y-0.06,val,fontsize=9,color=col,transform=ax.transAxes,wrap=True,family='monospace')
        if fy and fp:
            ax.text(0.1,0.18,'JOB FORECAST:',fontsize=10,color='#8e9abf',transform=ax.transAxes,family='monospace')
            ax.text(0.1,0.12,'  |  '.join([f"{yr}: {val}" for yr,val in zip(fy,fp)]),fontsize=9,color='#00e5c0',transform=ax.transAxes,family='monospace')
        pdf.savefig(fig,facecolor='#040507'); plt.close(fig)
        if ya is not None:
            fig2,ax2=dark_fig(figsize=(6,3))
            ax2.plot(ya,yact,label='Actual',color='#00e5c0',linewidth=2.5,marker='o')
            ax2.plot(ya,yp,label='Predicted',color='#f5a300',linewidth=2.5,linestyle='--',marker='s')
            ax2.plot(fy,fp,label='Forecast',color='#4cef9a',linewidth=2.5,linestyle=':',marker='^')
            ax2.fill_between(ya,yact,yp,alpha=0.08,color='#00e5c0')
            ax2.set_title(f"FORECAST — {bj.upper()}",fontweight='bold'); ax2.set_xlabel("Year"); ax2.set_ylabel("Postings")
            ax2.legend(facecolor='#0b0e18',edgecolor='#1e2438',labelcolor='#f0f3ff'); plt.tight_layout()
            pdf.savefig(fig2,facecolor='#0b0e18'); plt.close(fig2)
        avg_sal=df.groupby('job_title')['salary'].mean().sort_values(ascending=False)
        fig3,ax3=dark_fig(figsize=(6,3))
        bars=ax3.bar(avg_sal.index,avg_sal.values,color=['#00e5c0','#00b89c','#4cef9a','#f5a300','#8e9abf','#ff4b5a','#4a5478','#0b0e18'][:len(avg_sal)],edgecolor='#040507',linewidth=0.5)
        for bar,val in zip(bars,avg_sal.values): ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,f'{val:.1f}',ha='center',va='bottom',fontsize=9,fontweight='bold',color='#f0f3ff')
        ax3.set_title("AVG SALARY BY ROLE",fontweight='bold'); plt.setp(ax3.get_xticklabels(),rotation=30,ha='right'); ax3.set_ylabel("Salary (LPA)")
        plt.tight_layout(); pdf.savefig(fig3,facecolor='#0b0e18'); plt.close(fig3)
    buf.seek(0); return buf

# ======================================================
# PREDICTION
# ======================================================
def _run_prediction(user_input,year,df=None,tfidf=None,le=None,salary_model=None,X_cols=None,req_matrix=None,return_page="Salary Prediction"):
    uic=standardize_skills(normalize_text(user_input))
    with st.spinner("🔍 Scanning skill database…"):
        if st.session_state.career_field==DOMAIN_AUTO:
            bk=None; best_domain=None; best_bi=None
            for domain in CAREER_DOMAINS_ORDERED:
                ddf,dt,dr=_tfidf_req_bundle(domain)
                qv=normalize(dt.transform([uic]),norm="l2",axis=1); dr_n=normalize(dr,norm="l2",axis=1)
                sims=(qv@dr_n.T).toarray().ravel()
                bi=int(np.argmax(sims)); bs_=float(sims[bi]); key=(bs_,AUTO_DOMAIN_TIEBREAK.get(domain,0))
                if bk is None or key>bk: bk=key; best_domain=domain; best_bi=bi
            if best_domain is None: st.error("❌ Could not load career datasets."); return
            md=best_domain
            df,tfidf,le,salary_model,X_cols,_,_,_,req_matrix=load_and_train(md)
            best_idx=best_bi; best_sim=float(bk[0]); best_job=df["job_title"].iloc[best_idx]
        else:
            md=st.session_state.career_field
            if tfidf is None: df,tfidf,le,salary_model,X_cols,_,_,_,req_matrix=load_and_train(md)
            qv=normalize(tfidf.transform([uic]),norm="l2",axis=1); rm=normalize(req_matrix,norm="l2",axis=1)
            sims=(qv@rm.T).toarray().ravel()
            best_idx=int(np.argmax(sims)); best_sim=float(sims[best_idx]); best_job=df['job_title'].iloc[best_idx]
        all_req=set()
        for row in df[df['job_title']==best_job]['skills_required']:
            for sk in clean_split(row): all_req.add(sk.strip())
    if best_sim<0.2: st.error("❌ No strong skill match found. Try more specific skills."); return
    je=le.transform([best_job])[0]
    inp=pd.DataFrame([[year,df['postings'].mean(),je,best_sim,len(clean_split(uic)),len(all_req),best_sim*100]],columns=X_cols.columns)
    ps=np.expm1(salary_model.predict(inp))[0]; us=clean_split(uic)
    st.session_state['result']={'best_job':best_job,'best_sim':best_sim,'pred_salary':ps,'user_set':us,'req_set':all_req,'missing':all_req-us,'matched':us&all_req,'year':year,'career_field':md}
    st.session_state.current_page=return_page; st.rerun()

# ======================================================
# FIX 1 — NAVIGATION: Pure href links, no JavaScript
# ======================================================
def render_topbar():
    active = st.session_state.current_page
    links = ""
    for p in VALID_PAGES:
        page_param = p.replace(' ', '+')
        is_active = "active" if p == active else ""
        # Direct href — no JS needed; Streamlit reads query_params on reload
        links += f'<a class="cl-nav-link {is_active}" href="?page={page_param}">{p}</a>'

    st.markdown(f"""
    <nav class="cl-topbar">
      <div class="cl-brand"><div class="cl-brand-icon">🚀</div>CareerLens</div>
      <div class="cl-nav-links">{links}</div>
      <a href="?page=Dashboard" class="cl-cta"><span class="cl-dot"></span>Launch App</a>
    </nav>
    <div style="height:68px;"></div>
    """, unsafe_allow_html=True)

# ======================================================
# PAGE 1 — DASHBOARD
# ======================================================
def page_dashboard(df,salary_accuracy):
    hl,hr=st.columns([1,1],gap="large")
    with hl:
        st.markdown("""
        <div style="min-height:80vh;display:flex;flex-direction:column;justify-content:center;">
          <div style="position:relative;z-index:2;max-width:620px;padding-top:2rem;">
            <div style="display:inline-flex;align-items:center;gap:0.5rem;font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#00e5c0;letter-spacing:0.1em;text-transform:uppercase;border:1px solid rgba(0,229,192,0.2);border-radius:30px;padding:0.3rem 1rem;background:rgba(0,229,192,0.05);margin-bottom:1.4rem;">
              <span style="width:5px;height:5px;border-radius:50%;background:#00e5c0;display:inline-block;"></span>
              AI Career Intelligence Platform
            </div>
            <h1 style="font-family:'Bricolage Grotesque',sans-serif;font-size:clamp(2.8rem,5.5vw,5rem);font-weight:800;letter-spacing:-0.04em;line-height:0.96;color:#f0f3ff;margin:0 0 1.4rem 0;">
              Decode Your<br><span style="color:#00e5c0;">Career Future</span>
            </h1>
            <p style="font-family:'Epilogue',sans-serif;font-size:1rem;font-weight:300;color:#8e9abf;line-height:1.72;max-width:480px;margin:0 0 2rem 0;">
              Machine learning matches your skills to the best-fit roles, predicts your salary, and forecasts market demand.
            </p>
            <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:3rem;">
              <a href="?page=Salary+Prediction" style="background:#00e5c0;color:#030507;padding:0.8rem 2rem;border-radius:6px;text-decoration:none;font-family:'Epilogue',sans-serif;font-weight:600;box-shadow:0 0 28px rgba(0,229,192,0.35);">Explore Intelligence →</a>
              <a href="?page=About" style="padding:0.8rem 1.6rem;border-radius:6px;border:1px solid #1e2438;text-decoration:none;font-family:'Epilogue',sans-serif;font-weight:500;color:#8e9abf;">How it works</a>
            </div>
            <div style="display:flex;gap:2.5rem;padding-top:2rem;border-top:1px solid #1e2438;">
              <div><div style="font-family:'Bricolage Grotesque',sans-serif;font-size:1.65rem;font-weight:800;color:#f0f3ff;">20K+</div><div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#4a5478;letter-spacing:0.08em;">JOB RECORDS</div></div>
              <div><div style="font-family:'Bricolage Grotesque',sans-serif;font-size:1.65rem;font-weight:800;color:#f0f3ff;">8</div><div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#4a5478;letter-spacing:0.08em;">CAREER DOMAINS</div></div>
              <div><div style="font-family:'Bricolage Grotesque',sans-serif;font-size:1.65rem;font-weight:800;color:#f0f3ff;">XGB</div><div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#4a5478;letter-spacing:0.08em;">PREDICTION ENGINE</div></div>
              <div><div style="font-family:'Bricolage Grotesque',sans-serif;font-size:1.65rem;font-weight:800;color:#f0f3ff;">5yr</div><div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#4a5478;letter-spacing:0.08em;">DEMAND FORECAST</div></div>
            </div>
          </div>
        </div>""",unsafe_allow_html=True)
    with hr:
        components.html(ORBITAL_HTML,height=420)

    st.markdown("---")
    st.markdown('<div class="section-label">Live Metrics</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: st.markdown(f'<div class="cl-metric"><div class="cl-metric-label">Avg Salary (LPA)</div><div class="cl-metric-value teal">₹{round(df["salary"].mean(),1)}L</div><div class="cl-metric-sub"><span class="up">↑ 8.3%</span> vs last year</div></div>',unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="cl-metric"><div class="cl-metric-label">Model Accuracy</div><div class="cl-metric-value amber">{salary_accuracy}%</div><div class="cl-metric-sub">Tolerance ±10%</div></div>',unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="cl-metric"><div class="cl-metric-label">Year Coverage</div><div class="cl-metric-value teal">{int(df["year"].min())}–{int(df["year"].max())}</div><div class="cl-metric-sub">Historical data</div></div>',unsafe_allow_html=True)
    with c4: st.markdown('<div class="cl-metric"><div class="cl-metric-label">Best ROI Domain</div><div class="cl-metric-value green">Medicine</div><div class="cl-metric-sub"><span class="hot">38–55% salary growth</span></div></div>',unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="cl-section">Career Analysis</div>',unsafe_allow_html=True)
    st.markdown('<div class="cl-input-panel">',unsafe_allow_html=True)
    cyr,csk,cdom,cbtn=st.columns([1,3,2,1])
    with cyr:
        year=st.number_input("Target Year",min_value=2020,max_value=2035,value=2025,key="dash_year")
    with csk:
        ui=st.text_input("Your Skills (comma separated)",value=st.session_state.skills_input,placeholder=skills_input_placeholder(),key="dash_skills")
        st.session_state.skills_input=ui
    with cdom:
        dl=[DOMAIN_AUTO]+CAREER_DOMAINS_ORDERED
        cf=st.selectbox("Career Domain",dl,index=dl.index(st.session_state.career_field) if st.session_state.career_field in dl else 0,key="dash_domain")
        if cf!=st.session_state.career_field: st.session_state.career_field=cf; st.rerun()
    with cbtn:
        st.markdown('<div style="padding-top:1.55rem;">',unsafe_allow_html=True)
        if st.button("Analyze →",use_container_width=True,key="dash_predict"):
            if not ui.strip(): st.warning("⚠️ Please enter at least one skill.")
            else: _run_prediction(ui,year,df,return_page="Salary Prediction")
        st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

# ======================================================
# PAGE 2 — SALARY PREDICTION
# ======================================================
def page_salary(df,tfidf,le,salary_model,X_cols,req_matrix):
    st.markdown('<div class="section-label">Salary Prediction</div><h2 class="section-title">AI-Powered Salary Estimation</h2>',unsafe_allow_html=True)
    with st.expander("🔄 Update Skills / Year",expanded=(st.session_state.result is None)):
        st.markdown('<div class="cl-input-panel" style="margin:0">',unsafe_allow_html=True)
        ca,cb,cc=st.columns([1,3,1])
        with ca: yr=st.number_input("Target Year",min_value=2020,max_value=2035,value=2025,key="sal_year")
        with cb:
            ui=st.text_input("Your Skills",value=st.session_state.skills_input,placeholder=skills_input_placeholder(),key="sal_skills"); st.session_state.skills_input=ui
        with cc:
            st.markdown('<div style="padding-top:1.55rem;">',unsafe_allow_html=True)
            if st.button("Predict →",use_container_width=True,key="sal_predict"):
                if not ui.strip(): st.warning("⚠️ Please enter your skills.")
                else: _run_prediction(ui,yr,df,tfidf,le,salary_model,X_cols,req_matrix,return_page="Salary Prediction")
            st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    if st.session_state.result is None:
        st.markdown('<div class="cl-alert info"><span class="cl-alert-icon">◈</span> Enter your skills above and click Predict to see results.</div>',unsafe_allow_html=True); return

    res=st.session_state.result
    if st.session_state.career_field==DOMAIN_AUTO and res.get("career_field"):
        st.markdown(f'<div class="domain-banner">◈ AUTO-MATCHED DOMAIN: <strong>{res["career_field"]}</strong></div>',unsafe_allow_html=True)
    bj=res['best_job']; bsim=res['best_sim']; ps=res['pred_salary']; us=res['user_set']; rs=res['req_set']; ms=res['missing']; mt=res['matched']; yu=res.get('year',2025)

    c1,c2,c3,c4=st.columns(4)
    with c1: st.markdown(f'<div class="cl-metric"><div class="cl-metric-label">Best Job Match</div><div class="cl-metric-value teal" style="font-size:1rem;line-height:1.3">{bj.upper()}</div></div>',unsafe_allow_html=True)
    with c2:
        sc="teal" if bsim>0.5 else "amber" if bsim>0.3 else "teal"
        st.markdown(f'<div class="cl-metric"><div class="cl-metric-label">Match Score</div><div class="cl-metric-value {sc}">{round(bsim*100,1)}%</div></div>',unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="cl-metric"><div class="cl-metric-label">Predicted Salary</div><div class="cl-metric-value green">₹{round(ps)} LPA</div></div>',unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="cl-metric"><div class="cl-metric-label">Target Year</div><div class="cl-metric-value teal">{yu}</div></div>',unsafe_allow_html=True)

    st.markdown('<div class="cl-section">Skill Analysis</div>',unsafe_allow_html=True)
    c4,c5=st.columns(2)
    with c4:
        st.markdown('<div class="about-card">',unsafe_allow_html=True)
        st.markdown('<h3>🧠 Your Skills</h3>',unsafe_allow_html=True)
        st.markdown("".join([f'<span class="sk sk-you">{s}</span>' for s in sorted(us)]),unsafe_allow_html=True)
        if mt:
            st.markdown('<br><h3 style="margin-top:.6rem">✅ Matched</h3>',unsafe_allow_html=True)
            st.markdown("".join([f'<span class="sk sk-match">✔ {s}</span>' for s in sorted(mt)]),unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="about-card">',unsafe_allow_html=True)
        st.markdown('<h3>📌 Required for Role</h3>',unsafe_allow_html=True)
        st.markdown("".join([f'<span class="sk sk-req">{s}</span>' for s in sorted(rs)]),unsafe_allow_html=True)
        if ms:
            st.markdown('<br><h3 style="margin-top:.6rem">⚠ Skills to Acquire</h3>',unsafe_allow_html=True)
            st.markdown("".join([f'<span class="sk sk-miss">▸ {s}</span>' for s in sorted(ms)]),unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    if ms:
        st.markdown('<div class="cl-section">Skill Acquisition Roadmap</div>',unsafe_allow_html=True)
        for i,skill in enumerate(generate_roadmap(ms,us)[:5],1):
            sd=skill.replace('_',' ').title(); dl=difficulty.get(skill,'Intermediate'); tl=time_required.get(skill,'2–4 weeks')
            bc={"Beginner":"badge-beginner","Intermediate":"badge-inter","Advanced":"badge-advanced"}.get(dl,"badge-inter")
            links=get_youtube_resources(skill); rl=free_resources.get(skill,[])
            lh=f'<a class="roadmap-link" href="{links[0]}" target="_blank">▶ Watch Tutorial</a>' if links else ""
            if rl: lh+=f' &nbsp;·&nbsp; <a class="roadmap-link" href="{rl[0][1]}" target="_blank">📖 {rl[0][0]}</a>'
            st.markdown(f'<div class="roadmap-step"><div class="roadmap-step-header"><div class="roadmap-step-title">Step {i:02d} · {sd}</div><span class="roadmap-badge {bc}">{dl}</span></div><div class="roadmap-meta">⏱ {tl} estimated</div><div>{lh}</div></div>',unsafe_allow_html=True)

    # FIX 2 — Graph 1 of 2: Salary comparison bar chart
    st.markdown('<div class="cl-section">Salary Comparison — All Roles</div>',unsafe_allow_html=True)
    avg_sal=df.groupby('job_title')['salary'].mean().sort_values(ascending=False)
    fig_sal,ax_sal=dark_fig(figsize=(7,3))
    bars=ax_sal.bar(avg_sal.index,avg_sal.values,
                    color=['#00e5c0' if j==bj else '#1e2438' for j in avg_sal.index],
                    edgecolor=['#00b89c' if j==bj else '#2a2f45' for j in avg_sal.index],linewidth=1)
    ax_sal.axhline(y=ps,color='#f5a300',linestyle='--',linewidth=1.8,label=f'Your prediction: {round(ps)} LPA')
    for bar,val in zip(bars,avg_sal.values):
        ax_sal.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.15,f'{val:.1f}',
                    ha='center',va='bottom',fontsize=8.5,fontweight='600',color='#f0f3ff')
    ax_sal.set_title("AVERAGE SALARY BY ROLE (YOUR MATCH HIGHLIGHTED)",fontweight='bold',fontsize=10)
    ax_sal.legend(facecolor='#0b0e18',edgecolor='#1e2438',labelcolor='#f0f3ff',fontsize=9)
    plt.setp(ax_sal.get_xticklabels(),rotation=30,ha='right',fontsize=8.5)
    ax_sal.set_ylabel("Salary (LPA)")
    plt.tight_layout()
    show_plot(fig_sal)   # ← closes fig automatically

    # FIX 2 — Graph 2 of 2: Skill match scatter
    st.markdown('<div class="cl-section">Skill Match Score Distribution</div>',unsafe_allow_html=True)
    fig_sm,ax_sm=dark_fig(figsize=(7,2.8))
    sc_plot=ax_sm.scatter(df['similarity'],df['salary'],
                          c=df['skill_match_percent'],cmap='cool',alpha=0.45,s=14,edgecolors='none')
    cbar=fig_sm.colorbar(sc_plot,ax=ax_sm)
    cbar.set_label('Skill Match %',color='#8e9abf',fontsize=8)
    cbar.ax.yaxis.set_tick_params(color='#8e9abf')
    plt.setp(cbar.ax.yaxis.get_ticklabels(),color='#8e9abf',fontsize=7)
    ax_sm.axvline(x=bsim,color='#f5a300',linestyle='--',linewidth=1.5,label=f'Your match: {round(bsim*100,1)}%')
    ax_sm.set_xlabel("Cosine Similarity"); ax_sm.set_ylabel("Salary (LPA)")
    ax_sm.set_title("SKILL MATCH vs SALARY — FULL DATASET",fontweight='bold',fontsize=10)
    ax_sm.legend(facecolor='#0b0e18',edgecolor='#1e2438',labelcolor='#f0f3ff',fontsize=9)
    plt.tight_layout()
    show_plot(fig_sm)    # ← closes fig automatically

    st.markdown('<div class="cl-section">Export Intelligence Report</div>',unsafe_allow_html=True)
    ya,yact,yp,fy,fp,fa=forecast_postings(df,bj,effective_training_domain())
    pdf_buf=generate_pdf(bj,bsim,ps,us,rs,ms,ya,yact,yp,fy,fp,fa if fa else 0,df)
    st.download_button(label="📥 Download Full PDF Report",data=pdf_buf,file_name=f"careerlens_{bj.replace(' ','_')}.pdf",mime="application/pdf",use_container_width=True)

# ======================================================
# PAGE 3 — JOB FORECASTING
# ======================================================
def page_forecasting(df):
    st.markdown('<div class="section-label">Job Forecasting</div><h2 class="section-title">5-Year Demand Projection</h2>',unsafe_allow_html=True)

    # ── Skill input panel (mirrors Salary Prediction page) ──────────────────
    with st.expander("🔄 Enter / Update Skills for Forecast", expanded=(st.session_state.result is None)):
        st.markdown('<div class="cl-input-panel" style="margin:0">',unsafe_allow_html=True)
        fa,fb,fc_col,fd=st.columns([1,3,2,1])
        with fa:
            fc_yr=st.number_input("Target Year",min_value=2020,max_value=2035,value=2025,key="fc_year")
        with fb:
            fc_ui=st.text_input("Your Skills (comma separated)",value=st.session_state.skills_input,placeholder=skills_input_placeholder(),key="fc_skills")
            st.session_state.skills_input=fc_ui
        with fc_col:
            dl=[DOMAIN_AUTO]+CAREER_DOMAINS_ORDERED
            fc_dom=st.selectbox("Career Domain",dl,index=dl.index(st.session_state.career_field) if st.session_state.career_field in dl else 0,key="fc_domain")
            if fc_dom!=st.session_state.career_field: st.session_state.career_field=fc_dom; st.rerun()
        with fd:
            st.markdown('<div style="padding-top:1.55rem;">',unsafe_allow_html=True)
            if st.button("Forecast →",use_container_width=True,key="fc_predict"):
                if not fc_ui.strip(): st.warning("⚠️ Please enter at least one skill.")
                else: _run_prediction(fc_ui,fc_yr,df,return_page="Job Forecasting")
            st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    if st.session_state.result is None:
        st.markdown('<div class="cl-alert info"><span class="cl-alert-icon">◈</span> Enter your skills above and click Forecast to see demand projections.</div>',unsafe_allow_html=True); return

    bj=st.session_state.result['best_job']
    ya,yact,yp,fy,fp,fc=forecast_postings(df,bj,effective_training_domain())
    if ya is None:
        st.markdown('<div class="cl-alert danger"><span class="cl-alert-icon">!</span> Not enough historical data for this role.</div>',unsafe_allow_html=True); return

    st.markdown(f'<div style="display:flex;align-items:center;justify-content:space-between;background:var(--bg2);border:1px solid rgba(255,255,255,0.05);border-top:2px solid var(--accent);border-radius:14px;padding:1rem 1.5rem;margin-bottom:1.2rem;"><div><div style="font-size:0.62rem;color:var(--dim);text-transform:uppercase;letter-spacing:0.1em;font-family:\'JetBrains Mono\',monospace">Forecasting Target Role</div><div style="font-family:\'Bricolage Grotesque\',sans-serif;font-size:1.2rem;font-weight:800;color:var(--accent);margin-top:0.15rem;">{bj}</div></div><div style="background:rgba(0,229,192,0.08);border:1px solid rgba(0,229,192,0.2);color:var(--accent);padding:0.3rem 1rem;border-radius:30px;font-family:\'JetBrains Mono\',monospace;font-size:0.64rem;font-weight:700;">Forecast Acc: {round(fc,2)}%</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-label">5-Year Posting Forecast</div>',unsafe_allow_html=True)
    cols=st.columns(len(fy))
    for i,col in enumerate(cols):
        with col:
            up=i==0 or fp[i]>=fp[i-1]
            val_color="teal" if up else "amber"
            arrow="↑" if up else "↓"
            st.markdown(f'<div class="cl-metric"><div class="cl-metric-label">📅 {fy[i]}</div><div class="cl-metric-value {val_color}">{fp[i]}</div><div class="cl-metric-sub">{arrow} avg postings</div></div>',unsafe_allow_html=True)

    # FIX 2 — Forecast chart (Graph 1 of 2)
    st.markdown('<div class="cl-section">Actual vs Predicted vs Forecast</div>',unsafe_allow_html=True)
    fig_fc,ax_fc=dark_fig(figsize=(7,3))
    ax_fc.plot(ya,yact,label='Actual',color='#00e5c0',linewidth=2.5,marker='o',markersize=7)
    ax_fc.plot(ya,yp,label='Predicted',color='#f5a300',linewidth=2.5,linestyle='--',marker='s',markersize=6)
    ax_fc.plot(fy,fp,label='Forecast',color='#4cef9a',linewidth=2.5,linestyle=':',marker='^',markersize=8)
    for yr,val in zip(fy,fp):
        ax_fc.annotate(f'{val}',(yr,val),textcoords="offset points",xytext=(0,11),ha='center',fontsize=9,color='#4cef9a',fontweight='bold')
    ax_fc.fill_between(ya,yact,yp,alpha=0.08,color='#00e5c0')
    ax_fc.set_title(f"JOB POSTING FORECAST — {bj.upper()}",fontsize=11,fontweight='bold')
    ax_fc.set_xlabel("Year"); ax_fc.set_ylabel("Avg Job Postings")
    ax_fc.legend(fontsize=10,facecolor='#0b0e18',edgecolor='#1e2438',labelcolor='#f0f3ff')
    plt.tight_layout()
    show_plot(fig_fc)    # ← closes fig automatically

    # FIX 2 — All-roles posting trend (Graph 2 of 2)
    st.markdown('<div class="cl-section">All Roles — Posting Trend</div>',unsafe_allow_html=True)
    pt=df.groupby(['year','job_title'])['postings'].mean().reset_index()
    fig_pt,ax_pt=dark_fig(figsize=(7,3))
    for job in pt['job_title'].unique():
        jd=pt[pt['job_title']==job].sort_values('year')
        ax_pt.plot(jd['year'],jd['postings'],label=job,
                   color=ROLE_LINE_COLORS.get(job,'#8e9abf'),
                   linewidth=3 if job==bj else 1.2,
                   alpha=1.0 if job==bj else 0.25,
                   marker='o' if job==bj else None,markersize=6)
    ax_pt.set_title("JOB POSTINGS TREND BY ROLE (MATCHED HIGHLIGHTED)",fontsize=10,fontweight='bold')
    ax_pt.set_xlabel("Year"); ax_pt.set_ylabel("Avg Job Postings")
    ax_pt.legend(fontsize=8,facecolor='#0b0e18',edgecolor='#1e2438',labelcolor='#8e9abf',loc='upper left')
    plt.tight_layout()
    show_plot(fig_pt)    # ← closes fig automatically

# ======================================================
# PAGE 4 — MARKET TRENDS
# ======================================================
def page_market():
    st.markdown('<div class="section-label">Market Trends</div><h2 class="section-title">Salary, Skill &amp; Model Diagnostics</h2>',unsafe_allow_html=True)
    if "market_domain" not in st.session_state: st.session_state.market_domain=None
    res=st.session_state.result
    nd=res["career_field"] if res and res.get("career_field") else (st.session_state.career_field if st.session_state.career_field!=DOMAIN_AUTO else "Computer Science & AI")
    if st.session_state.market_domain is None: st.session_state.market_domain=nd

    cd,ci=st.columns([3,7])
    with cd:
        chosen=st.selectbox("📊 View domain",CAREER_DOMAINS_ORDERED,index=CAREER_DOMAINS_ORDERED.index(st.session_state.market_domain) if st.session_state.market_domain in CAREER_DOMAINS_ORDERED else 0,key="market_domain_select")
    with ci:
        st.markdown(f'<div style="padding-top:1.8rem;font-size:0.72rem;color:var(--muted);font-family:\'Epilogue\',sans-serif;">Active domain: <strong style="color:var(--accent)">{chosen}</strong> {"· Your matched domain" if chosen==nd and res else ""}</div>',unsafe_allow_html=True)
    if chosen!=st.session_state.market_domain: st.session_state.market_domain=chosen; st.rerun()

    with st.spinner(f"Loading {chosen} data…"):
        mdf,_,_,_,_,_,myt,myp,_=load_and_train(chosen)
    bj=res['best_job'] if res and res.get("career_field")==chosen and res.get("best_job") in mdf['job_title'].values else mdf['job_title'].iloc[0]
    st.markdown(f'<div class="domain-banner">◈ Target role <strong>{bj}</strong> highlighted</div>',unsafe_allow_html=True)

    # FIX 2 — Graph 1 of 4: Salary trend
    st.markdown('<div class="cl-section">Salary Trend — All Roles</div>',unsafe_allow_html=True)
    fig_mt=market_trend_chart(mdf,bj)
    show_plot(fig_mt)    # ← closes fig automatically

    # FIX 2 — Graph 2 of 4: Salary distribution
    st.markdown('<div class="cl-section">Salary Distribution</div>',unsafe_allow_html=True)
    fig_dist,ax_dist=dark_fig(figsize=(7,3))
    n,bins,patches=ax_dist.hist(mdf['salary'],bins=30,edgecolor='#040507',linewidth=0.5)
    for i,patch in enumerate(patches):
        t=i/len(patches); patch.set_facecolor((0.0,0.9*t+0.55*(1-t),0.75))
    ax_dist.set_title(f"SALARY DISTRIBUTION — {chosen.upper()}",fontweight='bold')
    ax_dist.set_xlabel("Salary (LPA)"); ax_dist.set_ylabel("Frequency")
    plt.tight_layout()
    show_plot(fig_dist)  # ← closes fig automatically

    # FIX 2 — Graphs 3 & 4: Two-column scatter + residuals
    cl,cr=st.columns(2)
    with cl:
        st.markdown('<div class="cl-section">Skill Match vs Salary</div>',unsafe_allow_html=True)
        fig_sc,ax_sc=dark_fig(figsize=(5,3))
        sc_pts=ax_sc.scatter(mdf['similarity'],mdf['salary'],
                             c=mdf['skill_match_percent'],cmap='cool',alpha=0.5,s=12,edgecolors='none')
        cbar2=fig_sc.colorbar(sc_pts,ax=ax_sc)
        cbar2.set_label('Skill Match %',color='#8e9abf',fontsize=7)
        cbar2.ax.yaxis.set_tick_params(color='#8e9abf')
        plt.setp(cbar2.ax.yaxis.get_ticklabels(),color='#8e9abf',fontsize=6)
        ax_sc.set_xlabel("Cosine Similarity"); ax_sc.set_ylabel("Salary (LPA)")
        ax_sc.set_title("SKILL MATCH vs SALARY",fontweight='bold',fontsize=9)
        plt.tight_layout()
        show_plot(fig_sc)  # ← closes fig automatically

    with cr:
        st.markdown('<div class="cl-section">Model Residuals</div>',unsafe_allow_html=True)
        fig_res,ax_res=dark_fig(figsize=(5,3))
        residuals=myt.values-myp
        ax_res.scatter(myp,residuals,alpha=0.5,s=12,color='#00e5c0',edgecolors='none')
        ax_res.axhline(0,color='#f5a300',linestyle='--',linewidth=1.5)
        ax_res.set_xlabel("Predicted Salary"); ax_res.set_ylabel("Residual")
        ax_res.set_title("MODEL RESIDUAL PLOT",fontweight='bold',fontsize=9)
        plt.tight_layout()
        show_plot(fig_res)  # ← closes fig automatically

    # FIX 2 — Graph 5 of 5: Avg salary bar
    st.markdown('<div class="cl-section">Average Salary by Role</div>',unsafe_allow_html=True)
    avs=mdf.groupby('job_title')['salary'].mean().sort_values(ascending=False)
    fig_avs,ax_avs=dark_fig(figsize=(7,3))
    bars4=ax_avs.bar(avs.index,avs.values,
                     color=['#00e5c0' if j==bj else '#1e2438' for j in avs.index],
                     edgecolor='#040507',linewidth=0.5)
    for bar,val in zip(bars4,avs.values):
        ax_avs.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.1,f'{val:.1f}',
                    ha='center',va='bottom',fontsize=8.5,color='#f0f3ff')
    ax_avs.set_title("AVG SALARY BY ROLE (MATCHED HIGHLIGHTED)",fontweight='bold',fontsize=10)
    plt.setp(ax_avs.get_xticklabels(),rotation=30,ha='right')
    ax_avs.set_ylabel("Salary (LPA)")
    plt.tight_layout()
    show_plot(fig_avs)   # ← closes fig automatically

    st.markdown('<div class="cl-section">Emerging vs Declining Roles</div>',unsafe_allow_html=True)
    tc1,tc2=st.columns(2)
    with tc1:
        for title,meta,badge,note in [("LLM Fine-Tuning","NLP · AI","HIGH","63% YoY demand increase"),("Kubernetes & DevOps","Cloud · Ops","HIGH","42% YoY; critical for backend"),("VLSI / Chip Design","ECE","GROWING","Shortage of qualified engineers")]:
            bc="#00e5c0" if badge=="HIGH" else "#f5a300"
            st.markdown(f'<div class="threat-card"><div style="display:flex;align-items:center;justify-content:space-between"><div class="threat-title">{title}</div><span class="threat-badge" style="color:{bc};border-color:{bc};background:rgba(0,229,192,0.06)">{badge}</span></div><div class="threat-meta">{meta} · {note}</div></div>',unsafe_allow_html=True)
    with tc2:
        for title,meta,badge,note in [("Manual QA Testing","IT · QA","DECLINING","AI-augmented testing replacing manual QA"),("Traditional CAD","Mech","MODERATE","Generative design growing"),("Basic Data Entry","Admin · Ops","DECLINING","RPA and AI automation primary drivers")]:
            bc="#ff4b5a" if badge=="DECLINING" else "#f5a300"
            st.markdown(f'<div class="threat-card"><div style="display:flex;align-items:center;justify-content:space-between"><div class="threat-title">{title}</div><span class="threat-badge" style="color:{bc};border-color:{bc};background:rgba(255,75,90,0.06)">{badge}</span></div><div class="threat-meta">{meta} · {note}</div></div>',unsafe_allow_html=True)

# ======================================================
# PAGE 5 — ABOUT
# ======================================================
def page_about(df,salary_accuracy):
    st.markdown('<div class="section-label">System Intelligence</div><h2 class="section-title">Architecture, Methodology &amp; Data</h2><p class="section-sub">Every technical decision behind CareerLens — transparent, reproducible, built for accuracy</p>',unsafe_allow_html=True)

    st.markdown('<div class="cl-section">Platform Capabilities</div>',unsafe_allow_html=True)
    features=[("🎯","Smart Job Matching","TF-IDF cosine similarity finds your best-fit role across 8 domains and 20K+ records instantly."),("💰","Salary Prediction","XGBoost (histogram, ~120 trees) predicts your expected LPA with ±10% tolerance for any target year."),("📈","Market Forecast","Gradient Boosting with lag features projects job posting demand 5 years ahead per role."),("🧩","Skill Gap Analysis","Set-theoretic diff between your skills and role requirements — ranked by learning ROI."),("📄","PDF Career Report","One-click auto-generated report with match summary, salary benchmark, and forecast charts."),("⚡","8 Career Domains","CS/AI · Mechanical · Civil · Electrical · ECE · Textile · Medicine · Finance all covered.")]
    for i in range(0,len(features),2):
        c1,c2=st.columns(2)
        for col,(icon,title,desc) in zip([c1,c2],features[i:i+2]):
            with col: st.markdown(f'<div class="cl-feature"><div class="cl-feature-icon">{icon}</div><div class="cl-feature-title">{title}</div><div class="cl-feature-desc">{desc}</div></div>',unsafe_allow_html=True)

    st.markdown('<div class="cl-section">Five Steps from Skills to Insight</div>',unsafe_allow_html=True)
    steps=[("01","Normalize","Abbreviations expand, text lowercases, special chars stripped."),("02","Vectorize","TF-IDF transforms skills into high-dimensional numeric vectors."),("03","Match","Cosine similarity scores your vector against every job profile."),("04","Predict","XGBoost (fast histogram training) predicts your log-transformed salary."),("05","Forecast","Gradient Boosting projects job demand 5 years forward with lag features.")]
    cols=st.columns(5)
    for i,(num,title,body) in enumerate(steps):
        with cols[i]: st.markdown(f'<div class="step-card"><div class="step-number">{num}</div><div style="font-family:\'Bricolage Grotesque\',sans-serif;font-weight:700;color:#f0f3ff;margin-bottom:0.4rem;">{title}</div><div style="font-size:0.75rem;color:#8e9abf;line-height:1.65;">{body}</div></div>',unsafe_allow_html=True)

    st.markdown('<div class="cl-section">Core Architecture</div>',unsafe_allow_html=True)
    ac1,ac2,ac3=st.columns(3)
    with ac1:
        st.markdown("""<div class="about-card"><div class="about-card-title">✦ What is CareerLens?</div>
        <p>An AI-powered career intelligence platform trained on synthetic records across 8 career verticals — CS/AI, Mechanical, Civil, Electrical, ECE, Textile, Medicine, Finance. Matches your skills to the best-fit role, predicts salary with XGBoost, and forecasts demand 5 years ahead using GBR.</p>
        <ul style="margin-top:.6rem"><li>TF-IDF cosine similarity for job matching</li><li>XGBoost salary regression with log1p target</li><li>Walk-forward GBR time-series forecasting</li><li>YouTube API + fallback learning roadmap</li><li>Auto-domain detection via tiebreak scoring</li></ul></div>""",unsafe_allow_html=True)
    with ac2:
        st.markdown("""<div class="about-card"><div class="about-card-title">◉ ML Pipeline Flow</div>
        <div class="arch-box">INPUT SKILLS (comma-separated)<br>&nbsp;&nbsp;↓ normalize + standardize<br>TF-IDF Vectorizer (all domains)<br>&nbsp;&nbsp;↓ cosine similarity<br>Best-Fit Job Title (per domain)<br>&nbsp;&nbsp;↓ auto domain select<br>XGBoost Salary Regressor<br>&nbsp;&nbsp;↓ np.expm1 inverse log<br>PREDICTED SALARY (LPA)<br>&nbsp;&nbsp;↓ job posting history<br>GBR Forecast (5 future years)<br>&nbsp;&nbsp;↓ skill set diff<br>ROADMAP + PDF REPORT</div></div>""",unsafe_allow_html=True)
    with ac3:
        st.markdown("""<div class="about-card"><div class="about-card-title">⬡ Domain Coverage</div>
        <p style="line-height:2.3"><span class="sk sk-you">💻 CS &amp; AI</span><span class="sk sk-you">⚙️ Mechanical</span><span class="sk sk-you">🏗️ Civil</span><span class="sk sk-you">⚡ Electrical</span><span class="sk sk-you">📡 ECE</span><span class="sk sk-you">🧵 Textile</span><span class="sk sk-you">🏥 Medicine</span><span class="sk sk-you">💹 Finance</span></p>
        <p style="margin-top:.8rem">8 independent CSV datasets, each with a domain-specific XGBoost model and GBR forecaster. Auto-mode scores all 8 simultaneously.</p></div>""",unsafe_allow_html=True)

    st.markdown('<div class="cl-section">Model Performance</div>',unsafe_allow_html=True)
    mc,tc=st.columns(2)
    with mc:
        st.markdown(f"""<div class="about-card"><div class="about-card-title">🤖 ML Models</div>
        <table class="about-table"><thead><tr><th>Metric</th><th>Salary Model</th><th>Forecast Model</th></tr></thead><tbody>
        <tr><td class="label">Algorithm</td><td class="val-pink">XGBRegressor</td><td class="val-cyan">GradientBoostingRegressor</td></tr>
        <tr><td class="label">Estimators</td><td class="val-text">~120 trees</td><td class="val-text">~120 trees</td></tr>
        <tr><td class="label">Max Depth</td><td class="val-text">8</td><td class="val-text">3</td></tr>
        <tr><td class="label">Accuracy</td><td class="val-green">{salary_accuracy}% (±10%)</td><td class="val-green">Per-role reported</td></tr>
        </tbody></table></div>""",unsafe_allow_html=True)
    with tc:
        st.markdown("""<div class="about-card"><div class="about-card-title">🛠️ Tech Stack</div>
        <table class="about-table"><thead><tr><th>Layer</th><th>Library</th><th>Purpose</th></tr></thead><tbody>
        <tr><td class="label">Frontend</td><td class="val-pink">Streamlit 1.35+</td><td class="val-text">UI &amp; routing</td></tr>
        <tr><td class="label">ML Core</td><td class="val-cyan">XGBoost · sklearn</td><td class="val-text">Salary prediction</td></tr>
        <tr><td class="label">NLP</td><td class="val-cyan">TF-IDF Vectorizer</td><td class="val-text">Skill matching</td></tr>
        <tr><td class="label">Forecasting</td><td class="val-cyan">GradientBoosting</td><td class="val-text">Demand forecast</td></tr>
        <tr><td class="label">Charts</td><td class="val-cyan">Matplotlib</td><td class="val-text">Dark visualisations</td></tr>
        <tr><td class="label">Export</td><td class="val-cyan">PdfPages</td><td class="val-text">PDF report generation</td></tr>
        </tbody></table></div>""",unsafe_allow_html=True)

    st.markdown('<div class="cl-section">Dataset Specifications</div>',unsafe_allow_html=True)
    st.markdown(f"""<div class="about-card"><div class="about-card-title">🗄️ Training Data Overview</div>
    <div style="display:flex;flex-wrap:wrap;gap:0.6rem;margin-top:0.7rem">
      <div style="flex:1;min-width:140px;background:var(--bg);border:1px solid var(--border2);border-radius:10px;padding:0.85rem 1rem"><div class="cl-metric-label">Total Records</div><div class="cl-metric-value teal" style="font-size:1.2rem">{len(df):,}+</div><div class="cl-metric-sub">per domain (synthetic)</div></div>
      <div style="flex:1;min-width:140px;background:var(--bg);border:1px solid var(--border2);border-radius:10px;padding:0.85rem 1rem"><div class="cl-metric-label">Year Range</div><div class="cl-metric-value amber" style="font-size:1.2rem">{int(df['year'].min())}–{int(df['year'].max())}</div><div class="cl-metric-sub">multi-year historical</div></div>
      <div style="flex:1;min-width:140px;background:var(--bg);border:1px solid var(--border2);border-radius:10px;padding:0.85rem 1rem"><div class="cl-metric-label">Salary Range</div><div class="cl-metric-value teal" style="font-size:1.2rem">3–23 LPA</div><div class="cl-metric-sub">base + similarity + noise</div></div>
      <div style="flex:1;min-width:140px;background:var(--bg);border:1px solid var(--border2);border-radius:10px;padding:0.85rem 1rem"><div class="cl-metric-label">Domains</div><div class="cl-metric-value teal" style="font-size:1.2rem">8</div><div class="cl-metric-sub">CS · Mech · Civil · EE · ECE · Textile · Med · Fin</div></div>
    </div></div>""",unsafe_allow_html=True)

    st.markdown('<div class="cl-section">Disclaimer</div>',unsafe_allow_html=True)
    st.markdown("""<div class="disclaimer"><strong>⚠ FOR INFORMATIONAL PURPOSES ONLY</strong><br><br>
    CareerLens is built on a <strong>synthetic dataset</strong> for demonstration and research. Salary estimates and forecasts reflect modelled statistical patterns and are <strong>not sourced from live market data</strong>. Real-world compensation varies significantly based on company size, location, experience, and negotiation.</div>""",unsafe_allow_html=True)
    st.markdown('<div class="footer">CareerLens AI System v2.5 &nbsp;·&nbsp; XGBoost + Gradient Boosting + TF-IDF &nbsp;·&nbsp; Built with Streamlit &nbsp;·&nbsp; Made with <span style="color:#ff4b5a;">♥</span> &nbsp;·&nbsp; © 2025</div>',unsafe_allow_html=True)

# ======================================================
# MAIN
# ======================================================
def main():
    # Sync page from query param — works because render_topbar uses href="?page=..."
    qp = st.query_params.get("page", None)
    if qp:
        decoded = qp.replace("+", " ")
        if decoded in VALID_PAGES and decoded != st.session_state.current_page:
            st.session_state.current_page = decoded

    render_topbar()

    page = st.session_state.current_page
    skip_heavy_load = page == "Job Forecasting" and st.session_state.result is None
    if not skip_heavy_load:
        with st.spinner("⚙️ Initialising intelligence modules…"):
            df,tfidf,le,salary_model,X_cols,salary_accuracy,_,_,req_matrix = load_and_train(effective_training_domain())
    else:
        df=tfidf=le=salary_model=X_cols=req_matrix=None
        salary_accuracy=0

    if   page == "Dashboard":          page_dashboard(df, salary_accuracy)
    elif page == "Salary Prediction":  page_salary(df, tfidf, le, salary_model, X_cols, req_matrix)
    elif page == "Job Forecasting":    page_forecasting(df)
    elif page == "Market Trends":      page_market()
    elif page == "About":              page_about(df, salary_accuracy)

    if page != "About":
        st.markdown('<div class="footer">CareerLens AI System v2.5 · XGBoost + Gradient Boosting · © 2025</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()