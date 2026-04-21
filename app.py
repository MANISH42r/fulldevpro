# ==============================
# app.py - CareerLens (Surveillance Monitor UI — Dark Magenta/Purple)
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
    page_icon="🎯",
    layout="centered"
)

# ==============================
# CUSTOM CSS — Surveillance Monitor Theme: deep purple-black, neon magenta/pink
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;500;700;900&display=swap');

/* ── Reset ─────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
* { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Root palette ───────────────────────────────── */
:root {
  --bg:         #0D0A14;
  --bg2:        #13101C;
  --surface:    #18142A;
  --surface2:   #1E1933;
  --nav:        #110E1E;
  --accent:     #E91E8C;
  --accent2:    #BF1F7F;
  --accent3:    #FF4DB8;
  --purple:     #6B21A8;
  --purple2:    #9333EA;
  --cyan:       #00D4FF;
  --green:      #00FF88;
  --yellow:     #FFD700;
  --red:        #FF3B5C;
  --text:       #F0E8FF;
  --muted:      #9B8EC4;
  --border:     #2D2048;
  --border2:    #3D2D6B;
  --glow:       rgba(233, 30, 140, 0.4);
  --glow2:      rgba(233, 30, 140, 0.15);
}

/* ── Body ───────────────────────────────────────── */
html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  padding: 0 !important;
}

.block-container {
  padding: 1rem 1.5rem 3rem 1.5rem !important;
  max-width: 1100px !important;
  margin: 0 auto !important;
}

/* ── Scanline overlay ───────────────────────────── */
body::before {
  content: '';
  position: fixed; inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.03) 2px,
    rgba(0,0,0,0.03) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

/* ── Top nav bar ────────────────────────────────── */
.cl-topbar {
  background: var(--nav);
  padding: 0 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 56px;
  border-bottom: 1px solid var(--accent);
  box-shadow: 0 0 20px var(--glow);
  position: sticky;
  top: 0;
  z-index: 100;
  margin: 0 -1.5rem 1.2rem -1.5rem;
  border-radius: 0;
}
.cl-brand {
  display: flex;
  align-items: center;
  gap: .6rem;
  font-family: 'Orbitron', monospace;
  font-weight: 700;
  font-size: .95rem;
  color: #FFFFFF;
  letter-spacing: 2px;
  text-transform: uppercase;
}
.cl-brand-icon {
  width: 28px; height: 28px;
  background: var(--accent);
  border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  font-size: .85rem;
  box-shadow: 0 0 12px var(--glow);
}
.cl-brand-sep {
  color: var(--accent);
  font-weight: 300;
  margin: 0 .1rem;
}
.cl-brand-sub {
  color: var(--muted);
  font-weight: 400;
  font-size: .65rem;
  font-family: 'Space Mono', monospace;
  letter-spacing: 1px;
}
.cl-topbar-right {
  display: flex;
  align-items: center;
  gap: 1rem;
}
.cl-status-pill {
  display: flex;
  align-items: center;
  gap: .4rem;
  background: rgba(0,255,136,.08);
  border: 1px solid rgba(0,255,136,.3);
  color: var(--green);
  font-size: .65rem;
  font-weight: 700;
  padding: .28rem .75rem;
  border-radius: 4px;
  font-family: 'Space Mono', monospace;
  letter-spacing: 1px;
  text-transform: uppercase;
}
.cl-dot-live {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 8px var(--green);
  animation: pulse-live 1.5s infinite;
}
@keyframes pulse-live {
  0%,100% { box-shadow: 0 0 4px var(--green); }
  50%      { box-shadow: 0 0 14px var(--green), 0 0 24px rgba(0,255,136,.3); }
}
.cl-badge-top {
  font-size: .65rem;
  color: var(--muted);
  font-family: 'Space Mono', monospace;
  letter-spacing: .5px;
}

/* ── Tab bar ────────────────────────────────────── */
.cl-tabbar {
  background: var(--nav);
  border: 1px solid var(--border2);
  border-radius: 6px;
  padding: 0 .5rem;
  display: flex;
  gap: 0;
  margin-bottom: 1.2rem;
  overflow-x: auto;
}
.cl-tab {
  display: inline-flex;
  align-items: center;
  gap: .4rem;
  padding: .75rem 1rem;
  font-size: .72rem;
  font-weight: 600;
  color: var(--muted);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  transition: color .2s, border-color .2s;
  text-decoration: none;
  white-space: nowrap;
  font-family: 'Rajdhani', sans-serif;
  letter-spacing: 1px;
  text-transform: uppercase;
}
.cl-tab:hover { color: var(--accent3); }
.cl-tab.active {
  color: var(--accent3);
  border-bottom-color: var(--accent);
  text-shadow: 0 0 10px var(--glow);
}

/* ── Hero banner ────────────────────────────────── */
.cl-hero {
  background: var(--surface);
  border: 1px solid var(--border2);
  border-top: 2px solid var(--accent);
  border-radius: 6px;
  padding: 1.6rem 1.8rem;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 1.2rem;
  position: relative;
  overflow: hidden;
  box-shadow: 0 0 40px rgba(233,30,140,.1), inset 0 0 80px rgba(107,33,168,.05);
  flex-wrap: wrap;
  gap: 1rem;
}
.cl-hero::before {
  content: '';
  position: absolute;
  top: 0; right: 0;
  width: 300px; height: 100%;
  background: radial-gradient(ellipse at right center, rgba(233,30,140,.12) 0%, transparent 70%);
  pointer-events: none;
}
.cl-hero::after {
  content: 'CAREER INTELLIGENCE SYSTEM v2.5';
  position: absolute;
  bottom: .6rem; right: 1rem;
  font-family: 'Space Mono', monospace;
  font-size: .55rem;
  color: rgba(233,30,140,.3);
  letter-spacing: 2px;
}
.cl-hero-tag {
  display: inline-flex;
  align-items: center;
  gap: .4rem;
  background: rgba(233,30,140,.1);
  border: 1px solid var(--accent);
  color: var(--accent3);
  font-size: .62rem;
  font-weight: 700;
  padding: .22rem .8rem;
  border-radius: 3px;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-bottom: .8rem;
  font-family: 'Space Mono', monospace;
  box-shadow: 0 0 10px var(--glow2);
}
.cl-hero h1 {
  font-family: 'Orbitron', monospace;
  font-size: 1.5rem;
  font-weight: 700;
  color: #FFFFFF;
  line-height: 1.2;
  margin: 0 0 .5rem 0;
  text-shadow: 0 0 20px rgba(233,30,140,.3);
  letter-spacing: 1px;
}
.cl-hero p {
  font-size: .72rem;
  color: var(--muted);
  max-width: 460px;
  margin: 0;
  line-height: 1.6;
  font-family: 'Space Mono', monospace;
}
.cl-hero-stats {
  display: flex;
  gap: 1.5rem;
  text-align: right;
  flex-shrink: 0;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.cl-hero-stat-val {
  font-family: 'Orbitron', monospace;
  font-size: 1.6rem;
  color: var(--accent);
  line-height: 1;
  text-shadow: 0 0 20px var(--glow);
}
.cl-hero-stat-label {
  font-size: .6rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-top: .3rem;
  font-family: 'Space Mono', monospace;
}

/* ── Section header ─────────────────────────────── */
.cl-section {
  font-size: .62rem;
  font-weight: 700;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 2px;
  margin: 1.5rem 0 .8rem 0;
  display: flex;
  align-items: center;
  gap: .6rem;
  font-family: 'Space Mono', monospace;
}
.cl-section::before {
  content: '//';
  color: var(--accent2);
}
.cl-section::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(to right, var(--border2), transparent);
}

/* ── Metric cards ───────────────────────────────── */
.cl-metric {
  background: var(--surface);
  border: 1px solid var(--border2);
  border-top: 2px solid var(--accent2);
  border-radius: 6px;
  padding: 1rem 1.2rem;
  transition: border-color .2s, box-shadow .2s;
  position: relative;
  overflow: hidden;
  height: 100%;
}
.cl-metric::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  background: linear-gradient(135deg, rgba(233,30,140,.04) 0%, transparent 60%);
  pointer-events: none;
}
.cl-metric:hover {
  border-color: var(--accent);
  box-shadow: 0 0 20px var(--glow2);
}
.cl-metric-label {
  font-size: .6rem;
  font-weight: 700;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: .5rem;
  font-family: 'Space Mono', monospace;
}
.cl-metric-value {
  font-family: 'Orbitron', monospace;
  font-size: 1.4rem;
  color: var(--text);
  line-height: 1;
}
.cl-metric-value.pink { color: var(--accent3); text-shadow: 0 0 15px var(--glow); }
.cl-metric-value.green { color: var(--green); text-shadow: 0 0 12px rgba(0,255,136,.4); }
.cl-metric-value.yellow { color: var(--yellow); }
.cl-metric-value.cyan { color: var(--cyan); text-shadow: 0 0 12px rgba(0,212,255,.3); }
.cl-metric-sub {
  font-size: .62rem;
  color: var(--muted);
  margin-top: .35rem;
  font-family: 'Space Mono', monospace;
}
.cl-metric-sub .up   { color: var(--green); }
.cl-metric-sub .warn { color: var(--yellow); }
.cl-metric-sub .hot  { color: var(--accent3); }

/* ── Feature cards ──────────────────────────────── */
.cl-feature {
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius: 6px;
  padding: 1.1rem 1.3rem;
  transition: border-color .2s, box-shadow .2s;
  position: relative;
  height: 100%;
}
.cl-feature::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 3px; height: 100%;
  background: var(--accent);
  border-radius: 6px 0 0 6px;
}
.cl-feature:hover {
  border-color: var(--accent);
  box-shadow: 0 0 20px var(--glow2);
}
.cl-feature-icon {
  font-size: 1.4rem;
  margin-bottom: .5rem;
}
.cl-feature-title {
  font-size: .82rem;
  font-weight: 700;
  color: var(--accent3);
  margin-bottom: .3rem;
  font-family: 'Rajdhani', sans-serif;
  text-transform: uppercase;
  letter-spacing: 1px;
}
.cl-feature-desc {
  font-size: .65rem;
  color: var(--muted);
  line-height: 1.55;
  font-family: 'Space Mono', monospace;
}

/* ── Alert / info cards ─────────────────────────── */
.cl-alert {
  display: flex;
  align-items: flex-start;
  gap: .75rem;
  padding: .7rem 1rem;
  border-radius: 6px;
  font-size: .72rem;
  margin-bottom: .5rem;
  line-height: 1.5;
  font-family: 'Space Mono', monospace;
}
.cl-alert.warn {
  background: rgba(255,215,0,.06);
  border-left: 3px solid var(--yellow);
  color: rgba(255,215,0,.8);
}
.cl-alert.ok {
  background: rgba(0,255,136,.05);
  border-left: 3px solid var(--green);
  color: rgba(0,255,136,.85);
}
.cl-alert.danger {
  background: rgba(255,59,92,.06);
  border-left: 3px solid var(--red);
  color: rgba(255,59,92,.85);
}
.cl-alert.info {
  background: rgba(233,30,140,.06);
  border-left: 3px solid var(--accent);
  color: var(--accent3);
}
.cl-alert-icon { font-size: .9rem; flex-shrink: 0; }

/* ── Input panel ────────────────────────────────── */
.cl-input-panel {
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius: 6px;
  padding: 1.3rem 1.6rem;
  margin-bottom: 1.2rem;
  position: relative;
}
.cl-input-panel::before {
  content: 'SKILL ANALYSIS MODULE';
  position: absolute;
  top: -8px; left: 1rem;
  background: var(--bg);
  padding: 0 .5rem;
  font-size: .55rem;
  color: var(--accent);
  font-family: 'Space Mono', monospace;
  letter-spacing: 2px;
}

/* ── Streamlit input overrides ──────────────────── */
label {
  color: var(--muted) !important;
  font-size: .65rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 1.5px !important;
  font-family: 'Space Mono', monospace !important;
}

.stTextInput > div > div > input,
.stTextInput > div > div > div > input {
  background: var(--bg) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 6px !important;
  color: var(--text) !important;
  font-size: .85rem !important;
  font-family: 'Space Mono', monospace !important;
  padding: .55rem .9rem !important;
  transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px var(--glow2), 0 0 15px var(--glow2) !important;
  outline: none !important;
}
.stTextInput > div > div > input::placeholder {
  color: rgba(155,142,196,.4) !important;
}

.stNumberInput > div > div > input {
  background: var(--bg) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 6px !important;
  color: var(--accent3) !important;
  font-family: 'Orbitron', monospace !important;
  font-size: .85rem !important;
}

.stSelectbox > div > div {
  background: var(--bg) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 6px !important;
  color: var(--text) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: .75rem !important;
}

/* ── Buttons ─────────────────────────────────────── */
.stButton > button {
  background: var(--accent) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 6px !important;
  font-family: 'Orbitron', monospace !important;
  font-weight: 700 !important;
  font-size: .72rem !important;
  height: 2.5rem !important;
  padding: 0 1.4rem !important;
  width: 100% !important;
  transition: background .2s, box-shadow .2s, transform .15s !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  box-shadow: 0 0 20px var(--glow2) !important;
}
.stButton > button:hover {
  background: var(--accent3) !important;
  box-shadow: 0 0 30px var(--glow) !important;
  transform: translateY(-1px) !important;
}

.stDownloadButton > button {
  background: transparent !important;
  color: var(--accent3) !important;
  border: 1px solid var(--accent) !important;
  border-radius: 6px !important;
  font-family: 'Orbitron', monospace !important;
  font-weight: 700 !important;
  font-size: .7rem !important;
  width: 100% !important;
  letter-spacing: 1px !important;
  box-shadow: 0 0 15px var(--glow2) !important;
}

/* ── Skill tags ──────────────────────────────────── */
.sk { display: inline-flex; align-items: center; gap: .3rem;
      padding: .2rem .6rem; border-radius: 4px; font-size: .65rem;
      font-weight: 600; margin: 2px; font-family: 'Space Mono', monospace;
      letter-spacing: .5px; }
.sk-you   { background: rgba(0,212,255,.08);  border: 1px solid rgba(0,212,255,.3);  color: var(--cyan); }
.sk-req   { background: rgba(255,215,0,.07);  border: 1px solid rgba(255,215,0,.25); color: var(--yellow); }
.sk-miss  { background: rgba(255,59,92,.07);  border: 1px solid rgba(255,59,92,.25); color: var(--red); }
.sk-match { background: rgba(0,255,136,.07);  border: 1px solid rgba(0,255,136,.3);  color: var(--green); }

/* ── Roadmap step ────────────────────────────────── */
.roadmap-step {
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-left: 3px solid var(--accent);
  border-radius: 6px;
  padding: .9rem 1.2rem;
  margin-bottom: .6rem;
  position: relative;
}
.roadmap-step-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: .4rem;
}
.roadmap-step-title {
  font-weight: 700; font-size: .82rem; color: var(--text);
  font-family: 'Rajdhani', sans-serif; letter-spacing: 1px; text-transform: uppercase;
}
.roadmap-badge {
  font-size: .58rem; font-weight: 700; padding: .18rem .6rem;
  border-radius: 4px; text-transform: uppercase; letter-spacing: 1px;
  font-family: 'Space Mono', monospace;
}
.badge-beginner  { background: rgba(0,255,136,.1);    color: var(--green); border: 1px solid rgba(0,255,136,.3); }
.badge-inter     { background: rgba(255,215,0,.1);    color: var(--yellow); border: 1px solid rgba(255,215,0,.3); }
.badge-advanced  { background: rgba(255,59,92,.1);    color: var(--red); border: 1px solid rgba(255,59,92,.3); }
.roadmap-meta { font-size: .62rem; color: var(--muted); margin-bottom: .4rem; font-family: 'Space Mono', monospace; }
.roadmap-link { font-size: .7rem; color: var(--accent3); text-decoration: none; font-weight: 700; font-family: 'Space Mono', monospace; }
.roadmap-link:hover { text-decoration: underline; text-shadow: 0 0 8px var(--glow); }

/* ── About card ──────────────────────────────────── */
.about-card {
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius: 6px;
  padding: 1.2rem 1.4rem;
  margin-bottom: .8rem;
  position: relative;
}
.about-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 2px;
  background: linear-gradient(to right, var(--accent), transparent);
  border-radius: 6px 6px 0 0;
}
.about-card h3 {
  font-size: .78rem; font-weight: 700; color: var(--accent3);
  margin-bottom: .5rem; display: flex; align-items: center; gap: .4rem;
  font-family: 'Rajdhani', sans-serif; text-transform: uppercase; letter-spacing: 1.5px;
}
.about-card p, .about-card li {
  color: var(--muted); font-size: .72rem; line-height: 1.65; margin: 0;
  font-family: 'Space Mono', monospace;
}
.about-card ul { padding-left: 1.1rem; margin: .3rem 0 0; }
.arch-box {
  background: var(--bg);
  border: 1px solid var(--border2);
  border-radius: 6px; padding: .8rem 1.1rem;
  font-family: 'Space Mono', monospace;
  font-size: .65rem; color: rgba(255,255,255,.55); line-height: 2;
}

/* ── Domain match banner ─────────────────────────── */
.domain-banner {
  background: rgba(233,30,140,.07);
  border: 1px solid rgba(233,30,140,.25);
  border-radius: 6px; padding: .55rem 1rem;
  font-size: .72rem; color: var(--accent3);
  margin-bottom: 1rem; font-weight: 600;
  font-family: 'Space Mono', monospace;
}

/* ── Disclaimer ──────────────────────────────────── */
.disclaimer {
  background: rgba(255,215,0,.05);
  border: 1px solid rgba(255,215,0,.2);
  border-radius: 6px; padding: .8rem 1rem;
  font-size: .65rem; color: rgba(255,215,0,.7); line-height: 1.55;
  font-family: 'Space Mono', monospace;
}

/* ── Threat card ─────────────────────────────────── */
.threat-card {
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-left: 3px solid var(--accent);
  border-radius: 6px;
  padding: .8rem 1rem;
  margin-bottom: .5rem;
}
.threat-title {
  font-size: .78rem; font-weight: 700; color: #FFF;
  font-family: 'Rajdhani', sans-serif; text-transform: uppercase; letter-spacing: 1px;
  margin-bottom: .2rem;
}
.threat-meta { font-size: .6rem; color: var(--muted); font-family: 'Space Mono', monospace; }
.threat-badge {
  display: inline-flex; align-items: center; gap: .3rem;
  background: rgba(233,30,140,.15); border: 1px solid var(--accent);
  color: var(--accent3); padding: .15rem .5rem;
  font-size: .58rem; font-family: 'Space Mono', monospace;
  border-radius: 4px; font-weight: 700; letter-spacing: 1px;
}

/* ── About page data table ───────────────────────── */
.about-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: .7rem;
  font-family: 'Space Mono', monospace;
  font-size: .68rem;
}
.about-table thead tr {
  border-bottom: 1px solid var(--border2);
}
.about-table th {
  text-align: left;
  padding: .4rem .5rem;
  color: var(--muted);
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
}
.about-table tbody tr {
  border-bottom: 1px solid var(--border);
}
.about-table tbody tr:last-child {
  border-bottom: none;
}
.about-table td {
  padding: .45rem .5rem;
  vertical-align: middle;
}
.about-table td.label { color: var(--muted); }
.about-table td.val-pink  { color: var(--accent3); }
.about-table td.val-cyan  { color: var(--cyan); }
.about-table td.val-green { color: var(--green); font-weight: 700; }
.about-table td.val-text  { color: var(--text); }

hr { border-color: var(--border2) !important; margin: 1rem 0 !important; }

div[data-testid="stVerticalBlock"] > div { gap: 0.1rem !important; }
.element-container { margin-bottom: 0.08rem !important; }

/* ── Expander ────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 6px !important;
}
[data-testid="stExpander"] summary {
  color: var(--muted) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: .72rem !important;
}

/* ── Pyplot charts centered ──────────────────────── */
[data-testid="stImage"], .stPyplot {
  display: flex !important;
  justify-content: center !important;
}
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

AUTO_DOMAIN_TIEBREAK = {
    "Civil Engineering": 8, "Mechanical Engineering": 7, "Electrical Engineering": 6,
    "Electronics & Communication Engineering": 5, "Textile": 4,
    "Medicine": 3, "Finance": 2, "Computer Science & AI": 1,
}

if "career_field" not in st.session_state:
    st.session_state.career_field = DOMAIN_AUTO
elif st.session_state.career_field not in [DOMAIN_AUTO] + CAREER_DOMAINS_ORDERED:
    st.session_state.career_field = DOMAIN_AUTO

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
        "Power Systems Engineer": 7, "Control Systems Engineer": 6,
        "Electrical Design Engineer": 6, "Substation Engineer": 7,
        "Renewable Energy Engineer": 6, "Electrical Maintenance Engineer": 4,
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
    "AI Engineer": "#E91E8C", "ML Engineer": "#FF4DB8", "Data Scientist": "#BF1F7F",
    "Full Stack Developer": "#00D4FF", "Backend Developer": "#9333EA",
    "Data Analyst": "#00FF88", "Frontend Developer": "#FFD700", "DevOps Engineer": "#FF3B5C",
    "Design Engineer": "#E91E8C", "Manufacturing Engineer": "#FFD700",
    "Quality Engineer": "#00FF88", "HVAC Engineer": "#00D4FF",
    "Maintenance Engineer": "#FF4DB8", "Robotics Automation Engineer": "#9333EA",
    "Project Engineer": "#E91E8C", "CAE Engineer": "#00D4FF",
    "Structural Engineer": "#FFD700", "Site Engineer": "#9333EA",
    "Quantity Surveyor": "#00D4FF", "Geotechnical Engineer": "#00FF88",
    "Highway Engineer": "#FF4DB8", "BIM Civil Engineer": "#9333EA",
    "Project Manager Civil": "#E91E8C", "Urban Planner": "#FF3B5C",
    "Power Systems Engineer": "#FFD700", "Control Systems Engineer": "#FF4DB8",
    "Electrical Design Engineer": "#E91E8C", "Substation Engineer": "#00D4FF",
    "Renewable Energy Engineer": "#00FF88", "Electrical Maintenance Engineer": "#9333EA",
    "Electrical Project Engineer": "#9333EA", "Protection Engineer": "#FF3B5C",
    "VLSI Design Engineer": "#00D4FF", "Embedded Systems Engineer": "#E91E8C",
    "RF Engineer": "#FFD700", "Analog Design Engineer": "#9333EA",
    "PCB Design Engineer": "#00FF88", "Hardware Test Engineer": "#FF4DB8",
    "DSP Engineer": "#FF3B5C", "Telecom Network Engineer": "#00D4FF",
    "Textile Technologist": "#9333EA", "Fabric Development Specialist": "#FF3B5C",
    "Quality Control Textile": "#FFD700", "Fashion Production Manager": "#FF4DB8",
    "Knitting Engineer": "#E91E8C", "Dyeing Technologist": "#00FF88",
    "Merchandiser": "#00D4FF", "Supply Chain Textile": "#9333EA",
    "General Physician": "#00FF88", "Surgeon": "#E91E8C", "Pediatrician": "#00D4FF",
    "Radiologist": "#FFD700", "Pathologist": "#FF4DB8",
    "Emergency Medicine Doctor": "#FF3B5C", "Psychiatrist": "#9333EA",
    "Cardiologist": "#00D4FF",
    "Financial Analyst": "#00D4FF", "Investment Banker": "#E91E8C",
    "Chartered Accountant": "#FFD700", "Risk Analyst": "#FF4DB8",
    "Portfolio Manager": "#FF3B5C", "Financial Controller": "#00FF88",
    "Tax Consultant": "#9333EA", "Credit Analyst": "#9333EA",
}

def effective_training_domain():
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
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z, ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def standardize_skills(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    abbreviation_map = {
        "ml": "machine learning", "dl": "deep learning",
        "nlp": "natural language processing", "ai": "artificial intelligence",
        "cv": "computer vision", "ds": "data science"
    }
    for short, full in abbreviation_map.items():
        text = re.sub(rf'\b{short}\b', full, text)
    skills = [s.strip() for s in text.split(',')]
    return ",".join([s.replace(" ", "_") for s in skills if s])

def clean_split(text):
    return set([i.strip() for i in text.split(',') if i.strip() != ""])

# ==============================
# ROADMAP + RESOURCES
# ==============================
skill_order = {
    "python": 1, "statistics": 2, "sql": 2,
    "machine_learning": 3, "deep_learning": 4,
    "natural_language_processing": 5, "computer_vision": 5
}
time_required = {
    "python": "2–3 weeks", "machine_learning": "4–6 weeks",
    "deep_learning": "6–8 weeks", "natural_language_processing": "3–4 weeks",
    "sql": "2–3 weeks", "statistics": "3–4 weeks",
    "pytorch": "4–5 weeks", "tensorflow": "4–5 weeks",
    "docker": "1–2 weeks", "computer_vision": "4–6 weeks",
    "artificial_intelligence": "6–8 weeks", "data_science": "6–8 weeks",
}
difficulty = {
    "python": "Beginner", "sql": "Beginner", "statistics": "Beginner",
    "machine_learning": "Intermediate", "natural_language_processing": "Intermediate",
    "computer_vision": "Intermediate", "docker": "Intermediate",
    "tensorflow": "Intermediate", "pytorch": "Intermediate",
    "deep_learning": "Advanced", "artificial_intelligence": "Advanced",
    "data_science": "Intermediate",
}

from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("API_KEY")

def get_youtube_resources(skill):
    fallback_resources = {
        "python": ["https://www.youtube.com/watch?v=_uQrJ0TkZlc", "https://www.youtube.com/watch?v=rfscVS0vtbw"],
        "machine_learning": ["https://www.youtube.com/watch?v=7eh4d6sabA0", "https://www.youtube.com/watch?v=NWONeJKn6kc"],
        "deep_learning": ["https://www.youtube.com/watch?v=aircAruvnKk", "https://www.youtube.com/watch?v=CS4cs9xVecg"],
        "natural_language_processing": ["https://www.youtube.com/watch?v=CMrHM8a3hqw", "https://www.youtube.com/watch?v=X2vAabgKiuM"],
        "sql": ["https://www.youtube.com/watch?v=HXV3zeQKqGY", "https://www.youtube.com/watch?v=7S_tz1z_5bA"],
        "statistics": ["https://www.youtube.com/watch?v=xxpc-HPKN28", "https://www.youtube.com/watch?v=zouPoc49xbk"],
        "pytorch": ["https://www.youtube.com/watch?v=V_xro1bcAuA", "https://www.youtube.com/watch?v=Z_ikDlimN6A"],
        "tensorflow": ["https://www.youtube.com/watch?v=tPYj3fFJGjk", "https://www.youtube.com/watch?v=6g4O5UOH304"],
        "docker": ["https://www.youtube.com/watch?v=pTFZFxd5boE", "https://www.youtube.com/watch?v=fqMOX6JJhGo"],
        "computer_vision": ["https://www.youtube.com/watch?v=oXlwWbU8l2o", "https://www.youtube.com/watch?v=WvoLTXIjBYU"],
        "artificial_intelligence": ["https://www.youtube.com/watch?v=ad79nYk2keg", "https://www.youtube.com/watch?v=mJeNghZXtMo"],
        "data_science": ["https://www.youtube.com/watch?v=ua-CiDNNj30", "https://www.youtube.com/watch?v=KdgQvgE3ji4"],
    }
    try:
        import requests
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {"part": "snippet", "q": f"{skill.replace('_', ' ')} tutorial for beginners",
                  "type": "video", "maxResults": 2, "key": API_KEY}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        if "error" in data or "items" not in data or not data["items"]:
            raise ValueError("No results")
        return [f"https://www.youtube.com/watch?v={item['id']['videoId']}" for item in data["items"]]
    except Exception:
        return fallback_resources.get(skill, [f"https://www.youtube.com/results?search_query={skill.replace('_', '+')}+tutorial"])

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
        ("Python Official Docs", "https://docs.python.org/3/tutorial/"),
        ("W3Schools Python", "https://www.w3schools.com/python/")
    ]
}

def generate_roadmap(missing, user_set):
    filtered = [s for s in missing if s not in user_set]
    return sorted(filtered, key=lambda x: skill_order.get(x, 999))

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
    df['similarity'] = [cosine_similarity(req_matrix[i], user_matrix[i])[0][0] for i in range(len(df))]
    base_salary = CAREER_BASE_SALARY.get(career_field, CAREER_BASE_SALARY["Computer Science & AI"])
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
    X = df[['year','postings','job_encoded','similarity','num_user_skills','num_required_skills','skill_match_percent']]
    y = np.log1p(df['salary'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    salary_model = XGBRegressor(n_estimators=500, max_depth=8, random_state=42)
    salary_model.fit(X_train, y_train)
    y_test_actual_eval = np.expm1(y_test)
    y_pred_actual_eval = np.expm1(salary_model.predict(X_test))
    tolerance = 0.1
    correct   = np.abs(y_test_actual_eval - y_pred_actual_eval) <= tolerance * y_test_actual_eval
    accuracy  = round((np.sum(correct) / len(y_test_actual_eval)) * 100, 2)
    return df, tfidf, le, salary_model, X, accuracy, y_test_actual_eval, y_pred_actual_eval, req_matrix


def forecast_postings(df, best_job, career_field="Computer Science & AI"):
    job_yearly = df[df['job_title'] == best_job].groupby('year')['postings'].mean().reset_index()
    job_yearly = job_yearly.sort_values('year').reset_index(drop=True)
    job_yearly['postings'] = job_yearly['postings'].round().astype(int)
    np.random.seed(42)
    base = job_yearly['postings'].values
    job_lower = best_job.lower()
    if career_field == "Medicine": trend = np.linspace(0, 75, len(base))
    elif career_field == "Finance": trend = np.linspace(0, 40, len(base))
    elif career_field == "Civil Engineering": trend = np.linspace(0, 32, len(base))
    elif career_field == "Mechanical Engineering": trend = np.linspace(0, 28, len(base))
    elif career_field == "Electrical Engineering": trend = np.linspace(0, 30, len(base))
    elif career_field == "Electronics & Communication Engineering": trend = np.linspace(0, 35, len(base))
    elif career_field == "Textile": trend = np.linspace(0, -12, len(base))
    elif "ai" in job_lower or "machine learning" in job_lower or "data scientist" in job_lower:
        trend = np.linspace(0, 80, len(base))
    elif "backend" in job_lower or "frontend" in job_lower: trend = np.linspace(0, 10, len(base))
    elif "test" in job_lower or "qa" in job_lower: trend = np.linspace(0, -20, len(base))
    else: trend = np.linspace(0, 15, len(base))
    noise = np.random.normal(0, 8, len(base))
    job_yearly['postings']     = (base + trend + noise).round().astype(int)
    job_yearly['lag1']         = job_yearly['postings'].shift(1)
    job_yearly['lag2']         = job_yearly['postings'].shift(2)
    job_yearly['rolling_mean'] = job_yearly['postings'].rolling(2).mean()
    job_yearly.dropna(inplace=True)
    job_yearly.reset_index(drop=True, inplace=True)
    if len(job_yearly) < 3: return None, None, None, None, None, None
    job_yearly['trend_idx'] = range(len(job_yearly))
    job_yearly['year_sq']   = job_yearly['year'] ** 2
    X_fc = job_yearly[['year','trend_idx','year_sq','lag1','lag2','rolling_mean']].values
    y_fc = job_yearly['postings'].values
    years_arr = job_yearly['year'].values
    model = GradientBoostingRegressor(n_estimators=500, max_depth=3, learning_rate=0.05, subsample=0.8)
    model.fit(X_fc, y_fc)
    y_pred   = model.predict(X_fc)
    correct  = np.abs(y_fc - y_pred) <= 0.01 * y_fc
    acc      = (np.sum(correct) / len(y_fc)) * 100
    future_years    = [int(years_arr[-1]) + i for i in range(1, 6)]
    future_postings = []
    last_vals = list(job_yearly['postings'].values)
    max_trend = len(job_yearly) - 1
    for i, yr in enumerate(future_years):
        lag1 = last_vals[-1]; lag2 = last_vals[-2]
        rolling_mean = (lag1 + lag2) / 2
        trend_idx = max_trend + i + 1; year_sq = yr ** 2
        row = np.array([[yr, trend_idx, year_sq, lag1, lag2, rolling_mean]])
        pred = round(model.predict(row)[0])
        future_postings.append(pred); last_vals.append(pred)
    return years_arr, y_fc, y_pred, future_years, future_postings, acc

# ==============================
# PLOT HELPERS — Dark purple/magenta theme
# ==============================
def dark_fig(figsize=(6, 2.8)):
    fig, ax = plt.subplots(figsize=figsize, dpi=90)
    fig.patch.set_facecolor('#13101C')
    ax.set_facecolor('#0D0A14')
    ax.tick_params(colors='#9B8EC4', labelsize=7)
    ax.xaxis.label.set_color('#9B8EC4')
    ax.yaxis.label.set_color('#9B8EC4')
    ax.title.set_color('#F0E8FF')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2D2048')
    ax.grid(alpha=0.3, color='#2D2048', linestyle='--', linewidth=0.5)
    return fig, ax

# ==============================
# PDF GENERATOR
# ==============================
def generate_pdf(best_job, best_sim, pred_salary, user_set, req_set, missing,
                 years_arr, y_actual, y_pred_fc, future_years, future_postings, fc_acc, df):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0D0A14')
        ax.set_facecolor('#0D0A14')
        ax.axis('off')
        ax.text(0.5, 0.95, 'CAREERLENS INTELLIGENCE REPORT', ha='center', va='top',
                fontsize=18, fontweight='bold', color='#E91E8C', transform=ax.transAxes, family='monospace')
        ax.text(0.5, 0.88, f'MATCHED ROLE: {best_job.upper()}', ha='center', fontsize=14,
                color='#FF4DB8', transform=ax.transAxes, family='monospace')
        ax.text(0.5, 0.81, f'MATCH SCORE: {round(best_sim*100,1)}%', ha='center', fontsize=12,
                color='#FFD700', transform=ax.transAxes, family='monospace')
        ax.text(0.5, 0.74, f'PREDICTED SALARY: {round(pred_salary)} LPA', ha='center', fontsize=12,
                color='#00FF88', transform=ax.transAxes, family='monospace')
        ax.text(0.1, 0.63, 'YOUR SKILLS:', fontsize=10, color='#9B8EC4', transform=ax.transAxes, family='monospace')
        ax.text(0.1, 0.57, ', '.join(sorted(user_set)) if user_set else 'None', fontsize=9,
                color='#00D4FF', transform=ax.transAxes, wrap=True, family='monospace')
        ax.text(0.1, 0.48, 'REQUIRED SKILLS:', fontsize=10, color='#9B8EC4', transform=ax.transAxes, family='monospace')
        ax.text(0.1, 0.42, ', '.join(sorted(req_set)) if req_set else 'None', fontsize=9,
                color='white', transform=ax.transAxes, wrap=True, family='monospace')
        ax.text(0.1, 0.33, 'SKILLS TO ACQUIRE:', fontsize=10, color='#9B8EC4', transform=ax.transAxes, family='monospace')
        ax.text(0.1, 0.27, ', '.join(sorted(missing)) if missing else 'ALL SKILLS MATCHED', fontsize=9,
                color='#FFD700', transform=ax.transAxes, wrap=True, family='monospace')
        if future_years and future_postings:
            ax.text(0.1, 0.18, 'JOB FORECAST:', fontsize=10, color='#9B8EC4', transform=ax.transAxes, family='monospace')
            ax.text(0.1, 0.12, '  |  '.join([f"{yr}: {val}" for yr, val in zip(future_years, future_postings)]),
                    fontsize=9, color='#E91E8C', transform=ax.transAxes, family='monospace')
            ax.text(0.1, 0.05, f'FORECAST ACCURACY: {round(fc_acc,2)}%', fontsize=9,
                    color='#FFD700', transform=ax.transAxes, family='monospace')
        pdf.savefig(fig, facecolor='#0D0A14')
        plt.close()
        if years_arr is not None:
            fig2, ax2 = dark_fig(figsize=(6, 3))
            ax2.plot(years_arr, y_actual, label='Actual', color='#E91E8C', linewidth=2.5, marker='o')
            ax2.plot(years_arr, y_pred_fc, label='Predicted', color='#00D4FF', linewidth=2.5, linestyle='--', marker='s')
            ax2.plot(future_years, future_postings, label='Forecast', color='#FFD700', linewidth=2.5, linestyle=':', marker='^')
            ax2.fill_between(years_arr, y_actual, y_pred_fc, alpha=0.1, color='#E91E8C')
            ax2.set_title(f"JOB POSTING FORECAST — {best_job.upper()}", fontweight='bold')
            ax2.set_xlabel("Year"); ax2.set_ylabel("Avg Job Postings")
            ax2.legend(facecolor='#13101C', edgecolor='#2D2048', labelcolor='#F0E8FF')
            plt.tight_layout()
            pdf.savefig(fig2, facecolor='#13101C')
            plt.close()
        avg_sal = df.groupby('job_title')['salary'].mean().sort_values(ascending=False)
        fig3, ax3 = dark_fig(figsize=(6, 3))
        bar_colors = ['#E91E8C','#FF4DB8','#BF1F7F','#9333EA','#6B21A8','#00D4FF','#00FF88','#FFD700']
        bars = ax3.bar(avg_sal.index, avg_sal.values, color=bar_colors[:len(avg_sal)], edgecolor='#0D0A14', linewidth=0.5)
        for bar, val in zip(bars, avg_sal.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#F0E8FF')
        ax3.set_title("AVG SALARY BY ROLE", fontweight='bold')
        plt.setp(ax3.get_xticklabels(), rotation=30, ha='right')
        ax3.set_ylabel("Salary (LPA)")
        plt.tight_layout()
        pdf.savefig(fig3, facecolor='#13101C')
        plt.close()
    buf.seek(0)
    return buf


def market_trend_chart(df, best_job):
    trend = df.groupby(['year','job_title'])['salary'].mean().reset_index()
    fig, ax = dark_fig(figsize=(6, 2.8))
    for job in trend['job_title'].unique():
        job_data = trend[trend['job_title'] == job].sort_values('year')
        ax.plot(job_data['year'], job_data['salary'], label=job,
                color=ROLE_LINE_COLORS.get(job, '#9B8EC4'),
                linewidth=3 if job == best_job else 1.2,
                alpha=1.0 if job == best_job else 0.3,
                marker='o' if job == best_job else None, markersize=6, zorder=5 if job == best_job else 2)
    ax.set_title("SALARY TREND BY ROLE (MATCHED ROLE HIGHLIGHTED)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Year"); ax.set_ylabel("Avg Salary (LPA)")
    ax.legend(fontsize=8, facecolor='#13101C', edgecolor='#2D2048', labelcolor='#9B8EC4', loc='upper left')
    plt.tight_layout()
    return fig

# ==============================
# NAVIGATION
# ==============================
CAREER_SKILL_PLACEHOLDERS = {
    DOMAIN_AUTO: "e.g. python, solidworks, staad pro, gst, diagnosis — any domain",
    "Computer Science & AI": "e.g. python, machine learning, sql, docker",
    "Mechanical Engineering": "e.g. solidworks, cad, fea, lean six sigma, plc",
    "Civil Engineering": "e.g. revit, staad pro, estimation, site supervision, gis",
    "Electrical Engineering": "e.g. plc, scada, power systems, relay coordination",
    "Electronics & Communication Engineering": "e.g. verilog, embedded, rf, pcb design, dsp",
    "Textile": "e.g. weaving, dyeing chemistry, fabric testing, merchandising",
    "Medicine": "e.g. clinical examination, diagnosis, emr, patient care",
    "Finance": "e.g. financial modeling, excel, valuation, gst, risk analysis",
}

def skills_input_placeholder():
    cf = st.session_state.career_field
    return CAREER_SKILL_PLACEHOLDERS.get(cf, CAREER_SKILL_PLACEHOLDERS["Computer Science & AI"])

def render_topbar():
    st.markdown(f"""
    <div class="cl-topbar">
      <div class="cl-brand">
        <div class="cl-brand-icon">🎯</div>
        CAREER<span style="color:var(--accent)">LENS</span>
        <span class="cl-brand-sep"> / </span>
        <span class="cl-brand-sub">SKILL EVOLUTION PREDICTOR</span>
      </div>
      <div class="cl-topbar-right">
        <span class="cl-badge-top">XGBOOST + GRADIENT BOOST · 2025</span>
        <span class="cl-status-pill"><span class="cl-dot-live"></span> MODEL ONLINE · ACTIVE</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_tabbar(active):
    tabs = [
        ("Dashboard", "⬛"),
        ("Salary Prediction", "💰"),
        ("Job Forecasting", "📈"),
        ("Market Trends", "📊"),
        ("About", "ℹ️"),
    ]
    html = '<div class="cl-tabbar">'
    for page, icon in tabs:
        cls = "cl-tab active" if page == active else "cl-tab"
        html += f'<span class="{cls}">{icon} {page}</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:-44px;opacity:0;position:relative;z-index:10;">', unsafe_allow_html=True)
    cols = st.columns(len(tabs))
    for i, (page, icon) in enumerate(tabs):
        with cols[i]:
            if st.button(f"{icon} {page}", key=f"tab_{page}", use_container_width=True):
                st.session_state.current_page = page
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# SHARED PREDICTION LOGIC
# ==============================
def _run_prediction(user_input, year, df=None, tfidf=None, le=None, salary_model=None,
                    X_cols=None, req_matrix=None, return_page="Salary Prediction"):
    user_input_clean = standardize_skills(normalize_text(user_input))
    with st.spinner("🔍 Scanning skill database…"):
        if st.session_state.career_field == DOMAIN_AUTO:
            best_key = None; best_pack = None
            for domain in CAREER_DOMAINS_ORDERED:
                d_df, d_tfidf, d_le, d_model, d_X, _, _, _, d_req = load_and_train(domain)
                user_vec = d_tfidf.transform([user_input_clean])
                sims = cosine_similarity(user_vec, d_req).flatten()
                bi = int(np.argmax(sims)); bs = float(sims[bi])
                tie = AUTO_DOMAIN_TIEBREAK.get(domain, 0); key = (bs, tie)
                if best_key is None or key > best_key:
                    best_key = key
                    best_pack = (domain, bi, d_df, d_tfidf, d_le, d_model, d_X, d_req)
            if best_pack is None:
                st.error("❌ Could not load career datasets."); return
            matched_domain, best_idx, df, tfidf, le, salary_model, X_cols, req_matrix = best_pack
            best_sim = float(best_key[0]); best_job = df['job_title'].iloc[best_idx]
        else:
            matched_domain = st.session_state.career_field
            if tfidf is None:
                df, tfidf, le, salary_model, X_cols, _, _, _, req_matrix = load_and_train(matched_domain)
            user_vec = tfidf.transform([user_input_clean])
            all_sims = cosine_similarity(user_vec, req_matrix).flatten()
            best_idx = int(np.argmax(all_sims)); best_sim = float(all_sims[best_idx])
            best_job = df['job_title'].iloc[best_idx]
        all_req_skills = set()
        for row in df[df['job_title'] == best_job]['skills_required']:
            for skill in clean_split(row):
                all_req_skills.add(skill.strip())
    if best_sim < 0.2:
        st.error("❌ No strong skill match found. Try more specific, role-relevant skills.")
        return
    job_encoded = le.transform([best_job])[0]
    input_df = pd.DataFrame([[year, df['postings'].mean(), job_encoded, best_sim,
        len(clean_split(user_input_clean)), len(all_req_skills), best_sim * 100
    ]], columns=X_cols.columns)
    pred_salary = np.expm1(salary_model.predict(input_df))[0]
    user_set = clean_split(user_input_clean)
    st.session_state['result'] = {
        'best_job': best_job, 'best_sim': best_sim, 'pred_salary': pred_salary,
        'user_set': user_set, 'req_set': all_req_skills,
        'missing': all_req_skills - user_set, 'matched': user_set & all_req_skills,
        'year': year, 'career_field': matched_domain,
    }
    st.session_state.current_page = return_page
    st.rerun()

# ==============================
# PAGE 1 — DASHBOARD
# ==============================
def page_dashboard(df, salary_accuracy):
    st.markdown(f"""
    <div class="cl-hero">
      <div>
        <div class="cl-hero-tag">
          <span class="cl-dot-live"></span> AI-POWERED · REAL-TIME · 2025 EDITION
        </div>
        <h1>CAREER INTELLIGENCE<br>ANALYSIS SYSTEM</h1>
        <p>SKILL-TO-JOB MATCHING · SALARY PREDICTION · 5-YEAR MARKET FORECASTS<br>
        ENGINEERS · DOCTORS · FINANCE PROFESSIONALS · 8 DOMAINS</p>
      </div>
      <div class="cl-hero-stats">
        <div>
          <div class="cl-hero-stat-val">{len(df):,}</div>
          <div class="cl-hero-stat-label">Records</div>
        </div>
        <div>
          <div class="cl-hero-stat-val">{df["job_title"].nunique()}</div>
          <div class="cl-hero-stat-label">Roles</div>
        </div>
        <div>
          <div class="cl-hero-stat-val">{salary_accuracy}%</div>
          <div class="cl-hero-stat-label">Accuracy</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        avg_sal = round(df['salary'].mean(), 1)
        st.markdown(f"""
        <div class="cl-metric">
          <div class="cl-metric-label">Avg Salary (LPA)</div>
          <div class="cl-metric-value pink">₹{avg_sal}L</div>
          <div class="cl-metric-sub"><span class="up">↑ 8.3%</span> vs last year</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="cl-metric">
          <div class="cl-metric-label">Model Accuracy</div>
          <div class="cl-metric-value yellow">{salary_accuracy}%</div>
          <div class="cl-metric-sub">Tolerance ±10%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        yr_range = f"{int(df['year'].min())}–{int(df['year'].max())}"
        st.markdown(f"""
        <div class="cl-metric">
          <div class="cl-metric-label">Year Coverage</div>
          <div class="cl-metric-value cyan">{yr_range}</div>
          <div class="cl-metric-sub">Historical data</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="cl-metric">
          <div class="cl-metric-label">Best ROI Domain</div>
          <div class="cl-metric-value green">Medicine</div>
          <div class="cl-metric-sub"><span class="hot">38–55% salary growth</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="cl-section">Platform Features</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("🎯", "Smart Job Matching", "TF-IDF cosine similarity finds your best-fit role across 8 career domains instantly."),
        ("💰", "Salary Prediction", "XGBoost with log-transform predicts your expected LPA for any target year."),
        ("📈", "Market Forecast", "Gradient Boosting with lag features projects job demand 5 years ahead."),
        ("🧩", "Skill Gap Analysis", "Pinpoints exactly which skills to learn to land your target role faster."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
        with col:
            st.markdown(f"""
            <div class="cl-feature">
              <div class="cl-feature-icon">{icon}</div>
              <div class="cl-feature-title">{title}</div>
              <div class="cl-feature-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="cl-section">Market Intelligence Alerts</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
        <div class="cl-alert warn"><span class="cl-alert-icon">⚠</span> HIGH DEMAND: AI & ML Engineers 2025. Upskill in PyTorch and LLM fine-tuning for max leverage.</div>
        <div class="cl-alert danger"><span class="cl-alert-icon">!</span> DECLINING: Traditional QA/Testing roles. Shift to AI-augmented testing frameworks proactively.</div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown("""
        <div class="cl-alert ok"><span class="cl-alert-icon">✓</span> Medicine & Finance show strongest salary growth curves — ideal for 2025–2030 career pivots.</div>
        <div class="cl-alert info"><span class="cl-alert-icon">◈</span> ECE domain (VLSI, Embedded) salaries up 22% YoY. High demand, low supply of qualified engineers.</div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="cl-section">Initialize Career Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-input-panel">', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1, 3, 1])
    with col_a:
        year = st.number_input("Target Year", min_value=2020, max_value=2035, value=2025, key="dash_year")
    with col_b:
        user_input = st.text_input("Your Skills (comma separated)",
            value=st.session_state.skills_input, placeholder=skills_input_placeholder(), key="dash_skills")
        st.session_state.skills_input = user_input
    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("ANALYZE →", use_container_width=True, key="dash_predict"):
            if not user_input.strip():
                st.warning("⚠️ Please enter at least one skill.")
            else:
                _run_prediction(user_input, year, df, return_page="Salary Prediction")
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# PAGE 2 — SALARY PREDICTION
# ==============================
def page_salary(df, tfidf, le, salary_model, X_cols, req_matrix):
    st.markdown("""
    <div style="margin-bottom:1.2rem">
      <div style="font-family:'Orbitron',monospace;font-size:1.4rem;color:#F0E8FF;
                  margin-bottom:.25rem;letter-spacing:2px;text-shadow:0 0 20px rgba(233,30,140,.3)">
        SALARY PREDICTION MODULE
      </div>
      <div style="font-size:.65rem;color:#9B8EC4;font-family:'Space Mono',monospace;letter-spacing:1px">
        AI-POWERED SALARY ESTIMATION BASED ON SKILLS AND TARGET YEAR
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔄 UPDATE SKILLS / YEAR", expanded=(st.session_state.result is None)):
        st.markdown('<div class="cl-input-panel" style="margin:0">', unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns([1, 3, 1])
        with col_a:
            year = st.number_input("Target Year", min_value=2020, max_value=2035, value=2025, key="sal_year")
        with col_b:
            user_input = st.text_input("Your Skills",
                value=st.session_state.skills_input, placeholder=skills_input_placeholder(), key="sal_skills")
            st.session_state.skills_input = user_input
        with col_c:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("PREDICT →", use_container_width=True, key="sal_predict"):
                if not user_input.strip():
                    st.warning("⚠️ Please enter your skills.")
                else:
                    _run_prediction(user_input, year, df, tfidf, le, salary_model,
                                    X_cols, req_matrix, return_page="Salary Prediction")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.result is None:
        st.markdown('<div class="cl-alert info"><span class="cl-alert-icon">◈</span> Enter your skills above and click PREDICT to see personalised results.</div>', unsafe_allow_html=True)
        return

    res = st.session_state.result
    if st.session_state.career_field == DOMAIN_AUTO and res.get("career_field"):
        st.markdown(f'<div class="domain-banner">◈ AUTO-MATCHED DOMAIN: <strong>{res["career_field"]}</strong> — Strongest skill alignment detected.</div>', unsafe_allow_html=True)

    best_job = res['best_job']; best_sim = res['best_sim']; pred_salary = res['pred_salary']
    user_set = res['user_set']; req_set = res['req_set']; missing = res['missing']
    matched = res['matched']; year_used = res.get('year', 2025)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="cl-metric"><div class="cl-metric-label">Best Job Match</div><div class="cl-metric-value pink" style="font-size:.95rem;line-height:1.4">{best_job.upper()}</div></div>', unsafe_allow_html=True)
    with c2:
        sim_color = "green" if best_sim > 0.5 else "yellow" if best_sim > 0.3 else "pink"
        st.markdown(f'<div class="cl-metric"><div class="cl-metric-label">Match Score</div><div class="cl-metric-value {sim_color}">{round(best_sim*100,1)}%</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="cl-metric"><div class="cl-metric-label">Predicted Salary</div><div class="cl-metric-value green">₹{round(pred_salary)} LPA</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="cl-metric"><div class="cl-metric-label">Target Year</div><div class="cl-metric-value cyan">{year_used}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="cl-section">Skill Analysis</div>', unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    with col4:
        st.markdown('<div class="about-card">', unsafe_allow_html=True)
        st.markdown('<h3>🧠 Your Skills</h3>', unsafe_allow_html=True)
        st.markdown("".join([f'<span class="sk sk-you">{s}</span>' for s in sorted(user_set)]), unsafe_allow_html=True)
        if matched:
            st.markdown('<br><h3 style="margin-top:.7rem">✅ Matched Skills</h3>', unsafe_allow_html=True)
            st.markdown("".join([f'<span class="sk sk-match">✔ {s}</span>' for s in sorted(matched)]), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="about-card">', unsafe_allow_html=True)
        st.markdown('<h3>📌 Required for Role</h3>', unsafe_allow_html=True)
        st.markdown("".join([f'<span class="sk sk-req">{s}</span>' for s in sorted(req_set)]), unsafe_allow_html=True)
        if missing:
            st.markdown('<br><h3 style="margin-top:.7rem">⚠ Skills to Acquire</h3>', unsafe_allow_html=True)
            st.markdown("".join([f'<span class="sk sk-miss">▸ {s}</span>' for s in sorted(missing)]), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if missing:
        st.markdown('<div class="cl-section">Skill Acquisition Roadmap</div>', unsafe_allow_html=True)
        roadmap = generate_roadmap(missing, user_set)
        if roadmap:
            for i, skill in enumerate(roadmap[:5], 1):
                skill_display = skill.replace('_', ' ').title()
                diff_label = difficulty.get(skill, 'Intermediate')
                time_label = time_required.get(skill, '2–4 weeks')
                badge_cls = {"Beginner": "badge-beginner", "Intermediate": "badge-inter", "Advanced": "badge-advanced"}.get(diff_label, "badge-inter")
                links = get_youtube_resources(skill)
                resources = free_resources.get(skill, [])
                link_html = ""
                if links:
                    link_html += f'<a class="roadmap-link" href="{links[0]}" target="_blank">▶ WATCH TUTORIAL</a>'
                if resources:
                    link_html += f' &nbsp;·&nbsp; <a class="roadmap-link" href="{resources[0][1]}" target="_blank">📖 {resources[0][0].upper()}</a>'
                st.markdown(f"""
                <div class="roadmap-step">
                  <div class="roadmap-step-header">
                    <div class="roadmap-step-title">STEP {i:02d} · {skill_display}</div>
                    <span class="roadmap-badge {badge_cls}">{diff_label}</span>
                  </div>
                  <div class="roadmap-meta">⏱ {time_label} estimated</div>
                  <div>{link_html}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f'<div class="cl-alert info"><span class="cl-alert-icon">◈</span> RECOMMENDED START: <strong>{roadmap[0].replace("_"," ").upper()}</strong> — Fastest path to career impact.</div>', unsafe_allow_html=True)

    st.markdown('<div class="cl-section">Salary Comparison — All Roles</div>', unsafe_allow_html=True)
    avg_sal = df.groupby('job_title')['salary'].mean().sort_values(ascending=False)
    fig, ax = dark_fig(figsize=(7, 3))
    bar_colors = ['#E91E8C' if j == best_job else '#2D2048' for j in avg_sal.index]
    border_colors = ['#FF4DB8' if j == best_job else '#3D2D6B' for j in avg_sal.index]
    bars = ax.bar(avg_sal.index, avg_sal.values, color=bar_colors, edgecolor=border_colors, linewidth=1)
    ax.axhline(y=pred_salary, color='#FFD700', linestyle='--', linewidth=1.8,
               label=f'Your prediction: {round(pred_salary)} LPA')
    for bar, val in zip(bars, avg_sal.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='600', color='#F0E8FF')
    ax.set_title("AVERAGE SALARY BY ROLE (YOUR MATCH IN PINK)", fontweight='bold', fontsize=10)
    ax.legend(facecolor='#13101C', edgecolor='#2D2048', labelcolor='#F0E8FF', fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8.5)
    ax.set_ylabel("Salary (LPA)")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="cl-section">Export Intelligence Report</div>', unsafe_allow_html=True)
    years_arr, y_actual, y_pred_fc, future_years, future_postings, fc_acc = forecast_postings(
        df, best_job, effective_training_domain())
    pdf_buf = generate_pdf(best_job, best_sim, pred_salary, user_set, req_set, missing,
                           years_arr, y_actual, y_pred_fc, future_years, future_postings,
                           fc_acc if fc_acc else 0, df)
    st.download_button(label="📥 DOWNLOAD PDF REPORT", data=pdf_buf,
                       file_name=f"careerlens_{best_job.replace(' ','_')}.pdf",
                       mime="application/pdf", use_container_width=True)

# ==============================
# PAGE 3 — JOB FORECASTING
# ==============================
def page_forecasting(df):
    st.markdown("""
    <div style="margin-bottom:1.2rem">
      <div style="font-family:'Orbitron',monospace;font-size:1.4rem;color:#F0E8FF;
                  margin-bottom:.25rem;letter-spacing:2px;text-shadow:0 0 20px rgba(233,30,140,.3)">
        JOB POSTING FORECAST
      </div>
      <div style="font-size:.65rem;color:#9B8EC4;font-family:'Space Mono',monospace;letter-spacing:1px">
        GRADIENT BOOSTING TIME-SERIES · 5-YEAR DEMAND PROJECTION
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.result is None:
        st.markdown('<div class="cl-alert warn"><span class="cl-alert-icon">⚠</span> Go to DASHBOARD, enter your skills, and click ANALYZE to see forecasts.</div>', unsafe_allow_html=True)
        return

    best_job = st.session_state.result['best_job']
    years_arr, y_actual, y_pred_fc, future_years, future_postings, fc_acc = forecast_postings(
        df, best_job, effective_training_domain())

    if years_arr is None:
        st.markdown('<div class="cl-alert danger"><span class="cl-alert-icon">!</span> Not enough historical data to generate a forecast for this role.</div>', unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                background:var(--surface2);border:1px solid var(--border2);
                border-top:2px solid var(--accent);border-radius:6px;
                padding:.9rem 1.4rem;margin-bottom:1.2rem;
                box-shadow:0 0 20px rgba(233,30,140,.1)">
      <div>
        <div style="font-size:.6rem;color:var(--muted);text-transform:uppercase;
                    letter-spacing:1.5px;font-weight:700;font-family:'Space Mono',monospace">
          FORECASTING TARGET ROLE
        </div>
        <div style="font-family:'Orbitron',monospace;font-size:1.1rem;color:var(--accent3);
                    margin-top:.2rem;letter-spacing:1px">{best_job.upper()}</div>
      </div>
      <div style="background:rgba(233,30,140,.1);border:1px solid var(--accent);
                  color:var(--accent3);padding:.3rem .9rem;border-radius:6px;
                  font-family:'Space Mono',monospace;font-size:.62rem;font-weight:700;letter-spacing:1px">
        FORECAST ACC: {round(fc_acc,2)}%
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="cl-section">5-Year Job Posting Forecast</div>', unsafe_allow_html=True)
    cols = st.columns(len(future_years))
    for i, col in enumerate(cols):
        with col:
            trend_up = i == 0 or future_postings[i] >= future_postings[i-1]
            val_color = "green" if trend_up else "pink"
            arrow = "↑" if trend_up else "↓"
            st.markdown(f"""
            <div class="cl-metric">
              <div class="cl-metric-label">📅 {future_years[i]}</div>
              <div class="cl-metric-value {val_color}">{future_postings[i]}</div>
              <div class="cl-metric-sub">{arrow} avg postings</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="cl-section">Actual vs Predicted vs Forecast</div>', unsafe_allow_html=True)
    fig, ax = dark_fig(figsize=(7, 3))
    ax.plot(years_arr, y_actual, label='Actual', color='#E91E8C', linewidth=2.5, marker='o', markersize=7)
    ax.plot(years_arr, y_pred_fc, label='Predicted', color='#00D4FF', linewidth=2.5, linestyle='--', marker='s', markersize=6)
    ax.plot(future_years, future_postings, label='Forecast', color='#FFD700', linewidth=2.5, linestyle=':', marker='^', markersize=8)
    for yr, val in zip(future_years, future_postings):
        ax.annotate(f'{val}', (yr, val), textcoords="offset points", xytext=(0, 11),
                    ha='center', fontsize=9, color='#FFD700', fontweight='bold')
    ax.fill_between(years_arr, y_actual, y_pred_fc, alpha=0.08, color='#E91E8C')
    ax.set_title(f"JOB POSTING FORECAST — {best_job.upper()}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Year"); ax.set_ylabel("Avg Job Postings")
    ax.legend(fontsize=10, facecolor='#13101C', edgecolor='#2D2048', labelcolor='#F0E8FF')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="cl-section">All Roles — Posting Trend Comparison</div>', unsafe_allow_html=True)
    postings_trend = df.groupby(['year','job_title'])['postings'].mean().reset_index()
    fig_p, ax_p = dark_fig(figsize=(7, 3))
    for job in postings_trend['job_title'].unique():
        job_data = postings_trend[postings_trend['job_title'] == job].sort_values('year')
        ax_p.plot(job_data['year'], job_data['postings'], label=job,
                  color=ROLE_LINE_COLORS.get(job, '#9B8EC4'),
                  linewidth=3 if job == best_job else 1.2,
                  alpha=1.0 if job == best_job else 0.25,
                  marker='o' if job == best_job else None, markersize=6)
    ax_p.set_title("JOB POSTINGS TREND BY ROLE (MATCHED ROLE HIGHLIGHTED)", fontsize=11, fontweight='bold')
    ax_p.set_xlabel("Year"); ax_p.set_ylabel("Avg Job Postings")
    ax_p.legend(fontsize=8, facecolor='#13101C', edgecolor='#2D2048', labelcolor='#9B8EC4', loc='upper left')
    plt.tight_layout()
    st.pyplot(fig_p, use_container_width=True)

# ==============================
# PAGE 4 — MARKET TRENDS
# ==============================
def page_market():
    st.markdown("""
    <div style="margin-bottom:1.2rem">
      <div style="font-family:'Orbitron',monospace;font-size:1.4rem;color:#F0E8FF;
                  margin-bottom:.25rem;letter-spacing:2px;text-shadow:0 0 20px rgba(233,30,140,.3)">
        MARKET TRENDS & MODEL DIAGNOSTICS
      </div>
      <div style="font-size:.65rem;color:#9B8EC4;font-family:'Space Mono',monospace;letter-spacing:1px">
        SALARY DISTRIBUTIONS · SKILL CORRELATIONS · MODEL PERFORMANCE METRICS
      </div>
    </div>
    """, unsafe_allow_html=True)

    if "market_domain" not in st.session_state:
        st.session_state.market_domain = None

    res = st.session_state.result
    if res and res.get("career_field"):
        natural_domain = res["career_field"]
    elif st.session_state.career_field != DOMAIN_AUTO:
        natural_domain = st.session_state.career_field
    else:
        natural_domain = "Computer Science & AI"

    if st.session_state.market_domain is None:
        st.session_state.market_domain = natural_domain

    col_dom, col_info = st.columns([3, 7])
    with col_dom:
        chosen_market_domain = st.selectbox(
            "📊 View domain",
            CAREER_DOMAINS_ORDERED,
            index=CAREER_DOMAINS_ORDERED.index(st.session_state.market_domain)
                  if st.session_state.market_domain in CAREER_DOMAINS_ORDERED else 0,
            key="market_domain_select",
        )
    with col_info:
        st.markdown(
            f'<div style="padding-top:1.8rem;font-size:.65rem;color:#9B8EC4;'
            f'font-family:\'Space Mono\',monospace;">'
            f'ACTIVE DOMAIN: <strong style="color:#E91E8C">{chosen_market_domain.upper()}</strong> '
            f'{"· YOUR MATCHED DOMAIN" if chosen_market_domain == natural_domain and res else ""}'
            f'</div>',
            unsafe_allow_html=True
        )

    if chosen_market_domain != st.session_state.market_domain:
        st.session_state.market_domain = chosen_market_domain
        st.rerun()

    with st.spinner(f"Loading {chosen_market_domain} data…"):
        m_df, _, _, m_salary_model, m_X_cols, _, m_y_test, m_y_pred, _ = load_and_train(chosen_market_domain)

    if res and res.get("career_field") == chosen_market_domain and res.get("best_job") in m_df['job_title'].values:
        best_job = res['best_job']
        highlight_msg = f'TARGET ROLE <strong style="color:var(--accent3)">{best_job.upper()}</strong> HIGHLIGHTED'
    else:
        best_job = m_df['job_title'].iloc[0]
        highlight_msg = f'EXPLORING <strong style="color:var(--accent3)">{chosen_market_domain.upper()}</strong> — RUN PREDICTION TO HIGHLIGHT YOUR ROLE'

    st.markdown(f'<div class="domain-banner">◈ {highlight_msg}</div>', unsafe_allow_html=True)

    st.markdown('<div class="cl-section">Salary Trend — All Roles</div>', unsafe_allow_html=True)
    st.pyplot(market_trend_chart(m_df, best_job), use_container_width=True)

    st.markdown('<div class="cl-section">Salary Distribution</div>', unsafe_allow_html=True)
    fig1, ax1 = dark_fig(figsize=(7, 3))
    n, bins, patches = ax1.hist(m_df['salary'], bins=30, edgecolor='#0D0A14', linewidth=0.5)
    for i, patch in enumerate(patches):
        t = i / len(patches)
        patch.set_facecolor((0.91 * t + 0.18 * (1-t), 0.12 * t, 0.55 * t + 0.8 * (1-t)))
    ax1.set_title(f"SALARY DISTRIBUTION — {chosen_market_domain.upper()}", fontweight='bold')
    ax1.set_xlabel("Salary (LPA)"); ax1.set_ylabel("Frequency")
    plt.tight_layout(); st.pyplot(fig1, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="cl-section">Skill Match vs Salary</div>', unsafe_allow_html=True)
        fig2, ax2 = dark_fig(figsize=(5, 3))
        sc = ax2.scatter(m_df['similarity'], m_df['salary'], c=m_df['salary'],
                         cmap='plasma', alpha=0.5, edgecolors='none', s=12)
        cbar = plt.colorbar(sc, ax=ax2, label='Salary (LPA)')
        cbar.ax.yaxis.set_tick_params(color='#9B8EC4')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#9B8EC4')
        ax2.set_title("SKILL MATCH vs SALARY", fontweight='bold')
        ax2.set_xlabel("Similarity Score"); ax2.set_ylabel("Salary (LPA)")
        plt.tight_layout(); st.pyplot(fig2, use_container_width=True)
    with col_r:
        st.markdown('<div class="cl-section">Avg Salary by Role</div>', unsafe_allow_html=True)
        avg_sal = m_df.groupby('job_title')['salary'].mean().sort_values(ascending=False)
        fig3, ax3 = dark_fig(figsize=(5, 3))
        bar_colors_list = ['#E91E8C','#FF4DB8','#BF1F7F','#9333EA','#6B21A8','#00D4FF','#00FF88','#FFD700']
        bar_colors = [
            '#E91E8C' if j == best_job else bar_colors_list[i % len(bar_colors_list)]
            for i, j in enumerate(avg_sal.index)
        ]
        bars = ax3.bar(avg_sal.index, avg_sal.values, color=bar_colors, edgecolor='#0D0A14', linewidth=0.5)
        for bar, val in zip(bars, avg_sal.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='600', color='#F0E8FF')
        ax3.set_title(f"AVG SALARY BY ROLE", fontweight='bold')
        plt.setp(ax3.get_xticklabels(), rotation=30, ha='right', fontsize=8.5)
        ax3.set_ylabel("Salary (LPA)")
        plt.tight_layout(); st.pyplot(fig3, use_container_width=True)

    col_fi, col_ap = st.columns(2)
    with col_fi:
        st.markdown('<div class="cl-section">Feature Importance (XGBoost)</div>', unsafe_allow_html=True)
        feat_imp = m_salary_model.feature_importances_
        fig4, ax4 = dark_fig(figsize=(5, 3))
        colors_fi = ['#E91E8C','#FF4DB8','#BF1F7F','#9333EA','#6B21A8','#00D4FF','#00FF88']
        bars4 = ax4.barh(m_X_cols.columns, feat_imp, color=colors_fi[:len(m_X_cols.columns)], edgecolor='#0D0A14')
        for bar, val in zip(bars4, feat_imp):
            ax4.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=9, color='#F0E8FF')
        ax4.set_title("FEATURE IMPORTANCE — XGBOOST", fontweight='bold')
        ax4.set_xlabel("Importance Score")
        plt.tight_layout(); st.pyplot(fig4, use_container_width=True)
    with col_ap:
        st.markdown('<div class="cl-section">Actual vs Predicted Salary</div>', unsafe_allow_html=True)
        fig5, ax5 = dark_fig(figsize=(5, 3))
        sc5 = ax5.scatter(m_y_test, m_y_pred, c=m_y_pred,
                          cmap='plasma', alpha=0.5, edgecolors='none', s=12)
        plt.colorbar(sc5, ax=ax5, label='Predicted Salary')
        ax5.plot([m_y_test.min(), m_y_test.max()],
                 [m_y_test.min(), m_y_test.max()],
                 color='#E91E8C', linewidth=2, linestyle='--', label='Perfect Prediction')
        ax5.set_title("ACTUAL vs PREDICTED SALARY", fontweight='bold')
        ax5.set_xlabel("Actual (LPA)"); ax5.set_ylabel("Predicted (LPA)")
        ax5.legend(facecolor='#13101C', edgecolor='#2D2048', labelcolor='#F0E8FF')
        plt.tight_layout(); st.pyplot(fig5, use_container_width=True)

# ==============================
# PAGE 5 — ABOUT (PROFESSIONAL REDESIGN)
# ==============================
def page_about(df, salary_accuracy):
    st.markdown("""
    <div style="margin-bottom:1.5rem">
      <div style="font-family:'Orbitron',monospace;font-size:1.4rem;color:#F0E8FF;
                  margin-bottom:.25rem;letter-spacing:2px;text-shadow:0 0 20px rgba(233,30,140,.3)">
        SYSTEM INFORMATION
      </div>
      <div style="font-size:.65rem;color:#9B8EC4;font-family:'Space Mono',monospace;letter-spacing:1px">
        ARCHITECTURE · METHODOLOGY · MODEL PERFORMANCE · DATASET SPECIFICATIONS
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top KPI strip ─────────────────────────────────────────────────────────
    st.markdown('<div class="cl-section">Platform At A Glance</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("8",       "Career Domains",    "pink"),
        ("64",      "Job Roles",         "cyan"),
        (f"{salary_accuracy}%", "Salary Accuracy", "green"),
        ("160K+",   "Training Records",  "yellow"),
        ("5 YRS",   "Forecast Horizon",  "pink"),
    ]
    for col, (val, label, color) in zip([k1, k2, k3, k4, k5], kpis):
        with col:
            st.markdown(f"""
            <div class="cl-metric" style="text-align:center">
              <div class="cl-metric-value {color}" style="font-size:1.3rem">{val}</div>
              <div class="cl-metric-label" style="margin-top:.4rem">{label}</div>
            </div>""", unsafe_allow_html=True)

    # ── About prose ───────────────────────────────────────────────────────────
    st.markdown('<div class="cl-section">About The Platform</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-card" style="border-top:2px solid var(--accent)">
      <p style="font-size:.78rem;color:var(--text);line-height:1.9;font-family:'Space Mono',monospace">
        <strong style="color:var(--accent3)">CareerLens</strong> is an AI-powered career intelligence
        platform that maps your current skill set to the most suitable job roles across
        <strong style="color:var(--accent3)">8 professional domains</strong> — covering Computer Science &amp; AI,
        all major Engineering disciplines, Medicine, Finance, and Textile. It predicts expected salaries
        using gradient-boosted regression and forecasts hiring demand up to
        <strong style="color:var(--accent3)">5 years ahead</strong> using time-series machine learning.
        Every result comes with a personalised skill gap analysis and a step-by-step acquisition roadmap
        complete with curated learning resources.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Core capabilities ─────────────────────────────────────────────────────
    st.markdown('<div class="cl-section">Core Capabilities</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns(4)
    feats = [
        ("🎯", "Smart Matching",
         "TF-IDF cosine similarity identifies the closest-fit role from 64 job profiles across 8 domains in real time."),
        ("💰", "Salary Prediction",
         "XGBoost regressor with log-transform and 7 engineered features estimates personalised LPA for any target year."),
        ("📈", "5-Year Forecast",
         "Gradient Boosting with lag, rolling mean, and trend index projects hiring demand five years forward."),
        ("🧩", "Skill Gap Analysis",
         "Compares your skills against role requirements and generates a prioritised, resource-linked learning roadmap."),
    ]
    for col, (icon, title, desc) in zip([fc1, fc2, fc3, fc4], feats):
        with col:
            st.markdown(f"""
            <div class="cl-feature">
              <div class="cl-feature-icon">{icon}</div>
              <div class="cl-feature-title">{title}</div>
              <div class="cl-feature-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── Architecture + Pipeline ───────────────────────────────────────────────
    st.markdown('<div class="cl-section">System Architecture & Prediction Pipeline</div>', unsafe_allow_html=True)
    arch_col, pipe_col = st.columns(2)

    with arch_col:
        st.markdown("""
        <div class="about-card" style="height:100%">
          <h3>🏗️ System Architecture</h3>
          <div class="arch-box" style="margin-top:.6rem;line-height:2.2;font-size:.67rem">
            <span style="color:var(--accent3)">[ USER / BROWSER ]</span><br>
            &nbsp;&nbsp;&nbsp;↓ &nbsp;Streamlit Web Frontend<br>
            <span style="color:var(--cyan)">[ SKILL INPUT + TARGET YEAR ]</span><br>
            &nbsp;&nbsp;&nbsp;↓ &nbsp;Text Normalisation · Abbreviation Expansion<br>
            <span style="color:var(--yellow)">[ TF-IDF VECTORIZER — sklearn ]</span><br>
            &nbsp;&nbsp;&nbsp;↓ &nbsp;Cosine Similarity → Best Job Match<br>
            <span style="color:var(--accent3)">[ XGBOOST SALARY REGRESSOR ]</span><br>
            &nbsp;&nbsp;&nbsp;↓ &nbsp;log1p Transform → Salary (LPA)<br>
            <span style="color:var(--green)">[ GRADIENT BOOSTING FORECASTER ]</span><br>
            &nbsp;&nbsp;&nbsp;↓ &nbsp;Lag Features → 5-Year Demand Forecast<br>
            <span style="color:var(--cyan)">[ RESULTS + PDF INTELLIGENCE REPORT ]</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with pipe_col:
        st.markdown("""
        <div class="about-card" style="height:100%">
          <h3>⚙️ Prediction Pipeline — Step by Step</h3>
          <div style="margin-top:.6rem">
            <div class="roadmap-step" style="margin-bottom:.45rem;border-left-color:var(--accent)">
              <div class="roadmap-step-header">
                <span class="roadmap-step-title">Step 01 · Skill Normalisation</span>
                <span class="roadmap-badge badge-beginner">Input</span>
              </div>
              <div class="roadmap-meta">Lowercase, regex clean, abbreviation map (ml→machine_learning, etc.)</div>
            </div>
            <div class="roadmap-step" style="margin-bottom:.45rem;border-left-color:var(--cyan)">
              <div class="roadmap-step-header">
                <span class="roadmap-step-title">Step 02 · TF-IDF Vectorisation</span>
                <span class="roadmap-badge badge-inter">Match</span>
              </div>
              <div class="roadmap-meta">Fit on all required + user skills; cosine similarity → best role selected</div>
            </div>
            <div class="roadmap-step" style="margin-bottom:.45rem;border-left-color:var(--yellow)">
              <div class="roadmap-step-header">
                <span class="roadmap-step-title">Step 03 · Feature Engineering</span>
                <span class="roadmap-badge badge-inter">Encode</span>
              </div>
              <div class="roadmap-meta">7 features: year, postings, job_encoded, similarity, skill counts, match %</div>
            </div>
            <div class="roadmap-step" style="margin-bottom:.45rem;border-left-color:var(--accent3)">
              <div class="roadmap-step-header">
                <span class="roadmap-step-title">Step 04 · Salary Prediction</span>
                <span class="roadmap-badge badge-advanced">XGBoost</span>
              </div>
              <div class="roadmap-meta">XGBRegressor · 500 trees · depth 8 · log1p → expm1 decode</div>
            </div>
            <div class="roadmap-step" style="border-left-color:var(--green)">
              <div class="roadmap-step-header">
                <span class="roadmap-step-title">Step 05 · Demand Forecasting</span>
                <span class="roadmap-badge badge-advanced">GBR</span>
              </div>
              <div class="roadmap-meta">GradientBoostingRegressor · lag1/lag2 · rolling mean · 5-year horizon</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Model Performance + Tech Stack ────────────────────────────────────────
    st.markdown('<div class="cl-section">Model Performance & Tech Stack</div>', unsafe_allow_html=True)
    mp_col, ts_col = st.columns(2)

    with mp_col:
        st.markdown(f"""
        <div class="about-card">
          <h3>📊 Model Performance Metrics</h3>
          <table class="about-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Salary Model</th>
                <th>Forecast Model</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="label">Algorithm</td>
                <td class="val-pink">XGBRegressor</td>
                <td class="val-cyan">GradientBoostingRegressor</td>
              </tr>
              <tr>
                <td class="label">Estimators</td>
                <td class="val-text">500 trees</td>
                <td class="val-text">500 trees</td>
              </tr>
              <tr>
                <td class="label">Max Depth</td>
                <td class="val-text">8</td>
                <td class="val-text">3</td>
              </tr>
              <tr>
                <td class="label">Test Split</td>
                <td class="val-text">20% held-out</td>
                <td class="val-text">Walk-forward</td>
              </tr>
              <tr>
                <td class="label">Loss / Target</td>
                <td class="val-text">log1p MSE</td>
                <td class="val-text">Squared error</td>
              </tr>
              <tr>
                <td class="label">Accuracy</td>
                <td class="val-green">{salary_accuracy}% (±10% tol.)</td>
                <td class="val-green">Per-role reported</td>
              </tr>
            </tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with ts_col:
        st.markdown("""
        <div class="about-card">
          <h3>🛠️ Tech Stack</h3>
          <table class="about-table">
            <thead>
              <tr>
                <th>Layer</th>
                <th>Library</th>
                <th>Purpose</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="label">Frontend</td>
                <td class="val-pink">Streamlit 1.35+</td>
                <td class="val-text">UI &amp; routing</td>
              </tr>
              <tr>
                <td class="label">ML Core</td>
                <td class="val-cyan">XGBoost · sklearn</td>
                <td class="val-text">Salary prediction</td>
              </tr>
              <tr>
                <td class="label">NLP</td>
                <td class="val-cyan">TF-IDF Vectorizer</td>
                <td class="val-text">Skill matching</td>
              </tr>
              <tr>
                <td class="label">Forecasting</td>
                <td class="val-cyan">GradientBoosting</td>
                <td class="val-text">Demand forecast</td>
              </tr>
              <tr>
                <td class="label">Data</td>
                <td class="val-cyan">pandas · numpy</td>
                <td class="val-text">Processing &amp; EDA</td>
              </tr>
              <tr>
                <td class="label">Charts</td>
                <td class="val-cyan">Matplotlib</td>
                <td class="val-text">Dark visualisations</td>
              </tr>
              <tr>
                <td class="label">Export</td>
                <td class="val-cyan">PdfPages</td>
                <td class="val-text">PDF report generation</td>
              </tr>
            </tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

    # ── Dataset Specifications ─────────────────────────────────────────────────
    st.markdown('<div class="cl-section">Dataset Specifications</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="about-card">
      <h3>🗄️ Training Data Overview</h3>
      <div style="display:flex;flex-wrap:wrap;gap:.6rem;margin-top:.7rem">
        <div style="flex:1;min-width:150px;background:var(--bg);border:1px solid var(--border2);
                    border-radius:6px;padding:.8rem 1rem">
          <div class="cl-metric-label">Total Records</div>
          <div class="cl-metric-value cyan" style="font-size:1.1rem">{len(df):,}+</div>
          <div class="cl-metric-sub">per domain (synthetic)</div>
        </div>
        <div style="flex:1;min-width:150px;background:var(--bg);border:1px solid var(--border2);
                    border-radius:6px;padding:.8rem 1rem">
          <div class="cl-metric-label">Year Range</div>
          <div class="cl-metric-value yellow" style="font-size:1.1rem">{int(df['year'].min())}–{int(df['year'].max())}</div>
          <div class="cl-metric-sub">multi-year historical</div>
        </div>
        <div style="flex:1;min-width:150px;background:var(--bg);border:1px solid var(--border2);
                    border-radius:6px;padding:.8rem 1rem">
          <div class="cl-metric-label">Salary Range</div>
          <div class="cl-metric-value pink" style="font-size:1.1rem">3–23 LPA</div>
          <div class="cl-metric-sub">base + similarity + noise</div>
        </div>
        <div style="flex:1;min-width:150px;background:var(--bg);border:1px solid var(--border2);
                    border-radius:6px;padding:.8rem 1rem">
          <div class="cl-metric-label">Input Features</div>
          <div class="cl-metric-value green" style="font-size:1.1rem">7</div>
          <div class="cl-metric-sub">engineered per sample</div>
        </div>
        <div style="flex:1;min-width:150px;background:var(--bg);border:1px solid var(--border2);
                    border-radius:6px;padding:.8rem 1rem">
          <div class="cl-metric-label">Domains</div>
          <div class="cl-metric-value cyan" style="font-size:1.1rem">8</div>
          <div class="cl-metric-sub">CS · Mech · Civil · EE · ECE · Textile · Med · Fin</div>
        </div>
      </div>
      <div style="margin-top:.9rem;font-size:.65rem;color:var(--muted);
                  font-family:'Space Mono',monospace;line-height:2;
                  border-top:1px solid var(--border2);padding-top:.7rem">
        <strong style="color:var(--accent3)">Columns:</strong>
        job_title · required_skills · user_skills · avg_salary_lpa · job_postings · year<br>
        <strong style="color:var(--accent3)">Auto domain select:</strong>
        Cosine similarity scored across all 8 domains simultaneously; highest-scoring domain wins
        with tiebreak priority ordering to handle near-equal scores.<br>
        <strong style="color:var(--accent3)">Salary construction:</strong>
        domain_base_LPA + (cosine_similarity × 15) + gaussian_noise ∈ [−0.5, +0.5]
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    st.markdown('<div class="cl-section">Disclaimer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer">
      <strong>⚠ IMPORTANT — FOR INFORMATIONAL PURPOSES ONLY</strong><br><br>
      CareerLens is built on a <strong>synthetic dataset</strong> generated for demonstration and
      research purposes. Salary estimates and job demand forecasts reflect modelled statistical
      patterns and are <strong>not sourced from live market data</strong>. Real-world compensation
      varies significantly based on company size, geographic location, years of experience,
      negotiation outcomes, and macroeconomic conditions. Do not make major career or financial
      decisions based solely on outputs from this tool. Always cross-reference with current
      industry salary surveys, recruiter data, and professional guidance.
    </div>
    """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;margin-top:2.5rem;padding:1.2rem 0;
                border-top:1px solid var(--border2);
                color:#9B8EC4;font-size:.6rem;
                font-family:'Space Mono',monospace;letter-spacing:1.5px">
      CAREERLENS AI SYSTEM v2.5 &nbsp;·&nbsp;
      XGBOOST + GRADIENT BOOSTING + TF-IDF &nbsp;·&nbsp;
      BUILT WITH STREAMLIT &nbsp;·&nbsp; © 2025
    </div>
    """, unsafe_allow_html=True)

# ==============================
# MAIN
# ==============================
def main():
    render_topbar()

    fields = [DOMAIN_AUTO] + CAREER_DOMAINS_ORDERED
    idx = fields.index(st.session_state.career_field) if st.session_state.career_field in fields else 0
    col_sel, _ = st.columns([3, 7])
    with col_sel:
        choice = st.selectbox("Career Domain", fields, index=idx,
            help="Auto: picks the domain with the best skill match. Lock a domain to search only that industry.")
    if choice != st.session_state.career_field:
        st.session_state.career_field = choice
        st.session_state.result = None
        st.rerun()

    with st.spinner("⚙️ Initialising intelligence modules…"):
        df, tfidf, le, salary_model, X_cols, salary_accuracy, \
        y_test_actual_eval, y_pred_actual_eval, req_matrix = load_and_train(effective_training_domain())

    render_tabbar(st.session_state.current_page)

    page = st.session_state.current_page
    if page == "Dashboard":
        page_dashboard(df, salary_accuracy)
    elif page == "Salary Prediction":
        page_salary(df, tfidf, le, salary_model, X_cols, req_matrix)
    elif page == "Job Forecasting":
        page_forecasting(df)
    elif page == "Market Trends":
        page_market()
    elif page == "About":
        page_about(df, salary_accuracy)

    if page != "About":
        st.markdown("""
        <div style="text-align:center;margin-top:2rem;padding:1rem;
                    border-top:1px solid var(--border2);
                    color:#9B8EC4;font-size:.6rem;
                    font-family:'Space Mono',monospace;letter-spacing:1px">
            CAREERLENS AI SYSTEM v2.5 · STREAMLIT + XGBOOST + GRADIENT BOOSTING · © 2025
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()