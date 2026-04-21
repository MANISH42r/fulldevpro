Project Documentation – CareerLens AI System
1. Project Title

CareerLens: AI-Based Job, Salary & Skill Prediction System





2. Objective

To develop an intelligent system that:

Predicts best job role based on user skills
Estimates expected salary (LPA)
Forecasts future job demand (5 years)
Provides skill gap analysis + learning roadmap






3. Problem Statement

Students and professionals often:

Don’t know which job fits their skills
Lack clarity on salary expectations
Don’t know what skills to learn next

👉 This project solves it using AI + ML models + data-driven insights







4. Tech Stack
Frontend
Streamlit (interactive UI)
Custom CSS (dark futuristic dashboard)
Backend / ML
Python
Pandas, NumPy
Scikit-learn
XGBoost
Visualization
Matplotlib






5. System Overview

The system works in 3 major stages:

1. Input
User enters skills (comma-separated)
Selects target year
2. Processing
Skills are cleaned & standardized
Converted into vectors using TF-IDF
Compared with dataset using cosine similarity
3. Output
Best matching job role
Predicted salary
Skill gap + roadmap
Job market forecast






6. Key Features
🎯 1. Smart Job Matching
Uses TF-IDF + Cosine Similarity
Finds best role across multiple domains
💰 2. Salary Prediction
Model used: XGBoost Regressor
Predicts salary using:
Skills match %
Job postings
Year
Skill count
📈 3. Job Forecasting
Model: Gradient Boosting Regressor
Uses:
Lag features (previous years data)
Trend analysis
Predicts next 5 years demand
🧩 4. Skill Gap Analysis
Compares:
User skills vs required skills
Outputs:
Missing skills
Matching skills
🗺️ 5. Learning Roadmap Generator
Orders skills by:
Difficulty
Importance
Provides:
Time estimates
YouTube resources







7. Machine Learning Models Used
1. TF-IDF Vectorizer

Used for:

Converting text (skills) → numerical vectors
2. Cosine Similarity

Used for:

Matching user skills with job requirements
3. XGBoost Regressor

Used for:

Salary prediction

Why XGBoost?

High accuracy
Handles non-linearity
Works well on tabular data
4. Gradient Boosting Regressor

Used for:

Job trend forecasting










8. Workflow
Step-by-Step Flow
User inputs skills
Text preprocessing:
Lowercase
Remove noise
Expand abbreviations (ML → Machine Learning)
TF-IDF transformation
Cosine similarity calculation
Best job role selected
Salary predicted (XGBoost)
Skill gap identified
Forecast generated (5 years)













9. Multi-Domain Support

System supports:

Computer Science & AI
Mechanical Engineering
Civil Engineering
Electrical Engineering
Electronics & Communication
Textile
Medicine
Finance

👉 Also supports AUTO domain detection














10. Important Logic (Very Important for Viva)
🔹 Skill Standardization
ml → machine_learning
ai → artificial_intelligence

👉 Improves matching accuracy

🔹 Salary Formula Logic

Salary is influenced by:

Base salary (domain-based)
Skill similarity
Random variation (realistic prediction)
🔹 Feature Engineering

Model uses:

similarity score
number of skills
job postings
encoded job role
🔹 Forecasting Logic

Uses:

lag1, lag2 (previous years)
rolling mean
trend index

👉 This makes prediction more realistic












11. Output Features

System provides:

✅ Best job role
✅ Match percentage
✅ Predicted salary (LPA)
✅ Skill gap
✅ Roadmap
✅ Job demand forecast
✅ PDF report generation












12. UI Features
Dark futuristic theme
Dashboard with:
Metrics
Alerts
Graphs
Interactive tabs:
Dashboard
Salary Prediction
Job Forecast
Market Trends











13. Model Performance
Salary Model Accuracy: ~ (calculated dynamically)
Uses tolerance-based accuracy (±10%)









14. Future Improvements
🔮 Real-time job data integration (LinkedIn API)
🤖 LLM-based career advisor
📊 Better recommendation system
🌍 Multi-language support















15. Key Highlights 

“I built an AI-based career prediction system using ML models like XGBoost and Gradient Boosting”
“It uses TF-IDF and cosine similarity for intelligent job matching”
“The system also provides skill gap analysis and personalized roadmap”
“It predicts salary and future job demand using machine learning models”