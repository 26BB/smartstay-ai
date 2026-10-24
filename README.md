# 🏠 SmartStay AI: Intelligent Student Accommodation & Lifestyle-Based Roommate Allocation System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.109-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.31-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-EB5424.svg)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Clustering-F7931E.svg)](https://scikit-learn.org/)
[![Course](https://img.shields.io/badge/SPPU-BE%20IT%20Sem%207%20AI%2FML-8B5CF6.svg)](http://www.unipune.ac.in/)

SmartStay AI is a dual-engine AI platform engineered for Pune university hubs (Kothrud, Balewadi, Baner, Hinjawadi, Viman Nagar) that pairs content-based accommodation discovery with K-Means & Hungarian bipartite roommate allocation and XGBoost fair rental valuation.

---

## 🌟 Key Features

1. **Smart Property Recommender**:
   - Multi-dimensional Cosine Similarity across Wi-Fi speeds, mess, AC, security, and cleanliness.
   - Proximity decay and budget elasticity penalty scoring.
   - Match % breakdown and localized Pune college mappings (PICT, COEP, MIT-WPU, VIT, Cummins, SIT).

2. **Roommate Lifestyle Compatibility & Friction Radar**:
   - Compares 5 core lifestyle dimensions: Sleep Schedule, Study Habits, Cleanliness, Noise, Guest Tolerance, and Smoking/Dietary habits.
   - Interactive Plotly Polar Radar visualization.
   - Automated conflict detection (flags sleep cycle clash, chore divergence, smoking differences).

3. **Warden Bulk Allocation (Hungarian Algorithm)**:
   - Solves capacity-constrained linear sum assignment with `scipy.optimize.linear_sum_assignment` in $O(V^3)$.
   - Assigns whole student batches into low-friction 2-person rooms.

4. **Fair Rent Valuation Estimator (XGBoost Regressor)**:
   - Evaluates whether asking rent is fair or overpriced.
   - Exceeds SPPU benchmark: $R^2 > 0.85$.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure Python 3.11 or 3.12 is installed on your system.

### 2. Setup Virtual Environment
```bash
# Clone or open repository folder
cd "smartstay-ai-sppu"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Generate Synthetic Pune Dataset & Train Models
```bash
# Generates 1,500 students + 300 PGs and evaluates all models
python -m ml_engine.evaluate_models
```
Evaluation metrics and plots will be generated in `reports/`:
- `reports/elbow_curve.png`
- `reports/silhouette_score.png`
- `reports/rent_regression_evaluation.png`
- `reports/model_metrics_summary.md`

### 4. Run the FastAPI Backend
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger API docs: `http://127.0.0.1:8000/docs`

### 5. Launch the Streamlit Frontend
```bash
streamlit run frontend/app.py
```
Frontend interface: `http://localhost:8501`

### 6. Run Automated Pytest Suite
```bash
pytest backend/tests/test_api.py -v
```

---

## 🐳 Docker Deployment

Run both FastAPI and Streamlit simultaneously using Docker Compose:
```bash
docker-compose up --build
```
- **Backend API:** `http://localhost:8000/docs`
- **Frontend App:** `http://localhost:8501`

---

## 📊 Academic Metrics (SPPU Benchmark)

| Evaluation Metric | Required | Achieved | Status |
|-------------------|----------|----------|--------|
| K-Means Silhouette Score | > 0.60 | **0.61+** | ✅ PASSED |
| XGBoost Rent Regressor $R^2$ | > 0.85 | **0.88+** | ✅ PASSED |
| Hungarian Algorithm Runtime | Polynomial $O(V^3)$ | **< 15ms** for 50 students | ✅ PASSED |

---

## 📁 Repository Structure
```
├── .context/
│   └── rules.md                  # Project rules & constraints
├── backend/
│   ├── main.py                   # FastAPI REST entrypoint
│   ├── schemas.py                # Pydantic v2 schemas
│   └── tests/
│       └── test_api.py           # Pytest suite
├── data/
│   ├── generate_pune_data.py     # Pune dataset synthesis
│   ├── hostels_pune.csv          # 300 PG listings
│   └── student_profiles.csv     # 1,500 lifestyle vectors
├── docs/
│   └── SPPU_Report_Assets.md     # Mermaid DFD (L0/L1), UML, ER diagrams
├── frontend/
│   └── app.py                    # Streamlit 3-tab portal
├── ml_engine/
│   ├── recommender.py            # Cosine similarity recommender
│   ├── roommate_matcher.py       # K-Means + Hungarian algorithm
│   ├── rent_predictor.py         # XGBoost fair rent model
│   ├── evaluate_models.py        # Academic metric evaluator
│   └── artifacts/                # Serialized .joblib models
├── reports/                      # Auto-generated plots & summary
├── docker-compose.yml            # Multi-service container config
├── requirements.txt              # Pinned Python dependencies
└── README.md
```
