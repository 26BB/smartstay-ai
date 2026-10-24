# **SmartStay AI: Intelligent Student Accommodation & Lifestyle-Based Roommate Allocation System**

**Course Context:** SPPU Third-Year Information Technology (AI/ML Project)

# **1\. Problem Statement & Project Overview**

Traditional hostel and PG allocation systems rely entirely on static first-come-first-serve queues, manual forms, or rigid roll-number batching. This blind allocation results in high interpersonal friction (divergent study hours, conflicting sleep cycles, differing social habits) and poor accommodation discovery.

SmartStay AI transforms the manual process into a dual-engine AI platform:

* **Property Discovery & Recommendation Engine:** Predicts optimal accommodation matches using location, price, and facility preferences.  
* **Roommate Compatibility & Conflict Prediction Engine:** Utilizes clustering and vector similarity to pair students with compatible living profiles before room assignment.

# **2\. Core AI/ML Modules & Tech Stack**

## **Core Modules**

### **Module 1: Smart Property Recommendation Engine**

* **Objective:** Rank available accommodations (hostels, PGs, flats) tailored to student preferences.  
* **Technique:** Content-based filtering using Cosine Similarity combined with Collaborative Filtering (Matrix Factorization / SVD).  
* **Features:** Proximity to campus (km), budget range, room sharing type (single/double/triple), Wi-Fi, mess rating, and transit accessibility.

### **Module 2: Lifestyle Compatibility & Roommate Matching**

* **Objective:** Form stable, low-friction room clusters based on student daily routines.  
* **Technique:** K-Means / Agglomerative Hierarchical Clustering for persona segmentation, coupled with Euclidean Distance / Weighted Cosine Similarity scoring and Hungarian Algorithm (linear sum assignment) for capacity-constrained pairing.  
* **Input Vector (Normalized 0 to 1):** Sleep Schedule (Early bird vs. Night owl), Study Habits (Quiet focus vs. Group study), Cleanliness & Chore Tolerance (Scale 1-5), Smoking / Dietary Habits (Binary / Categorical encoded), and Noise & Guest Tolerance (Scale 1-5).

### **Module 3: Rent Value Predictor (Fair Price Estimator)**

* **Objective:** Ensure transparent pricing and prevent local overcharging by predicting fair rent.  
* **Technique:** XGBoost Regressor / Random Forest Regressor evaluated using Root Mean Squared Error (RMSE) and R-squared (R2) score.

### **Module 4: Review Sentiment Analysis (Optional Extension)**

* **Objective:** Extract authentic feedback on local amenities and landlord reliability.  
* **Technique:** VADER / DistilBERT for aspect-based sentiment analysis on student reviews (food quality, water availability, security).

## **Tech Stack Architecture**

| Layer | Tools & Frameworks | Purpose |
| :---- | :---- | :---- |
| Language | Python 3.11 | Core language for ML models and API |
| ML Frameworks | Scikit-Learn, XGBoost, SciPy, NumPy, Pandas | Clustering, regression, optimal assignment, and data transformations |
| Backend API | FastAPI, Pydantic v2 | Model inference endpoints and DB orchestration |
| Vector Engine | FAISS or ChromaDB | Fast nearest-neighbor vector matching |
| Frontend | Streamlit (or React.js) | Dynamic UI for students and hostel warden/admin |
| Database | SQLite / PostgreSQL | User profiles, hostel listings, and allocation logs |

# **3\. Target Workspace Architecture in Google Antigravity**

Initialize the repository with the following folder structure so Antigravity's context engine indexes system boundaries cleanly:smartstay-ai-sppu/

├── .context/

│   └── rules.md                 \# Project guardrails, SPPU syllabus constraints & tech stack rules

├── data/

│   ├── generate\_pune\_data.py    \# Synthetic generator (Balewadi, Kothrud, Hinjawadi, Viman Nagar)

│   ├── hostels\_pune.csv         \# Generated PG/hostel listings

│   └── student\_profiles.csv     \# Lifestyle & preference vectors

├── ml\_engine/

│   ├── recommender.py           \# Cosine Similarity \+ SVD property engine

│   ├── roommate\_matcher.py      \# Weighted Cosine \+ K-Means \+ Hungarian room assignment

│   ├── rent\_predictor.py        \# XGBoost regressor for fair rent estimation

│   ├── evaluate\_models.py       \# Outputs Silhouette Score, RMSE, and R2 metrics

│   └── artifacts/               \# Saved .joblib models & scalers

├── backend/

│   ├── main.py                  \# FastAPI entrypoint

│   ├── schemas.py               \# Pydantic request/response validation

│   └── tests/                   \# Pytest suite managed by Antigravity test agent

├── frontend/

│   └── app.py                   \# Streamlit multi-tab student & warden portal

└── docker-compose.yml           \# One-command container orchestration

# **4\. Multi-Agent Workload Allocation**

| Agent Role | Antigravity View & Mode | Primary Responsibility | Key Artifact Produced |
| :---- | :---- | :---- | :---- |
| Agent 1: Data & ML Architect | Agent Manager (Turbo Mode for local Python scripts) | Synthesizes the Pune accommodation dataset, trains Scikit-Learn/XGBoost models, and validates metrics. | Model evaluation charts, .joblib files, & metric summary in walkthrough.md |
| Agent 2: Backend API Engineer | Editor View \+ Side Panel Agent (Strict/Supervised Mode) | Builds FastAPI endpoints, integrates saved ML models, and runs pytest suites in the integrated terminal. | OpenAPI schema, passing test logs, and task.md checklist |
| Agent 3: Frontend & QA Verifier | Agent Manager \+ Built-in Browser | Builds the Streamlit UI, launches the local server, and autonomously tests user flows in the Chrome preview. | UI screenshots, browser recordings, and final walkthrough.md |

# **5\. Step-by-Step Google Antigravity Execution Plan**

## **Step 1: Initialize Workspace Context & Agent Guardrails**

* Open an empty `smartstay-ai-sppu` folder in Antigravity and create `.context/rules.md` to seed the workspace knowledge base.  
* **Stack Lock:** Enforce Python 3.11, Scikit-Learn, XGBoost, FastAPI, Pydantic v2, and Streamlit so agents do not hallucinate heavy cloud dependencies.  
* **Academic Metrics Requirement:** Every ML module must export evaluation plots (Elbow curve, Silhouette score greater than 0.60, and XGBoost R2 score greater than 0.85) into a `reports/` directory for the SPPU project report.  
* **Supervision Mode:** Set terminal execution to Strict Mode during initial scaffolding so you approve package installations (`pip` / `uv`), then switch to Turbo Mode for iterative test-and-fix loops.

## **Step 2: Generate Pune Dataset & Train Core ML Models**

* Spawn **Agent 1** in the Agent Manager to build the data and ML layer.  
* **Synthetic Data Script:** Instruct the agent to write and execute `data/generate_pune_data.py` to create 1,500 student lifestyle records and 300 realistic Pune PG/hostel listings across Balewadi, Baner, Kothrud, Hinjawadi, and Viman Nagar (incorporating distance to college, mess availability, curfew rules, Wi-Fi speed, and deposit amounts).  
* **Model Pipeline:** Implement `recommender.py`, `roommate_matcher.py` (using `scipy.optimize.linear_sum_assignment` for conflict-free room capacity matching), and `rent_predictor.py`.  
* **Artifact Verification:** Review the generated Implementation Plan Artifact before approving execution. Let the agent run `evaluate_models.py` in the terminal and fix any feature-scaling bugs until target Silhouette and R2 thresholds pass.

## **Step 3: Scaffold the FastAPI Service & Automated Tests**

* Switch to the **Editor View** and use the side-panel agent to build the backend middleware once serialized `.joblib` models are saved in `ml_engine/artifacts/`.  
* **Endpoints:** Scaffold `POST /api/v1/recommend-hostels`, `POST /api/v1/match-roommates`, and `POST /api/v1/predict-rent` in `backend/main.py`.  
* **Self-Healing Test Loop:** Instruct the agent to generate `backend/tests/test_api.py` using FastAPI's `TestClient` and run `pytest` in the terminal. Allow the agent to read terminal tracebacks and patch `schemas.py` automatically if serialization issues occur.

## **Step 4: Build the Portal & Run Browser-Based Verification**

* Launch **Agent 3** to build the frontend and verify the end-to-end user journey using Antigravity's built-in Chrome browser.  
* **Three-Tab Interface (`frontend/app.py`):**  
  1. *Student Hostel Finder:* Filters \+ AI match percentage cards.  
  2. *Roommate Compatibility Matcher:* Interactive radar chart comparing two students' lifestyle vectors.  
  3. *Warden Bulk Allocation & Fair Rent Checker:* Uploads a CSV of students and outputs optimal room pairings.  
* **Autonomous Browser Check:** Instruct the agent to start both the FastAPI and Streamlit servers and open the local URL in Antigravity's built-in Chrome browser to test sliders, submit forms, capture screenshots, and attach a visual Walkthrough Artifact.

## **Step 5: Containerize & Generate SPPU Documentation Artifacts**

* **Dockerization:** Generate a multi-container `docker-compose.yml` running the FastAPI backend on port 8000 and the Streamlit UI on port 8501\.  
* **Report Diagrams:** Prompt the agent to scan the finalized codebase and generate Mermaid.js diagrams for the System Architecture, Data Flow Diagram (DFD Level 0 & Level 1), UML Sequence Diagram, and ER Diagram inside `docs/SPPU_Report_Assets.md`.

# **6\. Kickoff Prompt for Google Antigravity**

Copy and paste the following prompt directly into the Antigravity Agent panel to initialize the project:@workspace We are building "SmartStay AI", an AI Student Accommodation Recommendation, Roommate Compatibility Matching, and Fair Rent Prediction system for an SPPU 3rd-year IT project.

Please generate an Implementation Plan Artifact and Task List to build this end-to-end in four phases:

1\. Data & ML Engine (data/ and ml\_engine/):

   \- Create generate\_pune\_data.py to synthesize 1,500 student lifestyle profiles (sleep schedule, study habits, cleanliness, noise tolerance, guest tolerance, diet/smoking) and 300 Pune hostel/PG listings across Balewadi, Baner, Kothrud, Hinjawadi, and Viman Nagar.

   \- Build and train three models: Content-based Hostel Recommender (Cosine Similarity), Roommate Matcher (Weighted Cosine Similarity \+ K-Means Clustering \+ Scipy linear\_sum\_assignment for optimal pair allocation), and Fair Rent Predictor (XGBoost Regressor).

   \- Export all trained models/scalers to ml\_engine/artifacts/ and save evaluation plots (Elbow plot, Silhouette score, RMSE, R2) to reports/.

2\. FastAPI Backend (backend/):

   \- Create main.py and schemas.py exposing /api/v1/recommend-hostels, /api/v1/match-roommates, and /api/v1/predict-rent.

   \- Write and run a pytest suite to verify all endpoints return 200 OK with valid JSON payloads.

3\. Streamlit Frontend (frontend/app.py):

   \- Create a 3-tab dashboard (Hostel Finder, Roommate Radar Comparison, and Warden Bulk Room Allocator \+ Rent Estimator).

   \- Launch the backend and frontend servers, test the UI in the built-in browser, and capture screenshots in a Walkthrough Artifact.

4\. Packaging (docker-compose.yml & docs/):

   \- Add Docker Compose configuration and generate Mermaid DFD (Level 0/1) and ER diagrams in docs/.

Start by generating the Implementation Plan for review before executing Phase 1\.  
