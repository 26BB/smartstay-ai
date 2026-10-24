# SmartStay AI — Project Guardrails & Syllabus Rules

## Project Context
- **Title**: SmartStay AI — Intelligent Student Accommodation & Lifestyle-Based Roommate Allocation System
- **Context**: SPPU 3rd-Year Information Technology (AI/ML Project)
- **Target OS**: Windows / Cross-platform Linux containers

## Tech Stack Lock
- **Language**: Python 3.11+
- **ML Frameworks**: Scikit-Learn, SciPy, NumPy, Pandas, XGBoost, Joblib
- **API Framework**: FastAPI, Pydantic v2, Uvicorn
- **UI Framework**: Streamlit, Plotly
- **Testing**: Pytest, HTTPX / TestClient

## Academic Metric Constraints (SPPU)
1. **Roommate Lifestyle Clustering (K-Means)**:
   - Silhouette Score must exceed `0.60`.
   - Elbow curve analysis must be saved to `reports/`.
2. **Fair Rent Predictor (XGBoost / Random Forest Regressor)**:
   - Evaluated using RMSE and R-squared ($R^2$).
   - $R^2$ score must exceed `0.85`.
3. **Property Discovery Recommendation**:
   - Hybrid / Content-based filtering using Cosine Similarity on amenities, proximity, and budget.
4. **Room Allocation**:
   - Capacity-constrained bipartite matching using `scipy.optimize.linear_sum_assignment` (Hungarian Algorithm) to minimize lifestyle friction.

## Structural Boundaries
- Do not introduce heavy cloud dependencies or external proprietary vector databases that require cloud credentials.
- Keep ML models cleanly serialized in `ml_engine/artifacts/` using `.joblib`.
- API endpoints must validate all inputs via Pydantic v2 schemas.
- UI must provide multi-tab interactive flows for both students and hostel wardens.
