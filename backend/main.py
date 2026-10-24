"""
SmartStay AI — FastAPI Backend Application
Exposes REST endpoints for:
1. Property Discovery & Recommendation (/api/v1/recommend-hostels)
2. Roommate Lifestyle Compatibility & Ranking (/api/v1/match-roommates)
3. Warden Capacity-Constrained Bulk Room Allocation (/api/v1/bulk-allocate-rooms)
4. Fair Rent Valuation Predictor (/api/v1/predict-rent)
"""

import os
import sys
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure root directory is on Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.schemas import (
    HostelRecommendRequest, HostelRecommendResponse,
    RoommateMatchRequest, RoommateMatchResponse, RoommateMatchCandidate,
    BulkAllocationRequest, BulkAllocationResponse, RoomAllocationPair,
    RentPredictionRequest, RentPredictionResponse
)
from ml_engine.recommender import PropertyRecommender
from ml_engine.roommate_matcher import RoommateMatchingEngine, FEATURE_COLS
from ml_engine.rent_predictor import FairRentPredictor

# Global engine instances
recommender_engine = PropertyRecommender()
roommate_engine = RoommateMatchingEngine()
rent_engine = FairRentPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML artifacts
    print("[Startup] Loading SmartStay AI ML artifacts...")
    try:
        recommender_engine.load_artifacts()
    except Exception as e:
        print(f"[Warning] Recommender load failed: {e}")

    try:
        roommate_engine.load_artifacts()
    except Exception as e:
        print(f"[Warning] Roommate matcher load failed: {e}")

    try:
        rent_engine.load_artifacts()
    except Exception as e:
        print(f"[Warning] Rent predictor load failed: {e}")

    print("[Startup] ML Artifacts initialization check completed.")
    yield
    print("[Shutdown] Cleaning up SmartStay AI API...")


app = FastAPI(
    title="SmartStay AI — Student Accommodation & Roommate Allocation API",
    version="1.0.0",
    description="SPPU Third-Year IT Capstone Project: Content-based property discovery, K-Means & Hungarian roommate allocation, and XGBoost fair rent estimation.",
    lifespan=lifespan
)

# Enable CORS for Streamlit / React frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "online",
        "service": "SmartStay AI Engine",
        "version": "1.0.0",
        "recommender_ready": recommender_engine.is_fitted,
        "roommate_engine_ready": roommate_engine.is_fitted,
        "rent_engine_ready": rent_engine.is_fitted
    }


@app.post("/api/v1/recommend-hostels", response_model=HostelRecommendResponse)
def recommend_hostels(req: HostelRecommendRequest):
    if not recommender_engine.is_fitted:
        # Try loading on-demand
        recommender_engine.load_artifacts()
        if not recommender_engine.is_fitted:
            raise HTTPException(status_code=503, detail="Recommender model is not trained yet.")

    try:
        matches = recommender_engine.recommend(
            locality=req.locality,
            budget_max=req.budget_max,
            room_type=req.room_type,
            gender=req.gender,
            ac_preferred=req.ac_preferred,
            mess_preferred=req.mess_preferred,
            max_distance_km=req.max_distance_km,
            top_k=req.top_k
        )
        return {
            "success": True,
            "total_matches": len(matches),
            "results": matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/match-roommates", response_model=RoommateMatchResponse)
def match_roommates(req: RoommateMatchRequest):
    target = req.target_student
    v1 = np.array([getattr(target, col) for col in FEATURE_COLS])

    ranked = []
    for candidate in req.candidate_students:
        if candidate.student_id == target.student_id:
            continue
        v2 = np.array([getattr(candidate, col) for col in FEATURE_COLS])
        score, conflicts = roommate_engine.calculate_compatibility(v1, v2)
        ranked.append({
            "candidate": candidate,
            "compatibility_score": score,
            "conflicts": conflicts if conflicts else ["None — Highly Compatible Persona"]
        })

    # Sort descending by compatibility score
    ranked = sorted(ranked, key=lambda x: x["compatibility_score"], reverse=True)[:req.top_k]

    return {
        "success": True,
        "target_student_id": target.student_id,
        "matches": ranked
    }


@app.post("/api/v1/bulk-allocate-rooms", response_model=BulkAllocationResponse)
def bulk_allocate_rooms(req: BulkAllocationRequest):
    students_dict_list = [s.model_dump() for s in req.students]
    if len(students_dict_list) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 students for roommate allocation.")

    pairs = roommate_engine.allocate_rooms_hungarian(students_dict_list)

    valid_scores = [p["compatibility_score"] for p in pairs if p["student_2"] is not None]
    avg_score = round(float(np.mean(valid_scores)), 1) if valid_scores else 100.0

    return {
        "success": True,
        "total_students": len(req.students),
        "total_rooms_formed": len(pairs),
        "average_compatibility": avg_score,
        "allocations": pairs
    }


@app.post("/api/v1/predict-rent", response_model=RentPredictionResponse)
def predict_rent(req: RentPredictionRequest):
    if not rent_engine.is_fitted:
        rent_engine.load_artifacts()
        if not rent_engine.is_fitted:
            raise HTTPException(status_code=503, detail="Rent prediction model is not trained yet.")

    try:
        features = req.model_dump()
        result = rent_engine.predict_fair_rent(features)
        return {
            "success": True,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
