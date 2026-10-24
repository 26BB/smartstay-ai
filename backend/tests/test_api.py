"""
SmartStay AI — FastAPI Endpoint & ML Core Automated Test Suite
Tests schema validation, model inference, compatibility logic, and Hungarian assignment.
"""

import sys
import os
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.main import (
    health_check,
    recommend_hostels,
    match_roommates,
    bulk_allocate_rooms,
    predict_rent,
    recommender_engine,
    roommate_engine,
    rent_engine
)
from backend.schemas import (
    HostelRecommendRequest,
    RoommateMatchRequest,
    StudentProfile,
    BulkAllocationRequest,
    RentPredictionRequest
)


@pytest.fixture(scope="session", autouse=True)
def ensure_models_loaded():
    """Ensures pre-trained artifacts are loaded."""
    recommender_engine.load_artifacts()
    roommate_engine.load_artifacts()
    rent_engine.load_artifacts()


def test_health_check():
    res = health_check()
    assert res["status"] == "online"
    assert res["service"] == "SmartStay AI Engine"
    assert res["recommender_ready"] is True
    assert res["roommate_engine_ready"] is True
    assert res["rent_engine_ready"] is True


def test_recommend_hostels():
    req = HostelRecommendRequest(
        locality="Kothrud",
        budget_max=9500.0,
        room_type="Double",
        gender="Boys",
        ac_preferred=False,
        mess_preferred=True,
        max_distance_km=5.0,
        top_k=4
    )
    res = recommend_hostels(req)
    assert res["success"] is True
    assert len(res["results"]) > 0
    top = res["results"][0]
    assert top["match_percentage"] > 40.0
    assert top["monthly_rent"] > 0
    assert top["locality"] == "Kothrud"


def test_match_roommates():
    target = StudentProfile(
        student_id="STU_TEST_01",
        name="Piyush Patil",
        gender="Male",
        sleep_schedule=0.15,
        study_habit=0.15,
        cleanliness=0.90,
        noise_tolerance=0.15,
        guest_tolerance=0.15,
        smoking_habit=0,
        drinking_habit=0,
        dietary_habit="Veg"
    )

    candidates = [
        StudentProfile(
            student_id="STU_TEST_02",
            name="Compatible Scholar",
            gender="Male",
            sleep_schedule=0.16,
            study_habit=0.18,
            cleanliness=0.88,
            noise_tolerance=0.20,
            guest_tolerance=0.20,
            smoking_habit=0,
            drinking_habit=0,
            dietary_habit="Veg"
        ),
        StudentProfile(
            student_id="STU_TEST_03",
            name="Conflicting Night Owl",
            gender="Male",
            sleep_schedule=0.95,
            study_habit=0.85,
            cleanliness=0.30,
            noise_tolerance=0.90,
            guest_tolerance=0.90,
            smoking_habit=1,
            drinking_habit=1,
            dietary_habit="Non-Veg"
        )
    ]

    req = RoommateMatchRequest(
        target_student=target,
        candidate_students=candidates,
        top_k=2
    )
    res = match_roommates(req)
    assert res["success"] is True
    assert len(res["matches"]) == 2
    top = res["matches"][0]
    second = res["matches"][1]
    assert top["compatibility_score"] > second["compatibility_score"]
    assert top["compatibility_score"] >= 80.0
    # Conflicting candidate must have friction warnings
    assert len(second["conflicts"]) > 0


def test_bulk_room_allocation():
    students = [
        StudentProfile(
            student_id=f"STU_BULK_{i:02d}",
            name=f"Student {i}",
            gender="Male",
            sleep_schedule=0.15 if i % 2 == 0 else 0.85,
            study_habit=0.15 if i % 2 == 0 else 0.85,
            cleanliness=0.90 if i % 2 == 0 else 0.40,
            noise_tolerance=0.15 if i % 2 == 0 else 0.80,
            guest_tolerance=0.15 if i % 2 == 0 else 0.80,
            smoking_habit=0 if i % 2 == 0 else 1,
            drinking_habit=0 if i % 2 == 0 else 1,
            dietary_habit="Veg" if i % 2 == 0 else "Non-Veg"
        )
        for i in range(1, 7)
    ]

    req = BulkAllocationRequest(students=students)
    res = bulk_allocate_rooms(req)
    assert res["success"] is True
    assert res["total_students"] == 6
    assert res["total_rooms_formed"] == 3
    assert res["average_compatibility"] > 70.0


def test_predict_rent():
    req = RentPredictionRequest(
        locality="Kothrud",
        room_type="Double",
        gender_allowed="Boys",
        distance_to_college_km=1.5,
        wifi_speed_mbps=100,
        mess_included=1,
        ac_available=1,
        gym_available=0,
        laundry_available=1,
        power_backup=1,
        cctv_security=1,
        cleanliness_score=4.5,
        overall_rating=4.3,
        monthly_rent=13500
    )
    res = predict_rent(req)
    assert res["success"] is True
    assert res["predicted_fair_rent"] > 4000
    assert res["fair_range_min"] < res["predicted_fair_rent"] < res["fair_range_max"]
    assert "pricing_verdict" in res
