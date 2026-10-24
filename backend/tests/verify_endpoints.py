"""
SmartStay AI — Endpoints Direct Tester
Runs automated verification against all API routes using TestClient with explicit context.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from backend.main import app

def run_all_tests():
    print("[1/5] Testing Client Startup & Health Endpoint...")
    with TestClient(app) as client:
        res = client.get("/api/v1/health")
        print(f"Health Response: status={res.status_code}, json={res.json()}")
        assert res.status_code == 200
        assert res.json()["status"] == "online"

        print("\n[2/5] Testing Property Recommender Endpoint...")
        rec_payload = {
            "locality": "Kothrud",
            "budget_max": 9500.0,
            "room_type": "Double",
            "gender": "Boys",
            "ac_preferred": False,
            "mess_preferred": True,
            "max_distance_km": 5.0,
            "top_k": 3
        }
        res = client.post("/api/v1/recommend-hostels", json=rec_payload)
        assert res.status_code == 200
        data = res.json()
        print(f"Recommended {len(data['results'])} hostels. Top match: {data['results'][0]['name']} ({data['results'][0]['match_percentage']}%)")

        print("\n[3/5] Testing Roommate Compatibility Endpoint...")
        target = {
            "student_id": "STU_TEST_01",
            "name": "Piyush Patil",
            "gender": "Male",
            "sleep_schedule": 0.2,
            "study_habit": 0.2,
            "cleanliness": 0.9,
            "noise_tolerance": 0.2,
            "guest_tolerance": 0.3,
            "smoking_habit": 0,
            "drinking_habit": 0,
            "dietary_habit": "Veg"
        }
        candidates = [
            {
                "student_id": "STU_TEST_02",
                "name": "Compatible Scholar",
                "gender": "Male",
                "sleep_schedule": 0.22,
                "study_habit": 0.25,
                "cleanliness": 0.85,
                "noise_tolerance": 0.25,
                "guest_tolerance": 0.35,
                "smoking_habit": 0,
                "drinking_habit": 0,
                "dietary_habit": "Veg"
            },
            {
                "student_id": "STU_TEST_03",
                "name": "Conflicting Night Owl",
                "gender": "Male",
                "sleep_schedule": 0.95,
                "study_habit": 0.85,
                "cleanliness": 0.30,
                "noise_tolerance": 0.90,
                "guest_tolerance": 0.90,
                "smoking_habit": 1,
                "drinking_habit": 1,
                "dietary_habit": "Non-Veg"
            }
        ]
        match_payload = {
            "target_student": target,
            "candidate_students": candidates,
            "top_k": 2
        }
        res = client.post("/api/v1/match-roommates", json=match_payload)
        assert res.status_code == 200
        m_data = res.json()
        top_match = m_data["matches"][0]
        second_match = m_data["matches"][1]
        print(f"Candidate 1 Compatibility: {top_match['compatibility_score']}% | Conflicts: {top_match['conflicts']}")
        print(f"Candidate 2 Compatibility: {second_match['compatibility_score']}% | Conflicts: {second_match['conflicts']}")
        assert top_match["compatibility_score"] > second_match["compatibility_score"]

        print("\n[4/5] Testing Hungarian Bulk Room Allocation...")
        bulk_students = [
            {
                "student_id": f"STU_BULK_{i:02d}",
                "name": f"Student {i}",
                "gender": "Male",
                "sleep_schedule": 0.15 if i % 2 == 0 else 0.85,
                "study_habit": 0.15 if i % 2 == 0 else 0.85,
                "cleanliness": 0.90 if i % 2 == 0 else 0.40,
                "noise_tolerance": 0.15 if i % 2 == 0 else 0.80,
                "guest_tolerance": 0.15 if i % 2 == 0 else 0.80,
                "smoking_habit": 0 if i % 2 == 0 else 1,
                "drinking_habit": 0 if i % 2 == 0 else 1,
                "dietary_habit": "Veg" if i % 2 == 0 else "Non-Veg"
            }
            for i in range(1, 7)
        ]
        res = client.post("/api/v1/bulk-allocate-rooms", json={"students": bulk_students})
        assert res.status_code == 200
        alloc_data = res.json()
        print(f"Allocated {alloc_data['total_students']} students across {alloc_data['total_rooms_formed']} rooms. Avg score: {alloc_data['average_compatibility']}%")

        print("\n[5/5] Testing Fair Rent Valuation Predictor (XGBoost)...")
        rent_payload = {
            "locality": "Kothrud",
            "room_type": "Double",
            "gender_allowed": "Boys",
            "distance_to_college_km": 1.5,
            "wifi_speed_mbps": 100,
            "mess_included": 1,
            "ac_available": 1,
            "gym_available": 0,
            "laundry_available": 1,
            "power_backup": 1,
            "cctv_security": 1,
            "cleanliness_score": 4.5,
            "overall_rating": 4.3,
            "monthly_rent": 12500
        }
        res = client.post("/api/v1/predict-rent", json=rent_payload)
        assert res.status_code == 200
        r_data = res.json()
        print(f"Fair Rent Prediction: Rs. {r_data['predicted_fair_rent']}/mo | Band: Rs. {r_data['fair_range_min']} - {r_data['fair_range_max']} | Verdict: {r_data['pricing_verdict']}")

    print("\n[SUCCESS] All 5 API endpoint test groups passed perfectly!")

if __name__ == "__main__":
    run_all_tests()
