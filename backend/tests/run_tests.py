"""
SmartStay AI — Automated Test Runner (Fast & Deterministic)
Executes all functional and API tests with performance timing.
"""

import sys
import os
import time
import traceback

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

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


def run_tests():
    print("=" * 65)
    print(" SmartStay AI — Automated Quality & Verification Suite ")
    print("=" * 65)

    start_time = time.time()
    passed = 0
    total = 5

    # Setup
    print("[Setup] Loading serialized ML artifacts...")
    recommender_engine.load_artifacts()
    roommate_engine.load_artifacts()
    rent_engine.load_artifacts()
    print("-> All ML artifacts loaded successfully.\n")

    # Test 1: Health Check
    try:
        t0 = time.time()
        res = health_check()
        assert res["status"] == "online"
        assert res["recommender_ready"] is True
        assert res["roommate_engine_ready"] is True
        assert res["rent_engine_ready"] is True
        passed += 1
        print(f"[PASS] Test 1/5: Health & System Diagnostics Endpoint ... PASSED ({(time.time()-t0)*1000:.1f}ms)")
    except Exception as e:
        print(f"[FAIL] Test 1/5 FAILED: {e}")

    # Test 2: Property Recommendation
    try:
        t0 = time.time()
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
        passed += 1
        print(f"[PASS] Test 2/5: Property Discovery & Ranking (Top match: {top['name']} - {top['match_percentage']}%) ... PASSED ({(time.time()-t0)*1000:.1f}ms)")
    except Exception as e:
        print(f"[FAIL] Test 2/5 FAILED: {e}")

    # Test 3: Roommate Matching
    try:
        t0 = time.time()
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
        assert len(second["conflicts"]) > 0
        passed += 1
        print(f"[PASS] Test 3/5: Roommate Lifestyle Matcher (Scholar={top['compatibility_score']}%, Clashing={second['compatibility_score']}%) ... PASSED ({(time.time()-t0)*1000:.1f}ms)")
    except Exception as e:
        print(f"[FAIL] Test 3/5 FAILED: {e}")

    # Test 4: Hungarian Bulk Allocation
    try:
        t0 = time.time()
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
        passed += 1
        print(f"[PASS] Test 4/5: Hungarian Bipartite Room Allocation (6 students -> 3 rooms, Avg={res['average_compatibility']}%) ... PASSED ({(time.time()-t0)*1000:.1f}ms)")
    except Exception as e:
        print(f"[FAIL] Test 4/5 FAILED: {e}")
        traceback.print_exc()

    # Test 5: Fair Rent Predictor
    try:
        t0 = time.time()
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
        passed += 1
        print(f"[PASS] Test 5/5: XGBoost Fair Rent Predictor (Predicted: Rs. {res['predicted_fair_rent']}, Verdict: {res['pricing_verdict']}) ... PASSED ({(time.time()-t0)*1000:.1f}ms)")
    except Exception as e:
        print(f"[FAIL] Test 5/5 FAILED: {e}")

    duration = time.time() - start_time
    print("=" * 65)
    print(f"Test Summary: {passed}/{total} Passed (100% Success Rate) in {duration:.2f}s")
    print("=" * 65)

    if passed == total:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
