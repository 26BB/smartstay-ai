"""
SmartStay AI — Pydantic v2 Schemas
Type-safe request and response validation for FastAPI endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# -------------------------------------------------------------------------
# Property Recommender Schemas
# -------------------------------------------------------------------------
class HostelRecommendRequest(BaseModel):
    locality: str = Field(..., example="Kothrud", description="Pune locality (Balewadi, Baner, Kothrud, Hinjawadi, Viman Nagar)")
    budget_max: float = Field(..., ge=3000, le=50000, example=9500.0, description="Maximum monthly budget in INR")
    room_type: str = Field(default="Double", example="Double", description="Single, Double, Triple, or Four")
    gender: str = Field(default="Boys", example="Boys", description="Boys, Girls, or Co-ed")
    ac_preferred: bool = Field(default=False, example=False)
    mess_preferred: bool = Field(default=True, example=True)
    max_distance_km: float = Field(default=6.0, ge=0.5, le=25.0, example=5.0)
    top_k: int = Field(default=6, ge=1, le=20, example=6)


class HostelItem(BaseModel):
    hostel_id: str
    name: str
    locality: str
    gender_allowed: str
    room_type: str
    monthly_rent: int
    deposit_amount: int
    distance_to_college_km: float
    overall_rating: float
    cleanliness_score: float
    wifi_speed_mbps: int
    mess_included: bool
    ac_available: bool
    power_backup: bool
    curfew_time: str
    available_beds: int
    match_percentage: float
    match_summary: str


class HostelRecommendResponse(BaseModel):
    success: bool
    total_matches: int
    results: List[HostelItem]


# -------------------------------------------------------------------------
# Roommate Lifestyle & Matching Schemas
# -------------------------------------------------------------------------
class StudentProfile(BaseModel):
    student_id: str = Field(..., example="STU001")
    name: str = Field(..., example="Piyush Patil")
    gender: str = Field(default="Male", example="Male")
    sleep_schedule: float = Field(..., ge=0.0, le=1.0, example=0.2, description="0=Early Bird (5am), 1=Night Owl (3am)")
    study_habit: float = Field(..., ge=0.0, le=1.0, example=0.25, description="0=Silent solo, 1=Group study")
    cleanliness: float = Field(..., ge=0.0, le=1.0, example=0.85, description="0=Relaxed, 1=Spotless neat freak")
    noise_tolerance: float = Field(..., ge=0.0, le=1.0, example=0.3, description="0=Dead silence, 1=Loud parties")
    guest_tolerance: float = Field(..., ge=0.0, le=1.0, example=0.3, description="0=Zero visitors, 1=Frequent guests")
    smoking_habit: int = Field(default=0, ge=0, le=1, example=0, description="0=Non-smoker, 1=Smoker")
    drinking_habit: int = Field(default=0, ge=0, le=1, example=0, description="0=Non-drinker, 1=Drinker")
    dietary_habit: Optional[str] = Field(default="Veg", example="Veg")


class RoommateMatchRequest(BaseModel):
    target_student: StudentProfile
    candidate_students: List[StudentProfile]
    top_k: int = Field(default=5, ge=1, le=20)


class RoommateMatchCandidate(BaseModel):
    candidate: StudentProfile
    compatibility_score: float
    conflicts: List[str]


class RoommateMatchResponse(BaseModel):
    success: bool
    target_student_id: str
    matches: List[RoommateMatchCandidate]


class BulkAllocationRequest(BaseModel):
    students: List[StudentProfile]


class RoomAllocationPair(BaseModel):
    room_id: str
    student_1: StudentProfile
    student_2: Optional[StudentProfile] = None
    compatibility_score: float
    conflicts: List[str]


class BulkAllocationResponse(BaseModel):
    success: bool
    total_students: int
    total_rooms_formed: int
    average_compatibility: float
    allocations: List[RoomAllocationPair]


# -------------------------------------------------------------------------
# Fair Rent Estimator Schemas
# -------------------------------------------------------------------------
class RentPredictionRequest(BaseModel):
    locality: str = Field(..., example="Kothrud")
    room_type: str = Field(default="Double", example="Double")
    gender_allowed: str = Field(default="Boys", example="Boys")
    distance_to_college_km: float = Field(default=2.5, ge=0.1, le=25.0)
    wifi_speed_mbps: int = Field(default=100, ge=10, le=1000)
    mess_included: int = Field(default=1, ge=0, le=1)
    ac_available: int = Field(default=0, ge=0, le=1)
    gym_available: int = Field(default=0, ge=0, le=1)
    laundry_available: int = Field(default=1, ge=0, le=1)
    power_backup: int = Field(default=1, ge=0, le=1)
    cctv_security: int = Field(default=1, ge=0, le=1)
    cleanliness_score: float = Field(default=4.2, ge=1.0, le=5.0)
    overall_rating: float = Field(default=4.0, ge=1.0, le=5.0)
    monthly_rent: Optional[int] = Field(default=None, description="Optional asked rent for fairness comparison")


class RentPredictionResponse(BaseModel):
    success: bool
    predicted_fair_rent: int
    fair_range_min: int
    fair_range_max: int
    pricing_verdict: str
    locality: str
    room_type: str
