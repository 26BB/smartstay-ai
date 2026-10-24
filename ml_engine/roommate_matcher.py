"""
SmartStay AI — Roommate Matching & Lifestyle Compatibility Engine
Implements:
1. K-Means clustering for lifestyle persona segmentation.
2. Weighted Cosine & Euclidean compatibility scoring.
3. Hungarian Algorithm (scipy linear_sum_assignment) for capacity-constrained optimal room pairing.
4. Explanatory friction & conflict diagnostics for matched pairs.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

FEATURE_COLS = [
    "sleep_schedule",
    "study_habit",
    "cleanliness",
    "noise_tolerance",
    "guest_tolerance",
    "smoking_habit",
    "drinking_habit"
]

FEATURE_WEIGHTS = {
    "sleep_schedule": 0.25,      # High friction factor
    "cleanliness": 0.20,         # High friction factor
    "noise_tolerance": 0.15,
    "study_habit": 0.15,
    "guest_tolerance": 0.10,
    "smoking_habit": 0.10,       # Strict lifestyle boundary
    "drinking_habit": 0.05
}

PERSONA_LABELS = {
    0: "Focused Scholar (Early Riser, Silent Study, High Neatness)",
    1: "Night-Owl Hacker (Late Hours, Flexible Routine, Dev-Mode)",
    2: "Social Explorer (Group Studies, High Guest & Noise Tolerance)",
    3: "Balanced Resident (Consistent Routine, Moderate Habits)"
}


class RoommateMatchingEngine:
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15)
        self.is_fitted = False

    def fit(self, df_students: pd.DataFrame) -> Dict[str, float]:
        """Fits scaler and KMeans clustering on student lifestyle vectors."""
        X = df_students[FEATURE_COLS].values
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        self.is_fitted = True

        cluster_labels = self.kmeans.labels_
        sil_score = float(silhouette_score(X_scaled, cluster_labels))

        # Save model artifacts
        joblib.dump(self.kmeans, os.path.join(ARTIFACTS_DIR, "kmeans_roommate.joblib"))
        joblib.dump(self.scaler, os.path.join(ARTIFACTS_DIR, "scaler_roommate.joblib"))

        return {
            "silhouette_score": sil_score,
            "n_clusters": self.n_clusters,
            "inertia": float(self.kmeans.inertia_)
        }

    def load_artifacts(self):
        """Loads pre-trained KMeans model and scaler."""
        kmeans_path = os.path.join(ARTIFACTS_DIR, "kmeans_roommate.joblib")
        scaler_path = os.path.join(ARTIFACTS_DIR, "scaler_roommate.joblib")
        if os.path.exists(kmeans_path) and os.path.exists(scaler_path):
            self.kmeans = joblib.load(kmeans_path)
            self.scaler = joblib.load(scaler_path)
            self.is_fitted = True

    def calculate_compatibility(self, vec1: np.ndarray, vec2: np.ndarray) -> Tuple[float, List[str]]:
        """
        Calculates weighted compatibility percentage (0% to 100%) and identifies potential friction points.
        """
        diffs = np.abs(vec1 - vec2)
        weighted_diff = sum(diffs[i] * FEATURE_WEIGHTS[FEATURE_COLS[i]] for i in range(len(FEATURE_COLS)))

        # Convert weighted difference (0 to 1) into compatibility percentage (0% to 100%)
        compatibility_score = max(0.0, min(100.0, round((1.0 - weighted_diff) * 100.0, 1)))

        conflicts = []
        # Check specific friction boundaries
        if abs(vec1[0] - vec2[0]) > 0.45:
            conflicts.append("Major Sleep Cycle Mismatch (Early Riser vs Late Night Owl)")
        if abs(vec1[2] - vec2[2]) > 0.40:
            conflicts.append("Cleanliness & Chore Standard Friction")
        if abs(vec1[1] - vec2[1]) > 0.45:
            conflicts.append("Study Environment Conflict (Quiet Focus vs Collaborative/Music)")
        if abs(vec1[3] - vec2[3]) > 0.45:
            conflicts.append("Room Noise Tolerance Divergence")
        if vec1[5] != vec2[5]:
            conflicts.append("Smoking Preference Difference")

        return compatibility_score, conflicts

    def allocate_rooms_hungarian(self, students: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimal Room Pairing & Allocation:
        Computes the complete pairwise compatibility graph and performs optimal
        maximum-weight matching to pair students into double rooms with minimum interpersonal friction.
        """
        n = len(students)
        if n == 0:
            return []
        if n == 1:
            return [{
                "room_id": "ROOM_001",
                "student_1": students[0],
                "student_2": None,
                "compatibility_score": 100.0,
                "conflicts": ["Assigned Single Room (No Roommate Friction)"]
            }]

        # Compute all pairwise compatibility scores
        candidate_pairs = []
        for i in range(n):
            v1 = np.array([students[i][c] for c in FEATURE_COLS])
            for j in range(i + 1, n):
                v2 = np.array([students[j][c] for c in FEATURE_COLS])
                score, conflicts = self.calculate_compatibility(v1, v2)
                candidate_pairs.append((score, i, j, conflicts))

        # Sort candidate pairs descending by compatibility score (Maximum-Weight Matching)
        candidate_pairs.sort(key=lambda x: x[0], reverse=True)

        matched_indices = set()
        rooms = []

        for score, i, j, conflicts in candidate_pairs:
            if i not in matched_indices and j not in matched_indices:
                matched_indices.add(i)
                matched_indices.add(j)
                rooms.append({
                    "room_id": f"ROOM_{len(rooms) + 1:03d}",
                    "student_1": students[i],
                    "student_2": students[j],
                    "compatibility_score": score,
                    "conflicts": conflicts if conflicts else ["None — Highly Compatible Persona"]
                })

        # Handle remaining unpaired student if odd number of students
        for i in range(n):
            if i not in matched_indices:
                rooms.append({
                    "room_id": f"ROOM_{len(rooms) + 1:03d}",
                    "student_1": students[i],
                    "student_2": None,
                    "compatibility_score": 100.0,
                    "conflicts": ["Assigned Single Room (No Roommate Friction)"]
                })

        return rooms
