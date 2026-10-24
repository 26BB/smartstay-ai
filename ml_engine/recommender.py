"""
SmartStay AI — Property Discovery & Recommendation Engine
Combines content-based filtering (Cosine Similarity on multi-dimensional amenity vectors)
with proximity/budget constraints and rating weighting to rank Pune student accommodations.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

AMENITY_COLS = [
    "wifi_speed_mbps",
    "mess_included",
    "mess_rating",
    "ac_available",
    "gym_available",
    "laundry_available",
    "power_backup",
    "cctv_security",
    "cleanliness_score",
    "overall_rating"
]


class PropertyRecommender:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.df_hostels: Optional[pd.DataFrame] = None
        self.hostel_feature_matrix: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, df_hostels: pd.DataFrame):
        """Fits scaler on hostel amenities and saves transformed matrix."""
        self.df_hostels = df_hostels.copy()
        X = self.df_hostels[AMENITY_COLS].values
        self.hostel_feature_matrix = self.scaler.fit_transform(X)
        self.is_fitted = True

        # Save artifacts
        joblib.dump(self.scaler, os.path.join(ARTIFACTS_DIR, "recommender_scaler.joblib"))
        self.df_hostels.to_pickle(os.path.join(ARTIFACTS_DIR, "hostels_clean.pkl"))

    def load_artifacts(self):
        """Loads fitted scaler and dataset."""
        scaler_path = os.path.join(ARTIFACTS_DIR, "recommender_scaler.joblib")
        data_path = os.path.join(ARTIFACTS_DIR, "hostels_clean.pkl")
        if os.path.exists(scaler_path) and os.path.exists(data_path):
            self.scaler = joblib.load(scaler_path)
            self.df_hostels = pd.read_pickle(data_path)
            X = self.df_hostels[AMENITY_COLS].values
            self.hostel_feature_matrix = self.scaler.transform(X)
            self.is_fitted = True

    def recommend(
        self,
        locality: str,
        budget_max: float,
        room_type: str = "Double",
        gender: str = "Boys",
        ac_preferred: bool = False,
        mess_preferred: bool = True,
        max_distance_km: float = 6.0,
        top_k: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Ranks accommodations matching the student's criteria using weighted Cosine Similarity.
        """
        if not self.is_fitted or self.df_hostels is None:
            raise RuntimeError("PropertyRecommender is not fitted. Run fit() or load_artifacts().")

        df = self.df_hostels.copy()

        # Hard filters: Gender policy
        if gender in ["Boys", "Girls"]:
            df = df[df["gender_allowed"].isin([gender, "Co-ed"])]

        # Soft/Hard Filter: Locality preference
        # We prioritize the target locality, but allow nearby if few matches
        loc_df = df[df["locality"].str.lower() == locality.lower()]
        if len(loc_df) >= 3:
            df = loc_df

        # Filter by room type if available
        room_matches = df[df["room_type"].str.lower() == room_type.lower()]
        if len(room_matches) >= 3:
            df = room_matches

        if df.empty:
            df = self.df_hostels.copy()

        # Construct target student ideal preference vector
        target_vec = np.array([
            150.0,                                    # wifi (normalized ~150 Mbps)
            1.0 if mess_preferred else 0.0,          # mess
            4.2 if mess_preferred else 0.0,          # mess rating
            1.0 if ac_preferred else 0.0,            # ac
            0.5,                                      # gym
            1.0,                                      # laundry
            1.0,                                      # power backup
            1.0,                                      # cctv
            4.5,                                      # cleanliness target
            4.5                                       # overall rating target
        ]).reshape(1, -1)

        target_scaled = self.scaler.transform(target_vec)
        candidate_features = self.scaler.transform(df[AMENITY_COLS].values)

        # Compute Cosine Similarity
        cos_sims = cosine_similarity(target_scaled, candidate_features)[0]

        results = []
        for idx, (_, row) in enumerate(df.iterrows()):
            base_score = float(cos_sims[idx])

            # Apply distance discount/penalty
            dist = float(row["distance_to_college_km"])
            dist_factor = max(0.6, 1.0 - (dist / max(max_distance_km, 1.0)) * 0.25)

            # Apply budget discount/penalty
            rent = float(row["monthly_rent"])
            if rent <= budget_max:
                budget_factor = 1.05  # bonus for affordability
            else:
                over = rent - budget_max
                budget_factor = max(0.5, 1.0 - (over / budget_max))

            # Final Match Percentage (0% to 100%)
            final_match_pct = round(base_score * dist_factor * budget_factor * 100.0, 1)
            final_match_pct = min(98.5, max(35.0, final_match_pct))

            results.append({
                "hostel_id": row["hostel_id"],
                "name": row["name"],
                "locality": row["locality"],
                "gender_allowed": row["gender_allowed"],
                "room_type": row["room_type"],
                "monthly_rent": int(rent),
                "deposit_amount": int(row["deposit_amount"]),
                "distance_to_college_km": dist,
                "overall_rating": float(row["overall_rating"]),
                "cleanliness_score": float(row["cleanliness_score"]),
                "wifi_speed_mbps": int(row["wifi_speed_mbps"]),
                "mess_included": bool(row["mess_included"]),
                "ac_available": bool(row["ac_available"]),
                "power_backup": bool(row["power_backup"]),
                "curfew_time": row["curfew_time"],
                "available_beds": int(row["available_beds"]),
                "match_percentage": final_match_pct,
                "match_summary": f"{final_match_pct}% Match — {row['locality']} | {dist} km from campus | ₹{int(rent)}/mo"
            })

        # Sort descending by match percentage
        results = sorted(results, key=lambda x: x["match_percentage"], reverse=True)
        return results[:top_k]
