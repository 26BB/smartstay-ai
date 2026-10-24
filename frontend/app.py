"""
SmartStay AI — Interactive Streamlit Dashboard
Multi-tab Portal for Students and Hostel Wardens / Administrators
Tab 1: Smart Property Discovery & AI Accommodation Recommender
Tab 2: Roommate Lifestyle Compatibility & Radar Conflict Analysis
Tab 3: Warden Bulk Room Allocator (Hungarian Algorithm) & Fair Rent Valuation
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Ensure project root is in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml_engine.recommender import PropertyRecommender
from ml_engine.roommate_matcher import RoommateMatchingEngine, FEATURE_COLS
from ml_engine.rent_predictor import FairRentPredictor

st.set_page_config(
    page_title="SmartStay AI — Accommodation & Roommate Allocator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern university-grade UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E40AF 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .badge-match {
        background-color: #10B981;
        color: white;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-friction {
        background-color: #EF4444;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    """Caches in-memory ML engines for instant inference."""
    hostels_path = os.path.join(BASE_DIR, "data", "hostels_pune.csv")
    students_path = os.path.join(BASE_DIR, "data", "student_profiles.csv")

    rec = PropertyRecommender()
    rm = RoommateMatchingEngine()
    rent = FairRentPredictor()

    # Load artifacts or train if not present
    try:
        rec.load_artifacts()
    except Exception:
        if os.path.exists(hostels_path):
            rec.fit(pd.read_csv(hostels_path))

    try:
        rm.load_artifacts()
    except Exception:
        if os.path.exists(students_path):
            rm.fit(pd.read_csv(students_path))

    try:
        rent.load_artifacts()
    except Exception:
        if os.path.exists(hostels_path):
            rent.train(pd.read_csv(hostels_path))

    return rec, rm, rent


rec_engine, rm_engine, rent_engine = load_all_models()

# Header
st.markdown('<div class="main-header">SmartStay AI — Accommodation & Roommate System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">SPPU Third-Year IT Capstone | AI/ML-Powered Discovery, Lifestyle Matching & Fair Valuation</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🏠 Property Discovery",
    "🤝 Roommate Lifestyle Matcher",
    "📋 Warden Bulk Allocator & Fair Rent"
])

# =============================================================================
# TAB 1: PROPERTY DISCOVERY
# =============================================================================
with tabs[0]:
    st.subheader("Smart Accommodation Finder")
    st.write("Find Pune student hostels and PGs ranked by AI cosine similarity across amenities, distance, and budget.")

    col1, col2 = st.columns([1, 2.5])

    with col1:
        st.markdown("### 🎯 Your Preferences")
        locality = st.selectbox("Preferred Pune Locality", ["Kothrud", "Balewadi", "Baner", "Hinjawadi", "Viman Nagar"], index=0)
        budget = st.slider("Maximum Monthly Budget (₹)", min_value=5000, max_value=22000, value=9500, step=500)
        room_type = st.selectbox("Room Sharing Type", ["Single", "Double", "Triple", "Four"], index=1)
        gender = st.selectbox("Gender Accommodation", ["Boys", "Girls", "Co-ed"], index=0)
        max_dist = st.slider("Max Distance to Campus (km)", min_value=1.0, max_value=12.0, value=5.0, step=0.5)

        st.markdown("**Amenities Desired:**")
        mess_req = st.checkbox("Mess / Meal Facility Included", value=True)
        ac_req = st.checkbox("Air Conditioning (AC)", value=False)
        top_k = st.slider("Number of Recommendations", 3, 10, 6)

        search_btn = st.button("🔍 Find Matching Accommodations", use_container_width=True, type="primary")

    with col2:
        st.markdown("### 🏆 AI Recommended Accommodations")

        if search_btn or True:  # Default display
            try:
                results = rec_engine.recommend(
                    locality=locality,
                    budget_max=float(budget),
                    room_type=room_type,
                    gender=gender,
                    ac_preferred=ac_req,
                    mess_preferred=mess_req,
                    max_distance_km=max_dist,
                    top_k=top_k
                )

                if not results:
                    st.warning("No listings found matching exact criteria. Try increasing max budget or distance.")
                else:
                    for item in results:
                        with st.container():
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <h4 style="margin:0; color:#1F2937;">{item['name']}</h4>
                                    <span class="badge-match">{item['match_percentage']}% Match</span>
                                </div>
                                <p style="color:#6B7280; margin:4px 0 10px 0; font-size:0.95rem;">
                                    📍 <b>{item['locality']}</b> &nbsp;|&nbsp; 🚶 <b>{item['distance_to_college_km']} km</b> from college &nbsp;|&nbsp; 🛏️ <b>{item['room_type']} Room</b> ({item['gender_allowed']})
                                </p>
                                <div style="display:flex; gap:1.5rem; flex-wrap:wrap; font-size:0.9rem; margin-bottom:8px;">
                                    <div>💰 <b>₹{item['monthly_rent']:,}/month</b></div>
                                    <div>🔒 Deposit: ₹{item['deposit_amount']:,}</div>
                                    <div>📶 {item['wifi_speed_mbps']} Mbps Wi-Fi</div>
                                    <div>🍽️ Mess: {'Included' if item['mess_included'] else 'Not Included'}</div>
                                    <div>❄️ AC: {'Yes' if item['ac_available'] else 'No'}</div>
                                    <div>⭐ Cleanliness: {item['cleanliness_score']}/5.0</div>
                                    <div>⏰ Curfew: {item['curfew_time']}</div>
                                    <div>🛏️ Vacancies: {item['available_beds']} beds left</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Please ensure models are trained. Error: {e}")

# =============================================================================
# TAB 2: ROOMMATE LIFESTYLE MATCHER
# =============================================================================
with tabs[1]:
    st.subheader("Lifestyle Compatibility & Friction Radar")
    st.write("Compare lifestyle habits between prospective roommates to prevent interpersonal friction before room allocation.")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### 👤 Student 1 (You)")
        s1_name = st.text_input("Name", value="Piyush Patil", key="s1_name")
        s1_sleep = st.slider("Sleep Cycle", 0.0, 1.0, 0.25, 0.05,
                             help="0.0 = Early Riser (5 AM), 1.0 = Night Owl (3 AM)", key="s1_sleep")
        s1_study = st.slider("Study Habit", 0.0, 1.0, 0.20, 0.05,
                             help="0.0 = Silent Solo Focus, 1.0 = Group / Ambient Study", key="s1_study")
        s1_clean = st.slider("Cleanliness & Chore Standard", 0.0, 1.0, 0.85, 0.05,
                             help="0.0 = Relaxed, 1.0 = Spotless Neat Freak", key="s1_clean")
        s1_noise = st.slider("Room Noise Tolerance", 0.0, 1.0, 0.30, 0.05,
                             help="0.0 = Dead Silence, 1.0 = High Music/Voice Tolerance", key="s1_noise")
        s1_guest = st.slider("Guest & Visitor Tolerance", 0.0, 1.0, 0.35, 0.05,
                             help="0.0 = Zero Visitors, 1.0 = Frequent Social Visitors", key="s1_guest")
        s1_smoke = st.selectbox("Smoking Habit", [0, 1], format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker", key="s1_smoke")
        s1_drink = st.selectbox("Drinking Habit", [0, 1], format_func=lambda x: "Non-Drinker" if x == 0 else "Drinker", key="s1_drink")

    with colB:
        st.markdown("#### 👤 Student 2 (Candidate)")
        s2_name = st.text_input("Name", value="Rohan Deshmukh", key="s2_name")
        s2_sleep = st.slider("Sleep Cycle", 0.0, 1.0, 0.30, 0.05, key="s2_sleep")
        s2_study = st.slider("Study Habit", 0.0, 1.0, 0.35, 0.05, key="s2_study")
        s2_clean = st.slider("Cleanliness & Chore Standard", 0.0, 1.0, 0.80, 0.05, key="s2_clean")
        s2_noise = st.slider("Room Noise Tolerance", 0.0, 1.0, 0.40, 0.05, key="s2_noise")
        s2_guest = st.slider("Guest & Visitor Tolerance", 0.0, 1.0, 0.40, 0.05, key="s2_guest")
        s2_smoke = st.selectbox("Smoking Habit", [0, 1], format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker", key="s2_smoke")
        s2_drink = st.selectbox("Drinking Habit", [0, 1], format_func=lambda x: "Non-Drinker" if x == 0 else "Drinker", key="s2_drink")

    v1 = np.array([s1_sleep, s1_study, s1_clean, s1_noise, s1_guest, s1_smoke, s1_drink])
    v2 = np.array([s2_sleep, s2_study, s2_clean, s2_noise, s2_guest, s2_smoke, s2_drink])

    compat_score, conflicts = rm_engine.calculate_compatibility(v1, v2)

    st.markdown("---")
    res_col1, res_col2 = st.columns([1.2, 2])

    with res_col1:
        st.markdown("### Compatibility Verdict")
        if compat_score >= 80:
            badge_color = "#10B981"
            status = "🌟 Exceptional Living Compatibility"
        elif compat_score >= 60:
            badge_color = "#3B82F6"
            status = "👍 Moderate Compatibility (Minor Tradeoffs)"
        else:
            badge_color = "#EF4444"
            status = "⚠️ High Friction Risk — Not Recommended"

        st.markdown(f"""
        <div style="background-color:#F0FDF4; border-radius:12px; padding:1.5rem; text-align:center; border:2px solid {badge_color};">
            <h1 style="color:{badge_color}; margin:0; font-size:3.2rem;">{compat_score}%</h1>
            <h4 style="margin:8px 0 4px 0;">{status}</h4>
            <p style="color:#6B7280; font-size:0.9rem;">Calculated via Weighted Cosine & Normalized Distance Metrics</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🚨 Conflict & Friction Diagnostics:")
        if conflicts:
            for c in conflicts:
                st.markdown(f"- ⚠️ **{c}**")
        else:
            st.success("No critical friction points detected! Daily routines align comfortably.")

    with res_col2:
        st.markdown("### 🕸️ Lifestyle Dimension Radar")
        categories = ["Sleep Schedule", "Study Habit", "Cleanliness", "Noise Tolerance", "Guest Tolerance"]
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=[s1_sleep, s1_study, s1_clean, s1_noise, s1_guest],
            theta=categories,
            fill='toself',
            name=s1_name,
            line=dict(color='#2563EB', width=2)
        ))

        fig.add_trace(go.Scatterpolar(
            r=[s2_sleep, s2_study, s2_clean, s2_noise, s2_guest],
            theta=categories,
            fill='toself',
            name=s2_name,
            line=dict(color='#10B981', width=2)
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=340,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 3: WARDEN BULK ALLOCATION & FAIR RENT VALUATION
# =============================================================================
with tabs[2]:
    st.subheader("Warden & Administrator Management Center")

    warden_tab1, warden_tab2 = st.tabs(["⚖️ Fair Rent Valuation Predictor", "👥 Automated Bipartite Room Allocator"])

    with warden_tab1:
        st.markdown("#### Estimate Fair Rent Valuation (XGBoost Regressor)")
        st.caption("Benchmark PG and Hostel rates against Pune market standards to prevent student overcharging.")

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            r_loc = st.selectbox("Locality", ["Kothrud", "Balewadi", "Baner", "Hinjawadi", "Viman Nagar"], key="r_loc")
            r_type = st.selectbox("Room Type", ["Single", "Double", "Triple", "Four"], index=1, key="r_type")
            r_dist = st.slider("Distance to Campus (km)", 0.5, 12.0, 2.0, 0.5, key="r_dist")
            r_wifi = st.select_slider("Wi-Fi Speed (Mbps)", options=[30, 50, 100, 150, 200, 300], value=100, key="r_wifi")

        with rc2:
            r_mess = st.checkbox("Mess Included", value=True, key="r_mess")
            r_ac = st.checkbox("AC Available", value=False, key="r_ac")
            r_gym = st.checkbox("Gym Available", value=False, key="r_gym")
            r_laundry = st.checkbox("Laundry Facility", value=True, key="r_laundry")
            r_backup = st.checkbox("Power Backup", value=True, key="r_backup")
            r_cctv = st.checkbox("CCTV Security", value=True, key="r_cctv")

        with rc3:
            r_clean = st.slider("Cleanliness Rating (1-5)", 2.0, 5.0, 4.3, 0.1, key="r_clean")
            r_rate = st.slider("Overall Rating (1-5)", 2.0, 5.0, 4.2, 0.1, key="r_rate")
            r_asked = st.number_input("Landlord Asked Rent (₹, Optional)", value=9000, step=500, key="r_asked")

        eval_btn = st.button("📊 Calculate Fair Valuation", type="primary")

        if eval_btn or True:
            features = {
                "locality": r_loc,
                "room_type": r_type,
                "gender_allowed": "Boys",
                "distance_to_college_km": r_dist,
                "wifi_speed_mbps": r_wifi,
                "mess_included": 1 if r_mess else 0,
                "ac_available": 1 if r_ac else 0,
                "gym_available": 1 if r_gym else 0,
                "laundry_available": 1 if r_laundry else 0,
                "power_backup": 1 if r_backup else 0,
                "cctv_security": 1 if r_cctv else 0,
                "cleanliness_score": r_clean,
                "overall_rating": r_rate,
                "monthly_rent": int(r_asked) if r_asked > 0 else None
            }
            try:
                rent_eval = rent_engine.predict_fair_rent(features)
                st.markdown(f"""
                <div class="metric-card" style="border-left-color:#10B981; margin-top:1rem;">
                    <h3>Estimated Fair Monthly Rent: <span style="color:#059669;">₹{rent_eval['predicted_fair_rent']:,}</span></h3>
                    <p style="color:#4B5563; margin-bottom:6px;">
                        Recommended Fair Pricing Band: <b>₹{rent_eval['fair_range_min']:,} — ₹{rent_eval['fair_range_max']:,}</b>
                    </p>
                    <p><b>Market Assessment:</b> <span style="font-weight:700; color:#1E40AF;">{rent_eval['pricing_verdict']}</span></p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Rent predictor error: {e}")

    with warden_tab2:
        st.markdown("#### Automated Bipartite Room Allocation (Hungarian Algorithm)")
        st.caption("Solves linear sum assignment to pair an entire student intake into optimal 2-person rooms with minimum interpersonal friction.")

        students_csv_path = os.path.join(BASE_DIR, "data", "student_profiles.csv")
        if os.path.exists(students_csv_path):
            df_stud = pd.read_csv(students_csv_path)
            st.info(f"Loaded student applicant pool ({len(df_stud)} verified student profiles).")

            n_alloc = st.slider("Select batch size of students to allocate into rooms:", 4, 30, 10, 2)
            subset = df_stud.head(n_alloc).to_dict(orient="records")

            if st.button("⚡ Run Hungarian Bipartite Room Allocation", type="primary"):
                pairs = rm_engine.allocate_rooms_hungarian(subset)

                alloc_rows = []
                for p in pairs:
                    s1 = p["student_1"]["name"]
                    s2 = p["student_2"]["name"] if p["student_2"] else "Single Occupancy Room"
                    alloc_rows.append({
                        "Room": p["room_id"],
                        "Student 1": s1,
                        "Student 2": s2,
                        "Compatibility": f"{p['compatibility_score']}%",
                        "Friction Warnings": ", ".join(p["conflicts"])
                    })

                df_alloc = pd.DataFrame(alloc_rows)
                st.dataframe(df_alloc, use_container_width=True)

                avg_comp = np.mean([p["compatibility_score"] for p in pairs])
                st.success(f"Successfully allocated {n_alloc} students across {len(pairs)} rooms with an average compatibility score of {avg_comp:.1f}%!")
        else:
            st.warning("Please run data generator first to create student profiles.")
