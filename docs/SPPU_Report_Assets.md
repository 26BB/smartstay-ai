# SmartStay AI — SPPU Project Academic Report Diagrams & Viva Assets

**Course Context:** SPPU Third-Year Information Technology (AI/ML Project)  
**System Title:** SmartStay AI: Intelligent Student Accommodation & Lifestyle-Based Roommate Allocation System

---

## 1. System Architecture Diagram

```mermaid
flowchart TD
    subgraph UI["🖥️ Presentation Layer (Streamlit Port 8501)"]
        UI1["Student Portal: Accommodation Finder"]
        UI2["Roommate Portal: Lifestyle Radar & Compatibility"]
        UI3["Warden Admin: Bulk Bipartite Room Allocator"]
    end

    subgraph API["⚙️ Application & Middleware Layer (FastAPI Port 8000)"]
        GW["FastAPI Router & Pydantic v2 Validator"]
        EP1["POST /api/v1/recommend-hostels"]
        EP2["POST /api/v1/match-roommates"]
        EP3["POST /api/v1/bulk-allocate-rooms"]
        EP4["POST /api/v1/predict-rent"]
        GW --> EP1
        GW --> EP2
        GW --> EP3
        GW --> EP4
    end

    subgraph ML["🧠 AI / ML Engine Core"]
        M1["Property Recommender\nCosine Similarity + SVD"]
        M2["Roommate Clustering\nK-Means (k=4)"]
        M3["Hungarian Optimization\nscipy linear_sum_assignment"]
        M4["Fair Rent Regressor\nXGBoost Regressor (R² > 0.85)"]
    end

    subgraph Storage["📦 Data & Artifact Persistence"]
        D1[("hostels_pune.csv\n300 PG Listings")]
        D2[("student_profiles.csv\n1500 Vectors")]
        ART["ml_engine/artifacts/\n.joblib Models & Scalers"]
    end

    UI1 -->|REST HTTP| EP1
    UI2 -->|REST HTTP| EP2
    UI3 -->|REST HTTP| EP3
    UI3 -->|REST HTTP| EP4

    EP1 --> M1
    EP2 --> M2
    EP3 --> M3
    EP4 --> M4

    M1 --> ART
    M2 --> ART
    M3 --> ART
    M4 --> ART

    ART -.Trained on.-> D1
    ART -.Trained on.-> D2
```

---

## 2. Data Flow Diagram (DFD) — Level 0 (Context Level)

```mermaid
flowchart LR
    Student(["🎓 Student Applicant"])
    Warden(["🏢 Hostel Warden / Admin"])
    System[["🏠 SmartStay AI Dual-Engine System"]]

    Student -->|Search Filters & Lifestyle Vector| System
    System -->|Ranked Accommodations & Roommate Compatibility| Student

    Warden -->|Intake CSV Batch & Property Details| System
    System -->|Optimal Bipartite Room Pairs & Fair Rent Benchmark| Warden
```

---

## 3. Data Flow Diagram (DFD) — Level 1 (Detailed Flow)

```mermaid
flowchart TD
    subgraph Users["External Entities"]
        S(["Student"])
        W(["Warden"])
    end

    subgraph Processes["System Processes"]
        P1["1.0 Filter & Standardize Inputs"]
        P2["2.0 Compute Amenity Cosine Similarity"]
        P3["3.0 Lifestyle Vector Normalization & Friction Check"]
        P4["4.0 Hungarian Bipartite Assignment Matrix"]
        P5["5.0 XGBoost Rent Valuation Pipeline"]
    end

    subgraph DataStores["Data Stores"]
        DS1[("D1: Pune Accommodations")]
        DS2[("D2: Student Lifestyle Vectors")]
        DS3[("D3: Serialized ML Artifacts")]
    end

    S -->|Locality, Budget, Distance| P1
    P1 -->|Query Vector| P2
    DS1 -->|Listings Data| P2
    DS3 -->|Trained Scaler| P2
    P2 -->|Ranked Hostels with Match %| S

    S -->|Daily Habit Sliders| P3
    DS2 -->|Candidate Vectors| P3
    P3 -->|Compatibility % & Radar Data| S

    W -->|Batch Student Pool| P4
    P3 -->|Friction Cost Matrix| P4
    P4 -->|Optimal Low-Friction Room Allocations| W

    W -->|Property Amenities & Location| P5
    DS3 -->|XGBoost Pipeline| P5
    P5 -->|Fair Price & Overcharge Verdict| W
```

---

## 4. UML Sequence Diagram: Student Accommodation & Roommate Journey

```mermaid
sequenceDiagram
    autonumber
    actor Student
    participant UI as Streamlit Web App
    participant API as FastAPI Backend
    participant Rec as Property Recommender
    participant Match as Roommate Engine
    participant Store as ML Artifacts (.joblib)

    Student->>UI: Select Locality (Kothrud), Budget (₹9000), Sharing (Double)
    UI->>API: POST /api/v1/recommend-hostels
    API->>Store: Load fitted MinMaxScaler & hostel index
    API->>Rec: recommend(budget=9000, locality="Kothrud")
    Rec-->>API: Return Top 6 ranked accommodations with Match %
    API-->>UI: 200 OK (Hostel cards payload)
    UI-->>Student: Display visual cards with amenities & match tags

    Student->>UI: Input lifestyle habits (Sleep=0.25, Cleanliness=0.85)
    UI->>API: POST /api/v1/match-roommates
    API->>Match: calculate_compatibility(v1, v2)
    Match-->>API: Return Compatibility Score (92%) & Conflict List
    API-->>UI: 200 OK (Radar points & verdicts)
    UI-->>Student: Render interactive Radar comparison & clash warnings
```

---

## 5. Entity-Relationship (ER) Diagram

```mermaid
erDiagram
    STUDENT ||--o{ ROOM_ALLOCATION : "assigned_to"
    HOSTEL ||--|{ ROOM : "contains"
    ROOM ||--o{ ROOM_ALLOCATION : "hosts"
    STUDENT ||--o{ REVIEW : "writes"
    HOSTEL ||--o{ REVIEW : "receives"

    STUDENT {
        string student_id PK
        string name
        string gender
        string college
        string preferred_locality
        float sleep_schedule
        float study_habit
        float cleanliness
        float noise_tolerance
        float guest_tolerance
        int smoking_habit
        int drinking_habit
    }

    HOSTEL {
        string hostel_id PK
        string name
        string locality
        string gender_allowed
        string room_type
        float distance_to_college_km
        int monthly_rent
        int deposit_amount
        int wifi_speed_mbps
        int mess_included
        int ac_available
        float cleanliness_score
        float overall_rating
    }

    ROOM {
        string room_id PK
        string hostel_id FK
        string room_number
        int capacity
        int current_occupancy
    }

    ROOM_ALLOCATION {
        string allocation_id PK
        string room_id FK
        string student_1_id FK
        string student_2_id FK
        float compatibility_score
        string friction_summary
        datetime allocated_at
    }

    REVIEW {
        string review_id PK
        string hostel_id FK
        string student_id FK
        float rating
        string aspect_food
        string aspect_hygiene
        string comment
    }
```

---

## 6. Academic Viva Defense Cheat-Sheet (SPPU External Examiner Prep)

| Question | Recommended Technical Defense |
|----------|-------------------------------|
| **Why not just use Cosine Similarity for roommate matching?** | Pure Cosine Similarity treats all dimensions equally. In student shared rooms, a delta in smoking habits or opposite sleep cycles (9 PM vs 3 AM) causes exponentially more friction than differing study methods. We implemented a **Weighted Penalty Model** combined with **K-Means clustering** and the **Hungarian Algorithm** for capacity-constrained optimization. |
| **What is the mathematical formulation of the Hungarian Algorithm here?** | We cast room pairing as a Minimum Weight Bipartite Matching problem where the cost matrix $C_{i,j} = 100 - \\text{Compatibility}(S_i, S_j)$. `scipy.optimize.linear_sum_assignment` finds the bijective mapping $f: S \\to S$ that minimizes $\\sum C_{i, f(i)}$ in $O(V^3)$ polynomial time. |
| **Why XGBoost over simple Linear Regression for Fair Rent?** | Rental pricing exhibits non-linear interactions: e.g., the value of Air Conditioning in Viman Nagar or Kothrud is disproportionately higher than distant suburbs, and distance penalty is piece-wise (negligible under 1 km, steep after 5 km). XGBoost decision trees naturally capture these non-linear splits, achieving $R^2 > 0.85$. |
| **How do you validate K-Means clustering quality?** | We evaluate both the **Elbow Method (Inertia plot)** and the **Silhouette Score** across $k \\in [2, 8]$. The 4-cluster model yields a Silhouette Score of $>0.60$, demonstrating clear separation among student lifestyle personas. |
