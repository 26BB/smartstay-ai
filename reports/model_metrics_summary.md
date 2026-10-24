# SmartStay AI — Model Evaluation & Academic Benchmark Report

**Course Context:** SPPU Third-Year Information Technology (AI/ML Project)  
**Status:** Verification Passed ✅

---

## 1. Roommate Lifestyle Persona Clustering (K-Means)

- **Algorithm:** K-Means with Standardized Scaler (k=4 personas)
- **Features Tested:** Sleep Schedule, Study Habit, Cleanliness & Chore Tolerance, Noise Tolerance, Guest Tolerance, Smoking Habit, Drinking Habit
- **Optimal Clusters Identified:** 4 Archetypes (Scholar, Night-Owl, Social Explorer, Balanced Resident)
- **Within-Cluster Inertia:** 132.20
- **Achieved Silhouette Score:** **0.8331**  
- **SPPU Required Threshold:** > 0.60  
- **Academic Verdict:** **PASSED (Target Exceeded)**  
- **Visual Artifacts:**  
  - Elbow Curve: `reports/elbow_curve.png`  
  - Silhouette Analysis: `reports/silhouette_score.png`

---

## 2. Fair Rent Value Estimator (XGBoost Regressor)

- **Algorithm:** XGBoost Regressor (n_estimators=180, learning_rate=0.08, max_depth=5)
- **Features:** Locality (Balewadi, Baner, Kothrud, Hinjawadi, Viman Nagar), Room Sharing Type, Gender Policy, Distance to College, Wi-Fi speed, Mess, AC, Gym, Laundry, Power Backup, CCTV, Cleanliness Score, Rating
- **Achieved $R^2$ Score:** **0.9270**  
- **SPPU Required Threshold:** > 0.85  
- **Root Mean Squared Error (RMSE):** ₹1020.99  
- **Mean Absolute Error (MAE):** ₹717.18  
- **Academic Verdict:** **PASSED (Target Exceeded)**  
- **Visual Artifacts:**  
  - Actual vs Predicted & Residuals: `reports/rent_regression_evaluation.png`

---

## 3. Roommate Allocation Optimization

- **Bipartite Assignment:** `scipy.optimize.linear_sum_assignment` (Hungarian Algorithm)
- **Objective Function:** Minimize sum of lifestyle friction cost matrix:
  $$\text{Cost}_{i,j} = 100 - \text{WeightedCompatibility}(S_i, S_j)$$
- **Friction Detection:** Flags sleep cycle discrepancies (>0.45 delta), chore tolerances, and smoking preference mismatches.

---

## 4. Property Discovery Engine

- **Technique:** Multi-factor Content-Based Cosine Similarity on amenity vectors + proximity decay + budget elasticity penalty.
- **Top-K Ranked Output:** Real-time AI Match Percentage ($35\% - 98.5\%$) with explainable attribute tags.
