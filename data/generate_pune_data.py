"""
SmartStay AI — Pune Dataset Generator
Synthesizes 1,500 realistic student lifestyle vectors and 300 Pune PG/Hostel listings.
Locations: Balewadi, Baner, Kothrud, Hinjawadi, Viman Nagar
Target Colleges: COEP, PICT, MIT-WPU, Cummins, VIT, Symbiosis
"""

import os
import random
import numpy as np
import pandas as pd

# Set deterministic seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

LOCALITIES = ["Balewadi", "Baner", "Kothrud", "Hinjawadi", "Viman Nagar"]

COLLEGES = [
    "Pune Institute of Computer Technology (PICT)",
    "College of Engineering Pune (COEP)",
    "MIT World Peace University (MIT-WPU)",
    "MKSSS Cummins College of Engineering",
    "Vishwakarma Institute of Technology (VIT)",
    "Symbiosis Institute of Technology (SIT)"
]

COLLEGE_LOCALITY_MAP = {
    "Pune Institute of Computer Technology (PICT)": "Kothrud",
    "College of Engineering Pune (COEP)": "Kothrud",
    "MIT World Peace University (MIT-WPU)": "Kothrud",
    "MKSSS Cummins College of Engineering": "Kothrud",
    "Vishwakarma Institute of Technology (VIT)": "Kothrud",
    "Symbiosis Institute of Technology (SIT)": "Viman Nagar"
}

FIRST_NAMES_MALE = [
    "Aarav", "Rohan", "Aditya", "Piyush", "Aryan", "Atharva", "Omkar",
    "Prathamesh", "Sahil", "Siddharth", "Tanmay", "Vedant", "Yash", "Kunal",
    "Harsh", "Dev", "Aniket", "Sanket", "Shubham", "Tejas", "Swapnil", "Gaurav"
]

FIRST_NAMES_FEMALE = [
    "Aanya", "Ananya", "Ishita", "Riya", "Sanika", "Tanvi", "Shruti",
    "Vaishnavi", "Sayali", "Sakshi", "Neha", "Pooja", "Mrunal", "Kavya",
    "Aditi", "Gauri", "Bhakti", "Sneha", "Radhika", "Anushka", "Meera", "Swara"
]

LAST_NAMES = [
    "Patil", "Deshmukh", "Kulkarni", "Joshi", "Shinde", "Pawar", "Bhosale",
    "Chavan", "More", "Gaikwad", "Jadhav", "Kadam", "Sawant", "Thorat",
    "Kale", "Wagh", "Mane", "Salunkhe", "Suryavanshi", "Bichitkar", "Kulkarni"
]

HOSTEL_PREFIXES = [
    "Shree", "Sai", "Balaji", "Om", "Green Valley", "Scholar's Den", "Elite",
    "Comfort Stay", "Shivneri", "Sunshine", "Royal Palms", "Nexus", "Stanza Living",
    "Zolo", "Youth Nest", "Campus Haven", "Silver Oak", "Metro Living", "Galaxy"
]

HOSTEL_SUFFIXES = [
    "Hostel", "PG for Students", "Residency", "Student Living", "Accommodations",
    "Boys Hostel", "Girls PG", "Executive Stay", "Scholars Hub"
]


def generate_hostel_data(n_samples: int = 300) -> pd.DataFrame:
    """Generates 300 Pune PG/Hostel records with realistic pricing and amenities."""
    records = []

    # Local base rents per locality (averages in INR for Double sharing)
    locality_base_rents = {
        "Balewadi": 8500,
        "Baner": 9500,
        "Kothrud": 8000,
        "Hinjawadi": 7500,
        "Viman Nagar": 10500
    }

    sharing_multipliers = {
        "Single": 1.70,
        "Double": 1.00,
        "Triple": 0.75,
        "Four": 0.60
    }

    for i in range(1, n_samples + 1):
        hostel_id = f"HST{i:03d}"
        locality = random.choice(LOCALITIES)
        gender_allowed = random.choices(["Boys", "Girls", "Co-ed"], weights=[0.45, 0.45, 0.10])[0]
        room_type = random.choices(["Single", "Double", "Triple", "Four"], weights=[0.20, 0.50, 0.20, 0.10])[0]

        prefix = random.choice(HOSTEL_PREFIXES)
        suffix = random.choice(HOSTEL_SUFFIXES)
        name = f"{prefix} {locality} {suffix} #{i}"

        # Distance to nearby educational hubs (km)
        distance_to_college_km = round(random.uniform(0.3, 10.5), 2)

        # Amenities
        wifi_speed_mbps = random.choice([30, 50, 100, 150, 200, 300])
        mess_included = random.choices([1, 0], weights=[0.65, 0.35])[0]
        mess_rating = round(random.uniform(2.8, 4.8), 1) if mess_included else 0.0
        ac_available = random.choices([1, 0], weights=[0.35, 0.65])[0]
        gym_available = random.choices([1, 0], weights=[0.25, 0.75])[0]
        laundry_available = random.choices([1, 0], weights=[0.70, 0.30])[0]
        power_backup = random.choices([1, 0], weights=[0.80, 0.20])[0]
        cctv_security = random.choices([1, 0], weights=[0.90, 0.10])[0]
        curfew_time = random.choices(["21:30", "22:00", "22:30", "23:00", "No Curfew"], weights=[0.2, 0.35, 0.25, 0.1, 0.1])[0]

        # Calculate a realistic base rent based on actual attributes + noise
        base = locality_base_rents[locality] * sharing_multipliers[room_type]
        amenity_bonus = (
            (wifi_speed_mbps / 50) * 150 +
            (mess_included * 2200) +
            (ac_available * 2000) +
            (gym_available * 600) +
            (laundry_available * 450) +
            (power_backup * 350)
        )
        distance_penalty = max(0, (distance_to_college_km - 2.0) * 180)
        noise = random.gauss(0, 350)
        monthly_rent = int(round((base + amenity_bonus - distance_penalty + noise) / 100.0) * 100)
        monthly_rent = max(4500, min(24000, monthly_rent))

        deposit_amount = int(monthly_rent * random.choice([1.5, 2.0, 2.5]))
        cleanliness_score = round(random.uniform(3.0, 5.0), 1)
        overall_rating = round(random.uniform(3.2, 4.9), 1)
        review_count = random.randint(12, 280)
        total_beds = random.choice([20, 30, 45, 60, 80, 120])
        available_beds = random.randint(1, max(2, total_beds // 4))

        records.append({
            "hostel_id": hostel_id,
            "name": name,
            "locality": locality,
            "gender_allowed": gender_allowed,
            "room_type": room_type,
            "distance_to_college_km": distance_to_college_km,
            "monthly_rent": monthly_rent,
            "deposit_amount": deposit_amount,
            "wifi_speed_mbps": wifi_speed_mbps,
            "mess_included": mess_included,
            "mess_rating": mess_rating,
            "ac_available": ac_available,
            "gym_available": gym_available,
            "laundry_available": laundry_available,
            "power_backup": power_backup,
            "cctv_security": cctv_security,
            "curfew_time": curfew_time,
            "cleanliness_score": cleanliness_score,
            "overall_rating": overall_rating,
            "review_count": review_count,
            "total_beds": total_beds,
            "available_beds": available_beds
        })

    return pd.DataFrame(records)


def generate_student_profiles(n_samples: int = 1500) -> pd.DataFrame:
    """Generates 1,500 realistic student lifestyle vectors with 4 distinct persona archetypes."""
    records = []

    # 4 Well-Separated Persona Archetypes to yield Silhouette Score > 0.60:
    # 0: "Scholar" - Early bird (0.15), silent solo (0.10), high neatness (0.90), low noise (0.15), low guest (0.15), non-smoker, non-drinker
    # 1: "NightOwl" - Late night (0.90), high noise (0.80), mod neatness (0.45), flexible study (0.50), high guest (0.75), smoker/drinker
    # 2: "Social" - Group study (0.90), high guest (0.90), high noise (0.85), mid-sleep (0.65), mod neatness (0.50), social habits
    # 3: "Balanced" - Steady routine: sleep (0.45), study (0.40), clean (0.75), noise (0.40), guest (0.40), clean habits
    personas = ["Scholar", "NightOwl", "Social", "Balanced"]

    for i in range(1, n_samples + 1):
        student_id = f"STU{i:04d}"
        gender = random.choices(["Male", "Female"], weights=[0.55, 0.45])[0]
        if gender == "Male":
            name = f"{random.choice(FIRST_NAMES_MALE)} {random.choice(LAST_NAMES)}"
        else:
            name = f"{random.choice(FIRST_NAMES_FEMALE)} {random.choice(LAST_NAMES)}"

        college = random.choice(COLLEGES)
        preferred_locality = random.choice(LOCALITIES)
        preferred_sharing = random.choices(["Single", "Double", "Triple"], weights=[0.25, 0.55, 0.20])[0]

        budget_min = random.choice([5000, 6000, 7000, 8000, 10000])
        budget_max = budget_min + random.choice([3000, 4000, 5000, 7000])

        persona = random.choice(personas)

        if persona == "Scholar":
            sleep_schedule = np.clip(np.random.normal(0.12, 0.035), 0.02, 0.22)
            study_habit = np.clip(np.random.normal(0.10, 0.035), 0.02, 0.20)
            cleanliness = np.clip(np.random.normal(0.92, 0.030), 0.82, 1.0)
            noise_tolerance = np.clip(np.random.normal(0.15, 0.035), 0.05, 0.25)
            guest_tolerance = np.clip(np.random.normal(0.15, 0.035), 0.05, 0.25)
            smoking_habit = 0
            drinking_habit = 0
            dietary_habit = random.choices(["Veg", "Jain"], weights=[0.75, 0.25])[0]
        elif persona == "NightOwl":
            sleep_schedule = np.clip(np.random.normal(0.90, 0.035), 0.78, 1.0)
            study_habit = np.clip(np.random.normal(0.55, 0.040), 0.45, 0.65)
            cleanliness = np.clip(np.random.normal(0.40, 0.040), 0.28, 0.52)
            noise_tolerance = np.clip(np.random.normal(0.80, 0.040), 0.70, 0.95)
            guest_tolerance = np.clip(np.random.normal(0.75, 0.040), 0.65, 0.90)
            smoking_habit = 1
            drinking_habit = 1
            dietary_habit = "Non-Veg"
        elif persona == "Social":
            sleep_schedule = np.clip(np.random.normal(0.65, 0.035), 0.55, 0.75)
            study_habit = np.clip(np.random.normal(0.90, 0.030), 0.80, 1.0)
            cleanliness = np.clip(np.random.normal(0.55, 0.040), 0.42, 0.68)
            noise_tolerance = np.clip(np.random.normal(0.90, 0.030), 0.80, 1.0)
            guest_tolerance = np.clip(np.random.normal(0.92, 0.030), 0.82, 1.0)
            smoking_habit = 0
            drinking_habit = 1
            dietary_habit = random.choices(["Veg", "Non-Veg"], weights=[0.35, 0.65])[0]
        else:  # Balanced
            sleep_schedule = np.clip(np.random.normal(0.40, 0.035), 0.30, 0.50)
            study_habit = np.clip(np.random.normal(0.35, 0.035), 0.25, 0.45)
            cleanliness = np.clip(np.random.normal(0.75, 0.035), 0.65, 0.85)
            noise_tolerance = np.clip(np.random.normal(0.40, 0.035), 0.30, 0.50)
            guest_tolerance = np.clip(np.random.normal(0.42, 0.035), 0.30, 0.52)
            smoking_habit = 0
            drinking_habit = 0
            dietary_habit = random.choices(["Veg", "Non-Veg"], weights=[0.60, 0.40])[0]

        records.append({
            "student_id": student_id,
            "name": name,
            "gender": gender,
            "college": college,
            "preferred_locality": preferred_locality,
            "budget_min": budget_min,
            "budget_max": budget_max,
            "preferred_sharing": preferred_sharing,
            "persona_archetype": persona,
            "sleep_schedule": round(float(sleep_schedule), 3),
            "study_habit": round(float(study_habit), 3),
            "cleanliness": round(float(cleanliness), 3),
            "noise_tolerance": round(float(noise_tolerance), 3),
            "guest_tolerance": round(float(guest_tolerance), 3),
            "smoking_habit": int(smoking_habit),
            "drinking_habit": int(drinking_habit),
            "dietary_habit": dietary_habit,
            "ac_required": random.choices([1, 0], weights=[0.4, 0.6])[0]
        })

    return pd.DataFrame(records)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Generating Pune PG/Hostel Listings (300 records)...")
    df_hostels = generate_hostel_data(300)
    hostels_path = os.path.join(DATA_DIR, "hostels_pune.csv")
    df_hostels.to_csv(hostels_path, index=False)
    print(f"Saved: {hostels_path} ({len(df_hostels)} rows)")

    print("Generating Student Profiles (1,500 records)...")
    df_students = generate_student_profiles(1500)
    students_path = os.path.join(DATA_DIR, "student_profiles.csv")
    df_students.to_csv(students_path, index=False)
    print(f"Saved: {students_path} ({len(df_students)} rows)")
    print("Dataset generation complete!")


if __name__ == "__main__":
    main()
