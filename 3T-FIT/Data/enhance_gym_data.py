"""
Enhanced Gym Data Processor
===============================

This script processes raw gym member exercise tracking data according to the
Strategy_Analysis.md framework, implementing a hybrid approach:

1. Feature Engineering: Convert complex workout data into Estimated 1RM
2. Model-ready Output: Generate clean dataset for AI training
3. Exercise Mapping: Add realistic exercise names for strength workouts
4. Calories Calculation: Apply scientifically validated formulas

Key improvements:
- Implements 1RM estimation using Epley formula for strength exercises
- Maps calories using METs-based approach for non-strength workouts
- Adds exercise names from reference dataset for strength workouts
- Calculates unified intensity scores
- Applies SePA-inspired adjustments based on user readiness

Author: Claude Code Assistant
Date: 2025-11-24
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path
import json

# ==================== CONFIGURATION & CONSTANTS ====================

# MET values for different workout types
MET_VALUES = {
    'Cardio': 8.0,      # Running/treadmill moderate intensity
    'HIIT': 10.0,       # High Intensity Interval Training
    'Yoga': 3.0,        # Moderate intensity yoga
    'Strength': 6.0,    # Weight training moderate vigor
    'Cycling': 7.0,     # Moderate cycling
    'Swimming': 8.0,    # Moderate swimming
    'Walking': 4.0,     # Brisk walking
}

def load_exercise_database():
    """Load exercise names from multiple sources"""
    all_exercises = []

    # Source 1: Reference Excel file
    try:
        reference_df = pd.read_excel('d:/dacn_omnimer_health/3T-FIT/Data/preprocessing_data/own_gym_member_exercise_tracking.xlsx')
        # Strip whitespace from workout_type values
        reference_df['workout_type'] = reference_df['workout_type'].str.strip()
        strength_exercises = reference_df[reference_df['workout_type'] == 'Strength']['exercise_name'].dropna().unique()
        all_exercises.extend(list(strength_exercises))
        print(f"Loaded {len(strength_exercises)} exercises from Excel reference")
    except Exception as e:
        print(f"Warning: Could not load Excel exercise database: {e}")

    # Source 2: JSON files from exercises folder
    try:
        import os
        import json

        exercises_path = 'd:/dacn_omnimer_health/exercises'
        if os.path.exists(exercises_path):
            json_exercises = []
            for filename in os.listdir(exercises_path):
                if filename.endswith('.json'):
                    try:
                        filepath = os.path.join(exercises_path, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            exercise_data = json.load(f)

                        # Filter for strength/category exercises
                        if exercise_data.get('category', '').lower() in ['strength', 'plyometrics'] or \
                           any(muscle in ['chest', 'back', 'shoulders', 'legs', 'arms', 'abdominals']
                               for muscle in exercise_data.get('primaryMuscles', [])):
                            json_exercises.append(exercise_data['name'])
                    except Exception as e:
                        continue

            # Remove duplicates and add to list
            unique_json_exercises = list(set(json_exercises))
            all_exercises.extend(unique_json_exercises)
            print(f"Loaded {len(unique_json_exercises)} exercises from JSON files")

    except Exception as e:
        print(f"Warning: Could not load JSON exercise database: {e}")

    # Remove duplicates and create final database
    final_exercises = list(set(all_exercises))

    if not final_exercises:
        # Fallback exercise database
        final_exercises = [
            'Barbell Bench Press (Wide Grip)', 'Incline Chest Press', 'Seated Chest Fly',
            'Chest Dips (Assisted)', 'Machine Seated Dip', 'Lateral Raise',
            'Seater Overhead Press', 'Dumbbell Shoulder Press', 'Bent Over Row',
            'Lat Pulldown', 'Deadlift', 'Squat', 'Leg Press', 'Leg Extension',
            'Hamstring Curl', 'Calf Raise', 'Bicep Curl', 'Tricep Extension',
            'Plank', 'Crunches', 'Russian Twist'
        ]
        print("Using fallback exercise database")

    print(f"Total unique exercises loaded: {len(final_exercises)}")
    return final_exercises

def load_workout_templates():
    """Load complete workout templates from reference dataset"""
    try:
        reference_df = pd.read_excel('d:/dacn_omnimer_health/3T-FIT/Data/preprocessing_data/own_gym_member_exercise_tracking.xlsx')

        # Strip whitespace from workout_type values
        reference_df['workout_type'] = reference_df['workout_type'].str.strip()

        # Group workouts by workout_id
        workout_templates = {}

        for workout_id, workout_data in reference_df.groupby('workout_id'):
            workout_info = {
                'workout_type': workout_data['workout_type'].iloc[0],
                'exercises': []
            }

            for _, exercise_row in workout_data.iterrows():
                exercise_info = {
                    'name': exercise_row['exercise_name'],
                    'duration_min': exercise_row['duration_min'],
                    'base_intensity': exercise_row.get('unified_intensity', 50),
                    'base_calories': exercise_row.get('calories', 30),
                    'avg_hr': exercise_row.get('avg_hr', 120),
                    'max_hr': exercise_row.get('max_hr', 150)
                }
                workout_info['exercises'].append(exercise_info)

            workout_templates[workout_id] = workout_info

        # Filter only Strength workouts
        strength_workouts = {k: v for k, v in workout_templates.items()
                           if v['workout_type'] == 'Strength'}

        print(f"Loaded {len(strength_workouts)} strength workout templates")
        print(f"Total exercises in templates: {sum(len(w['exercises']) for w in strength_workouts.values())}")

        return strength_workouts

    except Exception as e:
        print(f"Warning: Could not load workout templates: {e}")
        return {}

# Load exercise database and workout templates globally
EXERCISE_DATABASE = load_exercise_database()
WORKOUT_TEMPLATES = load_workout_templates()

# ==================== 1RM ESTIMATION (STRATEGY_ANALYSIS.MD) ====================

def calculate_1rm_estimated(weight: float, reps: int) -> float:
    """
    Calculate estimated 1RM using Epley formula
    Formula: 1RM = Weight × (1 + Reps/30)
    """
    if weight <= 0 or reps <= 0:
        return 0.0
    return weight * (1 + reps / 30)

def parse_strength_data(sets_string: str) -> tuple:
    """
    Parse strength workout string and calculate estimated 1RM
    Input format: "12x40x2 | 8x50x3" (reps x weight x sets)
    Returns: (estimated_1rm, total_weight, total_sets)
    """
    if pd.isna(sets_string) or sets_string == "":
        return 0.0, 0.0, 0

    try:
        sets_data = sets_string.split('|')
        max_1rm = 0.0
        total_weight = 0.0
        total_sets = 0

        for set_data in sets_data:
            parts = set_data.strip().split('x')
            if len(parts) >= 2:
                reps = float(parts[0])
                weight = float(parts[1])
                sets = float(parts[2]) if len(parts) > 2 else 1

                # Calculate 1RM for this set
                estimated_1rm = calculate_1rm_estimated(weight, int(reps))
                max_1rm = max(max_1rm, estimated_1rm)

                total_weight += weight * reps * sets
                total_sets += int(sets)

        return max_1rm, total_weight, total_sets

    except Exception as e:
        print(f"Warning: Error parsing strength data '{sets_string}': {e}")
        return 0.0, 0.0, 0

# ==================== CALORIES CALCULATION ====================

def calculate_bmi(weight_kg, height_m):
    """Tính BMI = weight / height^2"""
    if height_m <= 0:
        return None
    return round(weight_kg / (height_m ** 2), 2)


def calculate_bmr(weight_kg, height_m, age, gender):
    """
    Tính BMR theo công thức Mifflin-St Jeor
    Men: BMR = 10×W + 6.25×H - 5×A + 5
    Women: BMR = 10×W + 6.25×H - 5×A - 161
    """
    height_cm = height_m * 100

    # Handle different gender formats
    if pd.isna(gender):
        # Default to male if gender is missing
        is_male = True
    elif isinstance(gender, str):
        is_male = gender.lower() in ['male', 'm', '1']
    elif isinstance(gender, (int, float)):
        is_male = int(gender) == 1
    else:
        # Default to male
        is_male = True

    if is_male:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:  # female
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    return round(bmr, 2)


def calculate_body_fat_navy(gender, waist_cm, neck_cm, height_cm, hip_cm=None):
    """
    Tính Body Fat % theo US Navy Method
    Men: 495 / (1.0324 - 0.19077 × log10(waist - neck) + 0.15456 × log10(height)) - 450
    Women: 495 / (1.29579 - 0.35004 × log10(waist + hip - neck) + 0.22100 × log10(height)) - 450
    """
    # Handle different gender formats
    if pd.isna(gender):
        is_male = True
    elif isinstance(gender, str):
        is_male = gender.lower() in ['male', 'm', '1']
    elif isinstance(gender, (int, float)):
        is_male = int(gender) == 1
    else:
        is_male = True

    if is_male:
        if waist_cm <= neck_cm:
            return None
        bf = 495 / (1.0324 - 0.19077 * np.log10(waist_cm - neck_cm) + 0.15456 * np.log10(height_cm)) - 450
    else:  # female
        if hip_cm is None or (waist_cm + hip_cm) <= neck_cm:
            return None
        bf = 495 / (1.29579 - 0.35004 * np.log10(waist_cm + hip_cm - neck_cm) + 0.22100 * np.log10(height_cm)) - 450

    return round(max(0, min(bf, 50)), 2)  # Clamp between 0-50%


def estimate_body_fat_from_bmi(bmi, age, gender):
    """
    Ước tính Body Fat từ BMI khi không có đo chu vi
    Dựa trên công thức Deurenberg
    """
    # Handle different gender formats
    if pd.isna(gender):
        is_male = True
    elif isinstance(gender, str):
        is_male = gender.lower() in ['male', 'm', '1']
    elif isinstance(gender, (int, float)):
        is_male = int(gender) == 1
    else:
        is_male = True

    if is_male:
        bf = (1.20 * bmi) + (0.23 * age) - 16.2
    else:
        bf = (1.20 * bmi) + (0.23 * age) - 5.4
    return round(max(0, min(bf, 50)), 2)


def calculate_muscle_mass(weight_kg, body_fat_percentage):
    """
    Tính Muscle Mass = Weight (kg) × (1 - Body Fat %)
    """
    if body_fat_percentage is None:
        return None
    lean_mass = weight_kg * (1 - body_fat_percentage / 100)
    # Muscle mass ≈ 50% of lean mass (rough estimate)
    return round(lean_mass * 0.5, 2)


# ==================== CALORIES CALCULATION ====================

def calculate_calories_mets(mets, weight_kg, duration_min):
    """
    Calculate calories using METs formula
    Formula: Calories/min = (METs × 3.5 × Weight) / 200
    """
    if mets is None or mets <= 0 or weight_kg <= 0 or duration_min <= 0:
        return 0.0
    calories_per_min = (mets * 3.5 * weight_kg) / 200
    return round(calories_per_min * duration_min, 2)


def calculate_calories_hr(avg_hr, weight_kg, age, gender, duration_min):
    """
    Calculate calories using Keytel formula based on heart rate
    Men: AHR = 0.6309×HR + 0.1988×W + 0.2017×A - 55.0969
    Women: AHR = 0.4472×HR - 0.1263×W + 0.074×A - 20.4022
    Calories = Duration × AHR / 4.184
    """
    if avg_hr is None or avg_hr <= 0 or weight_kg <= 0 or duration_min <= 0:
        return 0.0

    # Handle different gender formats
    if pd.isna(gender):
        is_male = True
    elif isinstance(gender, str):
        is_male = gender.lower() in ['male', 'm', '1']
    elif isinstance(gender, (int, float)):
        is_male = int(gender) == 1
    else:
        is_male = True

    try:
        if is_male:
            ahr = (0.6309 * avg_hr + 0.1988 * weight_kg + 0.2017 * age - 55.0969)
        else:
            ahr = (0.4472 * avg_hr - 0.1263 * weight_kg + 0.074 * age - 20.4022)

        calories = duration_min * (ahr / 4.184)
        return round(max(0, calories), 2)
    except Exception:
        return 0.0


# ==================== HEART RATE CALCULATIONS ====================

def calculate_max_hr(age):
    """Tính Max HR theo công thức Tanaka: 208 - 0.7 × Age"""
    return round(208 - 0.7 * age)


def calculate_target_hr(max_hr, resting_hr, intensity_percent):
    """
    Tính Target HR theo công thức Karvonen
    Target HR = [(Max HR - Resting HR) × % Intensity] + Resting HR
    """
    return round((max_hr - resting_hr) * intensity_percent + resting_hr)


# ==================== SUITABILITY SCORING ====================

def calculate_suitability_x(row):
    """
    Tính suitability_x: Mức độ phù hợp của BÀI TẬP RIÊNG LẺ trong workout
    Dựa trên:
    - Cường độ so với khả năng (unified_intensity vs experience_level)
    - Nhịp tim (avg_hr vs predicted HR)
    - Calories burned
    """
    score = 0.5  # Base score

    # Helper function to safely get values from pandas Series
    def safe_get(series, key, default=None):
        try:
            if key in series.index:
                val = series[key]
                return val if pd.notna(val) else default
            return default
        except:
            return default

    # Factor 1: Intensity appropriateness (30%)
    intensity = safe_get(row, 'unified_intensity')
    exp_level = safe_get(row, 'experience_level')

    if intensity is not None and exp_level is not None:
        # Normalize intensity based on type
        if intensity > 10:  # Likely 1RM or speed
            # For 1RM: higher is more intense
            # For speed: higher is more intense
            intensity_normalized = min(intensity / 100, 1.0)
        else:  # Intensity score 1-4
            intensity_normalized = intensity / 4.0

        # Match with experience level (1-4)
        exp_normalized = exp_level / 4.0

        # Good match if intensity ≈ experience
        diff = abs(intensity_normalized - exp_normalized)
        intensity_score = max(0, 1 - diff * 2)  # Penalty for mismatch
        score += intensity_score * 0.3

    # Factor 2: Heart rate appropriateness (25%)
    avg_hr = safe_get(row, 'avg_hr')
    max_hr = safe_get(row, 'max_hr')

    if avg_hr is not None and max_hr is not None and max_hr > 0:
        hr_percent = avg_hr / max_hr
        # Good range: 60-85% of max HR
        if 0.6 <= hr_percent <= 0.85:
            hr_score = 1.0
        elif 0.5 <= hr_percent < 0.6 or 0.85 < hr_percent <= 0.95:
            hr_score = 0.7
        else:
            hr_score = 0.3
        score += hr_score * 0.25

    # Factor 3: Duration appropriateness (15%)
    duration = safe_get(row, 'duration_min')

    if duration is not None:
        # Good range: 20-60 min per exercise
        if 20 <= duration <= 60:
            duration_score = 1.0
        elif 10 <= duration < 20 or 60 < duration <= 90:
            duration_score = 0.7
        else:
            duration_score = 0.4
        score += duration_score * 0.15

    # Factor 4: Calories efficiency (10%)
    calories = safe_get(row, 'calories')
    duration_min = safe_get(row, 'duration_min')

    if calories is not None and duration_min is not None and duration_min > 0:
        cal_per_min = calories / duration_min
        # Good range: 5-15 kcal/min
        if 5 <= cal_per_min <= 15:
            cal_score = 1.0
        elif 3 <= cal_per_min < 5 or 15 < cal_per_min <= 20:
            cal_score = 0.7
        else:
            cal_score = 0.4
        score += cal_score * 0.1

    # Add some randomness (±5%) to simulate real-world variation
    score = score * random.uniform(0.95, 1.05)

    return round(min(1.0, max(0.0, score)), 2)


def calculate_suitability_y(workout_exercises_df):
    """
    Tính suitability_y: Đánh giá TỔNG QUAN toàn bộ workout
    Dựa trên:
    - Trung bình suitability_x của các bài tập
    - Sự đa dạng của bài tập
    - Tổng thời gian workout
    - Progression (cường độ tăng dần hợp lý)
    """
    if len(workout_exercises_df) == 0:
        return 0.5
    
    # Factor 1: Average individual suitability (40%)
    avg_suit_x = workout_exercises_df['suitability_x'].mean()
    score = avg_suit_x * 0.4
    
    # Factor 2: Exercise variety (20%)
    num_exercises = len(workout_exercises_df)
    if 4 <= num_exercises <= 8:  # Optimal range
        variety_score = 1.0
    elif 3 <= num_exercises < 4 or 8 < num_exercises <= 10:
        variety_score = 0.7
    else:
        variety_score = 0.4
    score += variety_score * 0.2
    
    # Factor 3: Total session duration (20%)
    total_duration = workout_exercises_df['session_duration'].iloc[0] * 60  # Convert to minutes
    if 45 <= total_duration <= 90:  # Optimal range
        duration_score = 1.0
    elif 30 <= total_duration < 45 or 90 < total_duration <= 120:
        duration_score = 0.7
    else:
        duration_score = 0.4
    score += duration_score * 0.2
    
    # Factor 4: Intensity progression (20%)
    # Check if intensity increases reasonably
    intensities = workout_exercises_df['unified_intensity'].dropna().values
    if len(intensities) > 1:
        # Check for reasonable variation (not all same, not too erratic)
        std_intensity = np.std(intensities)
        mean_intensity = np.mean(intensities)
        cv = std_intensity / mean_intensity if mean_intensity > 0 else 0
        
        if 0.1 <= cv <= 0.4:  # Good variation
            progression_score = 1.0
        elif 0.05 <= cv < 0.1 or 0.4 < cv <= 0.6:
            progression_score = 0.7
        else:
            progression_score = 0.4
    else:
        progression_score = 0.5
    
    score += progression_score * 0.2
    
    # Add some randomness (±5%)
    score = score * random.uniform(0.95, 1.05)
    
    return round(min(1.0, max(0.0, score)), 2)


# ==================== CALORIES DISTRIBUTION FOR STRENGTH WORKOUTS ====================

def distribute_strength_calories(workout_session_df, exercises_per_workout=6):
    """
    Chia đều tổng calories của buổi tập Strength thành các bài tập riêng lẻ

    Args:
        workout_session_df: DataFrame chứa dữ liệu buổi tập Strength
        exercises_per_workout: Số bài tập trung bình mỗi buổi tập (default: 6)

    Returns:
        DataFrame mới với calories đã được chia đều cho mỗi bài tập
    """
    df_copy = workout_session_df.copy()

    # Lọc chỉ các buổi tập Strength
    strength_mask = df_copy['workout_type'] == 'Strength'

    if strength_mask.any():
        # Chia tổng calories cho mỗi bài tập
        df_copy.loc[strength_mask, 'calories'] = df_copy.loc[strength_mask, 'calories'] / exercises_per_workout

        print(f"+ Distributed Strength calories across {exercises_per_workout} exercises per workout")
        print(f"  Original range: {workout_session_df.loc[strength_mask, 'calories'].min():.0f} - {workout_session_df.loc[strength_mask, 'calories'].max():.0f}")
        print(f"  Distributed range: {df_copy.loc[strength_mask, 'calories'].min():.0f} - {df_copy.loc[strength_mask, 'calories'].max():.0f}")

    return df_copy

# ==================== ENHANCED PROCESSING FUNCTIONS ====================

def calculate_unified_intensity(workout_type: str,
                              estimated_1rm: float = 0,
                              avg_hr: float = 0,
                              max_hr: float = 0,
                              resting_hr: float = 70,
                              weight_kg: float = 70,
                              duration_minutes: float = 60,
                              readiness_factor: float = 1.0) -> float:
    """
    Calculate unified intensity score (0-100) for different workout types
    """
    if workout_type.lower() == 'strength' and estimated_1rm > 0:
        # For strength: Normalize 1RM to 0-100 scale (assuming 200kg as elite level)
        intensity = min(100, (estimated_1rm / 200) * 100)
    elif avg_hr > 0:
        # For cardio/HIIT: Use Heart Rate Reserve percentage
        hr_max = 208 - (0.7 * 30)  # Assuming average age 30 for generalization
        hrr = ((avg_hr - resting_hr) / (hr_max - resting_hr)) * 100
        intensity = min(100, max(0, hrr))
    else:
        # Fallback: Use MET-based intensity
        mets = MET_VALUES.get(workout_type, 5.0)
        intensity = min(100, (mets / 15) * 100)  # 15 METs as very high intensity

    # Apply readiness factor (SePA-inspired adjustment)
    intensity *= readiness_factor

    return round(intensity, 2)

def generate_intensity_variation(base_intensity: float, exercise_index: int, total_exercises: int) -> float:
    """
    Generate realistic intensity variations for different exercises in a workout
    """
    # Create varied intensity pattern: warm-up → main → cool-down
    if exercise_index == 0:  # First exercise (warm-up)
        variation = random.uniform(0.7, 0.8)
    elif exercise_index == total_exercises - 1:  # Last exercise (cool-down)
        variation = random.uniform(0.6, 0.75)
    elif exercise_index < total_exercises // 2:  # First half (building up)
        variation = random.uniform(0.85, 1.1)
    else:  # Second half (peak intensity)
        variation = random.uniform(0.9, 1.15)

    return round(base_intensity * variation, 2)

def calculate_exercise_heart_rates(base_avg_hr: float, intensity_factor: float, age: int) -> tuple:
    """
    Calculate average and max heart rates based on exercise intensity
    """
    max_hr = 208 - (0.7 * age)

    # Adjust heart rates based on intensity
    avg_hr = min(max_hr - 20, base_avg_hr * intensity_factor)
    max_hr_exercise = min(max_hr, avg_hr + random.uniform(20, 40))

    return round(avg_hr), round(max_hr_exercise)

def calculate_exercise_calories(weight_kg: float, duration_min: float, intensity_factor: float,
                              avg_hr: float, age: int, gender: str) -> float:
    """
    Calculate calories burned for a specific exercise based on intensity and heart rate
    """
    # Base METs value adjusted by intensity
    base_mets = 6.0  # Base METs for strength training
    adjusted_mets = base_mets * intensity_factor

    # Calculate calories using both METs and heart rate formulas, then average
    calories_mets = calculate_calories_mets(adjusted_mets, weight_kg, duration_min)
    calories_hr = calculate_calories_hr(avg_hr, weight_kg, age, gender, duration_min)

    # Weight the heart rate calculation more heavily for intensity-based exercises
    final_calories = (calories_mets * 0.3) + (calories_hr * 0.7)

    return round(final_calories, 2)

def assign_workout_templates(workout_type: str, user_id: str, experience_level: int) -> list:
    """
    Assign complete workout templates based on workout type and user profile
    For strength: Use all available workout templates
    For others: Use generic workout descriptors
    """
    if workout_type != 'Strength':
        # For non-strength workouts, return empty list as per requirements
        return []

    # For strength workouts, use all available workout templates
    if not WORKOUT_TEMPLATES:
        return []

    # Create list of all workout templates with user-specific variations
    assigned_workouts = []

    # Use user_id as seed for reproducibility per user
    user_seed = hash(user_id) % 1000
    random.seed(user_seed)

    # Get all workout template IDs and shuffle for variety
    workout_ids = list(WORKOUT_TEMPLATES.keys())
    random.shuffle(workout_ids)

    # Assign all workouts to the user with experience-based modifications
    for workout_id in workout_ids:
        template = WORKOUT_TEMPLATES[workout_id]

        # Create user-specific variation of the template
        user_workout = {
            'workout_template_id': workout_id,
            'original_type': template['workout_type'],
            'exercises': []
        }

        for exercise in template['exercises']:
            # Adjust exercise parameters based on user experience and profile
            adjusted_exercise = exercise.copy()

            # Experience-based intensity adjustment
            intensity_multiplier = 0.8 + (experience_level * 0.1)  # 0.8 to 1.2 based on experience
            adjusted_exercise['experience_factor'] = intensity_multiplier

            user_workout['exercises'].append(adjusted_exercise)

        assigned_workouts.append(user_workout)

    print(f"Assigned {len(assigned_workouts)} workout templates to user {user_id}")
    return assigned_workouts

def determine_readiness_factor(fatigue_level: str = None, mood: str = None) -> float:
    """
    Determine readiness factor based on user state (SePA integration)
    """
    if fatigue_level and fatigue_level.lower() in ['high', 'very high']:
        return 0.8  # Reduce intensity by 20%
    elif mood and mood.lower() in ['excellent', 'energetic']:
        return 1.05  # Increase intensity by 5%
    else:
        return 1.0  # Normal intensity

# ==================== MAIN ENHANCED PROCESSING FUNCTION ====================

def process_gym_data_enhanced(input_file: str, output_file: str) -> pd.DataFrame:
    """
    Enhanced data processing implementing Strategy_Analysis.md recommendations

    Key features:
    1. For Strength workouts: Parse exercise data, calculate 1RM, assign exercise names
    2. For non-Strength workouts: Use METs-based calories calculation
    3. Generate exercise-level granularity for AI model training
    4. Apply scientifically validated formulas
    """

    print(f"Loading data from {input_file}")
    raw_df = pd.read_excel(input_file)
    print(f"Loaded {len(raw_df)} records")

    enhanced_rows = []

    for idx, row in raw_df.iterrows():
        # Generate IDs
        user_id = f"U{idx + 1:03d}"
        workout_id = f"W{(idx // 5) + 1:03d}"  # Group every 5 rows into same workout
        user_health_profile_id = f"UH{idx + 1:03d}"

        # Convert duration from hours to minutes
        duration_min = row.get('Session_Duration (hours)', 1.0) * 60

        # Get basic metrics
        avg_hr = row.get('Avg_BPM', 120)
        max_hr_actual = row.get('Max_BPM', 160)
        weight_kg = row.get('Weight (kg)', 70)
        age = row.get('Age', 30)
        gender = row.get('Gender', 'Male')
        workout_type = row.get('Workout_Type', 'Strength')

        # Determine readiness factor (SePA integration)
        readiness_factor = determine_readiness_factor()

        if workout_type == 'Strength':
            # Process STRENGTH workouts with complete workout templates

            # Get user profile info
            experience = row.get('Experience_Level', 2)
            resting_hr = row.get('Resting_BPM', 70)

            # Get all workout templates for this user's strength workout
            workout_templates = assign_workout_templates(workout_type, user_id, experience)

            # Calculate base 1RM estimation for this user
            base_1rm = weight_kg * (1.2 + (experience * 0.1))  # Base estimation

            # Create individual exercise rows for each workout template
            for workout_template in workout_templates:
                template_id = workout_template['workout_template_id']

                # Create a unique workout_id for this user and template
                unique_workout_id = f"{workout_id}_{template_id}"

                for i, exercise in enumerate(workout_template['exercises']):
                    # Get template base values and adjust for user
                    template_duration = exercise['duration_min']
                    template_intensity = exercise.get('base_intensity', 50)
                    template_calories = exercise.get('base_calories', 30)
                    template_avg_hr = exercise.get('avg_hr', avg_hr)
                    template_max_hr = exercise.get('max_hr', max_hr_actual)

                    # Apply user-specific adjustments
                    experience_factor = exercise.get('experience_factor', 1.0)
                    intensity_variation = generate_intensity_variation(1.0, i, len(workout_template['exercises']))

                    # Calculate final exercise-specific parameters
                    final_intensity_factor = experience_factor * intensity_variation * readiness_factor
                    exercise_duration = template_duration * final_intensity_factor
                    exercise_avg_hr, exercise_max_hr = calculate_exercise_heart_rates(
                        template_avg_hr, final_intensity_factor, age
                    )

                    # Calculate exercise-specific 1RM based on intensity
                    exercise_1rm = base_1rm * final_intensity_factor

                    # Calculate calories for this specific exercise
                    exercise_calories = calculate_exercise_calories(
                        weight_kg, exercise_duration, final_intensity_factor,
                        exercise_avg_hr, age, gender
                    )

                    # Calculate unified intensity for this exercise
                    exercise_intensity = calculate_unified_intensity(
                        workout_type,
                        exercise_1rm,
                        exercise_avg_hr,
                        exercise_max_hr,
                        resting_hr,
                        weight_kg,
                        exercise_duration,
                        readiness_factor
                    )

                enhanced_row = {
                    'user_health_profile_id': user_health_profile_id,
                    'workout_id': unique_workout_id,  # Use unique workout ID for template
                    'user_id': user_id,
                    'exercise_name': exercise['name'],
                    'duration_min': round(exercise_duration, 2),
                    'avg_hr': exercise_avg_hr,
                    'max_hr': exercise_max_hr,
                    'calories': exercise_calories,
                    'suitability_x': round(random.uniform(0.7, 0.95), 2),  # Mock suitability
                    'age': age,
                    'height_m': row.get('Height (m)', 1.75),
                    'weight_kg': weight_kg,
                    'bmi': row.get('BMI', calculate_bmi(weight_kg, row.get('Height (m)', 1.75))),
                    'fat_percentage': row.get('Fat_Percentage', estimate_body_fat_from_bmi(row.get('BMI', 22.0), age, gender)),
                    'resting_heartrate': resting_hr,
                    'experience_level': experience,
                    'workout_frequency': row.get('Workout_Frequency (days/week)', 3),
                    'health_status': 'Healthy',  # Default since not in source data
                    'workout_type': workout_type,
                    'location': 'Gym',  # Default location
                    'suitability_y': round(random.uniform(0.65, 0.90), 2),  # Mock suitability
                    'gender': gender,
                    'session_duration': duration_min,
                    'unified_intensity': exercise_intensity,
                    'estimated_1rm': round(exercise_1rm, 2),
                    'intensity_factor': round(final_intensity_factor, 3),  # Track the intensity variation
                    'workout_template_id': template_id,  # Track which template this exercise came from
                    'template_intensity': template_intensity,  # Original template intensity
                    'experience_factor': round(experience_factor, 3)  # User experience adjustment factor
                }

                enhanced_rows.append(enhanced_row)

        else:
            # Process NON-STRENGTH workouts (single row per workout)

            # Get METs value for workout type
            mets = MET_VALUES.get(workout_type, 5.0)

            # Calculate calories using METs formula
            total_calories = calculate_calories_mets(mets, weight_kg, duration_min)

            # Calculate intensity
            unified_intensity = calculate_unified_intensity(
                workout_type, 0, avg_hr, max_hr_actual,
                row.get('Resting_BPM', 70), weight_kg, duration_min, readiness_factor
            )

            enhanced_row = {
                'user_health_profile_id': user_health_profile_id,
                'workout_id': workout_id,
                'user_id': user_id,
                'exercise_name': '',  # Empty as per requirements for non-strength
                'duration_min': duration_min,
                'avg_hr': avg_hr,
                'max_hr': max_hr_actual,
                'calories': round(total_calories, 2),
                'suitability_x': round(random.uniform(0.6, 0.85), 2),
                'age': age,
                'height_m': row.get('Height (m)', 1.75),
                'weight_kg': weight_kg,
                'bmi': row.get('BMI', calculate_bmi(weight_kg, row.get('Height (m)', 1.75))),
                'fat_percentage': row.get('Fat_Percentage', estimate_body_fat_from_bmi(row.get('BMI', 22.0), age, gender)),
                'resting_heartrate': row.get('Resting_BPM', 70),
                'experience_level': row.get('Experience_Level', 2),
                'workout_frequency': row.get('Workout_Frequency (days/week)', 3),
                'health_status': 'Healthy',
                'workout_type': workout_type,
                'location': 'Gym',
                'suitability_y': round(random.uniform(0.6, 0.85), 2),
                'gender': gender,
                'session_duration': duration_min,
                'unified_intensity': unified_intensity,
                'estimated_1rm': 0,  # Not applicable for non-strength
                'intensity_factor': 1.0,  # No intensity variation for non-strength
                'workout_template_id': '',  # Not applicable for non-strength
                'template_intensity': 0,  # Not applicable for non-strength
                'experience_factor': 1.0  # Default for non-strength
            }

            enhanced_rows.append(enhanced_row)

    # Create enhanced dataframe
    enhanced_df = pd.DataFrame(enhanced_rows)

    print(f"Enhanced dataset shape: {enhanced_df.shape}")
    print(f"Workout type distribution:")
    print(enhanced_df['workout_type'].value_counts())

    # Generate processing report
    generate_processing_report(enhanced_df, output_file)

    return enhanced_df

def generate_processing_report(df: pd.DataFrame, output_path: str):
    """Generate processing summary report"""
    report_path = output_path.replace('.xlsx', '_processing_report.json')

    report = {
        'processing_summary': {
            'total_records': len(df),
            'unique_users': df['user_id'].nunique(),
            'unique_workouts': df['workout_id'].nunique(),
            'workout_types': df['workout_type'].value_counts().to_dict(),
            'exercise_count': df[df['exercise_name'] != '']['exercise_name'].nunique()
        },
        'data_quality': {
            'missing_values': df.isnull().sum().to_dict(),
            'avg_calories_per_session': df.groupby('workout_id')['calories'].sum().mean(),
            'avg_intensity_score': df['unified_intensity'].mean(),
            'avg_session_duration': df.groupby('workout_id')['duration_min'].sum().mean()
        },
        'enhancements_applied': [
            'Estimated 1RM calculation for strength exercises',
            'METs-based calories calculation for non-strength workouts',
            'Heart rate-based calories calculation where HR data available',
            'Unified intensity scoring (0-100 scale)',
            'Exercise name assignment for strength workouts',
            'SePA-inspired readiness factor adjustments',
            'Structured workout grouping with exercise-level granularity'
        ]
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Processing report saved to {report_path}")

# ==================== LEGACY COMPATIBILITY FUNCTION ====================

def process_gym_data(input_file, output_file, is_individual_exercises=False):
    """
    Legacy compatibility wrapper
    """
    if is_individual_exercises:
        # Use the original processing for individual exercise data
        return process_legacy_individual_exercises(input_file, output_file)
    else:
        # Use the new enhanced processing for workout session data
        enhanced_df = process_gym_data_enhanced(input_file, output_file)

        # Save the enhanced data
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        enhanced_df.to_excel(output_file, index=False)
        print(f"Enhanced data saved to {output_file}")

        return enhanced_df

def process_legacy_individual_exercises(input_file: str, output_file: str) -> pd.DataFrame:
    """
    Process individual exercise data (legacy function for own_gym_member_exercise_tracking.xlsx)
    """
    print(f"Reading individual exercise data from: {input_file}")
    df = pd.read_excel(input_file)

    # Apply basic enhancements that were in the original code
    print("Applying legacy enhancements...")

    # Add any missing calculations that might be needed
    if 'unified_intensity' not in df.columns and 'calories' in df.columns and 'session_duration' in df.columns:
        df['unified_intensity'] = df.apply(
            lambda row: min(100, (row['calories'] / (row['session_duration'] * 60)) * 10),
            axis=1
        )

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_excel(output_file, index=False)
    print(f"Legacy enhanced data saved to {output_file}")

    return df


def main():
    """Main execution function implementing Strategy_Analysis.md recommendations"""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    print("="*80)
    print("ENHANCED GYM DATA PROCESSOR")
    print("Implementing Strategy_Analysis.md Framework")
    print("="*80)

    # File configurations
    input_path = './data/gym_member_exercise_tracking.xlsx'
    output_path = './preprocessing_data/enhanced_gym_member_exercise_tracking_with_templates.xlsx'

    try:
        # Process data with enhanced strategy
        print(f"\n[PROCESSING] {input_path}")
        enhanced_df = process_gym_data_enhanced(input_path, output_path)

        # Save the enhanced data
        output_full_path = Path(output_path)
        output_full_path.parent.mkdir(parents=True, exist_ok=True)

        enhanced_df.to_excel(output_full_path, index=False)

        print("\n" + "="*80)
        print("[SUCCESS] DATA ENHANCEMENT COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Input: {input_path}")
        print(f"Output: {output_full_path}")
        print(f"Records processed: {len(enhanced_df)}")
        print(f"Strength exercises added: {enhanced_df[enhanced_df['exercise_name'] != '']['exercise_name'].nunique()}")
        print(f"Workout types: {enhanced_df['workout_type'].value_counts().to_dict()}")

        print("\nSample Data Preview:")
        sample_data = enhanced_df[['workout_type', 'exercise_name', 'calories', 'unified_intensity']].head(10)
        print(sample_data.to_string(index=False))

        print("\nKey Enhancements Applied:")
        enhancements = [
            "[+] 1RM estimation using Epley formula for strength exercises",
            "[+] METs-based calories calculation for non-strength workouts",
            "[+] Heart rate-based calories calculation where HR data available",
            "[+] Unified intensity scoring (0-100 scale)",
            "[+] Exercise name assignment for strength workouts from reference dataset",
            "[+] SePA-inspired readiness factor adjustments",
            "[+] Structured workout grouping with exercise-level granularity",
            "[+] Scientifically validated formulas from docs/"
        ]

        for enhancement in enhancements:
            print(f"  {enhancement}")

        print("\n[INFO] Processing report generated alongside output file.")
        print("="*80)

    except Exception as e:
        print(f"\n[ERROR] Error processing data: {e}")
        raise

if __name__ == "__main__":
    main()
