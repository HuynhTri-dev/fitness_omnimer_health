import pandas as pd

# Tạo dữ liệu mẫu
data = [
    {
        'age': 25,
        'gender': 'Male',
        'weight_kg': 75,
        'height_m': 1.75,
        'experience_level': 2,  # Intermediate
        'workout_frequency': 3,
        'resting_heartrate': 65,
        'mood': 'Good',
        'fatigue': 'Low',
        'effort': 'Medium',
        'description': 'User 1: Thanh niên khỏe mạnh, trạng thái tốt'
    },
    {
        'age': 35,
        'gender': 'Female',
        'weight_kg': 60,
        'height_m': 1.65,
        'experience_level': 1,  # Beginner
        'workout_frequency': 2,
        'resting_heartrate': 72,
        'mood': 'Neutral',
        'fatigue': 'High',  # Đang mệt
        'effort': 'Low',
        'description': 'User 2: Nữ mới tập, đang mệt mỏi'
    },
    {
        'age': 28,
        'gender': 'Male',
        'weight_kg': 85,
        'height_m': 1.80,
        'experience_level': 4,  # Expert
        'workout_frequency': 5,
        'resting_heartrate': 55,
        'mood': 'Excellent',
        'fatigue': 'Very Low',
        'effort': 'High',
        'description': 'User 3: Vận động viên, sung sức'
    }
]

df = pd.DataFrame(data)

# Tính BMI (vì model cần)
df['bmi'] = df['weight_kg'] / (df['height_m'] ** 2)

# Lưu file
output_file = 'd:/dacn_omnimer_health/3T-FIT/ai_server/model/src/v3/sample_input.xlsx'
df.to_excel(output_file, index=False)
print(f"Created sample input file: {output_file}")
