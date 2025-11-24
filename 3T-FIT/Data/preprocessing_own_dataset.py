"""
Script để xử lý lại dữ liệu merged_omni_health_dataset.xlsx để tạo ra file gym_member_exercise_tracking.xlsx
Dữ liệu sẽ được loại bỏ các thông tin không cần thiết và được chuyển đổi thành dạng số liệu để dễ dàng xử lý và phân tích.

"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_1rm(weight, reps):
    """
    Tính 1RM ước tính theo công thức Epley: 1RM = Weight * (1 + Reps/30)
    """
    if weight == 0: return 0
    return weight * (1 + reps / 30)


def extract_intensity(row):
    """
    Tính toán chỉ số cường độ thống nhất (unified intensity) dựa trên loại bài tập.
    - Strength: Dùng Estimated 1RM (Max)
    - Cardio (có distance): Dùng Tốc độ trung bình (km/h)
    - Khác: Dùng Intensity Score (1-4)
    Tất cả giá trị được làm tròn đến 2 chữ số thập phân.
    """
    # 1. Xử lý Strength: sets/reps/weight/timeresteachset
    # Format dự kiến: "Reps x Weight x Rest" (VD: 12x40x2)
    if pd.notna(row.get('sets/reps/weight/timeresteachset')):
        try:
            data = str(row['sets/reps/weight/timeresteachset'])
            # Tách các set (có thể ngăn cách bởi | hoặc ,)
            sets = data.replace('|', ',').split(',')
            max_1rm = 0
            has_valid_set = False
            
            for s in sets:
                parts = s.strip().lower().split('x')
                if len(parts) >= 2:
                    try:
                        reps = float(parts[0])
                        weight = float(parts[1])
                        # Chỉ tính nếu weight > 0 (bài tập tạ)
                        if weight > 0:
                            rm = calculate_1rm(weight, reps)
                            if rm > max_1rm:
                                max_1rm = rm
                            has_valid_set = True
                    except ValueError:
                        continue
            
            if has_valid_set and max_1rm > 0:
                return round(max_1rm, 2)
        except Exception:
            pass

    # 2. Xử lý Cardio Distance: distance_km
    # Tính tốc độ km/h = distance_km / session_duration (giờ)
    if pd.notna(row.get('distance_km')) and row.get('distance_km') > 0:
        if pd.notna(row.get('session_duration')) and row.get('session_duration') > 0:
            speed = row['distance_km'] / row['session_duration']
            return round(speed, 2)

    # 3. Fallback: Intensity Score (1-4)
    # Bao gồm cả trường hợp sets/time_m/timeresteachset (Bodyweight/Static)
    if pd.notna(row.get('intensity')):
        return round(row['intensity'], 2)
    
    return 0.0


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu theo các yêu cầu sau:
    1. Loại bỏ các cột mà không có bất kì dữ liệu gì
    2. Xóa các cột: checkup_date, id, mood, recovery_h, effectiveness, equipment, target_muscle, secondary_muscles, 
                    bodypart_target, workout_goal_achieved, target_muscle_felt, category_exercise_want_todo, birthday,
                    fatigue, effort, whr, workout_date, injury_or_pain_notes, exercise_not_suitable, activity_level
    3. Loại bỏ các dòng của cột done có giá trị = 0 và sau đó loại bỏ cột done
    4. Sắp xếp các cột user_health_profile_id, workout_id, user_id ở đầu
    5. Đổi tên cột total_duration_min thành session_duration và chuyển đổi từ phút sang giờ
    6. Biến đổi dữ liệu experience_level và intensity sang dạng số
    7. Gộp các cột cường độ thành 1 cột duy nhất (unified_intensity)
    8. Đổi tên cột category_type_want_todo thành workout_type
    
    Args:
        df (pd.DataFrame): DataFrame cần làm sạch
        
    Returns:
        pd.DataFrame: DataFrame đã được làm sạch
    """
    # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
    df_cleaned = df.copy()
    
    print(f"Số lượng dòng ban đầu: {len(df_cleaned)}")
    print(f"Số lượng cột ban đầu: {len(df_cleaned.columns)}")
    print(f"Tên các cột ban đầu: {list(df_cleaned.columns)}\n")
    
    # Bước 1: Loại bỏ các cột mà không có bất kì dữ liệu gì
    # Kiểm tra các cột có tất cả giá trị là null hoặc rỗng
    empty_columns = []
    for col in df_cleaned.columns:
        if df_cleaned[col].isna().all() or (df_cleaned[col].astype(str).str.strip() == '').all():
            empty_columns.append(col)
    
    if empty_columns:
        print(f"Loại bỏ {len(empty_columns)} cột rỗng: {empty_columns}")
        df_cleaned = df_cleaned.drop(columns=empty_columns)
    else:
        print("Không có cột nào hoàn toàn rỗng")
    
    # Bước 2: Xóa các cột cụ thể
    columns_to_drop = [
        'checkup_date', 'id', 'mood', 'recovery_h', 'effectiveness', 
        'equipment', 'target_muscle', 'secondary_muscles', 'bodypart_target', 
        'workout_goal_achieved', 'target_muscle_felt', 'category_exercise_want_todo', 'birthday',
        'fatigue', 'effort',
        'whr', 'workout_date', 'injury_or_pain_notes', 'exercise_not_suitable', 'activity_level'
    ]
    existing_columns_to_drop = [col for col in columns_to_drop if col in df_cleaned.columns]
    
    if existing_columns_to_drop:
        print(f"\nXóa các cột: {existing_columns_to_drop}")
        df_cleaned = df_cleaned.drop(columns=existing_columns_to_drop)
    else:
        print(f"\nKhông tìm thấy các cột cần xóa trong danh sách: {columns_to_drop}")
    
    # Bước 3: Loại bỏ các dòng của cột done có giá trị = 0
    if 'done' in df_cleaned.columns:
        rows_before = len(df_cleaned)
        # Giữ lại các dòng có done != 0 (bao gồm cả done = 1 và các giá trị khác)
        df_cleaned = df_cleaned[df_cleaned['done'] != 0]
        rows_removed = rows_before - len(df_cleaned)
        print(f"\nLoại bỏ {rows_removed} dòng có done = 0")
        
        # Sau đó loại bỏ cột done
        df_cleaned = df_cleaned.drop(columns=['done'])
        print("Đã xóa cột 'done'")
    else:
        print("\nKhông tìm thấy cột 'done' trong dữ liệu")
    
    # Bước 4: Sắp xếp các cột ID ở đầu để dễ quản lý
    priority_columns = ['user_health_profile_id', 'workout_id', 'user_id']
    existing_priority_cols = [col for col in priority_columns if col in df_cleaned.columns]
    
    if existing_priority_cols:
        # Lấy các cột còn lại (không phải priority columns)
        other_columns = [col for col in df_cleaned.columns if col not in existing_priority_cols]
        
        # Sắp xếp lại: priority columns trước, các cột khác sau
        new_column_order = existing_priority_cols + other_columns
        df_cleaned = df_cleaned[new_column_order]
        
        print(f"\nSắp xếp lại thứ tự cột: {existing_priority_cols} được đặt ở đầu")
    else:
        print(f"\nKhông tìm thấy các cột ưu tiên: {priority_columns}")
    
    # Bước 5: Đổi tên cột total_duration_min thành session_duration và chuyển đổi từ phút sang giờ
    if 'total_duration_min' in df_cleaned.columns:
        # Chuyển đổi từ phút sang giờ (làm tròn 2 chữ số thập phân)
        df_cleaned['session_duration'] = (df_cleaned['total_duration_min'] / 60).round(2)
        
        # Xóa cột cũ
        df_cleaned = df_cleaned.drop(columns=['total_duration_min'])
        
        print(f"\nĐã đổi tên 'total_duration_min' thành 'session_duration' và chuyển đổi từ phút sang giờ")
        print(f"Ví dụ giá trị: {df_cleaned['session_duration'].head().tolist()}")
    else:
        print("\nKhông tìm thấy cột 'total_duration_min' trong dữ liệu")

    # Bước 6: Biến đổi dữ liệu experience_level và intensity
    # Mapping cho experience_level
    experience_map = {
        'beginner': 1,
        'intermediate': 2,
        'advanced': 3,
        'expert': 4
    }
    
    if 'experience_level' in df_cleaned.columns:
        # Chuyển về lowercase để map chính xác, sau đó map giá trị
        df_cleaned['experience_level'] = df_cleaned['experience_level'].astype(str).str.lower().map(experience_map)
        print(f"\nĐã biến đổi cột 'experience_level' sang dạng số: {experience_map}")
    else:
        print("\nKhông tìm thấy cột 'experience_level'")

    # Mapping cho intensity
    intensity_map = {
        'low': 1,
        'medium': 2,
        'high': 3,
        'maximal': 4
    }
    
    if 'intensity' in df_cleaned.columns:
        df_cleaned['intensity'] = df_cleaned['intensity'].astype(str).str.lower().map(intensity_map)
        print(f"\nĐã biến đổi cột 'intensity' sang dạng số: {intensity_map}")
    else:
        print("\nKhông tìm thấy cột 'intensity'")

    # Bước 7: Gộp các cột cường độ thành 1 cột duy nhất (unified_intensity)
    print("\nBắt đầu tính toán unified_intensity...")
    df_cleaned['unified_intensity'] = df_cleaned.apply(extract_intensity, axis=1)
    
    # Xóa các cột cũ
    cols_to_remove = [
        'sets/reps/weight/timeresteachset', 
        'sets/time_m/timeresteachset', 
        'distance_km', 
        'intensity'
    ]
    existing_cols_to_remove = [col for col in cols_to_remove if col in df_cleaned.columns]
    
    if existing_cols_to_remove:
        df_cleaned = df_cleaned.drop(columns=existing_cols_to_remove)
        print(f"Đã xóa các cột cường độ cũ: {existing_cols_to_remove}")
    
    print(f"Ví dụ giá trị unified_intensity: {df_cleaned['unified_intensity'].head().tolist()}")
    
    # Bước 8: Đổi tên cột category_type_want_todo thành workout_type
    if 'category_type_want_todo' in df_cleaned.columns:
        df_cleaned = df_cleaned.rename(columns={'category_type_want_todo': 'workout_type'})
        print(f"\nĐã đổi tên cột 'category_type_want_todo' thành 'workout_type'")
    else:
        print("\nKhông tìm thấy cột 'category_type_want_todo'")
    
    print(f"\nSố lượng dòng sau khi làm sạch: {len(df_cleaned)}")
    print(f"Số lượng cột sau khi làm sạch: {len(df_cleaned.columns)}")
    print(f"Tên các cột còn lại: {list(df_cleaned.columns)}")
    
    return df_cleaned


def process_dataset(input_file: str, output_file: str = None) -> pd.DataFrame:
    """
    Đọc file dữ liệu, làm sạch và lưu kết quả
    
    Args:
        input_file (str): Đường dẫn đến file dữ liệu đầu vào
        output_file (str, optional): Đường dẫn đến file dữ liệu đầu ra. 
                                     Nếu None, sẽ tự động tạo tên file
        
    Returns:
        pd.DataFrame: DataFrame đã được làm sạch
    """
    # Đọc dữ liệu
    print(f"Đọc dữ liệu từ: {input_file}")
    df = pd.read_excel(input_file)
    
    # Làm sạch dữ liệu
    print("\n" + "="*60)
    print("BẮT ĐẦU QUÁ TRÌNH LÀM SẠCH DỮ LIỆU")
    print("="*60 + "\n")
    
    df_cleaned = clean_dataset(df)
    
    # Lưu kết quả
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
    
    print(f"\n" + "="*60)
    print(f"Lưu dữ liệu đã làm sạch vào: {output_file}")
    print("="*60)
    
    df_cleaned.to_excel(output_file, index=False)
    
    return df_cleaned


if __name__ == "__main__":
    # Ví dụ sử dụng
    input_file = "./data/merged_omni_health_dataset.xlsx"
    output_file = "./preprocessing_data/own_gym_member_exercise_tracking.xlsx"
    
    # Kiểm tra file có tồn tại không
    if Path(input_file).exists():
        df_cleaned = process_dataset(input_file, output_file)
        print("\n✅ Hoàn thành quá trình làm sạch dữ liệu!")
    else:
        print(f"❌ Không tìm thấy file: {input_file}")
        print("Vui lòng cập nhật đường dẫn file trong script hoặc đặt file vào thư mục hiện tại.")