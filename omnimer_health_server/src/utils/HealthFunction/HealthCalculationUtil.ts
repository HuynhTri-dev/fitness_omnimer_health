import { GenderEnum } from "../../common/constants/EnumConstants";

export interface HealthInput {
  gender: GenderEnum;
  height: number; // cm
  weight: number; // kg
  neck: number; // cm
  waist: number; // cm
  hip: number; // cm
  birthday: Date;
  raceFactor?: number;
}

export interface HealthResult {
  age: number;
  bmi: number;
  bmr: number;
  bodyFatPercentage: number;
  muscleMass: number;
  whr: number | null;
}

/**
 * Tính toán các chỉ số sức khỏe dựa trên thông tin người dùng
 * Dùng trước khi gửi sang mô hình AI đánh giá.
 */
export function calculateHealthMetrics({
  gender,
  height,
  weight,
  neck,
  waist,
  hip,
  birthday,
  raceFactor = 0,
}: HealthInput): HealthResult {
  // Chuyển kiểu dữ liệu
  height = Number(height) || 0;
  weight = Number(weight) || 0;
  neck = Number(neck) || 0;
  waist = Number(waist) || 0;
  hip = Number(hip) || 0;

  // Tuổi
  const age =
    birthday instanceof Date
      ? Math.floor(
          (Date.now() - birthday.getTime()) / (1000 * 60 * 60 * 24 * 365.25)
        )
      : 0;

  // BMI
  const bmi = height > 0 ? weight / (height / 100) ** 2 : 0;

  // BMR
  const bmr =
    gender === GenderEnum.Male
      ? 88.362 + 13.397 * weight + 4.799 * height - 5.677 * age
      : 447.593 + 9.247 * weight + 3.098 * height - 4.33 * age;

  // Body Fat %
  let bodyFat = 0;
  try {
    if (gender === GenderEnum.Male) {
      bodyFat =
        495 /
          (1.0324 -
            0.19077 * Math.log10(Math.max(waist - neck, 1)) +
            0.15456 * Math.log10(height)) -
        450;
    } else {
      bodyFat =
        495 /
          (1.29579 -
            0.35004 * Math.log10(Math.max(waist + hip - neck, 1)) +
            0.221 * Math.log10(height)) -
        450;
    }
  } catch {
    bodyFat = 0;
  }

  // Muscle Mass
  const sex = gender === GenderEnum.Male ? 1 : 0;
  const muscleMass =
    0.244 * weight +
    7.8 * (height / 100) +
    6.6 * sex -
    0.098 * age +
    raceFactor -
    3.3;

  // WHR
  const whr = hip > 0 ? waist / hip : null;

  return {
    age,
    bmi: parseFloat(bmi.toFixed(2)),
    bmr: parseFloat(bmr.toFixed(2)),
    bodyFatPercentage: parseFloat(bodyFat.toFixed(2)),
    muscleMass: parseFloat(muscleMass.toFixed(2)),
    whr: whr !== null ? parseFloat(whr.toFixed(2)) : null,
  };
}
