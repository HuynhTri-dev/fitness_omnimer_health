import { IWorkout } from "../../domain/models";

/**
 * Tính toán summary cho 1 buổi tập Workout
 * Tổng hợp các chỉ số từ những set có done === true và deviceData (nếu có)
 */
export function calculateWorkoutSummary(workout: IWorkout, endTime?: Date) {
  const summary = {
    totalSets: 0,
    totalReps: 0,
    totalWeight: 0,
    totalDuration: 0,
    totalCalories: 0,
    totalDistance: 0,
    heartRateAvgAllWorkout: 0,
    heartRateMaxAllWorkout: 0,
  };

  let totalHeartRateAvg = 0;
  let heartRateAvgCount = 0;

  for (const exercise of workout.workoutDetail) {
    // Lấy các set đã hoàn thành
    const doneSets = exercise.sets.filter((set) => set.done);

    for (const set of doneSets) {
      summary.totalSets++;
      if (set.reps) summary.totalReps += set.reps;
      // Fix: Total Weight = Weight * Reps (Volume)
      if (set.weight && set.reps) {
        summary.totalWeight += set.weight * set.reps;
      }
      if (set.distance) summary.totalDistance += set.distance;
    }

    // Tổng thời gian bài tập (tích lũy từ bài tập lẻ)
    if (exercise.durationMin) summary.totalDuration += exercise.durationMin;

    // Dữ liệu thiết bị (nếu có)
    if (exercise.deviceData) {
      const { caloriesBurned, heartRateAvg, heartRateMax } =
        exercise.deviceData;

      if (caloriesBurned) summary.totalCalories += caloriesBurned;

      if (heartRateAvg) {
        totalHeartRateAvg += heartRateAvg;
        heartRateAvgCount++;
      }

      if (heartRateMax && heartRateMax > summary.heartRateMaxAllWorkout) {
        summary.heartRateMaxAllWorkout = heartRateMax;
      }
    }
  }

  // Override Total Duration if endTime is provided (Wall-clock time)
  if (endTime && workout.timeStart) {
    const start = new Date(workout.timeStart).getTime();
    const end = new Date(endTime).getTime();
    const durationMs = end - start;
    if (durationMs > 0) {
      summary.totalDuration = durationMs / 60000; // convert to minutes
    }
  }

  // Rounding values
  summary.totalDuration = parseFloat(summary.totalDuration.toFixed(2));
  summary.totalCalories = parseFloat(summary.totalCalories.toFixed(2));

  // Trung bình nhịp tim
  if (heartRateAvgCount > 0) {
    summary.heartRateAvgAllWorkout = Number(
      (totalHeartRateAvg / heartRateAvgCount).toFixed(1)
    );
  }

  return summary;
}
// utils/CalorieCalculator.ts

export interface IWorkoutDetailInfo {
  reps?: number;
  sets?: number;
  weight?: number; // kg
  distance?: number; // mét
  duration?: number; // giây
}

/**
 * 🔹 Tính hệ số cường độ MET (metFactor)
 * Dựa trên cường độ luyện tập (sức nặng, số set, số reps, vận tốc)
 * @param weight - Khối lượng tạ (kg)
 * @param sets - Số set
 * @param reps - Số lần lặp
 * @param v - Vận tốc (km/h)
 * @param duration - Thời lượng bài tập (giây)
 * @returns metFactor - hệ số nhân điều chỉnh MET gốc
 */
export function calculateMetFactor(
  detail: IWorkoutDetailInfo,
  durationMin: number
): number {
  let metFactor = 1;
  const weight = Number(detail.weight ?? 0);
  const sets = Number(detail.sets ?? 0);
  const reps = Number(detail.reps ?? 0);
  const distance = Number(detail.distance ?? 0); // mét
  const duration = Number(detail.duration ?? 0); // giây

  if (
    weight === 0 &&
    sets === 0 &&
    reps === 0 &&
    duration === 0 &&
    distance === 0
  ) {
    return 0;
  }

  // Cường độ do tạ và reps
  if (weight && sets && reps) {
    if (weight > 30 || sets > 5 || reps > 20) {
      metFactor *= 2;
    } else if (weight > 25 || sets > 4 || reps > 10) {
      metFactor *= 1.5;
    }

    return metFactor;
  }

  // Cường độ do vận tốc di chuyển
  if (distance) {
    const v = durationMin > 0 ? distance / 1000 / (durationMin / 60) : 0;
    if (v > 14) {
      metFactor *= 2;
    } else if (v > 10) {
      metFactor *= 1.5;
    } else if (v > 8) {
      metFactor *= 1.25;
    } else if (v < 5) {
      metFactor *= 0.75;
    }

    return metFactor;
  }

  // --- Cường độ do thời lượng ---
  if (duration) {
    if (duration > 90) metFactor *= 1.5;
    else if (duration > 60) metFactor *= 1.25;
    else if (duration > 30) metFactor *= 1.1;
    else if (duration < 10) metFactor *= 0.9;

    return metFactor;
  }

  return metFactor;
}

/**
 * 🔹 Tính lượng calo tiêu hao theo công thức MET
 * @param met - Giá trị MET (Metabolic Equivalent of Task)
 * @param weightKg - Cân nặng người tập (kg)
 * @param durationMin - Thời lượng bài tập (phút)
 * @param detail - Thông tin chi tiết bài tập (reps, sets, weight, distance, duration)
 * @returns Số calo ước tính tiêu hao
 */
export function calculateCaloriesByMET(
  met: number = 3,
  weightKg: number = 60,
  durationMin: number = 0,
  detail: IWorkoutDetailInfo = {}
): number {
  // Tính hệ số cường độ MET
  const metFactor = calculateMetFactor(detail, durationMin);

  if (metFactor === 0) return 0;

  // Công thức tính calo theo MET
  const calories = ((met * metFactor * weightKg * 3.5) / 200) * durationMin;

  return parseFloat(calories.toFixed(2));
}
