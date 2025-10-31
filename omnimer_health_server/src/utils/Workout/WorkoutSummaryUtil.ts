import { IWorkout } from "../../domain/models";

/**
 * Tính toán summary cho 1 buổi tập Workout
 * Tổng hợp các chỉ số từ những set có done === true và deviceData (nếu có)
 */
export function calculateWorkoutSummary(workout: IWorkout) {
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
      if (set.weight) summary.totalWeight += set.weight;
      if (set.distance) summary.totalDistance += set.distance;
    }

    // Tổng thời gian bài tập (nếu có)
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

  // Trung bình nhịp tim
  if (heartRateAvgCount > 0) {
    summary.heartRateAvgAllWorkout = Number(
      (totalHeartRateAvg / heartRateAvgCount).toFixed(1)
    );
  }

  return summary;
}
