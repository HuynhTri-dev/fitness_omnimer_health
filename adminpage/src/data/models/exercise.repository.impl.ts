import type { IExerciseRepository } from "../../domain/repositories/exercise.repository";
import type {
  Equipment,
  BodyPart,
  Muscle,
  ExerciseType,
  ExerciseCategory,
  Exercise,
  PaginationParams,
  PaginationResponse,
} from "../../shared/types";
import { apiService } from "../services/api";

export class ExerciseRepositoryImpl implements IExerciseRepository {
  // Equipment
  async getEquipment(
    params?: PaginationParams
  ): Promise<PaginationResponse<Equipment>> {
    const response = await apiService.get<Equipment[]>("/equipment/", params);
    return {
      data: response.data,
      total: response.data.length,
      page: params?.page || 1,
      limit: params?.limit || 10,
      totalPages: Math.ceil(response.data.length / (params?.limit || 10)),
    };
  }

  async getEquipmentById(id: string): Promise<Equipment> {
    const response = await apiService.get<Equipment>(`/equipment/${id}`);
    return response.data;
  }

  async createEquipment(equipmentData: FormData): Promise<Equipment> {
    const response = await apiService.uploadFile<Equipment>(
      "/equipment/",
      equipmentData
    );
    return response.data;
  }

  async updateEquipment(
    id: string,
    equipmentData: FormData | Partial<Equipment>
  ): Promise<Equipment> {
    if (equipmentData instanceof FormData) {
      const response = await apiService.uploadFile<Equipment>(
        `/equipment/${id}`,
        equipmentData
      );
      return response.data;
    } else {
      const response = await apiService.put<Equipment>(
        `/equipment/${id}`,
        equipmentData
      );
      return response.data;
    }
  }

  async deleteEquipment(id: string): Promise<void> {
    await apiService.delete(`/equipment/${id}`);
  }

  // Body Parts
  async getBodyParts(
    params?: PaginationParams
  ): Promise<PaginationResponse<BodyPart>> {
    const response = await apiService.get<BodyPart[]>("/body-part/", params);
    return {
      data: response.data,
      total: response.data.length,
      page: params?.page || 1,
      limit: params?.limit || 10,
      totalPages: Math.ceil(response.data.length / (params?.limit || 10)),
    };
  }

  async getBodyPartById(id: string): Promise<BodyPart> {
    const response = await apiService.get<BodyPart>(`/body-part/${id}`);
    return response.data;
  }

  async createBodyPart(bodyPartData: FormData): Promise<BodyPart> {
    const response = await apiService.uploadFile<BodyPart>(
      "/body-part/",
      bodyPartData
    );
    return response.data;
  }

  async updateBodyPart(
    id: string,
    bodyPartData: FormData | Partial<BodyPart>
  ): Promise<BodyPart> {
    if (bodyPartData instanceof FormData) {
      const response = await apiService.uploadFile<BodyPart>(
        `/body-part/${id}`,
        bodyPartData
      );
      return response.data;
    } else {
      const response = await apiService.put<BodyPart>(
        `/body-part/${id}`,
        bodyPartData
      );
      return response.data;
    }
  }

  async deleteBodyPart(id: string): Promise<void> {
    await apiService.delete(`/body-part/${id}`);
  }

  // Muscles
  async getMuscles(
    params?: PaginationParams
  ): Promise<PaginationResponse<Muscle>> {
    const response = await apiService.get<Muscle[]>("/muscle/", params);
    return {
      data: response.data,
      total: response.data.length,
      page: params?.page || 1,
      limit: params?.limit || 10,
      totalPages: Math.ceil(response.data.length / (params?.limit || 10)),
    };
  }

  async getMuscleById(id: string): Promise<Muscle> {
    const response = await apiService.get<Muscle>(`/muscle/${id}`);
    return response.data;
  }

  async createMuscle(muscleData: FormData): Promise<Muscle> {
    const response = await apiService.uploadFile<Muscle>(
      "/muscle/",
      muscleData
    );
    return response.data;
  }

  async updateMuscle(
    id: string,
    muscleData: FormData | Partial<Muscle>
  ): Promise<Muscle> {
    if (muscleData instanceof FormData) {
      const response = await apiService.uploadFile<Muscle>(
        `/muscle/${id}`,
        muscleData
      );
      return response.data;
    } else {
      const response = await apiService.put<Muscle>(
        `/muscle/${id}`,
        muscleData
      );
      return response.data;
    }
  }

  async deleteMuscle(id: string): Promise<void> {
    await apiService.delete(`/muscle/${id}`);
  }

  // Exercise Types
  async getExerciseTypes(
    params?: PaginationParams
  ): Promise<PaginationResponse<ExerciseType>> {
    const response = await apiService.get<ExerciseType[]>(
      "/exercise-type/",
      params
    );
    return {
      data: response.data,
      total: response.data.length,
      page: params?.page || 1,
      limit: params?.limit || 10,
      totalPages: Math.ceil(response.data.length / (params?.limit || 10)),
    };
  }

  async getExerciseTypeById(id: string): Promise<ExerciseType> {
    const response = await apiService.get<ExerciseType>(`/exercise-type/${id}`);
    return response.data;
  }

  async createExerciseType(typeData: any): Promise<ExerciseType> {
    const response = await apiService.post<ExerciseType>(
      "/exercise-type/",
      typeData
    );
    return response.data;
  }

  async updateExerciseType(
    id: string,
    typeData: Partial<ExerciseType>
  ): Promise<ExerciseType> {
    const response = await apiService.put<ExerciseType>(
      `/exercise-type/${id}`,
      typeData
    );
    return response.data;
  }

  async deleteExerciseType(id: string): Promise<void> {
    await apiService.delete(`/exercise-type/${id}`);
  }

  // Exercise Categories
  async getExerciseCategories(
    params?: PaginationParams
  ): Promise<PaginationResponse<ExerciseCategory>> {
    const response = await apiService.get<ExerciseCategory[]>(
      "/exercise-category/",
      params
    );
    return {
      data: response.data,
      total: response.data.length,
      page: params?.page || 1,
      limit: params?.limit || 10,
      totalPages: Math.ceil(response.data.length / (params?.limit || 10)),
    };
  }

  async getExerciseCategoryById(id: string): Promise<ExerciseCategory> {
    const response = await apiService.get<ExerciseCategory>(
      `/exercise-category/${id}`
    );
    return response.data;
  }

  async createExerciseCategory(
    categoryData: FormData
  ): Promise<ExerciseCategory> {
    const response = await apiService.uploadFile<ExerciseCategory>(
      "/exercise-category/",
      categoryData
    );
    return response.data;
  }

  async updateExerciseCategory(
    id: string,
    categoryData: FormData | Partial<ExerciseCategory>
  ): Promise<ExerciseCategory> {
    if (categoryData instanceof FormData) {
      const response = await apiService.uploadFile<ExerciseCategory>(
        `/exercise-category/${id}`,
        categoryData
      );
      return response.data;
    } else {
      const response = await apiService.put<ExerciseCategory>(
        `/exercise-category/${id}`,
        categoryData
      );
      return response.data;
    }
  }

  async deleteExerciseCategory(id: string): Promise<void> {
    await apiService.delete(`/exercise-category/${id}`);
  }

  // Exercises
  async getExercises(
    params?: PaginationParams
  ): Promise<PaginationResponse<Exercise>> {
    const response = await apiService.get<Exercise[]>("/exercise/", params);
    return {
      data: response.data,
      total: response.data.length,
      page: params?.page || 1,
      limit: params?.limit || 10,
      totalPages: Math.ceil(response.data.length / (params?.limit || 10)),
    };
  }

  async getExerciseById(id: string): Promise<Exercise> {
    const response = await apiService.get<Exercise>(`/exercise/${id}`);
    return response.data;
  }

  async createExercise(exerciseData: FormData): Promise<Exercise> {
    const response = await apiService.uploadFile<Exercise>(
      "/exercise/",
      exerciseData
    );
    return response.data;
  }

  async updateExercise(
    id: string,
    exerciseData: FormData | Partial<Exercise>
  ): Promise<Exercise> {
    if (exerciseData instanceof FormData) {
      const response = await apiService.uploadFile<Exercise>(
        `/exercise/${id}`,
        exerciseData
      );
      return response.data;
    } else {
      const response = await apiService.put<Exercise>(
        `/exercise/${id}`,
        exerciseData
      );
      return response.data;
    }
  }

  async deleteExercise(id: string): Promise<void> {
    await apiService.delete(`/exercise/${id}`);
  }
}
