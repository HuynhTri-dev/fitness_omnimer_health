import { logError, logAudit } from "../../../utils/LoggerUtil";
import { StatusLogEnum } from "../../../common/constants/AppConstants";
import { IHealthProfile } from "../../models";
import { HttpError } from "../../../utils/HttpError";
import { HealthProfileRepository, UserRepository } from "../../repositories";
import { PaginationQueryOptions } from "../../entities";
import { calculateHealthMetrics } from "../../../utils/HealthFunction/HealthCalculationUtil";
import { callOllamaEvaluation } from "../../../utils/HealthFunction/AiEvaluationUtil";
import { Types } from "mongoose";

export class HealthProfileService {
  private readonly healthProfileRepo: HealthProfileRepository;
  private readonly userRepo: UserRepository;

  constructor(
    healthProfileRepo: HealthProfileRepository,
    userRepo: UserRepository
  ) {
    this.healthProfileRepo = healthProfileRepo;
    this.userRepo = userRepo;
  }

  // =================== CREATE ===================
  async createHealthProfile(userId: string, data: Partial<IHealthProfile>) {
    try {
      const user = await this.userRepo.findById(
        data.userId?.toString() ?? userId
      );
      const metrics = calculateHealthMetrics({
        gender: user?.gender!,
        height: data.height!,
        weight: data.weight!,
        neck: data.neck!,
        waist: data.waist!,
        hip: data.hip!,
        birthday: user?.birthday!,
      });

      // 2. Chuẩn bị dữ liệu tổng hợp gửi sang AI
      const aiInput = {
        ...data,
        ...metrics,
        healthStatus: data.healthStatus,
      };

      // 3. Gọi mô hình AI đánh giá tổng quan
      const aiEvaluation = await callOllamaEvaluation(aiInput);

      const healthProfile = await this.healthProfileRepo.create({
        ...data,
        ...metrics,
        userId: new Types.ObjectId(data.userId ?? userId),
        checkupDate: new Date(),
        aiEvaluation,
      });

      await logAudit({
        userId,
        action: "createHealthProfile",
        message: `Tạo HealthProfile "${data.userId}-${data.checkupDate}" thành công`,
        status: StatusLogEnum.Success,
        targetId: healthProfile._id.toString(),
      });

      return healthProfile;
    } catch (err: any) {
      await logError({
        userId,
        action: "createHealthProfile",
        message: err.message || err,
        errorMessage: err.stack || err,
      });
      throw err;
    }
  }

  // =================== GET ALL ===================
  async getHealthProfiles(option?: PaginationQueryOptions) {
    try {
      return await this.healthProfileRepo.findAll({}, option);
    } catch (err: any) {
      await logError({
        action: "getHealthProfiles",
        message: err.message || err,
        errorMessage: err.stack || err,
      });
      throw err;
    }
  }

  // =================== GET BY ID ===================
  async getHealthProfileById(healthProfileId: string) {
    try {
      const healthProfile = await this.healthProfileRepo.findById(
        healthProfileId
      );
      if (!healthProfile) {
        throw new HttpError(404, "HealthProfile không tồn tại");
      }
      return healthProfile;
    } catch (err: any) {
      await logError({
        action: "getHealthProfileById",
        message: err.message || err,
        errorMessage: err.stack || err,
      });
      throw err;
    }
  }

  // =================== UPDATE ===================
  async updateHealthProfile(
    healthProfileId: string,
    data: Partial<IHealthProfile>,
    userId?: string
  ) {
    try {
      // 1. Tìm HealthProfile hiện tại
      const existing = await this.healthProfileRepo.findById(healthProfileId);
      if (!existing) {
        throw new HttpError(404, "HealthProfile không tồn tại");
      }

      // 2. Lấy thông tin người dùng để tái tính toán
      const user = await this.userRepo.findById(
        data.userId?.toString() ?? existing.userId?.toString() ?? userId
      );
      if (!user) {
        throw new HttpError(404, "Người dùng không tồn tại");
      }

      // 3. Tính lại chỉ số sức khỏe (metrics)
      const metrics = calculateHealthMetrics({
        gender: user.gender!,
        height: data.height ?? existing.height!,
        weight: data.weight ?? existing.weight!,
        neck: data.neck ?? existing.neck!,
        waist: data.waist ?? existing.waist!,
        hip: data.hip ?? existing.hip!,
        birthday: user.birthday!,
      });

      // 4. Chuẩn bị dữ liệu gửi sang AI đánh giá tổng quan
      const aiInput = {
        ...(existing.toObject?.() ?? existing),
        ...data,
        ...metrics,
        healthStatus: data.healthStatus ?? existing.healthStatus,
      };

      const aiEvaluation = await callOllamaEvaluation(aiInput);

      // 5. Cập nhật vào DB
      const updated = await this.healthProfileRepo.update(healthProfileId, {
        ...data,
        ...metrics,
        aiEvaluation,
        updatedAt: new Date(),
      });

      // 6. Ghi log audit
      await logAudit({
        userId,
        action: "updateHealthProfile",
        message: `Cập nhật HealthProfile "${healthProfileId}" thành công`,
        status: StatusLogEnum.Success,
        targetId: healthProfileId,
      });

      return updated;
    } catch (err: any) {
      await logError({
        userId,
        action: "updateHealthProfile",
        message: err.message || err,
        errorMessage: err.stack || err,
      });
      throw err;
    }
  }

  // =================== DELETE ===================
  async deleteHealthProfile(healthProfileId: string, userId?: string) {
    try {
      const deleted = await this.healthProfileRepo.delete(healthProfileId);

      if (!deleted) {
        await logAudit({
          userId,
          action: "deleteHealthProfile",
          message: `HealthProfile "${healthProfileId}" không tồn tại`,
          status: StatusLogEnum.Failure,
        });
        throw new HttpError(404, "HealthProfile không tồn tại");
      }

      await logAudit({
        userId,
        action: "deleteHealthProfile",
        message: `Xóa HealthProfile "${healthProfileId}" thành công`,
        status: StatusLogEnum.Success,
        targetId: healthProfileId,
      });

      return deleted;
    } catch (err: any) {
      await logError({
        userId,
        action: "deleteHealthProfile",
        message: err.message || err,
        errorMessage: err.stack || err,
      });
      throw err;
    }
  }
}
