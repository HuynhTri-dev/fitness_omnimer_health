import { Request, Response, NextFunction } from "express";
import { ExerciseTypeService } from "../../services";
import { RedisService } from "../../../redis/RedisService";
import {
  sendSuccess,
  sendCreated,
  sendUnauthorized,
} from "../../../utils/ResponseHelper";
import { DecodePayload } from "../../entities/DecodePayload.entity";
import { buildQueryOptions } from "../../../utils/BuildQueryOptions";

export class ExerciseTypeController {
  private readonly exerciseTypeService: ExerciseTypeService;
  private readonly redisService: RedisService;

  constructor(
    exerciseTypeService: ExerciseTypeService,
    redisService: RedisService
  ) {
    this.exerciseTypeService = exerciseTypeService;
    this.redisService = redisService;
  }

  // =================== CREATE ===================
  create = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const user = req.user as DecodePayload;
      const userId = user?.id?.toString();
      if (!userId) {
        sendUnauthorized(res);
        return;
      }

      const exerciseType = await this.exerciseTypeService.createExerciseType(
        userId,
        req.body
      );

      // Invalidate cache
      await this.redisService.del("exercise_type:list");

      return sendCreated(res, exerciseType, "Tạo loại bài tập thành công");
    } catch (err) {
      next(err);
    }
  };

  // =================== GET ALL ===================
  getAll = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const cacheKey = "exercise_type:list";

      // Try reading from cache first
      const cached = await this.redisService.get(cacheKey);
      if (cached) {
        return sendSuccess(
          res,
          JSON.parse(cached),
          "Danh sách loại bài tập (Cache)"
        );
      }

      const options = buildQueryOptions(req.params as any);
      const exerciseTypes = await this.exerciseTypeService.getExerciseTypes(
        options
      );

      // Save to cache
      await this.redisService.set(
        cacheKey,
        JSON.stringify(exerciseTypes),
        24 * 60 * 60
      );

      return sendSuccess(res, exerciseTypes, "Danh sách loại bài tập");
    } catch (err) {
      next(err);
    }
  };

  // =================== GET BY ID ===================
  getById = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { id } = req.params;
      const cacheKey = `exercise_type:${id}`;

      // Try cache first
      const cached = await this.redisService.get(cacheKey);
      if (cached) {
        return sendSuccess(
          res,
          JSON.parse(cached),
          "Chi tiết loại bài tập (Cache)"
        );
      }

      const exerciseType = await this.exerciseTypeService.getExerciseTypeById(
        req.params.id
      );

      // Save to cache
      await this.redisService.set(
        cacheKey,
        JSON.stringify(exerciseType),
        24 * 60 * 60
      );

      return sendSuccess(res, exerciseType, "Chi tiết loại bài tập");
    } catch (err) {
      next(err);
    }
  };

  // =================== UPDATE ===================
  update = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const user = req.user as DecodePayload;
      const userId = user?.id?.toString();
      if (!userId) {
        sendUnauthorized(res);
        return;
      }

      const { id } = req.params;

      const exerciseType = await this.exerciseTypeService.updateExerciseType(
        id,
        req.body,
        userId
      );

      // Invalidate cache
      await this.redisService.del("exercise_type:list");
      await this.redisService.del(`exercise_type:${id}`);

      return sendSuccess(res, exerciseType, "Cập nhật loại bài tập thành công");
    } catch (err) {
      next(err);
    }
  };

  // =================== DELETE ===================
  delete = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const user = req.user as DecodePayload;
      const userId = user?.id?.toString();
      if (!userId) {
        sendUnauthorized(res);
        return;
      }

      const { id } = req.params;

      const exerciseType = await this.exerciseTypeService.deleteExerciseType(
        id,
        userId
      );

      // Invalidate cache
      await this.redisService.del("exercise_type:list");
      await this.redisService.del(`exercise_type:${id}`);

      return sendSuccess(res, exerciseType, "Xóa loại bài tập thành công");
    } catch (err) {
      next(err);
    }
  };
}
