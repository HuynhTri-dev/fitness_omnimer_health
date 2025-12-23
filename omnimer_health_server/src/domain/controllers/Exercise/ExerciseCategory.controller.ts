import { Request, Response, NextFunction } from "express";
import { ExerciseCategoryService } from "../../services";
import { RedisService } from "../../../redis/RedisService";
import {
  sendSuccess,
  sendCreated,
  sendUnauthorized,
} from "../../../utils/ResponseHelper";
import { DecodePayload } from "../../entities/DecodePayload.entity";
import { buildQueryOptions } from "../../../utils/BuildQueryOptions";

export class ExerciseCategoryController {
  private readonly exerciseCategoryService: ExerciseCategoryService;
  private readonly redisService: RedisService;

  constructor(
    exerciseCategoryService: ExerciseCategoryService,
    redisService: RedisService
  ) {
    this.exerciseCategoryService = exerciseCategoryService;
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

      const exerciseCategory =
        await this.exerciseCategoryService.createExerciseCategory(
          userId,
          req.body
        );

      // Invalidate cache
      await this.redisService.del("exercise_category:list");

      return sendCreated(res, exerciseCategory, "Tạo loại bài tập thành công");
    } catch (err) {
      next(err);
    }
  };

  // =================== GET ALL ===================
  getAll = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const cacheKey = "exercise_category:list";

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
      const exerciseCategories =
        await this.exerciseCategoryService.getExerciseCategorys(options);

      // Save to cache
      await this.redisService.set(
        cacheKey,
        JSON.stringify(exerciseCategories),
        24 * 60 * 60
      );

      return sendSuccess(res, exerciseCategories, "Danh sách loại bài tập");
    } catch (err) {
      next(err);
    }
  };

  // =================== GET BY ID ===================
  getById = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { id } = req.params;
      const cacheKey = `exercise_category:${id}`;

      // Try cache first
      const cached = await this.redisService.get(cacheKey);
      if (cached) {
        return sendSuccess(
          res,
          JSON.parse(cached),
          "Chi tiết loại bài tập (Cache)"
        );
      }

      const exerciseCategory =
        await this.exerciseCategoryService.getExerciseCategoryById(id);

      // Save to cache
      await this.redisService.set(
        cacheKey,
        JSON.stringify(exerciseCategory),
        24 * 60 * 60
      );

      return sendSuccess(res, exerciseCategory, "Chi tiết loại bài tập");
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

      const exerciseCategory =
        await this.exerciseCategoryService.updateExerciseCategory(
          id,
          req.body,
          userId
        );

      // Invalidate cache
      await this.redisService.del("exercise_category:list");
      await this.redisService.del(`exercise_category:${id}`);

      return sendSuccess(
        res,
        exerciseCategory,
        "Cập nhật loại bài tập thành công"
      );
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

      const exerciseCategory =
        await this.exerciseCategoryService.deleteExerciseCategory(id, userId);

      // Invalidate cache
      await this.redisService.del("exercise_category:list");
      await this.redisService.del(`exercise_category:${id}`);

      return sendSuccess(res, exerciseCategory, "Xóa loại bài tập thành công");
    } catch (err) {
      next(err);
    }
  };
}
