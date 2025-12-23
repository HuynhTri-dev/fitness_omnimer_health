import { Request, Response, NextFunction } from "express";
import { ExerciseService } from "../../services";
import { RedisService } from "../../../redis/RedisService";
import {
  sendSuccess,
  sendCreated,
  sendUnauthorized,
} from "../../../utils/ResponseHelper";
import { DecodePayload } from "../../entities";
import { buildQueryOptions } from "../../../utils/BuildQueryOptions";

export class ExerciseController {
  private readonly exerciseService: ExerciseService;
  private readonly redisService: RedisService;

  constructor(exerciseService: ExerciseService, redisService: RedisService) {
    this.exerciseService = exerciseService;
    this.redisService = redisService;
  }

  // =================== CREATE ===================
  create = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const user = req.user as DecodePayload;
      const userId = user?.id?.toString();
      if (!userId) return sendUnauthorized(res);

      const files = req.files as
        | Record<string, Express.Multer.File[]>
        | undefined;
      const imageFiles = files?.image; // lấy tất cả ảnh
      const videoFile = files?.video?.[0];

      const exercise = await this.exerciseService.createExercise(
        userId,
        imageFiles,
        videoFile,
        req.body
      );

      // Invalidate cache
      await this.redisService.delPattern("exercise:list*");

      return sendCreated(res, exercise, "Create exercise success");
    } catch (err) {
      next(err);
    }
  };

  // =================== UPDATE ===================
  update = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const user = req.user as DecodePayload;
      const userId = user?.id?.toString();
      if (!userId) return sendUnauthorized(res);

      const { id } = req.params;
      const files = req.files as
        | Record<string, Express.Multer.File[]>
        | undefined;
      const imageFiles = files?.image;
      const videoFile = files?.video?.[0];

      const updated = await this.exerciseService.updateExercise(
        userId,
        id,
        imageFiles,
        videoFile,
        req.body
      );

      // Invalidate cache
      await this.redisService.delPattern("exercise:list*");
      await this.redisService.del(`exercise:${id}`);

      return sendSuccess(res, updated, "Update exercise success");
    } catch (err) {
      next(err);
    }
  };

  // =================== GET ALL ===================
  getAll = async (req: Request, res: Response, next: NextFunction) => {
    try {
      // Use query params hash or string for cache key
      const queryString = JSON.stringify(req.query);
      // Simple hash function for shorter keys (optional but good practice)
      // For simplicity here, just using base64 of query string or the string itself if short
      const queryHash = Buffer.from(queryString).toString("base64");
      const cacheKey = `exercise:list:${queryHash}`;

      const cached = await this.redisService.get(cacheKey);
      if (cached) {
        return sendSuccess(res, JSON.parse(cached));
      }

      const options = buildQueryOptions(req.query as any);
      const list = await this.exerciseService.getAllExercises(options);

      // 1 hour TTL for search results
      await this.redisService.set(cacheKey, JSON.stringify(list), 60 * 60);

      return sendSuccess(res, list);
    } catch (err) {
      next(err);
    }
  };

  // =================== GET BY ID ===================
  getById = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const id = req.params.id;
      const cacheKey = `exercise:${id}`;

      const cached = await this.redisService.get(cacheKey);
      if (cached) {
        return sendSuccess(res, JSON.parse(cached));
      }

      const exercise = await this.exerciseService.getExerciseById(id);

      await this.redisService.set(
        cacheKey,
        JSON.stringify(exercise),
        24 * 60 * 60
      );

      return sendSuccess(res, exercise);
    } catch (err) {
      next(err);
    }
  };

  // =================== DELETE ===================
  delete = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const user = req.user as DecodePayload;
      const userId = user?.id?.toString();
      if (!userId) return sendUnauthorized(res);

      const { id } = req.params;
      await this.exerciseService.deleteExercise(userId, id);

      // Invalidate cache
      await this.redisService.delPattern("exercise:list*");
      await this.redisService.del(`exercise:${id}`);

      return sendSuccess(res, true, "Delete exercise success");
    } catch (err) {
      next(err);
    }
  };
}
