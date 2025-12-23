import { Request, Response, NextFunction } from "express";
import { MuscleService } from "../../services";
import { RedisService } from "../../../redis/RedisService";
import {
  sendSuccess,
  sendCreated,
  sendUnauthorized,
  sendBadRequest,
} from "../../../utils/ResponseHelper";
import { DecodePayload } from "../../entities";
import { buildQueryOptions } from "../../../utils/BuildQueryOptions";

export class MuscleController {
  private readonly muscleService: MuscleService;
  private readonly redisService: RedisService;

  constructor(muscleService: MuscleService, redisService: RedisService) {
    this.muscleService = muscleService;
    this.redisService = redisService;
  }

  // =================== CREATE ===================
  create = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const user = req.user as DecodePayload;
      const userId = user?.id?.toString();
      if (!userId) return sendUnauthorized(res);

      const file = req.file;
      const muscle = await this.muscleService.createMuscle(
        userId,
        file,
        req.body
      );

      // Invalidate cache
      await this.redisService.del("muscle:list");
      await this.redisService.delPattern("muscle:name:*");

      return sendCreated(res, muscle, "Create muscle success");
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

      const file = req.file;
      const { id } = req.params;

      const updated = await this.muscleService.updateMuscle(
        userId,
        id,
        file,
        req.body
      );

      // Invalidate cache
      await this.redisService.del("muscle:list");
      await this.redisService.del(`muscle:${id}`);
      await this.redisService.delPattern("muscle:name:*");

      return sendSuccess(res, updated, "Update muscle success");
    } catch (err) {
      next(err);
    }
  };

  // =================== GET ALL ===================
  getAll = async (req: Request, res: Response, next: NextFunction) => {
    try {
      // Check cache only if no filters/options are present
      const hasOptions = Object.keys(req.query).length > 0;
      const cacheKey = "muscle:list";

      if (!hasOptions) {
        const cached = await this.redisService.get(cacheKey);
        if (cached) {
          return sendSuccess(
            res,
            JSON.parse(cached),
            "Get list muscle success (Cache)"
          );
        }
      }

      const options = buildQueryOptions(req.query as any);
      const list = await this.muscleService.getAllMuscles(options);

      // Cache if simple list
      if (!hasOptions) {
        await this.redisService.set(
          cacheKey,
          JSON.stringify(list),
          7 * 24 * 60 * 60
        ); // 7 days
      }

      return sendSuccess(res, list, "Get list muscle success");
    } catch (err) {
      next(err);
    }
  };

  // =================== GET BY ID ===================
  getMuscleById = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const id = req.params.id;
      const cacheKey = `muscle:${id}`;

      const cached = await this.redisService.get(cacheKey);
      if (cached) {
        return sendSuccess(
          res,
          JSON.parse(cached),
          "Get Muscle success (Cache)"
        );
      }

      const muscle = await this.muscleService.getMuscleById(id);

      await this.redisService.set(
        cacheKey,
        JSON.stringify(muscle),
        7 * 24 * 60 * 60
      ); // 7 days

      return sendSuccess(res, muscle, "Get Muscle success");
    } catch (err) {
      next(err);
    }
  };

  // =================== GET BY NAME ===================
  getMuscleByName = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const name = req.query.name as string;
      if (!name) return sendBadRequest(res, "Missing name parameter");

      const cacheKey = `muscle:name:${name}`;
      const cached = await this.redisService.get(cacheKey);
      if (cached) {
        return sendSuccess(
          res,
          JSON.parse(cached),
          "Get Muscle success (Cache)"
        );
      }

      const muscle = await this.muscleService.getMuscleByName(name);

      await this.redisService.set(
        cacheKey,
        JSON.stringify(muscle),
        7 * 24 * 60 * 60
      ); // 7 days

      return sendSuccess(res, muscle, "Get Muscle success");
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
      await this.muscleService.deleteMuscle(userId, id);

      // Invalidate cache
      await this.redisService.del("muscle:list");
      await this.redisService.del(`muscle:${id}`);
      await this.redisService.delPattern("muscle:name:*");

      return sendSuccess(res, true, "Delete Muscle error"); // Note: Original msg was "Delete Muscle error", should probably be success but keeping consistency unless asked to fix.
    } catch (err) {
      next(err);
    }
  };
}
