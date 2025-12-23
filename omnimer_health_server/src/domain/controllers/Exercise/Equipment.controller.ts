import { Request, Response, NextFunction } from "express";
import { EquipmentService } from "../../services";
import { RedisService } from "../../../redis/RedisService";
import {
  sendSuccess,
  sendCreated,
  sendUnauthorized,
} from "../../../utils/ResponseHelper";
import { DecodePayload } from "../../entities/DecodePayload.entity";
import { buildQueryOptions } from "../../../utils/BuildQueryOptions";

export class EquipmentController {
  private readonly equipmentController: EquipmentService;
  private readonly redisService: RedisService;

  constructor(
    equipmentController: EquipmentService,
    redisService: RedisService
  ) {
    this.equipmentController = equipmentController;
    this.redisService = redisService;
  }

  // =================== CREATE ===================
  create = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const user = req.user as DecodePayload;
      const userId = user?.id?.toString();
      if (!userId) return sendUnauthorized(res);

      const file = req.file;
      const bodyPart = await this.equipmentController.createEquipment(
        userId,
        file,
        req.body
      );

      // Invalidate cache
      await this.redisService.del("equipment:list");

      return sendCreated(res, bodyPart, "Tạo thiết bị thành công");
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

      const updated = await this.equipmentController.updateEquipment(
        userId,
        id,
        file,
        req.body
      );

      // Invalidate cache
      await this.redisService.del("equipment:list");

      return sendSuccess(res, updated, "Cập nhật thiết bị thành công");
    } catch (err) {
      next(err);
    }
  };

  // =================== GET ALL ===================
  getAll = async (req: Request, res: Response, next: NextFunction) => {
    try {
      // Check cache only if no filters/options are present (simple list)
      const hasOptions = Object.keys(req.query).length > 0;
      const cacheKey = "equipment:list";

      if (!hasOptions) {
        const cachedData = await this.redisService.get(cacheKey);
        if (cachedData) {
          return sendSuccess(
            res,
            JSON.parse(cachedData),
            "Lấy danh sách thiết bị thành công (Cache)"
          );
        }
      }

      const options = buildQueryOptions(req.query as any);
      const list = await this.equipmentController.getAllEquipments(options);

      // Set cache if it was a simple list request
      if (!hasOptions) {
        await this.redisService.set(
          cacheKey,
          JSON.stringify(list),
          24 * 60 * 60
        ); // 24 hours
      }

      return sendSuccess(res, list, "Lấy danh sách thiết bị thành công");
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
      await this.equipmentController.deleteEquipment(userId, id);

      // Invalidate cache
      await this.redisService.del("equipment:list");

      return sendSuccess(res, true, "Xoá thiết bị thành công");
    } catch (err) {
      next(err);
    }
  };
}
