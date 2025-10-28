import { Request, Response, NextFunction } from "express";
import { HealthProfileService } from "../../services";
import {
  sendSuccess,
  sendCreated,
  sendUnauthorized,
} from "../../../utils/ResponseHelper";
import { DecodePayload } from "../../entities";
import { buildQueryOptions } from "../../../utils/BuildQueryOptions";

export class HealthProfileController {
  private readonly healthProfileService: HealthProfileService;

  constructor(healthProfileService: HealthProfileService) {
    this.healthProfileService = healthProfileService;
  }

  // =================== CREATE ===================
  create = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const user = req.user as DecodePayload;
      const userId = user?.id?.toString();
      if (!userId) return sendUnauthorized(res);

      const created = await this.healthProfileService.createHealthProfile(
        userId,
        req.body
      );

      return sendCreated(res, created, "Tạo hồ sơ sức khỏe thành công");
    } catch (err) {
      next(err);
    }
  };

  // =================== GET ALL ===================
  getAll = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const options = buildQueryOptions(req.query as any);

      const list = await this.healthProfileService.getHealthProfiles(options);

      return sendSuccess(res, list, "Lấy danh sách hồ sơ sức khỏe thành công");
    } catch (err) {
      next(err);
    }
  };

  // =================== GET BY ID ===================
  getById = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { id } = req.params;

      const item = await this.healthProfileService.getHealthProfileById(id);

      return sendSuccess(res, item, "Lấy chi tiết hồ sơ sức khỏe thành công");
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

      const updated = await this.healthProfileService.updateHealthProfile(
        id,
        req.body,
        userId
      );

      return sendSuccess(res, updated, "Cập nhật hồ sơ sức khỏe thành công");
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

      const deleted = await this.healthProfileService.deleteHealthProfile(
        id,
        userId
      );

      return sendSuccess(res, deleted, "Xóa hồ sơ sức khỏe thành công");
    } catch (err) {
      next(err);
    }
  };
}
