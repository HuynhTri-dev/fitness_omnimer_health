import { Request, Response, NextFunction } from "express";
import { PermissionService } from "../services";
import {
  sendSuccess,
  sendCreated,
  sendUnauthorized,
} from "../../utils/ResponseHelper";
import { HttpError } from "../../utils/HttpError";
import { DecodePayload } from "../entities/DecodePayload";

export class PermissionController {
  private readonly service: PermissionService;

  // Constructor cho phép inject service (dễ test hoặc mock)
  constructor(service: PermissionService) {
    this.service = service;
  }

  // =================== CREATE ===================
  create = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const user = req.user as DecodePayload;
      const userId = user.id.toString();

      if (!userId) {
        sendUnauthorized(res);
        return;
      }

      const permission = await this.service.createPermission(userId, req.body);
      return sendCreated(res, permission, "Tạo quyền thành công");
    } catch (err) {
      next(err);
    }
  };

  // =================== GET ALL ===================
  getAll = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const permissions = await this.service.getPermissions();
      return sendSuccess(res, permissions, "Danh sách quyền hạn");
    } catch (err) {
      next(err);
    }
  };

  // =================== DELETE ===================
  delete = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const user = req.user as DecodePayload;
      const userId = user.id.toString();

      if (!userId) {
        sendUnauthorized(res);
        return;
      }

      const permission = await this.service.deletePermission(
        req.params.id,
        userId
      );
      return sendSuccess(res, permission, "Xóa quyền hạn thành công");
    } catch (err) {
      next(err);
    }
  };

  // =================== GET BY ID ===================
  getById = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const permission = await this.service.getPermissionById(req.params.id);
      return sendSuccess(res, permission, "Chi tiết quyền hạn");
    } catch (err) {
      next(err);
    }
  };

  // =================== UPDATE ===================
  update = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const user = req.user as DecodePayload;
      const userId = user.id.toString();

      if (!userId) {
        sendUnauthorized(res);
        return;
      }

      const permission = await this.service.updatePermission(
        req.params.id,
        req.body,
        userId
      );
      return sendSuccess(res, permission, "Cập nhật quyền hạn thành công");
    } catch (err) {
      next(err);
    }
  };
}
