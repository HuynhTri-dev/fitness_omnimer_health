import mongoose from "mongoose";
import { UserRepository } from "../../repositories";
import { IUser } from "../../models";
import {
  uploadToCloudflare,
  updateCloudflareImage,
  deleteFileFromCloudflare,
  uploadUserAvatar,
  updateUserAvatar,
} from "../../../utils/CloudflareUpload";
import { logError, logAudit } from "../../../utils/LoggerUtil";
import { StatusLogEnum } from "../../../common/constants/AppConstants";
import { HttpError } from "../../../utils/HttpError";
import { PaginationQueryOptions } from "../../entities";

export class UserService {
  private readonly userRepository: UserRepository;

  constructor(userRepository: UserRepository) {
    this.userRepository = userRepository;
  }

  // =================== UPDATE ===================
  async updateUser(
    userId: string,
    id: string,
    avatarImage: Express.Multer.File | undefined,
    data: Partial<IUser>
  ) {
    const session = await mongoose.startSession();
    session.startTransaction();
    try {
      const user = await this.userRepository.findById(id);
      if (!user) throw new HttpError(404, "User không tồn tại");

      let imageUrl = user.imageUrl;
      if (avatarImage) {
        if (imageUrl) {
          imageUrl = await updateUserAvatar(avatarImage, imageUrl, userId);
        } else {
          imageUrl = await uploadUserAvatar(avatarImage, user.id);
        }
      }

      const updated = await this.userRepository.updateWithSession(
        id,
        {
          ...data,
          imageUrl,
        },
        session
      );

      await session.commitTransaction();

      await logAudit({
        userId,
        action: "updateUser",
        message: `Cập nhật user "${updated?.fullname}" thành công`,
        status: StatusLogEnum.Success,
        targetId: id,
      });

      return updated;
    } catch (err: any) {
      await session.abortTransaction();
      await logError({
        userId,
        action: "updateUser",
        targetId: id,
        message: err.message || err,
        errorMessage: err.stack || err,
      });
      throw err;
    } finally {
      session.endSession();
    }
  }

  // =================== GET ALL ===================
  async getAllUsers(options?: PaginationQueryOptions) {
    try {
      const list = await this.userRepository.findAllUser(options);
      await logAudit({
        action: "getAllUsers",
        message: "Lấy danh sách users",
        status: StatusLogEnum.Success,
      });
      return list;
    } catch (err: any) {
      await logError({
        action: "getAllUsers",
        message: err.message || err,
        errorMessage: err.stack || err,
      });
      throw err;
    }
  }
}
