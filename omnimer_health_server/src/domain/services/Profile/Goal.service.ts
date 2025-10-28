import { logError, logAudit } from "../../../utils/LoggerUtil";
import { StatusLogEnum } from "../../../common/constants/AppConstants";
import { IGoal } from "../../models";
import { HttpError } from "../../../utils/HttpError";
import { GoalRepository } from "../../repositories";
import { PaginationQueryOptions } from "../../entities";

export class GoalService {
  private readonly goalRepo: GoalRepository;

  constructor(goalRepo: GoalRepository) {
    this.goalRepo = goalRepo;
  }

  // =================== CREATE ===================
  async createGoal(userId: string, data: Partial<IGoal>) {
    try {
      const goal = await this.goalRepo.create(data);

      await logAudit({
        userId,
        action: "createGoal",
        message: `Tạo Goal "${data.userId}" thành công`,
        status: StatusLogEnum.Success,
        targetId: goal._id.toString(),
      });

      return goal;
    } catch (err: any) {
      await logError({
        userId,
        action: "createGoal",
        message: err.message || err,
        errorMessage: err.stack || err,
      });
      throw err;
    }
  }

  // =================== GET ALL ===================
  async getGoals(option?: PaginationQueryOptions) {
    try {
      return await this.goalRepo.findAll({}, option);
    } catch (err: any) {
      await logError({
        action: "getGoals",
        message: err.message || err,
        errorMessage: err.stack || err,
      });
      throw err;
    }
  }

  // =================== GET BY ID ===================
  async getGoalById(goalId: string) {
    try {
      const goal = await this.goalRepo.findById(goalId);
      if (!goal) {
        throw new HttpError(404, "Goal không tồn tại");
      }
      return goal;
    } catch (err: any) {
      await logError({
        action: "getGoalById",
        message: err.message || err,
        errorMessage: err.stack || err,
      });
      throw err;
    }
  }

  // =================== UPDATE ===================
  async updateGoal(goalId: string, data: Partial<IGoal>, userId?: string) {
    try {
      const updated = await this.goalRepo.update(goalId, data);

      await logAudit({
        userId,
        action: "updateGoal",
        message: `Cập nhật Goal "${goalId}" thành công`,
        status: StatusLogEnum.Success,
        targetId: goalId,
      });

      return updated;
    } catch (err: any) {
      await logError({
        userId,
        action: "updateGoal",
        message: err.message || err,
        errorMessage: err.stack || err,
      });
      throw err;
    }
  }

  // =================== DELETE ===================
  async deleteGoal(goalId: string, userId?: string) {
    try {
      const deleted = await this.goalRepo.delete(goalId);

      if (!deleted) {
        await logAudit({
          userId,
          action: "deleteGoal",
          message: `Goal "${goalId}" không tồn tại`,
          status: StatusLogEnum.Failure,
        });
        throw new HttpError(404, "Goal không tồn tại");
      }

      await logAudit({
        userId,
        action: "deleteGoal",
        message: `Xóa Goal "${goalId}" thành công`,
        status: StatusLogEnum.Success,
        targetId: goalId,
      });

      return deleted;
    } catch (err: any) {
      await logError({
        userId,
        action: "deleteGoal",
        message: err.message || err,
        errorMessage: err.stack || err,
      });
      throw err;
    }
  }
}
