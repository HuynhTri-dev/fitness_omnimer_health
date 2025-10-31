import { FilterQuery, Model, Types } from "mongoose";
import { IWorkout } from "../../models";
import { BaseRepository } from "../Base.repository";
import { PaginationQueryOptions } from "../../entities";

export class WorkoutRepository extends BaseRepository<IWorkout> {
  constructor(model: Model<IWorkout>) {
    super(model);
  }

  /**
   * Lấy chi tiết WorkoutTemplate theo ID, bao gồm populate các reference.
   * @param id - ObjectId hoặc string của template
   */
  async getWorkoutById(id: string | Types.ObjectId) {
    return this.model
      .findById(id)
      .populate([
        { path: "workouttemplates", select: "_id name" },
        {
          path: "workOutDetail.exerciseId",
          select: "_id name difficulty met",
        },
      ])
      .lean()
      .exec();
  }

  /**
   * Tìm tất cả bản ghi, có thể truyền query filter, phân trang, sort.
   * @param filter - điều kiện tìm kiếm (optional)
   * @param options - optional: limit, page, sort
   */
  async findAllWorkout(
    filter: FilterQuery<IWorkout> = {},
    options?: PaginationQueryOptions
  ): Promise<IWorkout[]> {
    try {
      const page = options?.page ?? 1;
      const limit = options?.limit ?? 20;
      const skip = (page - 1) * limit;
      const sort = options?.sort ?? { createdAt: -1 };

      const finalFilter = {
        ...filter,
        ...(options?.filter || {}),
      };

      // Chỉ lấy các trường quan trọng
      const projection = "userId workoutTemplateId date summary";

      return this.model
        .find(finalFilter, projection)
        .populate([{ path: "workouttemplates", select: "_id name" }])
        .skip(skip)
        .limit(limit)
        .sort(sort)
        .exec();
    } catch (e) {
      throw e;
    }
  }

  async updateSetDone(
    workoutId: string,
    workoutDetailId: string,
    workoutSetId: string,
    done: boolean
  ) {
    return this.model.updateOne(
      {
        _id: new Types.ObjectId(workoutId),
      },
      {
        $set: {
          "workoutDetail.$[detail].sets.$[set].done": done,
        },
      },
      {
        arrayFilters: [
          { "detail._id": new Types.ObjectId(workoutDetailId) },
          { "set._id": new Types.ObjectId(workoutSetId) },
        ],
      }
    );
  }

  async updateExerciseInfo(
    workoutId: string,
    workoutDetailId: string,
    durationMin: number,
    deviceData?: any
  ) {
    const updateFields: any = {
      "workoutDetail.$[detail].durationMin": durationMin,
    };

    if (deviceData) {
      updateFields["workoutDetail.$[detail].deviceData"] = deviceData;
    }

    return this.model.updateOne(
      { _id: new Types.ObjectId(workoutId) },
      { $set: updateFields },
      {
        arrayFilters: [{ "detail._id": new Types.ObjectId(workoutDetailId) }],
      }
    );
  }
}
