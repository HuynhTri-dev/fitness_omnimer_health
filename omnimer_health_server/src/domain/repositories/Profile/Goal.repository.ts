import { Model } from "mongoose";
import { IGoal } from "../../models";
import { BaseRepository } from "../Base.repository";

export class GoalRepository extends BaseRepository<IGoal> {
  constructor(model: Model<IGoal>) {
    super(model);
  }
}
