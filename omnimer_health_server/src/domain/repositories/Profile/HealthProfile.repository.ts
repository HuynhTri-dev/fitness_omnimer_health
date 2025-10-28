import { Model } from "mongoose";
import { IHealthProfile } from "../../models";
import { BaseRepository } from "../Base.repository";

export class HealthProfileRepository extends BaseRepository<IHealthProfile> {
  constructor(model: Model<IHealthProfile>) {
    super(model);
  }
}
