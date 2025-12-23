import { Router } from "express";
import { ExerciseCategoryController } from "../controllers";
import { ExerciseCategoryService } from "../services";
import { ExerciseCategoryRepository } from "../repositories";
import { ExerciseCategory } from "../models";
import { verifyAccessToken } from "../../common/middlewares/auth.middleware";
import { RedisService } from "../../redis/RedisService";

const exerciseCategoryRepository = new ExerciseCategoryRepository(
  ExerciseCategory
);
const exerciseCategoryService = new ExerciseCategoryService(
  exerciseCategoryRepository
);
const redisService = new RedisService();
const exerciseCategoryController = new ExerciseCategoryController(
  exerciseCategoryService,
  redisService
);
const router = Router();

router.get("/", exerciseCategoryController.getAll);
router.post("/", verifyAccessToken, exerciseCategoryController.create);
router.delete("/:id", verifyAccessToken, exerciseCategoryController.delete);
router.get("/:id", exerciseCategoryController.getById);
router.put("/:id", verifyAccessToken, exerciseCategoryController.update);

export default router;
