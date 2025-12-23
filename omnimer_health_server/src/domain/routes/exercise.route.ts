import express from "express";
import { ExerciseController } from "../controllers";
import { ExerciseService } from "../services";
import { ExerciseRepository } from "../repositories";
import { verifyAccessToken } from "../../common/middlewares/auth.middleware";
import { uploadImageAndVideo } from "../../common/middlewares/upload.middleware";
import { Exercise, ExerciseRating } from "../models";
import { RedisService } from "../../redis/RedisService";

const router = express.Router();

const repo = new ExerciseRepository(Exercise, ExerciseRating);
const service = new ExerciseService(repo);
const redisService = new RedisService();
const controller = new ExerciseController(service, redisService);

router.post(
  "/",
  verifyAccessToken,
  uploadImageAndVideo("image", "video"),
  controller.create
);
router.put(
  "/:id",
  verifyAccessToken,
  uploadImageAndVideo("image", "video"),
  controller.update
);
router.get("/", controller.getAll);

router.get("/:id", controller.getById);

router.delete("/:id", verifyAccessToken, controller.delete);

export default router;
