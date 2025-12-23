import express from "express";
import { MuscleController } from "../controllers";
import { MuscleService } from "../services";
import { MuscleRepository } from "../repositories";
import { verifyAccessToken } from "../../common/middlewares/auth.middleware";
import { uploadImage } from "../../common/middlewares/upload.middleware";
import { Muscle } from "../models";
import { RedisService } from "../../redis/RedisService";

const router = express.Router();

const repo = new MuscleRepository(Muscle);
const service = new MuscleService(repo);
const redisService = new RedisService();
const controller = new MuscleController(service, redisService);

router.post("/", verifyAccessToken, uploadImage("image"), controller.create);
router.put("/:id", verifyAccessToken, uploadImage("image"), controller.update);
router.get("/", controller.getAll);
router.get("/name", controller.getMuscleByName);
router.get("/:id", controller.getMuscleById);
router.delete("/:id", verifyAccessToken, controller.delete);

export default router;
