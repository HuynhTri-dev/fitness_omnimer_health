// src/routes/AI.routes.ts
import { Router } from "express";
import { AIController } from "../controllers/AI.controller";
import { AIService } from "../services";
import { verifyAccessToken } from "../../common/middlewares/auth.middleware";
import { checkPermission } from "../../common/middlewares/checkPermission";
import { HttpError } from "../../utils/HttpError";
import dotenv from "dotenv";

dotenv.config();

const router = Router();

// === Validate AI API endpoint ===
const apiUrl = process.env.AI_API;
if (!apiUrl) {
  throw new HttpError(500, "Missing AI_API environment variable");
}

// === Instantiate Services ===
const aiService = new AIService(apiUrl);
const aiController = new AIController(aiService);

/**
 * AI Routes - Handles AI-powered exercise recommendations and workout evaluations
 * Base path: /api/ai
 */

// Main AI service endpoints
router.post(
  "/recommend",
  verifyAccessToken,
  aiController.recommendExercises.bind(aiController)
);

router.post(
  "/evaluate",
  verifyAccessToken,
  aiController.evaluateWorkout.bind(aiController)
);

router.post(
  "/process",
  verifyAccessToken,
  aiController.processRequest.bind(aiController)
);

// Utility endpoints (no auth required for health/info)
router.get(
  "/health",
  aiController.healthCheck.bind(aiController)
);

router.get(
  "/info",
  aiController.getServiceInfo.bind(aiController)
);

// Legacy endpoint for backward compatibility
router.post(
  "/recommend/legacy",
  verifyAccessToken,
  aiController.recommendExercisesLegacy.bind(aiController)
);

export default router;