// src/controllers/AI.controller.ts
import { Request, Response, NextFunction } from "express";
import { AIService } from "../services/AI.service";
import {
  sendSuccess,
  sendError,
  sendBadRequest,
} from "../../utils/ResponseHelper";
import {
  IRecommendInput,
  IRecommendOutput,
  IEvaluateInput,
  IEvaluateOutput,
  IAIServiceRequest,
  IAIServiceResponse,
} from "../entities/AI.entity";

export class AIController {
  private aiService: AIService;

  constructor(aiService: AIService) {
    this.aiService = aiService;
  }

  /**
   * POST /api/ai/recommend
   * Get AI-powered exercise recommendations based on user profile and goals
   */
  async recommendExercises(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const recommendInput: IRecommendInput = req.body;

      // Validate required fields
      if (
        !recommendInput.healthProfile ||
        !recommendInput.goals ||
        !recommendInput.exercises
      ) {
        sendBadRequest(
          res,
          "Missing required fields: healthProfile, goals, exercises"
        );
        return;
      }

      // Call AI service
      const result: IRecommendOutput =
        await this.aiService.recommendExercisesV4(recommendInput);

      sendSuccess(
        res,
        result,
        "Exercise recommendations generated successfully"
      );
    } catch (error) {
      next(error);
    }
  }

  /**
   * POST /api/ai/evaluate
   * Evaluate completed workout session and get intensity/suitability scores
   */
  async evaluateWorkout(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const evaluateInput: IEvaluateInput = req.body;

      // Validate required fields
      if (!evaluateInput.healthProfile || !evaluateInput.workoutDetail) {
        sendError(res, "Missing required fields: healthProfile, workoutDetail");
        return;
      }

      // Call AI service
      const result: IEvaluateOutput = await this.aiService.evaluateWorkout(
        evaluateInput
      );

      sendSuccess(res, result, "Workout evaluation completed successfully");
    } catch (error) {
      next(error);
    }
  }

  /**
   * POST /api/ai/process
   * Generic AI service processor that handles both recommend and evaluate requests
   */
  async processRequest(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const request: IAIServiceRequest = req.body;

      if (!request.type || !request.data) {
        sendBadRequest(res, "Missing required fields: type, data");
        return;
      }

      // Call AI service
      const result: IAIServiceResponse = await this.aiService.processRequest(
        request
      );

      if (result.success) {
        sendSuccess(
          res,
          result.data,
          `AI ${result.type} request completed successfully`
        );
      } else {
        sendError(res, result.message || `AI ${result.type} request failed`);
      }
    } catch (error) {
      next(error);
    }
  }

  /**
   * GET /api/ai/health
   * Health check endpoint for AI service
   */
  async healthCheck(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const isHealthy = await this.aiService.healthCheck();

      if (isHealthy) {
        sendSuccess(res, { status: "healthy" }, "AI service is healthy");
      } else {
        sendError(res, "AI service is unhealthy", 503);
      }
    } catch (error) {
      next(error);
    }
  }

  /**
   * GET /api/ai/info
   * Get AI service information and capabilities
   */
  async getServiceInfo(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const info = await this.aiService.getServiceInfo();
      sendSuccess(res, info, "AI service information retrieved successfully");
    } catch (error) {
      next(error);
    }
  }

  /**
   * POST /api/ai/recommend/legacy
   * Legacy RAG-based recommendation endpoint (for backward compatibility)
   */
  async recommendExercisesLegacy(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const context = req.body;

      if (!context.healthProfile) {
        sendBadRequest(res, "Missing required field: healthProfile");
        return;
      }

      // Call legacy AI service
      const result = await this.aiService.recommendExercises(context);

      sendSuccess(
        res,
        result,
        "Legacy exercise recommendations generated successfully"
      );
    } catch (error) {
      next(error);
    }
  }
}
