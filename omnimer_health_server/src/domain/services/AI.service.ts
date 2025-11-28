// src/services/AI.service.ts
import axios from "axios";
import {
  IRAGAIResponse,
  IRAGUserContext,
  IRecommendInput,
  IRecommendOutput,
  IEvaluateInput,
  IEvaluateOutput,
  IAIServiceRequest,
  IAIServiceResponse
} from "../entities/AI.entity";
import { normalizeUserContext } from "../../utils/normalizeRAGContext";

export class AIService {
  private readonly apiUrl: string;

  constructor(apiUrl: string) {
    this.apiUrl = apiUrl;
  }

  /**
   * Send user context + request to FastAPI RAG model and get exercise recommendation
   */
  async recommendExercises(context: IRAGUserContext): Promise<IRAGAIResponse> {
    try {
      const profile = normalizeUserContext(context);

      const payload = {
        profile,
        top_k: 5,
      };

      console.log("Payload sent to AI:", JSON.stringify(profile, null, 2));

      const { data } = await axios.post<IRAGAIResponse>(
        `${this.apiUrl}/recommend`,
        payload
      );

      console.log("Recommend Data Backend AI Service: ", { data });
      return data;
    } catch (err) {
      console.log(err);
      throw err;
    }
  }

  /**
   * NEW: Enhanced recommend API using v4 personal model
   * Send user profile, goals, and exercise candidates to get AI-powered recommendations
   */
  async recommendExercisesV4(input: IRecommendInput): Promise<IRecommendOutput> {
    try {
      const payload = {
        healthProfile: input.healthProfile,
        goals: input.goals,
        exercises: input.exercises,
        k: input.k || 5
      };

      console.log("Payload sent to AI v4 Recommend:", JSON.stringify(payload, null, 2));

      const { data } = await axios.post<IRecommendOutput>(
        `${this.apiUrl}/recommend`,
        payload
      );

      console.log("Recommend v4 Data Backend AI Service: ", { data });
      return data;
    } catch (err) {
      console.error("Error in recommendExercisesV4:", err);
      throw err;
    }
  }

  /**
   * NEW: Evaluate API using v4 personal model
   * Send completed workout data to get intensity and suitability evaluation
   */
  async evaluateWorkout(input: IEvaluateInput): Promise<IEvaluateOutput> {
    try {
      const payload = input;

      console.log("Payload sent to AI v4 Evaluate:", JSON.stringify(payload, null, 2));

      const { data } = await axios.post<IEvaluateOutput>(
        `${this.apiUrl}/evaluate`,
        payload
      );

      console.log("Evaluate v4 Data Backend AI Service: ", { data });
      return data;
    } catch (err) {
      console.error("Error in evaluateWorkout:", err);
      throw err;
    }
  }

  /**
   * Generic AI service method that handles both recommend and evaluate requests
   */
  async processRequest(request: IAIServiceRequest): Promise<IAIServiceResponse> {
    try {
      switch (request.type) {
        case 'recommend':
          const recommendResult = await this.recommendExercisesV4(request.data as IRecommendInput);
          return {
            type: 'recommend',
            data: recommendResult,
            success: true
          };

        case 'evaluate':
          const evaluateResult = await this.evaluateWorkout(request.data as IEvaluateInput);
          return {
            type: 'evaluate',
            data: evaluateResult,
            success: true
          };

        default:
          throw new Error(`Unsupported AI service request type: ${request.type}`);
      }
    } catch (err) {
      console.error("Error in AI service request:", err);
      return {
        type: request.type,
        data: null,
        success: false,
        message: err instanceof Error ? err.message : 'Unknown error occurred'
      };
    }
  }

  /**
   * Health check method to verify AI service availability
   */
  async healthCheck(): Promise<boolean> {
    try {
      await axios.get(`${this.apiUrl}/health`);
      return true;
    } catch (err) {
      console.error("AI service health check failed:", err);
      return false;
    }
  }

  /**
   * Get AI service info and capabilities
   */
  async getServiceInfo(): Promise<any> {
    try {
      const { data } = await axios.get(`${this.apiUrl}/info`);
      return data;
    } catch (err) {
      console.error("Failed to get AI service info:", err);
      throw err;
    }
  }
}
