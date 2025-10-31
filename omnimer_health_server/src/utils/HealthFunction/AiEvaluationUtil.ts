import axios from "axios";
import { RiskLevelEnum } from "../../common/constants/EnumConstants";

/**
 * Gọi AI Ollama (gpt-oss:120b-cloud) để đánh giá hồ sơ sức khỏe
 */
export async function callOllamaEvaluation(profileData: any) {
  try {
    const response = await axios.post("http://localhost:11434/api/generate", {
      model: "gpt-oss:120b-cloud",
      prompt: `
      Analyze the following health profile and provide a structured JSON response.
      Input data:
      ${JSON.stringify(profileData, null, 2)}

      Respond with:
      {
        "summary": "<health overview> in Vietnamese",
        "score": <0-100>,
        "riskLevel": "Low|Medium|High"
      }`,
      stream: false,
    });

    // Parse JSON trong response
    const text = response.data.response;
    const parsed = JSON.parse(text);

    return {
      summary: parsed.summary,
      score: parsed.score,
      riskLevel: parsed.riskLevel as RiskLevelEnum,
      updatedAt: new Date(),
      modelVersion: "gpt-oss:120b-cloud",
    };
  } catch (error) {
    console.error("AI Evaluation error:", error);
    return {
      summary: "AI evaluation failed.",
      score: null,
      riskLevel: RiskLevelEnum.Unknown,
      updatedAt: new Date(),
      modelVersion: "gpt-oss:120b-cloud",
    };
  }
}
