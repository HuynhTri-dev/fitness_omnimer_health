import axios from "axios";
import dotenv from "dotenv";

dotenv.config();

export const OLLAMA_BASE_URL =
  process.env.OLLAMA_BASE_URL || "http://host.docker.internal:11434";

/**
 * Kiểm tra kết nối tới Ollama server.
 */
export const connectOllama = async (): Promise<void> => {
  try {
    // Thường gọi root / để check xem server có sống không
    await axios.get(`${OLLAMA_BASE_URL}`);
    console.log(`✅ Ollama connected at ${OLLAMA_BASE_URL}`);
  } catch (error) {
    console.warn(
      `⚠️  Cannot connect to Ollama at ${OLLAMA_BASE_URL}. AI features may not work.`
    );
    // Không exit process vì AI có thể là optional feature
    // console.error(error);
  }
};
