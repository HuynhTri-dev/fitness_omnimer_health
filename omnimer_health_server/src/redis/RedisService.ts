import { RedisClientType } from "redis";
import { connectRedis } from "../common/configs/redisConnect";

export class RedisService {
  private client: RedisClientType | null = null;

  constructor() {
    this.init();
  }

  private async init() {
    try {
      this.client = await connectRedis();
    } catch (e) {
      console.error("Failed to connect to Redis in RedisService init", e);
    }
  }

  private async getClient(): Promise<RedisClientType> {
    if (!this.client || !this.client.isReady) {
      this.client = await connectRedis();
    }
    return this.client;
  }

  /**
   * Get value from Redis
   * @param key
   * @returns string | null
   */
  public async get(key: string): Promise<string | null> {
    const client = await this.getClient();
    try {
      return await client.get(key);
    } catch (error) {
      console.error(`Redis Get Error for key ${key}:`, error);
      return null;
    }
  }

  /**
   * Set value to Redis with optional TTL
   * @param key
   * @param value
   * @param ttlSeconds
   */
  public async set(
    key: string,
    value: string,
    ttlSeconds?: number
  ): Promise<void> {
    const client = await this.getClient();
    try {
      if (ttlSeconds) {
        await client.set(key, value, { EX: ttlSeconds });
      } else {
        await client.set(key, value);
      }
    } catch (error) {
      console.error(`Redis Set Error for key ${key}:`, error);
    }
  }

  /**
   * Delete key from Redis
   * @param key
   */
  public async del(key: string): Promise<void> {
    const client = await this.getClient();
    try {
      await client.del(key);
    } catch (error) {
      console.error(`Redis Del Error for key ${key}:`, error);
    }
  }

  /**
   * Delete keys matching pattern
   * @param pattern e.g. "exercise:*"
   */
  public async delPattern(pattern: string): Promise<void> {
    const client = await this.getClient();
    try {
      const keys = await client.keys(pattern);
      if (keys.length > 0) {
        await client.del(keys);
      }
    } catch (error) {
      console.error(`Redis DelPattern Error for pattern ${pattern}:`, error);
    }
  }
}
