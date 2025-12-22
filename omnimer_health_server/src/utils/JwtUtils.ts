import jwt, { SignOptions, Secret } from "jsonwebtoken";
import { StringValue } from "ms";
import crypto from "crypto";

if (!process.env.ACCESS_TOKEN_SECRET) {
  throw new Error(
    "ACCESS_TOKEN_SECRET is not defined in environment variables"
  );
}
if (!process.env.REFRESH_TOKEN_SECRET) {
  throw new Error(
    "REFRESH_TOKEN_SECRET is not defined in environment variables"
  );
}
if (!process.env.VERIFICATION_TOKEN_SECRET) {
  throw new Error(
    "VERIFICATION_TOKEN_SECRET is not defined in environment variables"
  );
}

const ACCESS_TOKEN_SECRET: Secret = process.env.ACCESS_TOKEN_SECRET;
const REFRESH_TOKEN_SECRET: Secret = process.env.REFRESH_TOKEN_SECRET;
const VERIFICATION_TOKEN_SECRET: Secret = process.env.VERIFICATION_TOKEN_SECRET;

export const JwtUtils = {
  generateAccessToken(payload: any, expiresIn: StringValue | number = "1h") {
    const options: SignOptions = { expiresIn };
    return jwt.sign(payload, ACCESS_TOKEN_SECRET, options);
  },

  generateRefreshToken(payload: any, expiresIn: StringValue | number = "1y") {
    const options: SignOptions = { expiresIn };
    return jwt.sign(payload, REFRESH_TOKEN_SECRET, options);
  },

  verifyAccessToken(token: string) {
    return jwt.verify(token, ACCESS_TOKEN_SECRET);
  },

  verifyRefreshToken(token: string) {
    return jwt.verify(token, REFRESH_TOKEN_SECRET);
  },

  /**
   * Tạo verification token cho email/phone verification
   * @param payload - Dữ liệu cần mã hóa (userId, email, etc.)
   * @param expiresIn - Thời gian hết hạn (mặc định 24h)
   */
  generateVerificationToken(
    payload: any,
    expiresIn: StringValue | number = "24h"
  ) {
    const options: SignOptions = { expiresIn };
    return jwt.sign(payload, VERIFICATION_TOKEN_SECRET, options);
  },

  /**
   * Xác thực verification token
   */
  verifyVerificationToken(token: string) {
    return jwt.verify(token, VERIFICATION_TOKEN_SECRET);
  },

  /**
   * Tạo random token (không dùng JWT, dùng crypto)
   * Dùng cho các trường hợp cần token ngắn gọn
   */
  generateRandomToken(length: number = 32): string {
    return crypto.randomBytes(length).toString("hex");
  },
};
