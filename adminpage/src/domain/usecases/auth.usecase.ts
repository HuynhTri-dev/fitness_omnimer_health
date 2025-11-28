import type { IAuthRepository } from "../repositories/auth.repository";
import type { User } from "../../shared/types";

export class AuthUseCase {
  constructor(private authRepository: IAuthRepository) {}

  async login(email: string, password: string) {
    const response = await this.authRepository.login(email, password);
    return response;
  }

  async register(userData: any) {
    const response = await this.authRepository.register(userData);
    return response;
  }

  async getCurrentUser() {
    return await this.authRepository.getCurrentUser();
  }

  async refreshToken() {
    const response = await this.authRepository.refreshToken();
    return response;
  }

  async logout() {
    await this.authRepository.logout();
  }

  isAuthenticated(): boolean {
    const token = localStorage.getItem("accessToken");
    return !!token;
  }

  getStoredToken(): string | null {
    return localStorage.getItem("accessToken");
  }
}
