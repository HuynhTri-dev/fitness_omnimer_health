import type { IAuthRepository } from "../../domain/repositories/auth.repository";
import type { User } from "../../shared/types";
import { apiService } from "../services/api";

export class AuthRepositoryImpl implements IAuthRepository {
  async login(email: string, password: string) {
    const response = await apiService.post("/auth/login", {
      email,
      password,
    });

    // Store tokens in api service
    if (response.data.accessToken) {
      apiService.setAccessToken(response.data.accessToken);
    }

    return response.data;
  }

  async register(userData: any) {
    const response = await apiService.post("/auth/register", userData);

    // Store tokens in api service
    if (response.data.accessToken) {
      apiService.setAccessToken(response.data.accessToken);
    }

    return response.data;
  }

  async getCurrentUser(): Promise<User> {
    const response = await apiService.get("/auth/");
    return response.data;
  }

  async refreshToken() {
    const response = await apiService.post("/auth/new-access-token", {
      refreshToken: localStorage.getItem("refreshToken"),
    });

    // Update tokens in api service
    if (response.data.accessToken) {
      apiService.setAccessToken(response.data.accessToken);
    }

    return response.data;
  }

  async logout(): Promise<void> {
    await apiService.post("/auth/logout");
    apiService.clearAccessToken();
    localStorage.removeItem("refreshToken");
  }
}
