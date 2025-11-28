import type {
  ApiResponse,
  PaginationParams,
  ErrorResponse,
} from "../../shared/types";

export class ApiService {
  private baseURL: string;
  private accessToken: string | null = null;

  constructor(baseURL: string = "http://localhost:5000/api/v1") {
    this.baseURL = baseURL;
    this.accessToken = localStorage.getItem("accessToken");
  }

  setAccessToken(token: string) {
    this.accessToken = token;
    localStorage.setItem("accessToken", token);
  }

  clearAccessToken() {
    this.accessToken = null;
    localStorage.removeItem("accessToken");
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`;

    const headers: HeadersInit = {
      "Content-Type": "application/json",
      ...options.headers,
    };

    if (this.accessToken) {
      headers.Authorization = `Bearer ${this.accessToken}`;
    }

    const config: RequestInit = {
      ...options,
      headers,
    };

    try {
      const response = await fetch(url, config);
      const data = await response.json();

      if (!response.ok) {
        const errorResponse: ErrorResponse = {
          error: data.error || "Something went wrong",
          statusCode: response.status,
        };
        throw errorResponse;
      }

      return data;
    } catch (error) {
      if (error instanceof Error) {
        throw {
          error: error.message,
          statusCode: 500,
        };
      }
      throw error;
    }
  }

  // Generic GET request
  async get<T>(
    endpoint: string,
    params?: PaginationParams
  ): Promise<ApiResponse<T>> {
    const url = new URL(endpoint, this.baseURL);

    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          url.searchParams.append(key, value.toString());
        }
      });
    }

    return this.request<T>(url.pathname + url.search);
  }

  // Generic POST request
  async post<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: "POST",
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  // Generic PUT request
  async put<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: "PUT",
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  // Generic PATCH request
  async patch<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: "PATCH",
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  // Generic DELETE request
  async delete<T>(endpoint: string): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: "DELETE",
    });
  }

  // File upload request
  async uploadFile<T>(
    endpoint: string,
    formData: FormData
  ): Promise<ApiResponse<T>> {
    const headers: HeadersInit = {};

    if (this.accessToken) {
      headers.Authorization = `Bearer ${this.accessToken}`;
    }

    return this.request<T>(endpoint, {
      method: "POST",
      headers,
      body: formData,
    });
  }
}

export const apiService = new ApiService();
