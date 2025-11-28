import type { IPermissionRepository } from "../../domain/repositories/permission.repository";
import type {
  Permission,
  PaginationParams,
  PaginationResponse,
} from "../../shared/types";
import { apiService } from "../services/api";

export class PermissionRepositoryImpl implements IPermissionRepository {
  async getPermissions(
    params?: PaginationParams
  ): Promise<PaginationResponse<Permission>> {
    const response = await apiService.get<Permission[]>("/permission/", params);

    return {
      data: response.data,
      total: response.data.length,
      page: params?.page || 1,
      limit: params?.limit || 10,
      totalPages: Math.ceil(response.data.length / (params?.limit || 10)),
    };
  }

  async getPermissionById(id: string): Promise<Permission> {
    const response = await apiService.get<Permission>(`/permission/${id}`);
    return response.data;
  }

  async createPermission(permissionData: any): Promise<Permission> {
    const response = await apiService.post<Permission>(
      "/permission/",
      permissionData
    );
    return response.data;
  }

  async updatePermission(
    id: string,
    permissionData: Partial<Permission>
  ): Promise<Permission> {
    const response = await apiService.put<Permission>(
      `/permission/${id}`,
      permissionData
    );
    return response.data;
  }

  async deletePermission(id: string): Promise<void> {
    await apiService.delete(`/permission/${id}`);
  }
}
