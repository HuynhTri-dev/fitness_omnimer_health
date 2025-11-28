import type { IRoleRepository } from "../../domain/repositories/role.repository";
import type {
  Role,
  Permission,
  PaginationParams,
  PaginationResponse,
} from "../../shared/types";
import { apiService } from "../services/api";

export class RoleRepositoryImpl implements IRoleRepository {
  async getRoles(params?: PaginationParams): Promise<PaginationResponse<Role>> {
    const response = await apiService.get<Role[]>("/role/", params);

    return {
      data: response.data,
      total: response.data.length,
      page: params?.page || 1,
      limit: params?.limit || 10,
      totalPages: Math.ceil(response.data.length / (params?.limit || 10)),
    };
  }

  async getRoleById(id: string): Promise<Role> {
    const response = await apiService.get<Role>(`/role/${id}`);
    return response.data;
  }

  async getRolesWithoutAdmin(): Promise<Role[]> {
    const response = await apiService.get<Role[]>("/role/without-admin");
    return response.data;
  }

  async createRole(roleData: any): Promise<Role> {
    const response = await apiService.post<Role>("/role/", roleData);
    return response.data;
  }

  async updateRole(id: string, roleData: Partial<Role>): Promise<Role> {
    const response = await apiService.put<Role>(`/role/${id}`, roleData);
    return response.data;
  }

  async updateRolePermissions(
    id: string,
    permissions: string[]
  ): Promise<Role> {
    const response = await apiService.patch<Role>(`/role/${id}`, {
      permissions,
    });
    return response.data;
  }

  async deleteRole(id: string): Promise<void> {
    await apiService.delete(`/role/${id}`);
  }
}
