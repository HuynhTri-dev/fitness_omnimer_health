import 'package:omnihealthmobileflutter/core/api/api_client.dart';
import 'package:omnihealthmobileflutter/core/api/endpoints.dart';
import 'package:omnihealthmobileflutter/data/models/user_model.dart';
import 'package:omnihealthmobileflutter/services/firebase_auth_service.dart';

abstract class AuthDataSource {
  Future<UserModel> login(LoginRequestModel loginRequest);
  Future<void> logout();
}

class AuthDataSourceImpl implements AuthDataSource {
  final ApiClient apiClient;
  final FirebaseAuthService firebaseAuth;

  AuthDataSourceImpl({required this.apiClient, required this.firebaseAuth});

  @override
  Future<UserModel> login(LoginRequestModel loginRequest) async {
    // 1. Login với Firebase để lấy token
    final firebaseToken = await firebaseAuth.signInAndGetToken(
      loginRequest.email,
      loginRequest.password,
    );

    // 2. Gọi API backend với firebase token
    final response = await apiClient.post<UserModel>(
      Endpoints.login,
      data: {...loginRequest.toJson(), 'firebase_token': firebaseToken},
      parser: (json) => UserModel.fromJson(json),
    );

    if (!response.success || response.data == null) {
      throw Exception(response.message);
    }

    return response.data!;
  }

  @override
  Future<void> logout() async {
    await firebaseAuth.signOut();
    await apiClient.post(Endpoints.logout);
  }
}
