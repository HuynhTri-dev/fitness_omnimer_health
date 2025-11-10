import '../entities/user_entity.dart';

abstract class AuthRepository {
  /// Login with email and password
  Future<UserEntity> login(LoginEntity loginEntity);

  /// Logout current user
  Future<void> logout();

  /// Get current user
  UserEntity? getCurrentUser();

  /// Send password reset email
  Future<void> sendPasswordResetEmail(String email);
}
