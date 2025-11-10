import 'package:omnihealthmobileflutter/data/datasources/auth_datasource.dart';
import 'package:omnihealthmobileflutter/data/models/user_model.dart';
import 'package:omnihealthmobileflutter/domain/abstracts/auth_repository.dart';
import 'package:omnihealthmobileflutter/domain/entities/user_entity.dart';
import 'package:omnihealthmobileflutter/services/firebase_auth_service.dart';

class AuthRepositoryImpl implements AuthRepository {
  final AuthDataSource authDataSource;
  final FirebaseAuthService firebaseAuth;

  AuthRepositoryImpl({
    required this.authDataSource,
    required this.firebaseAuth,
  });

  @override
  Future<UserEntity> login(LoginEntity loginEntity) async {
    try {
      // Convert entity to model
      final loginModel = LoginRequestModel.fromEntity(loginEntity);

      // Call API through data source
      final userModel = await authDataSource.login(loginModel);

      // Convert model back to entity
      return userModel.toEntity();
    } catch (e) {
      // Re-throw with meaningful message
      throw Exception('Đăng nhập thất bại: ${e.toString()}');
    }
  }

  @override
  Future<void> logout() async {
    await authDataSource.logout();
  }

  @override
  UserEntity? getCurrentUser() {
    final firebaseUser = firebaseAuth.getCurrentUser();
    if (firebaseUser == null) return null;

    return UserEntity(
      email: firebaseUser.email ?? '',
      displayName: firebaseUser.displayName,
      photoUrl: firebaseUser.photoURL,
      phoneNumber: firebaseUser.phoneNumber,
    );
  }

  @override
  Future<void> sendPasswordResetEmail(String email) async {
    await firebaseAuth.sendPasswordResetEmail(email);
  }
}
